"""CPU numerical check of the three reference methods, alone and combined (OpenPose + Canny at once), against Forge Neo's own Krea2 model code.

    pip install torch einops rich gguf comfy-kitchen    (CPU builds are fine)
    FORGE_DIR=/path/to/sd-webui-forge-classic python tests/test_forward.py
"""
import os, sys, itertools
sys.argv = ["x", "--cpu"]
FORGE = os.environ.get("FORGE_DIR")
assert FORGE, "set FORGE_DIR to your Forge Neo checkout (neo branch)"
sys.path.insert(0, FORGE)
sys.path.insert(0, os.path.join(FORGE, 'modules_forge', 'packages'))  # vendored gguf, as modules_forge/initialization.py does
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn.functional as F
from einops import rearrange
import backend.nn.krea as K
from backend.args import dynamic_args
from backend.nn.flux import timestep_embedding
from backend.quant_ops import ck
from lib_krea2_control import forward as KF

torch.manual_seed(0)
dm = K.SingleStreamDiT(features=64, tdim=32, txtdim=16, heads=4, kvheads=2, multiplier=2, layers=3,
                       patch=2, channels=4, txtlayers=3, txtheads=2, txtkvheads=2).eval()
with torch.no_grad():
    for p in dm.parameters():
        torch.nn.init.normal_(p, std=0.08)

def native(x, t, ctx, refs=None):
    dynamic_args.ref_latents = list(refs) if refs else []
    try:
        return K.SingleStreamDiT.forward(dm, x.clone(), t, ctx.clone())
    finally:
        dynamic_args.ref_latents = []

# ---- independent reference: explicit per-token modulation + boolean attention mask ----
def reference(x, t, ctx, ref, t0_refs, isolate):
    was5d = x.ndim == 5
    if was5d: x = x.squeeze(2)
    bs, c, H, W = x.shape; P = dm.patch; h_, w_ = H // P, W // P
    img = dm.first(rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=P, pw=P))
    rtok = dm.first(rearrange(ref.expand(bs, -1, -1, -1), "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=P, pw=P))
    tt = dm.tmlp(timestep_embedding(t, dm.tdim).unsqueeze(1)); vec = dm.tproj(tt)
    t0 = dm.tmlp(timestep_embedding(torch.zeros_like(t), dm.tdim).unsqueeze(1)); vec0 = dm.tproj(t0)
    c2 = dm.txtmlp(dm.txtfusion(ctx.clone(), mask=None))
    L_t, L_i, L_r = c2.shape[1], img.shape[1], rtok.shape[1]
    seq = torch.cat((c2, img, rtok), 1); L = seq.shape[1]
    def ids(frame, n_h, n_w):
        g = torch.zeros(n_h, n_w, 3); g[..., 0] = frame
        g[..., 1] = torch.arange(n_h)[:, None].float(); g[..., 2] = torch.arange(n_w)[None, :].float()
        return g.reshape(1, -1, 3).repeat(bs, 1, 1)
    pos = torch.cat((torch.zeros(bs, L_t, 3), ids(0, h_, w_), ids(1, h_, w_)), 1)
    freqs = dm.pe_embedder(pos)
    isref = torch.zeros(L, dtype=torch.bool); isref[L_t + L_i:] = True
    allowed = torch.ones(L, L, dtype=torch.bool)
    if isolate: allowed[isref] = isref  # ref queries see only ref keys
    tokvec = torch.where(isref[None, :, None], (vec0 if t0_refs else vec).expand(bs, L, -1), vec.expand(bs, L, -1))
    for blk in dm.blocks:
        ps, psh, pg, qs, qsh, qg = tokvec.chunk(6, -1)
        lin = blk.mod.lin
        ps, psh, pg, qs, qsh, qg = [a + b for a, b in zip((ps, psh, pg, qs, qsh, qg), lin.chunk(6, -1))]
        hN = psh + (1 + ps) * blk.prenorm(seq)
        a = blk.attn
        q = rearrange(a.wq(hN), "B L (H D) -> B H L D", H=a.heads)
        k = rearrange(a.wk(hN), "B L (H D) -> B H L D", H=a.kvheads)
        v = rearrange(a.wv(hN), "B L (H D) -> B H L D", H=a.kvheads)
        q, k = a.qknorm(q, k); q, k = ck.apply_rope(q, k, freqs)
        rep = a.heads // a.kvheads; k = k.repeat_interleave(rep, 1); v = v.repeat_interleave(rep, 1)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed)
        o = rearrange(o, "B H L D -> B L (H D)")
        seq = seq + pg * a.wo(o * torch.sigmoid(a.gate(hN)))
        seq = seq + qg * blk.mlp(qsh + (1 + qs) * blk.postnorm(seq))
    out = dm.last(seq[:, L_t:L_t + L_i], tt)
    out = rearrange(out, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=h_, w=w_, ph=P, pw=P, c=dm.channels)
    return out.unsqueeze(2) if was5d else out

def reference_multi(x, t, ctx, refs, frames):
    """Independent implementation for any mix of methods (explicit per-token modulation + boolean attention mask).
    JOINT: current t, sees everything. JOINT_T0: t=0, sees everything. CACHED_T0: t=0 and isolated (its queries see only
    its own keys, so it is the same as running it alone), while every other token can attend to its keys."""
    was5d = x.ndim == 5
    if was5d: x = x.squeeze(2)
    bs, c, H, W = x.shape; P = dm.patch; h_, w_ = H // P, W // P
    img = dm.first(rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=P, pw=P))
    toks = [dm.first(rearrange(r.expand(bs, -1, -1, -1), "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=P, pw=P)) for r, _ in refs]
    tt = dm.tmlp(timestep_embedding(t, dm.tdim).unsqueeze(1)); vec = dm.tproj(tt)
    t0 = dm.tmlp(timestep_embedding(torch.zeros_like(t), dm.tdim).unsqueeze(1)); vec0 = dm.tproj(t0)
    c2 = dm.txtmlp(dm.txtfusion(ctx.clone(), mask=None))
    L_t, L_i = c2.shape[1], img.shape[1]
    seq = torch.cat((c2, img, *toks), 1); L = seq.shape[1]
    def ids(frame, n_h, n_w):
        g = torch.zeros(n_h, n_w, 3); g[..., 0] = frame
        g[..., 1] = torch.arange(n_h)[:, None].float(); g[..., 2] = torch.arange(n_w)[None, :].float()
        return g.reshape(1, -1, 3).repeat(bs, 1, 1)
    pos = torch.cat((torch.zeros(bs, L_t, 3), ids(0, h_, w_), *[ids(f, h_, w_) for f in frames]), 1)
    freqs = dm.pe_embedder(pos)
    allowed = torch.ones(L, L, dtype=torch.bool)
    use_t0 = torch.zeros(L, dtype=torch.bool)
    start = L_t + L_i
    for tk, (_, m) in zip(toks, refs):
        a, b = start, start + tk.shape[1]; start = b
        if m in (KF.JOINT_T0, KF.CACHED_T0): use_t0[a:b] = True
        if m == KF.CACHED_T0:
            allowed[a:b, :] = False; allowed[a:b, a:b] = True
    tokvec = torch.where(use_t0[None, :, None], vec0.expand(bs, L, -1), vec.expand(bs, L, -1))
    for blk in dm.blocks:
        ps, psh, pg, qs, qsh, qg = tokvec.chunk(6, -1)
        lin = blk.mod.lin
        ps, psh, pg, qs, qsh, qg = [a + b for a, b in zip((ps, psh, pg, qs, qsh, qg), lin.chunk(6, -1))]
        hN = psh + (1 + ps) * blk.prenorm(seq)
        a = blk.attn
        q = rearrange(a.wq(hN), "B L (H D) -> B H L D", H=a.heads)
        k = rearrange(a.wk(hN), "B L (H D) -> B H L D", H=a.kvheads)
        v = rearrange(a.wv(hN), "B L (H D) -> B H L D", H=a.kvheads)
        q, k = a.qknorm(q, k); q, k = ck.apply_rope(q, k, freqs)
        rep = a.heads // a.kvheads; k = k.repeat_interleave(rep, 1); v = v.repeat_interleave(rep, 1)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed)
        o = rearrange(o, "B H L D -> B L (H D)")
        seq = seq + pg * a.wo(o * torch.sigmoid(a.gate(hN)))
        seq = seq + qg * blk.mlp(qsh + (1 + qs) * blk.postnorm(seq))
    out = dm.last(seq[:, L_t:L_t + L_i], tt)
    out = rearrange(out, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=h_, w=w_, ph=P, pw=P, c=dm.channels)
    return out.unsqueeze(2) if was5d else out

def mk(bs, H=8, W=12, T=5, five=True):
    x = torch.randn(bs, 4, 1, H, W) if five else torch.randn(bs, 4, H, W)
    ctx = torch.randn(bs, T, 3, 16)
    ref = torch.randn(1, 4, H, W)
    t = torch.rand(bs)
    return x, t, ctx, ref

def close(a, b, name, tol=2e-4):
    err = (a - b).abs().max().item()
    ok = err < tol
    print(f"{'PASS' if ok else 'FAIL'}  {name:55s} max|diff|={err:.2e}")
    assert ok

with torch.no_grad():
    for bs, five in itertools.product((1, 2), (True, False)):
        x, t, ctx, ref = mk(bs, five=five)
        tag = f"bs={bs} {'5D' if five else '4D'}"
        # 1) JOINT == Forge native dynamic_args.ref_latents path
        if bs <= 2:
            nat = native(x, t, ctx, [ref.clone()])
            close(KF.control_forward(dm, x.clone(), t, ctx.clone(), {}, ref, KF.JOINT), nat if five else nat.squeeze(2), f"JOINT == Forge native refs [{tag}]")
        # 2) three methods vs independent reference
        close(KF.control_forward(dm, x.clone(), t, ctx.clone(), {}, ref, KF.JOINT), reference(x, t, ctx, ref, False, False), f"JOINT == reference(t, joint) [{tag}]")
        close(KF.control_forward(dm, x.clone(), t, ctx.clone(), {}, ref, KF.JOINT_T0), reference(x, t, ctx, ref, True, False), f"JOINT_T0 == reference(t0, joint) [{tag}]")
        store = {}
        c1 = KF.control_forward(dm, x.clone(), t, ctx.clone(), {}, ref, KF.CACHED_T0, store)
        close(c1, reference(x, t, ctx, ref, True, True), f"CACHED_T0 == reference(t0, isolated) [{tag}]")
        # cache reuse on a second step with a different t gives the same as a fresh computation
        t2 = torch.rand(bs)
        c2 = KF.control_forward(dm, x.clone(), t2, ctx.clone(), {}, ref, KF.CACHED_T0, store)
        close(c2, reference(x, t2, ctx, ref, True, True), f"CACHED_T0 reused K/V at new t [{tag}]")
        c3 = KF.control_forward(dm, x.clone(), t2, ctx.clone(), {}, ref, KF.CACHED_T0, {}, offload_kv=True)
        close(c3, c2, f"CACHED_T0 offload_kv [{tag}]", 1e-6)

    # 3) odd latent size -> padding path agrees with native
    x, t, ctx, ref = mk(1, H=7, W=9)
    close(KF.control_forward(dm, x.clone(), t, ctx.clone(), {}, ref, KF.JOINT), native(x, t, ctx, [ref.clone()]), "JOINT == native, odd latent 7x9 (padding/resize)")
    # 4) ref at a different size gets resized like native
    x, t, ctx, _ = mk(1, H=8, W=12); ref_small = torch.randn(1, 4, 4, 6)
    close(KF.control_forward(dm, x.clone(), t, ctx.clone(), {}, ref_small, KF.JOINT), native(x, t, ctx, [ref_small.clone()]), "JOINT == native, ref resized 4x6 -> 8x12")

    # 5) hook: inactive -> identical to native; armed -> control path
    KF.install_hook(dm); KF.install_hook(dm)
    x, t, ctx, ref = mk(2)
    close(dm(x.clone(), t, ctx.clone()), native(x, t, ctx), "hook inert when STATE disarmed", 1e-7)
    KF.STATE.arm(ref, KF.CACHED_T0, False)
    close(dm(x.clone(), t, ctx.clone()), reference(x, t, ctx, ref, True, True), "hook armed -> CACHED_T0")
    assert KF.STATE.calls == 1
    KF.STATE.reset()
    close(dm(x.clone(), t, ctx.clone()), native(x, t, ctx), "hook inert after reset", 1e-7)
# ---- several controls at once: any mix of methods, each reference keeps its own -------------------------------------
COMBOS = {
    "CACHED_T0 + JOINT   (OpenPose default + Canny)": lambda a, b: [(a, KF.CACHED_T0), (b, KF.JOINT)],
    "JOINT_T0 + JOINT    (OpenPose joint t=0 + Canny)": lambda a, b: [(a, KF.JOINT_T0), (b, KF.JOINT)],
    "JOINT + JOINT": lambda a, b: [(a, KF.JOINT), (b, KF.JOINT)],
    "CACHED_T0 + JOINT_T0": lambda a, b: [(a, KF.CACHED_T0), (b, KF.JOINT_T0)],
    "JOINT_T0 + CACHED_T0": lambda a, b: [(a, KF.JOINT_T0), (b, KF.CACHED_T0)],
    "CACHED_T0 + CACHED_T0": lambda a, b: [(a, KF.CACHED_T0), (b, KF.CACHED_T0)],
    "JOINT_T0 + JOINT_T0": lambda a, b: [(a, KF.JOINT_T0), (b, KF.JOINT_T0)],
}

def mk2(bs, H=8, W=12, five=True):
    x, t, ctx, a = mk(bs, H, W, five=five)
    return x, t, ctx, a, torch.randn(1, 4, H, W)

with torch.no_grad():
    for name, make in COMBOS.items():
        worst, n = 0.0, 0
        for bs, five, frames in itertools.product((1, 2), (True, False), ([1, 1], [1, 2])):
            x, t, ctx, a, b = mk2(bs, five=five)
            refs = make(a, b)
            out = KF.control_forward(dm, x.clone(), t, ctx.clone(), {}, refs, frames=frames)
            err = (out - reference_multi(x, t, ctx, refs, frames)).abs().max().item()
            worst, n = max(worst, err), n + 1
        print(f"{'PASS' if worst < 2e-4 else 'FAIL'}  {name:55s} max|diff|={worst:.2e} over {n} settings (bs 1/2, 5D/4D, frames same/separate)")
        assert worst < 2e-4

    # two JOINT references at frames 1, 2 == Forge's native multi-reference path
    for bs in (1, 2):
        x, t, ctx, a, b = mk2(bs)
        close(KF.control_forward(dm, x.clone(), t, ctx.clone(), {}, [(a, KF.JOINT), (b, KF.JOINT)], frames=[1, 2]), native(x, t, ctx, [a.clone(), b.clone()]), f"two JOINT refs @ frames 1,2 == Forge native refs [bs={bs}]")

    # a list with one reference takes exactly the single-reference path
    x, t, ctx, a, b = mk2(1)
    for m in KF.METHODS:
        single = KF.control_forward(dm, x.clone(), t, ctx.clone(), {}, a, m, {})
        as_list = KF.control_forward(dm, x.clone(), t, ctx.clone(), {}, [(a, m)], None, {})
        assert torch.equal(single, as_list), m
    print("PASS  one-element list == single-reference call (all three methods, bit-exact)")

    # K/V cache with several references: reuse at a new t, invalidation when frames change, offload
    x, t, ctx, a, b = mk2(2)
    refs = COMBOS["CACHED_T0 + CACHED_T0"](a, b)
    store = {}
    KF.control_forward(dm, x.clone(), t, ctx.clone(), {}, refs, None, store, frames=[1, 1])
    t2 = torch.rand(2)
    close(KF.control_forward(dm, x.clone(), t2, ctx.clone(), {}, refs, None, store, frames=[1, 1]), reference_multi(x, t2, ctx, refs, [1, 1]), "two cached refs: K/V reused at a new t")
    close(KF.control_forward(dm, x.clone(), t2, ctx.clone(), {}, refs, None, store, frames=[1, 2]), reference_multi(x, t2, ctx, refs, [1, 2]), "...and rebuilt when the frame indices change (no stale cache)")
    close(KF.control_forward(dm, x.clone(), t2, ctx.clone(), {}, refs, None, {}, True, frames=[1, 2]), reference_multi(x, t2, ctx, refs, [1, 2]), "two cached refs: offload_kv", 1e-5)

    # the references really matter, and so does their position
    x, t, ctx, a, b = mk2(1)
    both = KF.control_forward(dm, x.clone(), t, ctx.clone(), {}, [(a, KF.JOINT), (b, KF.JOINT)], frames=[1, 1])
    only_a = KF.control_forward(dm, x.clone(), t, ctx.clone(), {}, a, KF.JOINT)
    apart = KF.control_forward(dm, x.clone(), t, ctx.clone(), {}, [(a, KF.JOINT), (b, KF.JOINT)], frames=[1, 2])
    assert (both - only_a).abs().max() > 1e-3 and (both - apart).abs().max() > 1e-3
    print(f"PASS  both references change the output ({(both - only_a).abs().max():.3f}) and so does their frame index ({(both - apart).abs().max():.3f})")

    # hook with several references
    KF.install_hook(dm)
    x, t, ctx, a, b = mk2(2)
    refs = COMBOS["CACHED_T0 + JOINT   (OpenPose default + Canny)"](a, b)
    KF.STATE.arm(refs, None, False, [1, 1])
    close(dm(x.clone(), t, ctx.clone()), reference_multi(x, t, ctx, refs, [1, 1]), "hook armed with two references")
    assert KF.STATE.calls == 1 and KF.STATE.ref is a and KF.STATE.method == KF.CACHED_T0
    KF.STATE.reset()
    close(dm(x.clone(), t, ctx.clone()), native(x, t, ctx), "hook inert after reset (multi)", 1e-7)
    KF.STATE.arm(a, KF.JOINT, False)  # single-control API still works
    close(dm(x.clone(), t, ctx.clone()), reference(x, t, ctx, a, False, False), "hook armed the old way (one latent + method)")
    KF.STATE.reset()

    # bad input is refused loudly
    for bad, what in (([(a, "nope")], "unknown method"), ([], "no references")):
        try:
            KF.control_forward(dm, x.clone(), t, ctx.clone(), {}, bad)
            raise AssertionError(what)
        except ValueError:
            pass
    try:
        KF.control_forward(dm, x.clone(), t, ctx.clone(), {}, [(a, KF.JOINT)], frames=[1, 2])
        raise AssertionError("frames length")
    except ValueError:
        pass
    print("PASS  unknown method / no references / wrong number of frame indices raise ValueError")
print("MULTI-REFERENCE TESTS PASSED")

print("ALL TESTS PASSED")

with torch.no_grad():
    x, t, ctx, ref = mk(1)
    a = KF.control_forward(dm, x.clone(), t, ctx.clone(), {}, ref, KF.JOINT)
    b = KF.control_forward(dm, x.clone(), t, ctx.clone(), {}, ref, KF.JOINT_T0)
    c = KF.control_forward(dm, x.clone(), t, ctx.clone(), {}, ref, KF.CACHED_T0, {})
    n = native(x, t, ctx)
    print("methods really differ:", f"JOINT vs JOINT_T0 {(a-b).abs().max():.3f}", f"JOINT_T0 vs CACHED {(b-c).abs().max():.3f}", f"ref vs no-ref {(a-n).abs().max():.3f}")
