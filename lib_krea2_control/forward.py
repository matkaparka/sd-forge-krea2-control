"""
Krea 2 control-reference forward passes for Forge Neo.

Mirrors ``backend/nn/krea.py :: SingleStreamDiT.forward`` and only changes how the
control (reference) latent enters the sequence. Three methods:

JOINT      Reference tokens are appended to the sequence and modulated with the SAME
           timestep as the target. This is how NK2E in-context LoRAs (canny) are trained.
           Equivalent to Forge's native ``dynamic_args.ref_latents`` path.

JOINT_T0   Reference tokens are appended and modulated at t=0 ("index_timestep_zero").
           Ostris-edit non-cached path.

CACHED_T0  Reference tokens run ONCE through the blocks at t=0, attending only to each
           other; every block's post-RoPE K/V is recorded and appended as extra keys in
           each denoising step. Ostris-edit ``kv_cache`` path; the official workflow of the
           thedeoxen OpenPose LoRA enables it.

Several controls at once (OpenPose + Canny): ``control_forward`` takes a list of ``(latent, method)`` pairs.
Every reference keeps its own method: tokens of JOINT references (current t) and JOINT_T0 references (t=0) are
appended to the sequence, CACHED_T0 references contribute cached K/V as extra keys. ``frames[i]`` is the RoPE
frame index of reference i (default: all at frame 1, the index each control LoRA was trained with on its own;
Forge's native multi-reference path uses 1, 2, 3, ...).

Ported from:
  https://github.com/Haoming02/sd-webui-forge-classic (neo)  backend/nn/krea.py   (AGPL-3.0)
  https://github.com/ostris/comfyui-krea2-ostris-edit        nodes.py             (MIT)
"""

import torch
import torch.nn.functional as F
from einops import rearrange

from backend.attention import attention_function
from backend.misc.image_resize import adaptive_resize
from backend.nn.flux import timestep_embedding
from backend.quant_ops import ck
from backend.utils import pad_to_patch_size

JOINT = "joint"
JOINT_T0 = "joint_t0"
CACHED_T0 = "cached_t0"
METHODS = (JOINT, JOINT_T0, CACHED_T0)


def _grid_ids(bs: int, frame: int, h: int, w: int, device) -> torch.Tensor:
    ids = torch.zeros(h, w, 3, device=device, dtype=torch.float32)
    ids[..., 0] = frame
    ids[..., 1] = torch.arange(h, device=device, dtype=torch.float32)[:, None]
    ids[..., 2] = torch.arange(w, device=device, dtype=torch.float32)[None, :]
    return ids.reshape(1, h * w, 3).repeat(bs, 1, 1)


def _patchify(x: torch.Tensor, patch: int) -> torch.Tensor:
    return rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch, pw=patch)


def _time(dm, timesteps: torch.Tensor, dtype: torch.dtype):
    t = dm.tmlp(timestep_embedding(timesteps, dm.tdim).unsqueeze(1).to(dtype))
    return t, dm.tproj(t)


def _attention(attn, x, freqs, transformer_options, capture=None, extra_kv=None):
    """``Attention.forward`` with optional K/V capture and extra (cached) K/V.
    Both happen post-RoPE and before the GQA head expansion."""
    q, k, v, gate = attn.wq(x), attn.wk(x), attn.wv(x), attn.gate(x)
    q = rearrange(q, "B L (H D) -> B H L D", H=attn.heads)
    k = rearrange(k, "B L (H D) -> B H L D", H=attn.kvheads)
    v = rearrange(v, "B L (H D) -> B H L D", H=attn.kvheads)
    q, k = attn.qknorm(q, k)
    if freqs is not None:
        q, k = ck.apply_rope(q, k, freqs)
    if capture is not None:
        capture.append((k, v))
    if extra_kv is not None:
        rk, rv = extra_kv
        rk = rk.to(device=k.device, dtype=k.dtype, non_blocking=True)
        rv = rv.to(device=v.device, dtype=v.dtype, non_blocking=True)
        if rk.shape[0] != k.shape[0]:
            rk = rk.expand(k.shape[0], -1, -1, -1)
            rv = rv.expand(v.shape[0], -1, -1, -1)
        k = torch.cat((k, rk), dim=2)
        v = torch.cat((v, rv), dim=2)
    if attn.kvheads != attn.heads:
        rep = attn.heads // attn.kvheads
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
    out = attention_function(q, k, v, attn.heads, mask=None, skip_reshape=True, transformer_options=transformer_options)
    return attn.wo(out * F.sigmoid(gate))


def _block(block, x, vec, freqs, transformer_options, capture=None, extra_kv=None):
    """``SingleStreamBlock.forward`` (single modulation span) with K/V capture / injection."""
    prescale, preshift, pregate, postscale, postshift, postgate = block.mod(vec)
    x.addcmul_(pregate, _attention(block.attn, torch.addcmul(preshift, 1 + prescale, block.prenorm(x)), freqs, transformer_options, capture, extra_kv))
    x.addcmul_(postgate, block.mlp(torch.addcmul(postshift, 1 + postscale, block.postnorm(x))))
    return x


def _block_split(block, x, vec, vec0, split, freqs, transformer_options, extra_kv=None):
    """``SingleStreamBlock.forward`` with two modulation spans:
    tokens [:split] (text + target + current-t references) use ``vec``, tokens [split:] (t=0 references) use ``vec0``.
    ``extra_kv``: cached reference K/V appended as extra keys (see ``_attention``)."""
    m = block.mod(vec)
    r = block.mod(vec0)

    def mod(h, scale, shift):
        return torch.cat((torch.addcmul(m[shift], 1 + m[scale], h[:, :split]), torch.addcmul(r[shift], 1 + r[scale], h[:, split:])), dim=1)

    def gate(h, g):
        return torch.cat((m[g] * h[:, :split], r[g] * h[:, split:]), dim=1)

    modulated = mod(block.prenorm(x), 0, 1)
    if extra_kv is None:
        attn_out = block.attn(modulated, freqs, None, transformer_options=transformer_options)
    else:
        attn_out = _attention(block.attn, modulated, freqs, transformer_options, extra_kv=extra_kv)
    x = x + gate(attn_out, 2)
    x = x + gate(block.mlp(mod(block.postnorm(x), 3, 4)), 5)
    return x


def _reference_kv(dm, ref_tok, refpos, timesteps, dtype, transformer_options, offload: bool):
    """Run only the clean reference tokens through every block at t=0 and record
    each block's post-RoPE K/V (batch 1)."""
    h = dm.first(ref_tok)
    _, vec0 = _time(dm, torch.zeros_like(timesteps[:1]), h.dtype)
    freqs = dm.pe_embedder(refpos)
    kvs = []
    last = len(dm.blocks) - 1
    for i, block in enumerate(dm.blocks):
        cap = []
        if i < last:
            h = _block(block, h, vec0, freqs, transformer_options, capture=cap)
        else:  # the last block's output is never used, only its K/V
            prescale, preshift, *_ = block.mod(vec0)
            _attention(block.attn, torch.addcmul(preshift, 1 + prescale, block.prenorm(h)), freqs, transformer_options, capture=cap)
        k, v = cap[0]
        if offload:
            k, v = k.to("cpu"), v.to("cpu")
        kvs.append((k, v))
    return kvs


class _Ref:
    """One control reference, prepared for the sequence."""

    def __init__(self, latent, method, frame):
        self.latent, self.method, self.frame = latent, method, frame
        self.tok = None  # (1, Lr, C*p*p), patchified
        self.pos = None  # (1, Lr, 3), RoPE ids


def _normalize_refs(ref, method, frames):
    """-> [(latent, method)], frames: a lone tensor with ``method``, or an iterable of (latent, method) pairs."""
    refs = [(ref, method)] if torch.is_tensor(ref) else [tuple(r) for r in ref]
    if not refs:
        raise ValueError("control_forward needs at least one reference")
    for _, m in refs:
        if m not in METHODS:
            raise ValueError(f"Unknown reference method: {m}")
    frames = [1] * len(refs) if frames is None else list(frames)
    if len(frames) != len(refs):
        raise ValueError(f"{len(refs)} references but {len(frames)} frame indices")
    return refs, frames


def control_forward(dm, x, timesteps, context, transformer_options, ref, method=None, kv_store=None, offload_kv=False, frames=None):
    """
    dm       : Forge ``SingleStreamDiT``
    x        : (B, C, 1, H, W) or (B, C, H, W) noisy latent
    ref      : (1, C, h, w) or (1, C, 1, h, w) control latent, already ``process_in``-ed, together with ``method``;
               or a list of (latent, method) pairs to apply several controls at once
    method   : JOINT | JOINT_T0 | CACHED_T0 (per reference when ``ref`` is a list)
    kv_store : dict, reused across the steps of one sampling run (CACHED_T0 references only)
    frames   : RoPE frame index per reference (default: all 1)
    """
    refs, frames = _normalize_refs(ref, method, frames)

    was5d = x.ndim == 5
    if was5d:
        x = x.squeeze(2)
    bs, _, H0, W0 = x.shape
    patch = dm.patch
    device = x.device

    x = pad_to_patch_size(x, (patch, patch))
    H, W = x.shape[-2], x.shape[-1]
    h_, w_ = H // patch, W // patch

    entries = []
    for (latent, m), frame in zip(refs, frames):
        e = _Ref(latent.to(device=device, dtype=x.dtype), m, frame)
        if e.latent.ndim == 5:
            e.latent = e.latent.squeeze(2)
        e.latent = e.latent[:1]
        if e.latent.shape[-2:] != (H, W):
            e.latent = adaptive_resize(e.latent, W, H, "area", "center")
        e.latent = pad_to_patch_size(e.latent, (patch, patch), padding_mode="replicate")
        rh, rw = e.latent.shape[-2] // patch, e.latent.shape[-1] // patch
        e.tok = _patchify(e.latent, patch)
        e.pos = _grid_ids(1, frame, rh, rw, device)  # RoPE axis-0 index = frame, own grid from 0
        entries.append(e)

    cached = [e for e in entries if e.method == CACHED_T0]
    current = [e for e in entries if e.method == JOINT]
    at_zero = [e for e in entries if e.method == JOINT_T0]

    img = dm.first(_patchify(x, patch))
    t, vec = _time(dm, timesteps, img.dtype)

    ctx = dm.txtfusion(context, mask=None, transformer_options=transformer_options)
    ctx = dm.txtmlp(ctx)

    txtlen, imglen = ctx.shape[1], img.shape[1]
    txtpos = torch.zeros(bs, txtlen, 3, device=device, dtype=torch.float32)
    imgpos = _grid_ids(bs, 0, h_, w_, device)

    kv = None
    if cached:
        if kv_store is None:
            kv_store = {}
        key = tuple((tuple(e.latent.shape), str(e.latent.dtype), str(device), e.frame) for e in cached)
        if kv_store.get("key") != key:
            per_ref = [_reference_kv(dm, e.tok, e.pos, timesteps, img.dtype, transformer_options, offload_kv) for e in cached]
            if len(per_ref) == 1:
                kv_store["kv"] = per_ref[0]
            else:  # each reference ran on its own (isolated); their K/V are simply laid side by side
                kv_store["kv"] = [(torch.cat([r[b][0] for r in per_ref], dim=2), torch.cat([r[b][1] for r in per_ref], dim=2)) for b in range(len(dm.blocks))]
            kv_store["key"] = key
        kv = kv_store["kv"]

    # tokens appended to the sequence: current-t references first, then the t=0 ones (they get their own modulation span)
    joint = current + at_zero
    combined = torch.cat((ctx, img, *[dm.first(e.tok).expand(bs, -1, -1) for e in joint]), dim=1)
    freqs = dm.pe_embedder(torch.cat((txtpos, imgpos, *[e.pos.expand(bs, -1, -1) for e in joint]), dim=1))

    if at_zero:
        _, vec0 = _time(dm, torch.zeros_like(timesteps), img.dtype)
        split = txtlen + imglen + sum(e.tok.shape[1] for e in current)
        for i, block in enumerate(dm.blocks):
            combined = _block_split(block, combined, vec, vec0, split, freqs, transformer_options, kv[i] if kv is not None else None)
    elif kv is not None:
        for block, block_kv in zip(dm.blocks, kv):
            combined = _block(block, combined, vec, freqs, transformer_options, extra_kv=block_kv)
    else:
        for block in dm.blocks:
            combined = block(combined, vec, freqs, None, transformer_options=transformer_options)

    out = dm.last(combined[:, txtlen : txtlen + imglen, :], t)
    out = rearrange(out, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=h_, w=w_, ph=patch, pw=patch, c=dm.channels)
    out = out[:, :, :H0, :W0]
    return out.unsqueeze(2) if was5d else out


# ---------------------------------------------------------------------------
# Global state + forward hook
# ---------------------------------------------------------------------------


class _State:
    def __init__(self):
        self.reset()

    def reset(self):
        self.active = False
        self.refs = []  # [(latent, method)]
        self.frames = None
        self.kv = {}
        self.offload_kv = False
        self.calls = 0

    def arm(self, ref, method=None, offload_kv=False, frames=None):
        """One control: ``arm(latent, method, offload)``. Several: ``arm([(latent, method), ...], None, offload, frames)``."""
        self.refs, self.frames = _normalize_refs(ref, method, frames)
        self.offload_kv = offload_kv
        self.kv = {}
        self.calls = 0
        self.active = True

    @property
    def ref(self):  # first reference, kept for callers of the single-control API
        return self.refs[0][0] if self.refs else None

    @property
    def method(self):
        return self.refs[0][1] if self.refs else None


STATE = _State()

_ORIG_ATTR = "_krea2_control_original_forward"


def install_hook(dm):
    """Wrap ``dm.forward`` once per model instance; inert unless STATE is armed."""
    if _ORIG_ATTR in dm.__dict__:
        return
    original = dm.forward

    def forward(x, timesteps, context, attention_mask=None, transformer_options={}, **kwargs):
        st = STATE
        if st.active and st.refs:
            st.calls += 1
            return control_forward(dm, x, timesteps, context, transformer_options, st.refs, None, st.kv, st.offload_kv, st.frames)
        return original(x, timesteps, context, attention_mask=attention_mask, transformer_options=transformer_options, **kwargs)

    dm.__dict__[_ORIG_ATTR] = original
    dm.forward = forward
