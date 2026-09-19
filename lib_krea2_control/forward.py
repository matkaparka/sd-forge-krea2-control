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


def _block_split(block, x, vec, vec0, split, freqs, transformer_options):
    """``SingleStreamBlock.forward`` with two modulation spans:
    tokens [:split] (text + target) use ``vec``, tokens [split:] (reference) use ``vec0``."""
    m = block.mod(vec)
    r = block.mod(vec0)

    def mod(h, scale, shift):
        return torch.cat((torch.addcmul(m[shift], 1 + m[scale], h[:, :split]), torch.addcmul(r[shift], 1 + r[scale], h[:, split:])), dim=1)

    def gate(h, g):
        return torch.cat((m[g] * h[:, :split], r[g] * h[:, split:]), dim=1)

    x = x + gate(block.attn(mod(block.prenorm(x), 0, 1), freqs, None, transformer_options=transformer_options), 2)
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


def control_forward(dm, x, timesteps, context, transformer_options, ref, method, kv_store=None, offload_kv=False):
    """
    dm       : Forge ``SingleStreamDiT``
    x        : (B, C, 1, H, W) or (B, C, H, W) noisy latent
    ref      : (1, C, h, w) or (1, C, 1, h, w) control latent, already ``process_in``-ed
    method   : JOINT | JOINT_T0 | CACHED_T0
    kv_store : dict, reused across the steps of one sampling run (CACHED_T0 only)
    """
    was5d = x.ndim == 5
    if was5d:
        x = x.squeeze(2)
    bs, _, H0, W0 = x.shape
    patch = dm.patch
    device = x.device

    x = pad_to_patch_size(x, (patch, patch))
    H, W = x.shape[-2], x.shape[-1]
    h_, w_ = H // patch, W // patch

    ref = ref.to(device=device, dtype=x.dtype)
    if ref.ndim == 5:
        ref = ref.squeeze(2)
    ref = ref[:1]
    if ref.shape[-2:] != (H, W):
        ref = adaptive_resize(ref, W, H, "area", "center")
    ref = pad_to_patch_size(ref, (patch, patch), padding_mode="replicate")
    rh, rw = ref.shape[-2] // patch, ref.shape[-1] // patch
    ref_tok = _patchify(ref, patch)  # (1, Lr, C*p*p)
    refpos = _grid_ids(1, 1, rh, rw, device)  # RoPE axis-0 index 1, own grid from 0

    img = dm.first(_patchify(x, patch))
    t, vec = _time(dm, timesteps, img.dtype)

    ctx = dm.txtfusion(context, mask=None, transformer_options=transformer_options)
    ctx = dm.txtmlp(ctx)

    txtlen, imglen = ctx.shape[1], img.shape[1]
    txtpos = torch.zeros(bs, txtlen, 3, device=device, dtype=torch.float32)
    imgpos = _grid_ids(bs, 0, h_, w_, device)

    if method == CACHED_T0:
        if kv_store is None:
            kv_store = {}
        key = (tuple(ref.shape), str(ref.dtype), str(device))
        if kv_store.get("key") != key:
            kv_store["kv"] = _reference_kv(dm, ref_tok, refpos, timesteps, img.dtype, transformer_options, offload_kv)
            kv_store["key"] = key
        kv = kv_store["kv"]

        combined = torch.cat((ctx, img), dim=1)
        freqs = dm.pe_embedder(torch.cat((txtpos, imgpos), dim=1))
        for block, block_kv in zip(dm.blocks, kv):
            combined = _block(block, combined, vec, freqs, transformer_options, extra_kv=block_kv)

    else:
        rtok = dm.first(ref_tok).expand(bs, -1, -1)
        combined = torch.cat((ctx, img, rtok), dim=1)
        freqs = dm.pe_embedder(torch.cat((txtpos, imgpos, refpos.expand(bs, -1, -1)), dim=1))

        if method == JOINT:
            for block in dm.blocks:
                combined = block(combined, vec, freqs, None, transformer_options=transformer_options)
        elif method == JOINT_T0:
            _, vec0 = _time(dm, torch.zeros_like(timesteps), img.dtype)
            split = txtlen + imglen
            for block in dm.blocks:
                combined = _block_split(block, combined, vec, vec0, split, freqs, transformer_options)
        else:
            raise ValueError(f"Unknown reference method: {method}")

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
        self.ref = None
        self.method = None
        self.kv = {}
        self.offload_kv = False
        self.calls = 0

    def arm(self, ref, method, offload_kv):
        self.ref = ref
        self.method = method
        self.offload_kv = offload_kv
        self.kv = {}
        self.calls = 0
        self.active = True


STATE = _State()

_ORIG_ATTR = "_krea2_control_original_forward"


def install_hook(dm):
    """Wrap ``dm.forward`` once per model instance; inert unless STATE is armed."""
    if _ORIG_ATTR in dm.__dict__:
        return
    original = dm.forward

    def forward(x, timesteps, context, attention_mask=None, transformer_options={}, **kwargs):
        st = STATE
        if st.active and st.ref is not None:
            st.calls += 1
            return control_forward(dm, x, timesteps, context, transformer_options, st.ref, st.method, st.kv, st.offload_kv)
        return original(x, timesteps, context, attention_mask=attention_mask, transformer_options=transformer_options, **kwargs)

    dm.__dict__[_ORIG_ATTR] = original
    dm.forward = forward
