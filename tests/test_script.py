"""Integration check of scripts/krea2_control.py on Forge Neo's real ``modules`` (panel construction + the hook sequence of
a txt2img / img2img / Hires. fix run), with Forge's real small-random-weights Krea2 DiT and a fake engine / VAE /
text-processing engine. CPU only, no model files.

    FORGE_DIR=/path/to/forge-neo python tests/test_script.py        (use the Forge venv's python)
"""
import atexit
import importlib.util
import logging
import os
import shutil
import sys
import tempfile
import types
from pathlib import Path

FORGE = os.environ.get("FORGE_DIR")
assert FORGE, "set FORGE_DIR to your Forge Neo checkout (neo branch)"
EXT = Path(__file__).resolve().parents[1]
DATA = tempfile.mkdtemp(prefix="krea2_control_test_")  # keeps config.json / ui-config.json of the real install untouched
atexit.register(shutil.rmtree, DATA, ignore_errors=True)
sys.argv = ["x", "--cpu", "--data-dir", DATA]
sys.path.insert(0, FORGE)
sys.path.insert(0, os.path.join(FORGE, "modules_forge", "packages"))  # vendored gguf, as modules_forge/initialization.py does
sys.path.insert(0, str(EXT))

import gradio as gr
import numpy as np
import torch
from PIL import Image

from modules import initialize

initialize.imports()  # what webui.py does first: builds shared.opts / shared.state

import backend.nn.krea as K
from backend.args import dynamic_args
from lib_krea2_control import ControlError
from lib_krea2_control import forward as KF
from lib_krea2_control import imaging as IM
from modules import scripts as forge_scripts
from modules import shared
from modules.processing import StableDiffusionProcessingImg2Img

spec = importlib.util.spec_from_file_location("krea2_control_script", EXT / "scripts" / "krea2_control.py")
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)

torch.manual_seed(0)
ok = 0
records: list[logging.LogRecord] = []


class _Cap(logging.Handler):
    def emit(self, record):
        records.append(record)


logging.getLogger("Krea2 Control").addHandler(_Cap())


def logged(needle, level=None):
    return any(needle in r.getMessage() and (level is None or r.levelno == level) for r in records)


def check(cond, msg):
    global ok
    assert cond, "FAIL: " + msg
    ok += 1
    print("ok  ", msg)


# --- fakes -------------------------------------------------------------------------------------------------
DM = K.SingleStreamDiT(features=64, tdim=32, txtdim=16, heads=4, kvheads=2, multiplier=2, layers=3, patch=2, channels=4, txtlayers=3, txtheads=2, txtkvheads=2).eval()
with torch.no_grad():
    for prm in DM.parameters():
        torch.nn.init.normal_(prm, std=0.08)
CTX = torch.randn(1, 5, 3, 16)
T = torch.tensor([0.4])


def run_dm(x):
    n = x.shape[0]
    with torch.no_grad():
        return DM(x.clone(), T.repeat(n), CTX.repeat(n, 1, 1, 1))


class FakeFSM:
    @staticmethod
    def process_in(s):
        return s * 0.5 + 1.0


class FakeVAE:
    first_stage_model = FakeFSM()
    fail = False

    def encode(self, s):  # (1, H, W, 3) in [0, 1] -> (1, 4, 1, H/8, W/8)
        if FakeVAE.fail:
            raise RuntimeError("vae exploded")
        g = torch.Generator().manual_seed(int(s.shape[1] * 1000 + s.shape[2]))
        return torch.randn(1, 4, 1, s.shape[1] // 8, s.shape[2] // 8, generator=g) + s.mean()


class FakeTPE:
    vision_block = "<|vision_start|>"
    calls: list = []

    def __call__(self, prompt, images=None):
        FakeTPE.calls.append((prompt, None if images is None else tuple(images[0].shape), self.vision_block))
        return "cond:" + str(prompt)


class FakeEngine:
    def __init__(self):
        self.text_processing_engine_qwen = FakeTPE()
        self.forge_objects = types.SimpleNamespace(vae=FakeVAE(), clip=types.SimpleNamespace(patcher=object()), unet=types.SimpleNamespace(model=types.SimpleNamespace(diffusion_model=DM)))

    def get_learned_conditioning(self, prompt):
        return "orig:" + str(prompt)


S.memory_management.load_model_gpu = lambda *a, **k: None


def make_p(w=96, h=64, hr=False, img2img=False):
    p = object.__new__(StableDiffusionProcessingImg2Img) if img2img else types.SimpleNamespace()
    p.width, p.height, p.enable_hr = w, h, hr
    p.all_prompts = ["1girl", "2girls <lora:nk2e:0.2>"]
    p.all_hr_prompts = ["hr one", "hr two"] if hr else []
    p.extra_generation_params = {}
    p.is_hr_pass = False
    p.sd_model = FakeEngine()
    p.init_images = []
    p.clear_prompt_cache = lambda: None
    return p


rng = np.random.default_rng(1)
PHOTO = (rng.random((90, 120, 3)) * 255).astype(np.uint8)
PHOTO[20:70, 30:90] = 200
PHOTO2 = np.ascontiguousarray(PHOTO[::-1, ::-1])

KNOWN = {"pose-lora", "nk2e", "other-lora"}
S._lora_exists = lambda name: name in KNOWN

script = S.Krea2Control()
POSE_DEFAULT = dict(image=None, pre=S.NONE, res=512, resize=IM.CROP_AND_RESIZE, lora="pose-lora", weight=0.9)
CANNY_DEFAULT = dict(image=None, pre=S.BUILTIN_CANNY, res=64, low=100, high=200, resize=IM.CROP_AND_RESIZE, lora="nk2e", weight=0.7)


def proc(p, pose=None, canny=None, image=PHOTO, enable=True, types_=None, method=S.DEFAULT_METHOD, offload=False, positions=S.SAME_FRAME):
    """pose / canny: None = that Control Type is not ticked ({} = ticked with defaults); a dict = overrides of its settings."""
    q = {**POSE_DEFAULT, **(pose or {})}
    c = {**CANNY_DEFAULT, **(canny or {})}
    ticked = types_ if types_ is not None else ([S.POSE] if pose is not None else []) + ([S.CANNY] if canny is not None else [])
    a = dict(enable=enable, types=ticked, image=image, p_image=q["image"], c_image=c["image"], method=method, offload=offload, positions=positions)
    a.update({f"p_{k}": v for k, v in q.items() if k != "image"})
    a.update({f"c_{k}": v for k, v in c.items() if k != "image"})
    script.process(p, *[a[k] for k in S.KEYS])


def reset_world():
    S._cleanup()
    S._last_cond_key = None
    shared.state.interrupted = False
    FakeVAE.fail = False
    FakeTPE.calls = []
    dynamic_args.krea2 = True
    records.clear()


def expected_latent(kind, w, h, image=PHOTO, **over):
    if kind == S.POSE:
        a = {**POSE_DEFAULT, **over}
        det = S._preprocess(S.POSE, image, a["pre"], a["res"])
        return IM.encode_latent(FakeVAE(), IM.fit_to(det, w, h, a["resize"]))
    a = {**CANNY_DEFAULT, **over}
    det = S._preprocess(S.CANNY, image, a["pre"], a["res"], a["low"], a["high"])
    return IM.encode_latent(FakeVAE(), IM.fit_to(det, w, h, a["resize"]))


def sampling(p, x):
    script.process_before_every_sampling(p, x=x, noise=x)


def finish(p):
    script.postprocess_batch(p)
    script.postprocess(p, None)


X = torch.randn(1, 4, 1, 8, 12)  # a 96x64 image
BASE = run_dm(X)

# =============================================================================================================
# 1. UI
# =============================================================================================================
for img2img in (False, True):
    with gr.Blocks() as blocks:
        comps = script.ui(img2img)
    name = "img2img" if img2img else "txt2img"
    check(len(comps) == len(S.KEYS), f"ui({name}) returns {len(comps)} components == len(KEYS)")
    seen, dups = {}, []

    def walk(x):
        if hasattr(x, "children"):
            for c in x.children:
                walk(c)
        else:
            key = x.label if getattr(x, "label", None) is not None else (x.value if isinstance(x, gr.Button) else None)
            if key is not None:
                if key in seen:
                    dups.append(key)
                seen[key] = True

    walk(blocks)
    check(all(k == "\U0001f504" for k in dups), f"ui({name}): no two controls share a label (only the two refresh buttons share their icon): {ascii(dups)}")

labels = [getattr(c, "label", None) or "" for c in comps]
check(len([l for l in labels if l.startswith("Krea2 Control")]) == 1, "ONE panel: a single 'Krea2 Control' enable box")
defaults = {l: c.value for l, c in zip(labels, comps)}
check(defaults["Control Type"] == [S.POSE] and type(comps[S.KEYS.index("types")]).__name__ == "CheckboxGroup", "Control Type is a CheckboxGroup (non-exclusive), OpenPose ticked by default")
check(defaults["Pose LoRA Weight"] == 0.9 and defaults["Canny LoRA Weight"] == 0.7, "the two LoRA weight sliders keep their own defaults (0.9 / 0.7)")
for ticked, want in (([S.POSE], [True, False, True, False]), ([S.CANNY], [False, True, False, True]), ([S.POSE, S.CANNY], [True] * 4), ([], [False] * 4)):
    check([u["visible"] for u in S._on_types(ticked)] == want, f"ticking {ticked or 'nothing'} shows exactly those settings and previews")

names = ["anima-preview-canny-v0.2", "NK2E-canny-v0.1", "krea2_turbo_openpose_controlnet", "animaControlPose_preview2", "style"]
check(S._guess_lora(S.POSE, names) == "krea2_turbo_openpose_controlnet", "default pose LoRA: the Krea 2 one, not Anima's pose adapter")
check(S._guess_lora(S.CANNY, names) == "NK2E-canny-v0.1", "default canny LoRA: NK2E, not Anima's canny LoRA")
check(S._guess_lora(S.CANNY, ["anima-preview-canny-v0.2", "style"]) == S.NONE, "no guess beats a wrong one")
check(S._canny_preview(None, None, S.BUILTIN_CANNY, 64, 100, 200) is None, "Canny Preview with no image warns and returns nothing")
check(min(S._canny_preview(PHOTO, None, S.BUILTIN_CANNY, 64, 100, 200).shape[:2]) == 64, "Canny Preview returns the map")
check(np.array_equal(S._pose_preview(SKEL := PHOTO, None, S.NONE, 512), PHOTO) and np.array_equal(S._canny_preview(PHOTO2, PHOTO, S.NONE, 64, 100, 200), PHOTO), "None passes the image through; the type's own image wins over the shared one")

# =============================================================================================================
# 2. OpenPose alone
# =============================================================================================================
reset_world()
p = make_p(hr=True)
proc(p, pose={})
check(set(S.JOB.controls) == {S.POSE} and not shared.state.interrupted and S.JOB.vl is not None, "OpenPose only: one control, the pose map goes to Qwen3-VL")
check(S.JOB.vl.shape[0] == 1 and S.JOB.vl.shape[1] * S.JOB.vl.shape[2] <= 384 * 384, "the vision copy is at most 384x384 pixels")
check(p.all_prompts == ["1girl <lora:pose-lora:0.9>", "2girls <lora:nk2e:0.2> <lora:pose-lora:0.9>"], "LoRA tag appended")
check(p.all_hr_prompts[0] == "hr one <lora:pose-lora:0.9>", "...and in the Hires prompt")
check("OpenPose" in p.extra_generation_params["Krea2 Control"] and "Canny" not in p.extra_generation_params["Krea2 Control"], f"PNG info: {p.extra_generation_params['Krea2 Control']!r}")
script.process_batch(p)
check(p.sd_model.get_learned_conditioning("a prompt") == "cond:a prompt" and FakeTPE.calls[-1][1] is not None and FakeTPE.calls[-1][2].startswith(S.PICTURE_PREFIX), "conditioning: 'Picture 1:' + the pose map through the text-processing engine")
sampling(p, X)
check(KF.STATE.active and [m for _, m in KF.STATE.refs] == [KF.CACHED_T0] and KF.STATE.frames == [1], "armed: one reference, Cached K/V t=0 (official workflow)")
out = run_dm(X)
want = KF.control_forward(DM, X.clone(), T.clone(), CTX.clone(), {}, expected_latent(S.POSE, 96, 64), KF.CACHED_T0, {})
check(torch.allclose(out, want, atol=1e-6) and not torch.allclose(out, BASE, atol=1e-4), "the forward through the hook equals control_forward with that pose latent")
check(KF.STATE.calls == 1, "the hook was called once")
script.postprocess_batch(p)
check(not KF.STATE.active and torch.equal(run_dm(X), BASE) and p.sd_model.get_learned_conditioning("z") == "orig:z" and S._last_cond_key is None, "postprocess_batch: disarmed, conditioning back to the engine's own, cond cache key cleared")
script.postprocess(p, None)
check(S.JOB is None, "postprocess drops the job")

# =============================================================================================================
# 3. Canny alone
# =============================================================================================================
reset_world()
p = make_p()
proc(p, canny={})
check(set(S.JOB.controls) == {S.CANNY} and S.JOB.vl is None, "Canny only: text-only conditioning (NK2E was trained text-only)")
script.process_batch(p)
p.sd_model.get_learned_conditioning("a prompt")
check(FakeTPE.calls[-1][1] is None, "the text encoder sees text only")
sampling(p, X)
check([m for _, m in KF.STATE.refs] == [KF.JOINT], "armed: Joint, current t")
out = run_dm(X)
check(torch.allclose(out, KF.control_forward(DM, X.clone(), T.clone(), CTX.clone(), {}, expected_latent(S.CANNY, 96, 64), KF.JOINT), atol=1e-6), "forward == control_forward with the edge latent")
finish(p)

# =============================================================================================================
# 4. Both ticked
# =============================================================================================================
reset_world()
p = make_p(hr=True)
proc(p, pose={}, canny={})
check(set(S.JOB.controls) == {S.POSE, S.CANNY} and not shared.state.interrupted, "both ticked: two controls")
check(p.all_prompts[0] == "1girl <lora:pose-lora:0.9> <lora:nk2e:0.7>" and p.all_prompts[1] == "2girls <lora:nk2e:0.2> <lora:pose-lora:0.9>", "both LoRAs are added, OpenPose first; the user's own tag wins")
info = p.extra_generation_params["Krea2 Control"]
check(info.count(" + ") == 1 and "OpenPose" in info and "Canny" in info and info.endswith("same frame"), f"one PNG info line carries both: {info!r}")
script.process_batch(p)
p.sd_model.get_learned_conditioning("x")
check(FakeTPE.calls[-1][1] is not None, "with both ticked the pose map still goes into the text conditioning")
sampling(p, X)
check([m for _, m in KF.STATE.refs] == [KF.CACHED_T0, KF.JOINT] and KF.STATE.frames == [1, 1], "armed: OpenPose Cached K/V + Canny Joint, both at frame 1")
both = run_dm(X)
refs = [(expected_latent(S.POSE, 96, 64), KF.CACHED_T0), (expected_latent(S.CANNY, 96, 64), KF.JOINT)]
check(torch.allclose(both, KF.control_forward(DM, X.clone(), T.clone(), CTX.clone(), {}, refs, None, {}, frames=[1, 1]), atol=1e-6), "forward == control_forward with both references")
pose_alone = KF.control_forward(DM, X.clone(), T.clone(), CTX.clone(), {}, refs[0][0], KF.CACHED_T0, {})
canny_alone = KF.control_forward(DM, X.clone(), T.clone(), CTX.clone(), {}, refs[1][0], KF.JOINT)
check(not torch.allclose(both, pose_alone, atol=1e-4) and not torch.allclose(both, canny_alone, atol=1e-4), "...and it differs from either control alone")
xb = torch.randn(2, 4, 1, 8, 12)
check(torch.allclose(run_dm(xb), torch.cat([run_dm(xb[i : i + 1]) for i in range(2)]), atol=1e-5), "batch 2 gives the same rows as batch-1 runs")
finish(p)

reset_world()
p = make_p()
proc(p, pose={}, canny={}, positions=S.SEPARATE_FRAMES, method="Joint, t=0")
sampling(p, X)
check(KF.STATE.frames == [1, 2] and [m for _, m in KF.STATE.refs] == [KF.JOINT_T0, KF.JOINT], "separate positions: frames 1,2; the chosen OpenPose method (Joint, t=0) + Canny Joint")
check(p.extra_generation_params["Krea2 Control"].endswith("frames 1,2"), "PNG info says so")
sep = run_dm(X)
check(torch.allclose(sep, KF.control_forward(DM, X.clone(), T.clone(), CTX.clone(), {}, [(expected_latent(S.POSE, 96, 64), KF.JOINT_T0), (expected_latent(S.CANNY, 96, 64), KF.JOINT)], None, {}, frames=[1, 2]), atol=1e-6), "forward == control_forward with frames [1, 2]")
finish(p)

# Hires. fix: both references are re-encoded at the hires size
reset_world()
p = make_p(hr=True)
proc(p, pose={}, canny={})
sampling(p, X)
run_dm(X)
p.is_hr_pass = True
xh = torch.randn(1, 4, 1, 16, 24)
sampling(p, xh)
check(sorted(S.JOB.refs) == [(S.CANNY, (96, 64)), (S.CANNY, (192, 128)), (S.POSE, (96, 64)), (S.POSE, (192, 128))], "references cached per control type and size")
check(all(tuple(r.shape[-2:]) == (16, 24) for r, _ in KF.STATE.refs), "hires pass: both latents at the hires size")
check(logged("[hires]") and logged("OpenPose + Canny"), "...and logged")
check(torch.allclose(run_dm(xh), KF.control_forward(DM, xh.clone(), T.clone(), CTX.clone(), {}, [(expected_latent(S.POSE, 192, 128), KF.CACHED_T0), (expected_latent(S.CANNY, 192, 128), KF.JOINT)], None, {}, frames=[1, 1]), atol=1e-6), "hires forward equals the control_forward reference")
finish(p)

# =============================================================================================================
# 5. Images
# =============================================================================================================
reset_world()
p = make_p()
proc(p, pose={}, canny={}, image=PHOTO)
check(np.array_equal(S.JOB.controls[S.POSE].detected, PHOTO) and np.array_equal(S.JOB.controls[S.CANNY].detected, IM.canny(PHOTO, 100, 200, 64)), "only the shared Control Image given: both types use it, each with its own preprocessor")
S._cleanup()
reset_world()
proc(make_p(), pose={"image": PHOTO2}, canny={}, image=PHOTO)
check(np.array_equal(S.JOB.controls[S.POSE].detected, PHOTO2) and np.array_equal(S.JOB.controls[S.CANNY].detected, IM.canny(PHOTO, 100, 200, 64)), "a type's own image wins; the other type still uses the shared one")
S._cleanup()
reset_world()
p = make_p(img2img=True)
p.init_images = [Image.fromarray(PHOTO)]
proc(p, pose={}, canny={}, image=None)
check(set(S.JOB.controls) == {S.POSE, S.CANNY} and not shared.state.interrupted, "img2img: with no image anywhere both types use the img2img input")
S._cleanup()
reset_world()
p = make_p()
proc(p, pose={}, canny={}, image=None)
check(shared.state.interrupted and S.JOB is None and logged("no control image", logging.ERROR) and not any(r.exc_info for r in records), "no image anywhere (txt2img): one clear error line + interrupt")

# =============================================================================================================
# 6. Failures are visible, nothing is half-applied
# =============================================================================================================
reset_world()
p = make_p()
proc(p, types_=[])
check(shared.state.interrupted and S.JOB is None and logged("no Control Type is ticked", logging.ERROR) and p.all_prompts[0] == "1girl", "panel on but nothing ticked: error + interrupt, nothing injected")

reset_world()
p = make_p()
proc(p, pose={"lora": "nope"}, canny={})
check(shared.state.interrupted and S.JOB is None and logged("OpenPose: LoRA \"nope\" was not found", logging.ERROR) and p.all_prompts[0] == "1girl", "unknown pose LoRA: error naming the control, and the canny tag was NOT left in the prompt")

reset_world()
p = make_p()
proc(p, pose={}, canny={"lora": "pose-lora"})
check(shared.state.interrupted and logged("two different LoRA files", logging.ERROR) and p.all_prompts[0] == "1girl", "the same LoRA for both types is refused")

reset_world()
p = make_p()
proc(p, pose={"lora": "None"}, canny={"lora": "None"})
check(S.JOB is not None and logged('"Pose LoRA" is None', logging.WARNING) and logged('"Canny LoRA" is None', logging.WARNING) and p.all_prompts[0] == "1girl", 'LoRA "None": allowed (the user may write the tags), warned, nothing appended')
S._cleanup()

reset_world()
p = make_p()
proc(p, pose={}, canny={})
FakeVAE.fail = True
sampling(p, X)
check(shared.state.interrupted and not KF.STATE.active and logged("failed to prepare the reference"), "VAE failure at sampling time: error + interrupt, nothing armed")
script.postprocess(p, None)

reset_world()
dynamic_args.krea2 = False
p = make_p()
proc(p, pose={}, canny={})
check(S.JOB is None and not shared.state.interrupted and logged("not Krea 2", logging.WARNING) and p.all_prompts[0] == "1girl", "non-Krea2 checkpoint: warns and does nothing")

reset_world()
S._last_cond_key = ("pose", "stale")
p = make_p()
proc(p, pose={}, canny={}, enable=False)
check(S.JOB is None and not records and p.all_prompts[0] == "1girl" and S._last_cond_key is None, "panel off: inert, and a stale pose conditioning cache is dropped")

reset_world()
shared.opts.data["krea2_do_reference"] = True
proc(make_p(), canny={})
check(logged("Enable Reference is ON", logging.WARNING), "[Krea2] Enable Reference ON is warned about")
shared.opts.data["krea2_do_reference"] = False
S._cleanup()

reset_world()
KF.STATE.arm(torch.randn(1, 4, 8, 12), KF.JOINT, False)
S.JOB = S._Job({}, KF.JOINT, False, True, None)
script.before_process(make_p())
check(not KF.STATE.active and S.JOB is None, "before_process clears the leftovers of an interrupted job")

reset_world()
p = make_p()
proc(p, pose={}, canny={})
script.process_batch(p)
sampling(p, X)
boom = DM.last.register_forward_pre_hook(lambda m, a: (_ for _ in ()).throw(RuntimeError("boom")))  # the output layer: every method goes through it
try:
    run_dm(X)
    raised = False
except RuntimeError:
    raised = True
boom.remove()
script.postprocess(p, None)
check(raised and not KF.STATE.active and S.JOB is None and "get_learned_conditioning" not in p.sd_model.__dict__, "exception during sampling: state, job and the conditioning patch are all cleaned up")

print(f"\nall {ok} checks passed")
