import logging
import re
import weakref

import gradio as gr
import torch

from backend import memory_management
from backend.args import dynamic_args
from lib_krea2_control import ControlError
from lib_krea2_control import forward as KF
from lib_krea2_control import imaging as IM
from modules import scripts, shared
from modules.processing import StableDiffusionProcessing, StableDiffusionProcessingImg2Img
from modules.ui_components import FormRow, InputAccordion, ToolButton

try:
    from backend.logging import setup_logger

    logger = logging.getLogger("Krea2 Control")
    setup_logger(logger)
except Exception:
    logger = logging.getLogger("Krea2 Control")

POSE = "OpenPose"
CANNY = "Canny"
TYPES = (POSE, CANNY)  # the "Control Type" choices; any subset can be ticked

NONE = "None"
BUILTIN_CANNY = "canny (NK2E recipe)"
POSE_PREPROCESSORS = ("dw_openpose_full", "openpose_full", "openpose", "openpose_hand", "openpose_face", "animal_openpose")

METHOD_LABELS = {
    "Cached K/V, t=0 (official workflow)": KF.CACHED_T0,
    "Joint, t=0": KF.JOINT_T0,
    "Joint, current t": KF.JOINT,
}
DEFAULT_METHOD = next(iter(METHOD_LABELS))

SAME_FRAME = "Same position (both at frame 1)"
SEPARATE_FRAMES = "Separate positions (frames 1 and 2)"
POSITION_LABELS = (SAME_FRAME, SEPARATE_FRAMES)

DEFAULT_WEIGHT = {POSE: 0.9, CANNY: 0.7}
# every word must be in the file name; no guess beats a wrong one (the LoRA list is shared with other models, e.g. Anima's own canny LoRA)
LORA_WORDS = {POSE: (("krea", "pose"),), CANNY: (("nk2e", "canny"), ("krea", "canny"))}

# One panel, any combination of control types. `ui()` returns the components in this order and `process()` names the
# flat argument list again with it. Labels must be unique inside the panel: Forge's ui-config.json is keyed by
# "<script>/<label>", and a repeated label would hand the second component the first one's default.
KEYS = (
    "enable", "types", "image", "p_image", "c_image",
    "p_pre", "p_res", "p_resize", "p_lora", "p_weight",
    "c_pre", "c_res", "c_low", "c_high", "c_resize", "c_lora", "c_weight",
    "method", "offload", "positions",
)  # fmt: skip

PICTURE_PREFIX = "Picture 1: "  # TextEncodeKrea2OstrisEdit: "Picture {n}: <|vision_start|>..."
OPT_F = 8


# ---------------------------------------------------------------------------
# module-level job state (shared by the txt2img and img2img instances)
# ---------------------------------------------------------------------------


class _Control:
    """One ticked control type."""

    def __init__(self, kind, detected, resize_mode, lora, weight):
        self.kind = kind
        self.detected = detected  # preprocessed map at the preprocessing resolution
        self.resize_mode = resize_mode
        self.lora = lora
        self.weight = weight


class _Job:
    def __init__(self, controls, method, offload, same_frame, vl):
        self.controls: dict[str, _Control] = controls
        self.method = method  # OpenPose reference method (Canny always uses JOINT)
        self.offload = offload
        self.same_frame = same_frame  # both references at RoPE frame 1, or frames 1 and 2
        self.vl = vl  # pose map for Qwen3-VL (None: text-only conditioning)
        self.refs: dict[tuple[str, tuple[int, int]], torch.Tensor] = {}  # (type, size) -> reference latent


JOB: _Job | None = None
_last_cond_key = None
_patched_engines: list[weakref.ref] = []


def _restore_engines():
    for ref in _patched_engines:
        engine = ref()
        if engine is not None and "get_learned_conditioning" in engine.__dict__:
            del engine.__dict__["get_learned_conditioning"]
    _patched_engines.clear()


def _cleanup():
    global JOB
    JOB = None
    KF.STATE.reset()
    _restore_engines()


def _toast(msg: str):
    try:
        gr.Warning(msg)
    except Exception:
        pass  # outside a Gradio request (API / tests): the log line is enough


def _clear_cond_caches(p: StableDiffusionProcessing):
    p.clear_prompt_cache()
    try:
        from modules.processing import StableDiffusionProcessingTxt2Img as T2I

        T2I.cached_hr_c = [None, None, None]
        T2I.cached_hr_uc = [None, None, None]
        if isinstance(p, T2I):
            p.cached_hr_c = T2I.cached_hr_c
            p.cached_hr_uc = T2I.cached_hr_uc
    except Exception:
        pass


def _sync_cond_cache(p: StableDiffusionProcessing, key):
    """Forge caches conditioning by prompt; the OpenPose path puts an image into the
    text encoder, so the cache must be dropped whenever that image changes (or goes away)."""
    global _last_cond_key
    if key != _last_cond_key:
        _clear_cond_caches(p)
        _last_cond_key = key


def _patch_conditioning(engine, vl):
    """Replace Krea2.get_learned_conditioning for this job:
    vl is None  -> plain text encoding (NK2E canny was trained text-only)
    vl tensor   -> "Picture 1: <vision>" + prompt through Qwen3-VL (Ostris edit layout)"""
    tpe = engine.text_processing_engine_qwen

    def get_learned_conditioning(prompt):
        with torch.inference_mode():
            memory_management.load_model_gpu(engine.forge_objects.clip.patcher)
            if vl is None:
                return tpe(prompt)
            original = tpe.vision_block
            tpe.vision_block = PICTURE_PREFIX + original
            try:
                return tpe(prompt, images=[vl])
            except Exception:
                logger.error("Encoding the pose map through Qwen3-VL failed. The text encoder file must include the vision weights " "(e.g. qwen3vl_4b_fp8_scaled.safetensors from Comfy-Org/Krea-2).")
                raise
            finally:
                tpe.vision_block = original

    engine.__dict__["get_learned_conditioning"] = get_learned_conditioning
    if not any(r() is engine for r in _patched_engines):
        _patched_engines.append(weakref.ref(engine))


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _list_loras(refresh: bool = False) -> list[str]:
    try:
        import networks

        if refresh:
            networks.list_available_networks()
        names = sorted(networks.available_networks.keys(), key=str.lower)
    except Exception:
        names = []
    return [NONE] + names


def _guess_lora(kind: str, names: list[str]) -> str:
    for words in LORA_WORDS[kind]:
        for n in names:
            low = n.lower()
            if all(w in low for w in words):
                return n
    return NONE


def _lora_exists(name: str) -> bool:
    import networks

    return name in networks.available_networks or name in getattr(networks, "available_network_aliases", {})


def _pose_choices() -> list[str]:
    return IM.available_forge_preprocessors(POSE_PREPROCESSORS) + [NONE]


def _preprocess(kind, image, preprocessor, detect_res, low=100, high=200):
    if preprocessor == NONE:
        return image
    if kind == CANNY:
        return IM.canny(image, low, high, detect_res)
    return IM.run_forge_preprocessor(preprocessor, image, detect_res)


def _has_tag(prompt: str, name: str) -> bool:
    return re.search(r"<lora:" + re.escape(name) + r"[:>]", prompt, re.IGNORECASE) is not None


def _inject_lora(p: StableDiffusionProcessing, name: str, weight: float):
    tag = f"<lora:{name}:{weight:g}>"

    def add(prompt: str) -> str:
        if _has_tag(prompt, name):
            return prompt  # the user already wrote their own tag
        return f"{prompt} {tag}" if prompt.strip() else tag

    p.all_prompts = [add(x) for x in p.all_prompts]
    if getattr(p, "enable_hr", False) and getattr(p, "all_hr_prompts", None):
        p.all_hr_prompts = [add(x) for x in p.all_hr_prompts]


def _is_krea2(engine) -> bool:
    return bool(getattr(dynamic_args, "krea2", False)) and hasattr(engine, "text_processing_engine_qwen")


def _on_types(types):
    types = types or []
    pose, canny = POSE in types, CANNY in types
    return [gr.update(visible=pose), gr.update(visible=canny), gr.update(visible=pose), gr.update(visible=canny)]  # previews, then settings


def _on_refresh(current_lora):
    return gr.update(choices=_list_loras(refresh=True), value=current_lora)


def _pose_preview(image, own_image, preprocessor, detect_res):
    src = IM.to_rgb_array(own_image) if own_image is not None else IM.to_rgb_array(image)
    if src is None:
        gr.Warning("Upload a control image first")
        return None
    try:
        return _preprocess(POSE, src, preprocessor, detect_res)
    except Exception as e:
        gr.Warning(f"Preprocessing failed: {e}")
        return None


def _canny_preview(image, own_image, preprocessor, detect_res, low, high):
    src = IM.to_rgb_array(own_image) if own_image is not None else IM.to_rgb_array(image)
    if src is None:
        gr.Warning("Upload a control image first")
        return None
    try:
        return _preprocess(CANNY, src, preprocessor, detect_res, low, high)
    except Exception as e:
        gr.Warning(f"Preprocessing failed: {e}")
        return None


INFO = """
Needs a control LoRA in <code>models/Lora</code> (not included) for each ticked Control Type:<br>
<b>OpenPose</b>: <code>krea2_turbo_openpose_controlnet</code> by thedeoxen
(<a href="https://huggingface.co/thedeoxen/Krea-2-pose-controlnet" target="_blank">HF</a>) &nbsp;|&nbsp;
<b>Canny</b>: <code>NK2E-canny-v0.1</code> by Nynxz
(<a href="https://huggingface.co/nynxz/NK2E" target="_blank">HF</a>, folder <code>comfy/canny_v0.1</code>)<br>
<b>Tick one or both.</b> With both, OpenPose uses the method under Advanced and Canny uses Joint, current t; the two LoRAs were trained separately, so start with lower weights.<br>
Keep <b>[Krea2] Enable Reference</b> OFF and ImageStitch disabled.
Known issue: <b>Canny + Hires. fix</b> paints the edge map into the image as ghost outlines, so leave Hires. fix off for Canny.
"""


class Krea2Control(scripts.Script):
    def title(self):
        return "Krea2 Control"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        loras = _list_loras()
        pose_choices = _pose_choices()
        suffix = " (empty = use img2img input)" if is_img2img else ""

        with InputAccordion(value=False, label=self.title()) as enable:
            gr.HTML(INFO)
            types = gr.CheckboxGroup(list(TYPES), value=[POSE], label="Control Type", info="tick one or both; each ticked type shows its own settings below")
            with gr.Row():
                image = gr.Image(label="Control Image" + suffix, sources=["upload", "clipboard"], type="numpy", image_mode="RGB", height=320)
                with gr.Column():
                    p_preview = gr.Image(label="Pose Map", type="numpy", interactive=False, height=320)
                    c_preview = gr.Image(label="Canny Map", type="numpy", interactive=False, height=320, visible=False)

            # ---------------- OpenPose ----------------
            with gr.Column() as pose_box:
                gr.HTML("<b>OpenPose</b>")
                with FormRow():
                    p_pre = gr.Dropdown(pose_choices, value=pose_choices[0], label="Pose Preprocessor", allow_custom_value=True)
                    p_res = gr.Slider(128, 2048, value=512, step=8, label="Pose Preprocessor Resolution", info="short side used for detection")
                    p_btn = gr.Button("Preview Pose", size="sm")
                p_resize = gr.Radio(list(IM.RESIZE_MODES), value=IM.CROP_AND_RESIZE, label="Pose Resize Mode")
                with FormRow():
                    p_lora = gr.Dropdown(loras, value=_guess_lora(POSE, loras), label="Pose LoRA", allow_custom_value=True)
                    p_refresh = ToolButton("🔄", elem_id=self.elem_id("pose_refresh"))
                    p_weight = gr.Slider(0.0, 2.0, value=DEFAULT_WEIGHT[POSE], step=0.05, label="Pose LoRA Weight")

            # ---------------- Canny ----------------
            with gr.Column(visible=False) as canny_box:
                gr.HTML("<b>Canny</b>")
                with FormRow():
                    c_pre = gr.Dropdown([BUILTIN_CANNY, NONE], value=BUILTIN_CANNY, label="Canny Preprocessor", allow_custom_value=True)
                    c_res = gr.Slider(128, 2048, value=512, step=8, label="Canny Preprocessor Resolution", info="short side; NK2E canny was trained at 512")
                    c_btn = gr.Button("Preview Canny", size="sm")
                with FormRow():
                    c_low = gr.Slider(1, 255, value=100, step=1, label="Canny Low Threshold")
                    c_high = gr.Slider(1, 255, value=200, step=1, label="Canny High Threshold")
                c_resize = gr.Radio(list(IM.RESIZE_MODES), value=IM.CROP_AND_RESIZE, label="Canny Resize Mode")
                with FormRow():
                    c_lora = gr.Dropdown(loras, value=_guess_lora(CANNY, loras), label="Canny LoRA", allow_custom_value=True)
                    c_refresh = ToolButton("🔄", elem_id=self.elem_id("canny_refresh"))
                    c_weight = gr.Slider(0.0, 2.0, value=DEFAULT_WEIGHT[CANNY], step=0.05, label="Canny LoRA Weight")

            with gr.Accordion("Advanced", open=False):
                method = gr.Radio(list(METHOD_LABELS), value=DEFAULT_METHOD, label="Reference Method (OpenPose only; Canny always uses Joint, current t)")
                offload = gr.Checkbox(False, label="Keep reference K/V in system RAM (Cached method; saves VRAM at large sizes, slower)")
                positions = gr.Radio(list(POSITION_LABELS), value=SAME_FRAME, label="Reference positions (only when both are ticked)", info="where the two references sit on the RoPE frame axis. Each LoRA was trained alone with its reference at frame 1; Forge's own multi-reference path uses 1, 2, ...")
                gr.HTML("Each ticked type uses the Control Image above unless you give it its own image here.")
                with gr.Row():
                    p_image = gr.Image(label="OpenPose Image (optional)", sources=["upload", "clipboard"], type="numpy", image_mode="RGB", height=240)
                    c_image = gr.Image(label="Canny Image (optional)", sources=["upload", "clipboard"], type="numpy", image_mode="RGB", height=240)

        types.change(_on_types, inputs=[types], outputs=[p_preview, c_preview, pose_box, canny_box], queue=False, show_progress=False)
        p_refresh.click(_on_refresh, inputs=[p_lora], outputs=[p_lora], queue=False, show_progress=False)
        c_refresh.click(_on_refresh, inputs=[c_lora], outputs=[c_lora], queue=False, show_progress=False)
        p_btn.click(_pose_preview, inputs=[image, p_image, p_pre, p_res], outputs=[p_preview])
        c_btn.click(_canny_preview, inputs=[image, c_image, c_pre, c_res, c_low, c_high], outputs=[c_preview])

        return [
            enable, types, image, p_image, c_image,
            p_pre, p_res, p_resize, p_lora, p_weight,
            c_pre, c_res, c_low, c_high, c_resize, c_lora, c_weight,
            method, offload, positions,
        ]  # fmt: skip

    # ------------------------------------------------------------------ hooks

    def before_process(self, p, *args, **kwargs):
        _cleanup()  # leftovers from an interrupted / crashed job

    def process(self, p, *args, **kwargs):
        _cleanup()
        try:
            self._setup(p, dict(zip(KEYS, args)))
        except ControlError as e:
            logger.error(f"Krea2 Control: {e} -- interrupting instead of generating without the control")
            _cleanup()
            _toast(f"Krea2 Control: {e}")
            shared.state.interrupt()
        except Exception as e:
            logger.exception("Krea2 Control setup failed -- interrupting")
            _cleanup()
            _toast(f"Krea2 Control: {e}")
            shared.state.interrupt()

    @staticmethod
    def _source(p, own_image, shared_image):
        """The type's own image, else the shared Control Image, else (img2img) the img2img input."""
        src = IM.to_rgb_array(own_image)
        if src is None:
            src = IM.to_rgb_array(shared_image)
        if src is None and isinstance(p, StableDiffusionProcessingImg2Img) and p.init_images:
            src = IM.to_rgb_array(p.init_images[0])
        if src is None:
            raise ControlError("no control image: upload one" + (" (an empty control image falls back to the img2img input, which is empty too)" if isinstance(p, StableDiffusionProcessingImg2Img) else ""))
        return src

    @classmethod
    def _build(cls, kind, p, a):
        pose = kind == POSE
        src = cls._source(p, a["p_image"] if pose else a["c_image"], a["image"])

        lora = a["p_lora"] if pose else a["c_lora"]
        use_lora = bool(lora) and lora != NONE
        if use_lora:
            if not _lora_exists(lora):
                raise ControlError(f'LoRA "{lora}" was not found -- press the refresh button, or check that the file is in your LoRA folder')
        else:
            logger.warning(f'Krea2 Control: "{"Pose" if pose else "Canny"} LoRA" is None -- make sure the control LoRA is in your prompt')

        if pose:
            detected = _preprocess(POSE, src, a["p_pre"], a["p_res"])
            return _Control(POSE, detected, a["p_resize"], lora if use_lora else None, float(a["p_weight"]))
        detected = _preprocess(CANNY, src, a["c_pre"], a["c_res"], a["c_low"], a["c_high"])
        return _Control(CANNY, detected, a["c_resize"], lora if use_lora else None, float(a["c_weight"]))

    def _setup(self, p, a):
        global JOB

        if not a.get("enable"):
            _sync_cond_cache(p, None)
            return

        if not getattr(dynamic_args, "krea2", False):
            msg = "Krea2 Control is enabled, but the loaded checkpoint is not Krea 2 -- skipped"
            logger.warning(msg)
            _toast(msg)
            _sync_cond_cache(p, None)
            return

        ticked = [t for t in TYPES if t in (a.get("types") or [])]
        if not ticked:
            raise ControlError('no Control Type is ticked -- tick OpenPose and/or Canny, or switch "Krea2 Control" off')

        # build and validate everything first, so a failing control never leaves the other one's tag in the prompt
        controls: dict[str, _Control] = {}
        for kind in ticked:
            try:
                controls[kind] = self._build(kind, p, a)
            except ControlError as e:
                raise ControlError(f"{kind}: {e}") from e

        names = [c.lora for c in controls.values() if c.lora]
        if len(set(names)) != len(names):
            raise ControlError(f'OpenPose and Canny are both set to the LoRA "{names[0]}". They need two different LoRA files')

        pose = controls.get(POSE)
        vl, cond_key = None, None
        if pose is not None:
            target = IM.fit_to(pose.detected, p.width, p.height, pose.resize_mode)
            vl = IM.vl_image(target)
            cond_key = ("pose", IM.image_digest(target))
        _sync_cond_cache(p, cond_key)

        if getattr(shared.opts, "krea2_do_reference", False):
            logger.warning("Krea2 Control: [Krea2] Enable Reference is ON -- turn it off; this extension supplies its own reference")

        for c in controls.values():
            if c.lora:
                _inject_lora(p, c.lora, c.weight)

        method_id = METHOD_LABELS.get(a["method"], KF.CACHED_T0)
        JOB = _Job(controls, method_id, bool(a["offload"]), a["positions"] != SEPARATE_FRAMES, vl)

        infos = []
        if pose is not None:
            infos.append(f"{POSE}, {a['p_pre']}, {int(a['p_res'])}, {pose.resize_mode}, {method_id}")
        if CANNY in controls:
            canny = controls[CANNY]
            head = f"{CANNY}, {a['c_pre']}, {int(a['c_res'])}, {canny.resize_mode}"
            if a["c_pre"] != NONE:
                head += f", {int(a['c_low'])}/{int(a['c_high'])}"
            infos.append(head)
        info = " + ".join(infos)
        if len(controls) > 1:
            info += f", {'same frame' if JOB.same_frame else 'frames 1,2'}"
        p.extra_generation_params["Krea2 Control"] = info

    def process_batch(self, p, *args, **kwargs):
        # after any model reload and right before setup_conds()
        if JOB is None:
            return
        engine = p.sd_model
        if not _is_krea2(engine):
            logger.warning("Krea2 Control: model is not Krea 2 at conditioning time -- skipped")
            _cleanup()
            return
        _patch_conditioning(engine, JOB.vl)

    def process_before_every_sampling(self, p, *args, **kwargs):
        if JOB is None:
            return
        try:
            x = kwargs["x"]
            lat_h, lat_w = int(x.shape[-2]), int(x.shape[-1])
            size = (lat_w * OPT_F, lat_h * OPT_F)
            vae = p.sd_model.forge_objects.vae

            refs, frames, kinds = [], [], []
            for kind in (t for t in TYPES if t in JOB.controls):
                control = JOB.controls[kind]
                ref = JOB.refs.get((kind, size))
                if ref is None:
                    target = IM.fit_to(control.detected, size[0], size[1], control.resize_mode)
                    ref = IM.encode_latent(vae, target)
                    JOB.refs[(kind, size)] = ref
                refs.append((ref, JOB.method if kind == POSE else KF.JOINT))
                frames.append(1 if JOB.same_frame else len(refs))
                kinds.append(kind)

            dm = p.sd_model.forge_objects.unet.model.diffusion_model
            KF.install_hook(dm)
            KF.STATE.arm(refs, None, JOB.offload, frames)
            stage = "hires" if getattr(p, "is_hr_pass", False) else "base"
            detail = ", ".join(f"{k}: latent {tuple(r.shape)} method {m} frame {f}" for k, (r, m), f in zip(kinds, refs, frames))
            logger.info(f"Krea2 Control [{stage}] {' + '.join(kinds)}: {size[0]}x{size[1]}, {detail}")
        except Exception:
            logger.exception("Krea2 Control failed to prepare the reference -- interrupting instead of sampling with the LoRA but no control")
            KF.STATE.reset()
            _toast("Krea2 Control: failed to prepare the reference (see the console)")
            shared.state.interrupt()

    def postprocess_batch(self, p, *args, **kwargs):
        # Sampling (incl. hires) is done: disarm before postprocess_image hooks such as
        # ADetailer run their own passes, and don't let pose-conditioned conds stay cached.
        if JOB is None:
            return
        if KF.STATE.active and KF.STATE.calls == 0:
            logger.warning("Krea2 Control: the control forward was never called -- the reference was NOT applied")
        KF.STATE.reset()
        _restore_engines()
        if JOB.vl is not None:
            _sync_cond_cache(p, None)

    def postprocess(self, p, processed, *args, **kwargs):
        _cleanup()
