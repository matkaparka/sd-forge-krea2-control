import logging
import re
import weakref

import gradio as gr
import torch

from backend import memory_management
from backend.args import dynamic_args
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
MODES = (POSE, CANNY)

NONE = "None"
BUILTIN_CANNY = "canny (NK2E recipe)"
POSE_PREPROCESSORS = ("dw_openpose_full", "openpose_full", "openpose", "openpose_hand", "openpose_face", "animal_openpose")

METHOD_LABELS = {
    "Cached K/V, t=0 (official workflow)": KF.CACHED_T0,
    "Joint, t=0": KF.JOINT_T0,
    "Joint, current t": KF.JOINT,
}
DEFAULT_METHOD = next(iter(METHOD_LABELS))

DEFAULT_WEIGHT = {POSE: 0.9, CANNY: 0.7}
LORA_HINTS = {POSE: ("openpose", "pose"), CANNY: ("canny",)}

PICTURE_PREFIX = "Picture 1: "  # TextEncodeKrea2OstrisEdit: "Picture {n}: <|vision_start|>..."
OPT_F = 8


# ---------------------------------------------------------------------------
# module-level job state (shared by the txt2img and img2img instances)
# ---------------------------------------------------------------------------


class _Job:
    def __init__(self, mode, method, detected, resize_mode, offload, vl):
        self.mode = mode
        self.method = method
        self.detected = detected
        self.resize_mode = resize_mode
        self.offload = offload
        self.vl = vl
        self.refs: dict[tuple[int, int], torch.Tensor] = {}


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


def _guess_lora(mode: str, names: list[str]) -> str:
    for hint in LORA_HINTS[mode]:
        for n in names:
            if hint in n.lower():
                return n
    return NONE


def _preprocessor_choices(mode: str) -> list[str]:
    if mode == CANNY:
        return [BUILTIN_CANNY, NONE]
    return IM.available_forge_preprocessors(POSE_PREPROCESSORS) + [NONE]


def _preprocess(image, mode, preprocessor, detect_res, low, high):
    if preprocessor == NONE:
        return image
    if mode == CANNY:
        return IM.canny(image, low, high, detect_res)
    return IM.run_forge_preprocessor(preprocessor, image, detect_res)


def _inject_lora(p: StableDiffusionProcessing, name: str, weight: float):
    tag = f"<lora:{name}:{weight:g}>"
    pattern = re.compile(r"<lora:" + re.escape(name) + r":", re.IGNORECASE)

    def add(prompt: str) -> str:
        if pattern.search(prompt):
            return prompt
        return f"{prompt} {tag}" if prompt.strip() else tag

    p.all_prompts = [add(x) for x in p.all_prompts]
    if getattr(p, "enable_hr", False) and getattr(p, "all_hr_prompts", None):
        p.all_hr_prompts = [add(x) for x in p.all_hr_prompts]


def _is_krea2(engine) -> bool:
    return bool(getattr(dynamic_args, "krea2", False)) and hasattr(engine, "text_processing_engine_qwen")


def _on_mode(mode, current_lora):
    names = _list_loras()
    choices = _preprocessor_choices(mode)
    guess = _guess_lora(mode, names)
    return [
        gr.update(choices=choices, value=choices[0]),
        gr.update(visible=(mode == CANNY)),
        gr.update(choices=names, value=guess if guess != NONE else current_lora),
        gr.update(value=DEFAULT_WEIGHT[mode]),
    ]


def _on_refresh(current_lora):
    return gr.update(choices=_list_loras(refresh=True), value=current_lora)


def _on_preview(image, mode, preprocessor, detect_res, low, high):
    image = IM.to_rgb_array(image)
    if image is None:
        gr.Warning("Upload a control image first")
        return None
    try:
        return _preprocess(image, mode, preprocessor, detect_res, low, high)
    except Exception as e:
        gr.Warning(f"Preprocessing failed: {e}")
        return None


INFO = """
Needs a control LoRA in <code>models/Lora</code> (not included):<br>
<b>OpenPose</b>: <code>krea2_turbo_openpose_controlnet</code> by thedeoxen
(<a href="https://huggingface.co/thedeoxen/Krea-2-pose-controlnet" target="_blank">HF</a>) &nbsp;|&nbsp;
<b>Canny</b>: <code>NK2E-canny-v0.1</code> by Nynxz
(<a href="https://huggingface.co/nynxz/NK2E" target="_blank">HF</a>, folder <code>comfy/canny_v0.1</code>)<br>
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
        with InputAccordion(value=False, label=self.title()) as enable:
            gr.HTML(INFO)
            mode = gr.Radio(list(MODES), value=POSE, label="Control Type")
            with gr.Row():
                image = gr.Image(
                    label="Control Image" + (" (empty = use img2img input)" if is_img2img else ""),
                    sources=["upload", "clipboard"],
                    type="numpy",
                    image_mode="RGB",
                    height=320,
                )
                preview = gr.Image(label="Preprocessed Map", type="numpy", interactive=False, height=320)
            with FormRow():
                pose_choices = _preprocessor_choices(POSE)
                preprocessor = gr.Dropdown(pose_choices, value=pose_choices[0], label="Preprocessor", allow_custom_value=True)
                detect_res = gr.Slider(128, 2048, value=512, step=8, label="Preprocessor Resolution", info="short side; NK2E canny was trained at 512")
                btn_preview = gr.Button("Preview", size="sm")
            with FormRow(visible=False) as canny_row:
                low = gr.Slider(1, 255, value=100, step=1, label="Canny Low Threshold")
                high = gr.Slider(1, 255, value=200, step=1, label="Canny High Threshold")
            resize_mode = gr.Radio(list(IM.RESIZE_MODES), value=IM.CROP_AND_RESIZE, label="Resize Mode")
            with FormRow():
                lora = gr.Dropdown(loras, value=_guess_lora(POSE, loras), label="Control LoRA", allow_custom_value=True)
                btn_refresh = ToolButton("🔄")
                weight = gr.Slider(0.0, 2.0, value=DEFAULT_WEIGHT[POSE], step=0.05, label="LoRA Weight")
            with gr.Accordion("Advanced", open=False):
                method = gr.Radio(list(METHOD_LABELS), value=DEFAULT_METHOD, label="Reference Method (OpenPose only; Canny always uses Joint, current t)")
                offload = gr.Checkbox(False, label="Keep reference K/V in system RAM (Cached method; saves VRAM at large sizes, slower)")

        mode.change(_on_mode, inputs=[mode, lora], outputs=[preprocessor, canny_row, lora, weight], queue=False, show_progress=False)
        btn_refresh.click(_on_refresh, inputs=[lora], outputs=[lora], queue=False, show_progress=False)
        btn_preview.click(_on_preview, inputs=[image, mode, preprocessor, detect_res, low, high], outputs=[preview])

        return [enable, mode, image, preprocessor, detect_res, low, high, resize_mode, lora, weight, method, offload]

    # ------------------------------------------------------------------ hooks

    def before_process(self, p, *args, **kwargs):
        _cleanup()  # leftovers from an interrupted / crashed job

    def process(self, p, enable, mode, image, preprocessor, detect_res, low, high, resize_mode, lora, weight, method, offload, *args, **kwargs):
        _cleanup()
        try:
            self._setup(p, enable, mode, image, preprocessor, detect_res, low, high, resize_mode, lora, weight, method, offload)
        except Exception:
            logger.exception("Krea2 Control setup failed -- interrupting")
            _cleanup()
            shared.state.interrupt()

    def _setup(self, p, enable, mode, image, preprocessor, detect_res, low, high, resize_mode, lora, weight, method, offload):
        global JOB

        if not enable:
            _sync_cond_cache(p, None)
            return

        if not getattr(dynamic_args, "krea2", False):
            logger.warning("Krea2 Control is enabled, but the loaded checkpoint is not Krea 2 -- skipped")
            _sync_cond_cache(p, None)
            return

        src = IM.to_rgb_array(image)
        if src is None and isinstance(p, StableDiffusionProcessingImg2Img) and p.init_images:
            src = IM.to_rgb_array(p.init_images[0])
        if src is None:
            logger.error("Krea2 Control: no control image -- skipped")
            _sync_cond_cache(p, None)
            return

        detected = _preprocess(src, mode, preprocessor, detect_res, low, high)
        target = IM.fit_to(detected, p.width, p.height, resize_mode)

        pose = mode == POSE
        method_id = METHOD_LABELS.get(method, KF.CACHED_T0) if pose else KF.JOINT
        vl = IM.vl_image(target) if pose else None

        _sync_cond_cache(p, ("pose", IM.image_digest(target)) if pose else None)
        JOB = _Job(mode, method_id, detected, resize_mode, bool(offload), vl)

        if lora and lora != NONE:
            _inject_lora(p, lora, float(weight))
        else:
            logger.warning('Krea2 Control: "Control LoRA" is None -- make sure the control LoRA is in your prompt')

        if getattr(shared.opts, "krea2_do_reference", False):
            logger.warning("Krea2 Control: [Krea2] Enable Reference is ON -- turn it off; this extension supplies its own reference")

        info = f"{mode}, {preprocessor}, {int(detect_res)}, {resize_mode}"
        if mode == CANNY and preprocessor != NONE:
            info += f", {int(low)}/{int(high)}"
        if pose:
            info += f", {method_id}"
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

            ref = JOB.refs.get(size)
            if ref is None:
                target = IM.fit_to(JOB.detected, size[0], size[1], JOB.resize_mode)
                ref = IM.encode_latent(p.sd_model.forge_objects.vae, target)
                JOB.refs[size] = ref

            dm = p.sd_model.forge_objects.unet.model.diffusion_model
            KF.install_hook(dm)
            KF.STATE.arm(ref, JOB.method, JOB.offload)
            stage = "hires" if getattr(p, "is_hr_pass", False) else "base"
            logger.info(f"Krea2 Control [{stage}] {JOB.mode}: {size[0]}x{size[1]}, reference latent {tuple(ref.shape)}, method {JOB.method}")
        except Exception:
            logger.exception("Krea2 Control failed to prepare the reference -- interrupting instead of sampling with the LoRA but no control")
            KF.STATE.reset()
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
