# sd-forge-krea2-control

**OpenPose** and **Canny** control for **Krea 2** in [Forge Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo). Tick one control type, or **both at once**.

[中文说明](README_zh.md)

Krea 2 has no real ControlNet. What exists are *control LoRAs*, and each one was trained with its own way of feeding the control image into the model. This extension reproduces each LoRA's training-time conditioning inside Forge Neo, with a ControlNet-style panel: control image, preprocessor, preview, resize mode, and the LoRA picked from a dropdown.

Built against Forge Neo `neo` branch, commit `41359cd` (2026-09-18). Version 0.2.0 (both control types at once; the 0.2.0 changes were tested on commit `534e6ecc`, 2026-09-13).

## LoRAs used

This extension contains **no model weights**. It drives these two community LoRAs, and all the actual control comes from them. Download them from their authors and put them in `models/Lora`:

| Control | LoRA | Author | File | Trained on | License |
| --- | --- | --- | --- | --- | --- |
| OpenPose | [Krea-2-pose-controlnet](https://huggingface.co/thedeoxen/Krea-2-pose-controlnet) | thedeoxen | `krea2_turbo_openpose_controlnet.safetensors` | Krea 2 Turbo | Apache-2.0 |
| Canny | [NK2E](https://huggingface.co/nynxz/NK2E) (canny v0.1) | Nynxz | `comfy/canny_v0.1/NK2E-canny-v0.1.safetensors` | Krea 2 Raw | Krea 2 Community License |

The NK2E repo also contains *edit* LoRAs (`NK2E-v0.3` etc.). Those are not the canny LoRA.

How each one is conditioned (reproduced from the authors' ComfyUI workflows):

- **OpenPose (thedeoxen)**, which uses [Ostris' Krea 2 edit nodes](https://github.com/ostris/comfyui-krea2-ostris-edit). The pose map is fed to Qwen3-VL as `Picture 1: <image>` in front of the prompt, downscaled to ≤384×384 pixels. The same map is VAE-encoded as a reference latent. That reference runs once at t=0, attending only to itself, and its per-block K/V are cached and appended to every denoising step. This matches the node's `kv_cache = true`, which is how the official workflow is set up.
- **Canny (NK2E)**, an in-context reference. The edge map's latent tokens are appended to the sequence and share the target's timestep. The text encoder sees text only. Edges are extracted with the NK2E training recipe: grayscale, `cv2.Canny(100, 200)`, white on black, trained at 512².

Authors' recommended settings:

- **OpenPose**: Krea 2 Turbo, ~10 steps, CFG 1, euler + simple, LoRA weight 0.8–1.0 (use 0.6–0.8 if the pose is too rigid). DWPose on a black background works best. No trigger word.
- **Canny**: LoRA weight around 0.7 to start. It is experimental: it follows outlines but is looser than an SDXL ControlNet.

## Requirements

- Forge Neo with a Krea 2 checkpoint (Turbo or Raw).
- A text encoder **that includes the Qwen3-VL vision weights**, e.g. `qwen3vl_4b_fp8_scaled.safetensors` from Comfy-Org. The OpenPose path needs Qwen3-VL to see the pose map.
- DWPose models download automatically on first use into `models/ControlNetPreprocessor/openpose/` (`yolox_l.onnx` 217 MB, `dw-ll_ucoco_384.onnx` 134 MB).

## Install

**Extensions → Install from URL**: `https://github.com/matkaparka/sd-forge-krea2-control`, then restart.

Or clone it into `extensions/`.

## Usage

1. In txt2img, open **Krea2 Control** and enable it.
2. Under **Control Type** tick **OpenPose**, **Canny**, or both. Each ticked type shows its own settings.
3. Drop in the **Control Image**. Every ticked type processes it with its own preprocessor.
   - Photos are preprocessed automatically. Use the **Preview** buttons to check the result first.
   - For a ready-made skeleton or edge map, set that type's preprocessor to `None`.
   - To give one type a different image, use *Separate images* under **Advanced**.
   - In img2img, an empty control image means the img2img input is used.
4. Pick the LoRA(s) under **Pose LoRA** / **Canny LoRA**. `<lora:…>` is appended to the prompt (and the Hires prompt). If the prompt already contains that LoRA, your own tag and weight are kept.
5. Write the prompt and generate.

Keep **Settings → Stable Diffusion → [Krea2] Enable Reference** OFF, and don't use ImageStitch at the same time. This extension supplies its own reference.

## Options

**Shared**

- **Control Type**: tick **OpenPose**, **Canny**, or both. OpenPose is ticked by default.
- **Control Image**: one image for every ticked type; each type applies its own preprocessor to it.
- **Separate images** (Advanced): give OpenPose and/or Canny its own image. An empty slot falls back to the Control Image (in img2img, to the img2img input).

**OpenPose** (settings shown when ticked)

- **Pose Preprocessor** and **Pose Preprocessor Resolution**: one of Forge's pose preprocessors (DWPose first), and the short side used for detection.
- **Pose Resize Mode**: same meaning as in ControlNet. `Resize and Fill` pads with black, which means "no signal" for pose maps.
- **Pose LoRA** and **Pose LoRA Weight** (0.9 by default).

**Canny** (settings shown when ticked)

- **Canny Preprocessor** and **Canny Preprocessor Resolution**: short side used for detection. NK2E was trained at 512.
- **Canny Low / High Threshold** (100 / 200), **Canny Resize Mode** (`Resize and Fill` pads with black, "no signal" for edge maps).
- **Canny LoRA** and **Canny LoRA Weight** (0.7 by default).

**Advanced**

- **Reference Method** (OpenPose only):
  - `Cached K/V, t=0` (default): the official workflow.
  - `Joint, t=0`: reference tokens stay in the sequence every step.
  - `Joint, current t`: the NK2E method. Canny always uses this.
- **Keep reference K/V in system RAM**: the cached K/V take about 0.7 GB at 1024² and about 1.6 GB at 1536². Turn this on if VRAM is tight. It is slower.
- **Reference positions** (only when both are ticked): where the two references sit on the RoPE frame axis. *Same position* (default) puts both at frame 1, the index each LoRA was trained with on its own; *Separate positions* uses frames 1 and 2, like Forge's own multi-reference path.

## Using both at once

- Each LoRA keeps the conditioning it was trained with: OpenPose uses the Reference Method above (default: its reference runs once at t=0 and its K/V are extra attention keys), Canny uses Joint, current t (its edge tokens are appended to the sequence). The pose map still goes through Qwen3-VL; the edge map does not (NK2E was trained with a text-only encoder).
- Both LoRAs are added to the prompt and have to be two different files. If one of them fails (no image, unknown LoRA...) the job is interrupted before anything is added to the prompt.
- The default LoRA of each type is only pre-selected when its file name fits (OpenPose: `krea` and `pose`; Canny: `nk2e` and `canny`), so another model's LoRA with a similar name is not picked by mistake.
- **Not known:** how the pictures look. Neither author trained or tested for stacking two control LoRAs, and this combination has not been run on a GPU. Start with lower weights (e.g. 0.6 / 0.5).
- What *is* verified is the model code: every pair of methods matches an independent implementation, see [Verification](#verification).

## Known issues

### Canny + Hires. fix paints the edge map into the image

With the NK2E canny LoRA (weight 0.7) and **Hires. fix** enabled, the edge map shows up in the result as light, jagged ghost outlines:

![Canny + Hires. fix ghosting](docs/known-issue-canny-hires-fix.webp)

The second pass runs at low denoise, so the composition is already fixed. Re-applying the control there mostly traces the edges into the picture. Turning Hires. fix off removes the artifact.

Until there is an option to skip control in the Hires. fix pass, you can:

- generate at the final size without Hires. fix, or
- upscale afterwards (img2img / upscaler) with Krea2 Control disabled.

Lowering the weight to 0.5–0.6 reduces the artifact but loosens the structure. It is not yet known whether the OpenPose path shows the same effect.

### Other notes

- **ADetailer**: control is removed after each batch, before ADetailer runs, so ADetailer's own passes are uncontrolled. If ADetailer reuses the main prompt, it inherits the control LoRA without the reference, so give ADetailer its own prompt.
- **Conditioning cache**: when OpenPose is ticked the pose map is part of the text conditioning, so the prompt is re-encoded on every generation.
- **Hires. fix with both ticked**: both references are re-encoded at the upscaled size and applied in the second pass, so the Canny ghosting above applies there too.
- **LoRA Control syntax** (`<lora:name:[w@t, …]>`) works if you set *Control LoRA* to `None` and write the tag yourself. It needs a "(fp16 LoRA)" option under *Diffusion in Low Bits*. The reference itself stays active for all steps.
- PNG info records a `Krea2 Control: …` line, but pasting parameters does not restore the panel.

## Verification

- `tests/test_forward.py` runs Forge Neo's real `backend/nn/krea.py` (small, random weights, CPU) and checks the three reference methods, alone and in every pair:
  - `Joint, current t` matches Forge's native `dynamic_args.ref_latents` path, and two Joint references at frames 1 and 2 match its multi-reference path;
  - every method, and every pair of methods (7 combinations), matches an independent implementation (per-token modulation + attention mask), max error about 6e-7;
  - coverage: batch 1/2, 4D/5D input, odd sizes, reference resizing, K/V reuse (also when the frame indices change) and offload; a one-element list is bit-identical to the single-reference call.
- `tests/test_script.py` (61 checks) runs the panel on Forge's real modules with the real small Krea2 DiT and a fake engine / VAE: panel construction (unique labels, non-exclusive Control Type), each type alone and both together, the conditioning patch and its cleanup, shared / own / img2img images, Hires. fix, every failure path (nothing ticked, same LoRA twice, an unknown LoRA never leaves the other one's tag in the prompt).
- Checked earlier against Forge's real modules (0.1.0): Qwen3-VL token layout, identical token-for-token to Ostris' node; Forge's DWPose preprocessor.
- Real-GPU use so far: Canny mode on its own. OpenPose reports are welcome, and using both at once has not been run on a GPU. Logs start with `Krea2 Control`.

## Credits and license

- Model forward ported from Forge Neo `backend/nn/krea.py` (AGPL-3.0, Haoming02) and [ostris/comfyui-krea2-ostris-edit](https://github.com/ostris/comfyui-krea2-ostris-edit) (MIT).
- Canny recipe from [Nynxz/NK2E](https://github.com/Nynxz/NK2E).
- Pose LoRA by [thedeoxen](https://huggingface.co/thedeoxen/Krea-2-pose-controlnet); canny LoRA by [Nynxz](https://huggingface.co/nynxz/NK2E).
- This extension: AGPL-3.0. The LoRAs and Krea 2 keep their own licenses.
