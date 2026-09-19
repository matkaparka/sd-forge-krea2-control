# sd-forge-krea2-control

为 [Forge Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo) 里的 **Krea 2** 提供 **OpenPose** 和 **Canny** 控制。

[English](README.md)

Krea 2 目前没有真正的 ControlNet，现有的都是"控制 LoRA"，而且每个 LoRA 训练时把控制图喂进模型的方式都不一样。这个扩展在 Forge Neo 里按每个 LoRA 各自的训练方式复现了条件注入，界面和 ControlNet 类似：控制图、预处理器、预览、Resize 模式，LoRA 从下拉框里选。

适配 Forge Neo `neo` 分支，提交 `41359cd`（2026-09-18）。版本 0.1.0。

## 使用的 LoRA

扩展本身**不含任何模型权重**，控制效果完全来自下面两个社区 LoRA。请到作者页面下载，放进 `models/Lora`：

| 控制类型 | LoRA | 作者 | 文件 | 基于 | 许可 |
| --- | --- | --- | --- | --- | --- |
| OpenPose | [Krea-2-pose-controlnet](https://huggingface.co/thedeoxen/Krea-2-pose-controlnet) | thedeoxen | `krea2_turbo_openpose_controlnet.safetensors` | Krea 2 Turbo | Apache-2.0 |
| Canny | [NK2E](https://huggingface.co/nynxz/NK2E)（canny v0.1） | Nynxz | `comfy/canny_v0.1/NK2E-canny-v0.1.safetensors` | Krea 2 Raw | Krea 2 Community License |

NK2E 仓库里的 `NK2E-v0.3` 等文件是图像编辑 LoRA，不是 canny，别下错。

两个 LoRA 的条件注入方式（按作者的 ComfyUI 工作流复现）：

- **OpenPose（thedeoxen）**，依赖 [Ostris 的 Krea 2 edit 节点](https://github.com/ostris/comfyui-krea2-ostris-edit)：
  - 姿态图以 `Picture 1: <图>` 的格式放在 prompt 前面送进 Qwen3-VL，先缩到 384×384 以内；
  - 同一张图再 VAE 编码成参考 latent，在 t=0 单独跑一遍，只和自己做注意力，每层 K/V 缓存下来，拼进之后每一步的注意力里；
  - 这对应节点的 `kv_cache = true`，官方工作流就是这样设置的。
- **Canny（NK2E）**：in-context 参考。
  - 边缘图的 latent token 接在序列后面，和目标图用同一个 timestep；
  - 文本编码只输入文字；
  - 边缘提取按 NK2E 训练脚本：灰度图、`cv2.Canny(100, 200)`、白线黑底，训练尺寸 512²。

作者的推荐设置：

- **OpenPose**：Krea 2 Turbo，约 10 步，CFG 1，euler + simple，LoRA 权重 0.8–1.0，姿态太僵就降到 0.6–0.8。黑底 DWPose 骨架图效果最好，不需要触发词。
- **Canny**：LoRA 权重从 0.7 起步。实验版，能跟着轮廓走，但约束比 SDXL ControlNet 松。

## 前置要求

- Forge Neo 和 Krea 2 模型（Turbo 或 Raw）。
- **带 Qwen3-VL 视觉权重的文本编码器**，例如 Comfy-Org 的 `qwen3vl_4b_fp8_scaled.safetensors`。OpenPose 需要让 Qwen3-VL 看姿态图。
- DWPose 模型首次使用时自动下载到 `models/ControlNetPreprocessor/openpose/`（`yolox_l.onnx` 217 MB、`dw-ll_ucoco_384.onnx` 134 MB）。

## 安装

**Extensions → Install from URL**，填 `https://github.com/matkaparka/sd-forge-krea2-control`，然后重启。

也可以直接 clone 到 `extensions/`。

## 用法

1. txt2img 里展开 **Krea2 Control** 并启用。
2. 选择 **Control Type**。
3. 放入控制图：
   - 照片会自动预处理，可以先点 **Preview** 看效果；
   - 已经是骨架图或边缘图的，Preprocessor 设为 `None`；
   - img2img 里不放控制图时，用 img2img 的输入图。
4. 在 **Control LoRA** 里选 LoRA。扩展会在 prompt（和 Hires prompt）末尾加 `<lora:…>`。如果 prompt 里已经有同名 LoRA，就保留你写的标签和权重。
5. 正常写 prompt，生成。

**Settings → Stable Diffusion → [Krea2] Enable Reference** 要保持关闭，也不要同时用 ImageStitch。

## 选项

- **Preprocessor Resolution**：检测时的短边尺寸。NK2E 训练用的是 512。
- **Resize Mode**：和 ControlNet 相同。`Resize and Fill` 用黑色填充，对 canny 和 pose 来说黑色就是"无信号"。
- **Reference Method**（Advanced，只对 OpenPose 生效）：
  - `Cached K/V, t=0`（默认）：官方工作流的方式；
  - `Joint, t=0`：参考 token 每一步都在序列里；
  - `Joint, current t`：NK2E 的方式，Canny 固定用这个。
- **Keep reference K/V in system RAM**：缓存的 K/V 在 1024² 时约 0.7 GB，1536² 时约 1.6 GB。显存紧张时打开，会慢一点。

## 已知问题

### Canny + 高清修复会把边缘图画进画面

NK2E canny LoRA（权重 0.7）配合**高清修复**时，边缘图会以浅色、带锯齿的重影线条出现在结果里：

![Canny + 高清修复重影](docs/known-issue-canny-hires-fix.webp)

原因是高清修复第二轮去噪强度低，构图已经定了，这时再施加控制，主要效果就是往画面上描线。关掉高清修复后重影消失。

在加入"高清修复时跳过控制"选项之前，可以：

- 直接按最终尺寸出图，不开高清修复；
- 或者出图后关掉 Krea2 Control，再用 img2img 或放大器放大。

权重降到 0.5–0.6 也能减轻重影，但结构约束会变松。OpenPose 是否有同样的问题还没有确认。

### 其它说明

- **ADetailer**：每批出图后、ADetailer 运行之前，控制就已撤掉，所以 ADetailer 自己的修复不带控制。但如果 ADetailer 沿用主 prompt，会继承 prompt 里的控制 LoRA（此时没有参考图），请给 ADetailer 单独写 prompt。
- **条件缓存**：OpenPose 模式下姿态图属于文本条件的一部分，每次生成都会重新编码 prompt。
- **LoRA Control 语法**（`<lora:名字:[w@t, …]>`）可以用：把 Control LoRA 设为 `None`，自己写标签，并在 *Diffusion in Low Bits* 里选带 "(fp16 LoRA)" 的选项。注意这只能让 LoRA 淡出，参考图仍会作用到最后一步。
- PNG info 会记录一行 `Krea2 Control: …`，但粘贴参数时不会恢复面板设置。

## 验证情况

- `tests/test_forward.py` 用 Forge Neo 真实的 `backend/nn/krea.py`（缩小尺寸、随机权重、CPU）检查三种参考方法：
  - `Joint, current t` 与 Forge 原生的 `dynamic_args.ref_latents` 路径一致；
  - 三种方法都与一份独立实现（逐 token 调制 + 注意力掩码）一致，最大误差约 5e-7；
  - 覆盖 batch 1/2、4D/5D 输入、奇数尺寸、参考图缩放、K/V 复用与 offload。
- 在 Forge 真实模块上验证过：
  - 界面构建，以及完整一次 txt2img + 高清修复的钩子流程；
  - Qwen3-VL 的 token 排布，与 Ostris 节点逐 token 相同；
  - Forge 的 DWPose 预处理。
- 真实 GPU 上目前跑过 Canny 模式。欢迎反馈 OpenPose 的使用情况，日志以 `Krea2 Control` 开头。

## 致谢与许可

- 模型前向移植自 Forge Neo `backend/nn/krea.py`（AGPL-3.0，Haoming02）与 [ostris/comfyui-krea2-ostris-edit](https://github.com/ostris/comfyui-krea2-ostris-edit)（MIT）。
- Canny 参数来自 [Nynxz/NK2E](https://github.com/Nynxz/NK2E)。
- Pose LoRA 作者 [thedeoxen](https://huggingface.co/thedeoxen/Krea-2-pose-controlnet)，Canny LoRA 作者 [Nynxz](https://huggingface.co/nynxz/NK2E)。
- 本扩展以 AGPL-3.0 发布，LoRA 和 Krea 2 遵循各自的许可。
