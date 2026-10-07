# sd-forge-krea2-control

为 [Forge Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo) 里的 **Krea 2** 提供 **OpenPose** 和 **Canny** 控制。可以只勾一种，也可以**两种同时勾选**。

[English](README.md)

Krea 2 目前没有真正的 ControlNet，现有的都是"控制 LoRA"，而且每个 LoRA 训练时把控制图喂进模型的方式都不一样。这个扩展在 Forge Neo 里按每个 LoRA 各自的训练方式复现了条件注入，界面和 ControlNet 类似：控制图、预处理器、预览、Resize 模式，LoRA 从下拉框里选。

适配 Forge Neo `neo` 分支，提交 `41359cd`（2026-09-18）。版本 0.2.0（两种控制可同时使用；0.2.0 的改动在提交 `534e6ecc`（2026-09-13）上测试）。

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
2. 在 **Control Type** 里勾选 **OpenPose**、**Canny**，或者两个都勾。每个被勾选的类型会显示它自己的设置。
3. 放入 **Control Image**。每个被勾选的类型都用自己的预处理器处理它：
   - 照片会自动预处理，可以先点 **Preview** 按钮看效果；
   - 已经是骨架图或边缘图的，把该类型的预处理器设为 `None`；
   - 想给某个类型单独用另一张图，用 **Advanced** 里的 *Separate images*；
   - img2img 里不放控制图时，用 img2img 的输入图。
4. 在 **Pose LoRA** / **Canny LoRA** 里选 LoRA。扩展会在 prompt（和 Hires prompt）末尾加 `<lora:…>`。如果 prompt 里已经有同名 LoRA，就保留你写的标签和权重。
5. 正常写 prompt，生成。

**Settings → Stable Diffusion → [Krea2] Enable Reference** 要保持关闭，也不要同时用 ImageStitch。

## 选项

**通用**

- **Control Type**：勾选 **OpenPose**、**Canny**，或者两个都勾。默认勾选 OpenPose。
- **Control Image**：所有被勾选的类型共用的一张图，每个类型用自己的预处理器处理它。
- **Separate images**（Advanced）：给 OpenPose 和 / 或 Canny 单独指定图。留空就退回到 Control Image（img2img 里再退回到 img2img 的输入图）。

**OpenPose**（勾选后显示）

- **Pose Preprocessor** 和 **Pose Preprocessor Resolution**：Forge 的姿态预处理器之一（DWPose 在前），以及检测时的短边尺寸。
- **Pose Resize Mode**：和 ControlNet 相同。`Resize and Fill` 用黑色填充，对姿态图来说黑色就是"无信号"。
- **Pose LoRA** 和 **Pose LoRA Weight**（默认 0.9）。

**Canny**（勾选后显示）

- **Canny Preprocessor** 和 **Canny Preprocessor Resolution**：检测时的短边尺寸。NK2E 训练用的是 512。
- **Canny Low / High Threshold**（100 / 200）、**Canny Resize Mode**（`Resize and Fill` 用黑色填充，对边缘图来说黑色就是"无信号"）。
- **Canny LoRA** 和 **Canny LoRA Weight**（默认 0.7）。

**Advanced**

- **Reference Method**（只对 OpenPose 生效）：
  - `Cached K/V, t=0`（默认）：官方工作流的方式；
  - `Joint, t=0`：参考 token 每一步都在序列里；
  - `Joint, current t`：NK2E 的方式，Canny 固定用这个。
- **Keep reference K/V in system RAM**：缓存的 K/V 在 1024² 时约 0.7 GB，1536² 时约 1.6 GB。显存紧张时打开，会慢一点。
- **Reference positions**（只在两个都勾选时有意义）：两个参考在 RoPE 帧轴上的位置。*Same position*（默认）把两个都放在第 1 帧，也就是每个 LoRA 单独训练时用的位置；*Separate positions* 用第 1、2 帧，和 Forge 自己的多参考路径一样。

## 同时使用两种控制

- 每个 LoRA 保持它训练时的条件注入方式：OpenPose 用上面的 Reference Method（默认：参考在 t=0 单独跑一遍，每层 K/V 作为额外的注意力键），Canny 用 Joint, current t（边缘图的 token 接在序列后面）。姿态图仍然送进 Qwen3-VL，边缘图不送（NK2E 训练时文本编码器只输入文字）。
- 两个 LoRA 都会加进 prompt，必须是两个不同的文件。其中一个出错（没有图、LoRA 找不到……）时，任务会在往 prompt 里加任何标签之前就被中断。
- 每种类型的默认 LoRA 只有在文件名吻合时才会被预选（OpenPose：同时含 `krea` 和 `pose`；Canny：同时含 `nk2e` 和 `canny`），不会误选到别的模型的同类 LoRA。
- **不知道的是：** 出图效果。两位作者都没有为叠加两个控制 LoRA 训练或测试过，这个组合也没有在 GPU 上跑过。建议先把权重调低（比如 0.6 / 0.5）。
- 已经验证的是模型代码：每一对方法都和一份独立实现一致，见[验证情况](#验证情况)。

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
- **条件缓存**：勾选 OpenPose 时姿态图属于文本条件的一部分，每次生成都会重新编码 prompt。
- **两个都勾选时的高清修复**：两个参考都会按放大后的尺寸重新编码并作用到第二轮，所以上面说的 Canny 重影在这里同样存在。
- **LoRA Control 语法**（`<lora:名字:[w@t, …]>`）可以用：把 Control LoRA 设为 `None`，自己写标签，并在 *Diffusion in Low Bits* 里选带 "(fp16 LoRA)" 的选项。注意这只能让 LoRA 淡出，参考图仍会作用到最后一步。
- PNG info 会记录一行 `Krea2 Control: …`，但粘贴参数时不会恢复面板设置。

## 验证情况

- `tests/test_forward.py` 用 Forge Neo 真实的 `backend/nn/krea.py`（缩小尺寸、随机权重、CPU）检查三种参考方法，单独用和两两组合都测：
  - `Joint, current t` 与 Forge 原生的 `dynamic_args.ref_latents` 路径一致，两个 Joint 参考在第 1、2 帧时与它的多参考路径一致；
  - 每种方法、每一对方法组合（7 种）都与一份独立实现（逐 token 调制 + 注意力掩码）一致，最大误差约 6e-7；
  - 覆盖 batch 1/2、4D/5D 输入、奇数尺寸、参考图缩放、K/V 复用（包括帧索引变化时）与 offload；只有一个元素的列表和单参考调用逐位相同。
- `tests/test_script.py`（61 项）用 Forge 真实的模块、真实的缩小版 Krea2 DiT 和假的引擎 / VAE 跑面板：界面构建（标签唯一、Control Type 可多选）、每种类型单独用和一起用、条件编码补丁及其清理、共用图 / 各自的图 / img2img 的输入图、高清修复、各种失败路径（什么都没勾、同一个 LoRA 用两次；一个出错不会把另一个的标签留在 prompt 里）。
- 早前（0.1.0）在 Forge 真实模块上验证过：Qwen3-VL 的 token 排布，与 Ostris 节点逐 token 相同；Forge 的 DWPose 预处理。
- 真实 GPU 上目前只单独跑过 Canny 模式。欢迎反馈 OpenPose 的使用情况；两种一起用还没有在 GPU 上跑过。日志以 `Krea2 Control` 开头。

## 致谢与许可

- 模型前向移植自 Forge Neo `backend/nn/krea.py`（AGPL-3.0，Haoming02）与 [ostris/comfyui-krea2-ostris-edit](https://github.com/ostris/comfyui-krea2-ostris-edit)（MIT）。
- Canny 参数来自 [Nynxz/NK2E](https://github.com/Nynxz/NK2E)。
- Pose LoRA 作者 [thedeoxen](https://huggingface.co/thedeoxen/Krea-2-pose-controlnet)，Canny LoRA 作者 [Nynxz](https://huggingface.co/nynxz/NK2E)。
- 本扩展以 AGPL-3.0 发布，LoRA 和 Krea 2 遵循各自的许可。
