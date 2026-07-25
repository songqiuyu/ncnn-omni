# ncnn-omni

ncnn-omni 是一个处于早期阶段、面向端侧的轻量级 C++ 多模态模型推理运行时，
使用 [ncnn](https://github.com/Tencent/ncnn) 作为张量计算图后端。

项目面向手机、PC 和嵌入式设备。架构以可组合 Pipeline 表达不同模型拓扑，
不把所有多模态模型都简化为“文本 LLM 加一个视觉分支”。首批架构验证模型为
Qwen3-ASR、一个 Qwen VLM 和 Qwen3-TTS。

## 支持的模型

| 模型 | 任务 | ncnn 模型 |
|---|---|---|
| Qwen3-ASR-0.6B | 自动语音识别 | [Qwen3-ASR-0.6B-ncnn](https://huggingface.co/songqiuyu/Qwen3-ASR-0.6B-ncnn) |

更多转换后的 ncnn 模型及相关资源可以访问
[songqiuyu 的 Hugging Face 主页](https://huggingface.co/songqiuyu)。

## 当前状态

Qwen3-ASR-0.6B 的第一版桌面 CPU FP32 纵向链路已经可以运行，包括可复用的音频
I/O 与 Whisper 兼容 Log-Mel、常驻加载的五个 ncnn 子模型、独立的通用文本
KV-cache 解码器和 token 级对齐工具；完整音频前处理 tensor 也已完成数值对齐。
公共 API 目前仍为实验状态。

- [设计文档索引](docs/README.md)
- [总体架构](docs/architecture/overview.md)
- [推理引擎调研](docs/research/inference-engine-survey.md)
- [模型包设计](docs/architecture/model-package.md)
- [实现路线](docs/roadmap.md)
- [Qwen3-ASR 第一版运行指南](docs/guides/qwen3-asr-first-run.md)
- [音频前处理对齐报告](docs/diagnostics/qwen3-asr-audio-frontend-parity.md)
- [长音频根因报告](docs/diagnostics/qwen3-asr-long-prefill-divergence.md)

## 实现亮点

- **纯 C++ 部署：** PyTorch 和 Transformers 只用于离线验证，不进入运行时。
- **五个 ncnn 子图独立验证：** Audio Conv、Audio Transformer、Token
  Embedding、Text Decoder/KV Cache 和 LM Head 均有差分证据。
- **图外语义正确：** final-hop 兼容处理、带 bias 的 Conv 尾块 padding、
  CPU/SDPA 全局 Audio Attention、embedding 融合、RoPE 和 sentinel KV mask
  都由清晰的 C++ 逻辑负责。
- **严格对齐：** 归一化 PCM 逐样本相等，完整 Log-Mel 最大误差不超过
  `2.23e-5`，现有长音频的每一个 greedy token 均与 Transformers 相等。
- **正确性优先且可检查：** CPU FP32 是基线；其他后端和精度必须建立自己的
  parity 结果，不能继承结论。
- **精简热路径：** Whisper DFT 系数、RoPE 频率和音频位置编码只预计算一次；
  Conv 输出直接写入 Audio Transformer 输入，不再经过中间 tensor 复制。

## 当前目录

```text
include/ncnn_omni/   通用音频/ASR 类型与 Qwen3-ASR 任务 API
src/models/          Qwen3-ASR 模型适配与 embedding 编排
src/processors/      WAV、tokenizer、Log-Mel 与模型音频策略
src/generation/      与任务无关的文本自回归/KV 解码循环
src/runtime/ncnn/    模块规格、逻辑端口、运行选项与执行
examples/asr/        可直接运行的 CLI
tools/parity/        前处理与端到端差分工具
tests/unit/          确定性的处理器/tokenizer 测试
docs/                架构、调研、指南和验证证据
```

物理目录只表达已经实现的代码。通用多模态 Engine、VLM、TTS、bindings 和平台层
在出现真实实现并验证边界之前，只保留在架构文档中，不用空目录假装已经落地。

## Star 趋势

<p align="center">
  <a href="https://www.star-history.com/songqiuyu/ncnn-omni">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=songqiuyu/ncnn-omni&type=Date&theme=dark" />
      <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=songqiuyu/ncnn-omni&type=Date" />
      <img alt="ncnn-omni Star 趋势图" src="https://api.star-history.com/svg?repos=songqiuyu/ncnn-omni&type=Date" />
    </picture>
  </a>
</p>
