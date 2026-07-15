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

Qwen3-ASR-0.6B 的第一版桌面 CPU FP32 纵向链路已经可以运行，包括 WAV
读取、Whisper 兼容 Log-Mel、五个 ncnn 子模型、greedy KV-cache 解码和 token
级对齐工具。公共 API 目前仍为实验状态。

- [设计文档索引](docs/README.md)
- [总体架构](docs/architecture/overview.md)
- [推理引擎调研](docs/research/inference-engine-survey.md)
- [模型包设计](docs/architecture/model-package.md)
- [实现路线](docs/roadmap.md)
- [Qwen3-ASR 第一版运行指南](docs/guides/qwen3-asr-first-run.md)

## 初始目录

```text
include/ncnn_omni/   公共 C++ 接口
src/core/            Engine、Session 和资源管理
src/runtime/ncnn/    直接调用 ncnn 的执行适配层
src/pipeline/        Stage、拓扑、队列和调度策略
src/processors/      文本、图像和音频预处理
src/generation/      文本与音频生成引擎
src/models/          模型族适配器
bindings/            C ABI 与平台语言绑定
tools/               转换、检查和性能工具
tests/               单元、集成和精度对齐测试
```

第一条实现路径是 Qwen3-ASR CPU 纵向切片。在 ASR、VLM 和流式 TTS 共同验证
公共抽象之前，所有接口均保持实验状态。

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
