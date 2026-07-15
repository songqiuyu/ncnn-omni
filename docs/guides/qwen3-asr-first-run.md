# Qwen3-ASR 第一版运行指南

第一版完成了 Qwen3-ASR-0.6B 的桌面 CPU FP32 纵向链路：

```text
PCM WAV -> Whisper Log-Mel -> Audio Conv -> Audio Transformer
        -> Prompt/Embedding Fusion -> Decoder/KV Cache -> LM Head
        -> greedy token generation -> ASR text
```

## 构建

项目只使用 CMake。可以传入已经安装的 ncnn CMake package：

```bash
cd ncnn-omni
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -Dncnn_DIR=/path/to/ncnn/lib/cmake/ncnn
cmake --build build -j4
```

也可以直接传入 ncnn 源码目录，由同一个 CMake 工程编译：

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DNCNN_OMNI_NCNN_SOURCE=/path/to/ncnn
cmake --build build -j4
```

## 模型和 tokenizer 资源

`--model` 指向包含五组 `.ncnn.param/.ncnn.bin` 的目录。`--assets` 指向原始
Qwen3-ASR-0.6B 模型资源目录，第一版从中读取 `vocab.json` 和 `merges.txt`，不会
加载原始 Transformers 权重。

## 推理

```bash
./build/ncnn-omni-asr \
  --model /path/to/qwen3-asr-0.6b-ncnn \
  --assets /path/to/Qwen3-ASR-0.6B \
  --audio /path/to/16k-mono.wav \
  --max-new-tokens 64 \
  --threads 8
```

可使用 `--language Chinese` 强制语言，或使用 `--context TEXT` 提供上下文。

## 当前输入约束

- 单条本地 WAV；
- mono 16 kHz；
- PCM16 或 float32；
- CPU FP32；
- greedy decode；
- 无 batch、流式、时间戳、重采样和 Vulkan。

若采样点数不是 160 的整数倍，运行时只在末尾补不足一个 hop 的零。这样
Whisper STFT 帧数和 Qwen3-ASR 的 feature attention mask 长度一致，同时避免
上游对任意长度 WAV 可能出现的 `split_with_sizes` 错误。

## 对齐验证

单元测试包含 Whisper Log-Mel 数值采样点和 Qwen2 tokenizer token ID 对齐：

```bash
QWEN3_ASR_ASSETS=/path/to/Qwen3-ASR-0.6B \
  ./build/ncnn-omni-unit-tests
```

完整 token 级对齐：

```bash
python tools/parity/qwen3_asr_e2e.py \
  --binary build/ncnn-omni-asr \
  --ncnn-model /path/to/qwen3-asr-0.6b-ncnn \
  --transformers-model /path/to/Qwen3-ASR-0.6B \
  --audio /path/to/16k-mono.wav
```

脚本只在开发验证时加载 Transformers 模型；ncnn-omni 的实际推理不依赖
PyTorch、Transformers、NumPy 或 SoundFile。
