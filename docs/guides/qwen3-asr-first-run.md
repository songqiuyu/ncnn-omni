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

如果只需要 CLI，不编译单元测试和诊断工具，可以使用最小构建：

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -Dncnn_DIR=/path/to/ncnn/lib/cmake/ncnn \
  -DNCNN_OMNI_BUILD_TESTS=OFF \
  -DNCNN_OMNI_BUILD_TOOLS=OFF
cmake --build build --target ncnn-omni-asr -j4
```

单配置构建器的产物是：

```text
build/ncnn-omni-asr
```

Visual Studio 等多配置构建器通常生成：

```text
build/Release/ncnn-omni-asr.exe
```

若导入的 ncnn package 在链接接口中启用了 OpenMP，CMake 还必须能找到对应的
OpenMP runtime。AppleClang 环境可安装 `libomp`，必要时显式传入：

```bash
cmake -S . -B build \
  -Dncnn_DIR=/path/to/ncnn/lib/cmake/ncnn \
  -DOpenMP_CXX_FLAGS="-Xpreprocessor -fopenmp" \
  -DOpenMP_CXX_LIB_NAMES=omp \
  -DOpenMP_omp_LIBRARY=/path/to/libomp.dylib
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

`--model` 目录必须包含：

```text
audio_conv.ncnn.param
audio_conv.ncnn.bin
audio_transformer.ncnn.param
audio_transformer.ncnn.bin
embed_token.ncnn.param
embed_token.ncnn.bin
decoder.ncnn.param
decoder.ncnn.bin
lm_head.ncnn.param
lm_head.ncnn.bin
```

`--assets` 运行时实际需要：

```text
vocab.json
merges.txt
```

## 推理

先确认 CLI 可以启动：

```bash
./build/ncnn-omni-asr --help
```

自动识别语言：

```bash
./build/ncnn-omni-asr \
  --model /path/to/qwen3-asr-0.6b-ncnn \
  --assets /path/to/Qwen3-ASR-0.6B \
  --audio /path/to/16k-mono.wav \
  --max-new-tokens 64 \
  --threads 8
```

强制中文并提供上下文：

```bash
./build/ncnn-omni-asr \
  --model /path/to/qwen3-asr-0.6b-ncnn \
  --assets /path/to/Qwen3-ASR-0.6B \
  --audio /path/to/16k-mono.wav \
  --language Chinese \
  --context "会议讨论移动端模型部署。" \
  --max-new-tokens 128 \
  --threads 8
```

参数含义：

| 参数 | 必需 | 说明 |
|---|---|---|
| `--model DIR` | 是 | 五个 ncnn 子模型目录 |
| `--assets DIR` | 是 | `vocab.json` 和 `merges.txt` 所在目录 |
| `--audio FILE` | 是 | mono 16 kHz PCM16/float32 WAV |
| `--language NAME` | 否 | 强制语言；不传则由模型识别 |
| `--context TEXT` | 否 | 提供转写上下文，不是输出后处理 |
| `--max-new-tokens N` | 否 | 最大生成 token 数，默认 256 |
| `--threads N` | 否 | ncnn CPU 线程数，默认使用硬件并发数 |

如果输入音频不是 16 kHz mono，可以先转换：

```bash
ffmpeg -i input.wav -ar 16000 -ac 1 -c:a pcm_s16le audio_16k_mono.wav
```

## 输出说明

CLI 输出示例：

```text
language: English
text: Hmm. Oh yeah, yeah. ...
raw: language English<asr_text>Hmm. Oh yeah, yeah. ...
tokens: 47
token_ids: 11528 6364 151704 ...
timing_ms: frontend=80.508 audio_encoder=590.83 prefill=875.349 decode=2560.78
```

- `language`：自动识别或强制指定的语言；
- `text`：解析后的最终转写文本；
- `raw`：模型原始解码结果，用于排查 tokenizer/后处理问题；
- `tokens/token_ids`：生成序列，用于严格 parity；
- `timing_ms`：前处理、音频编码、prefill 和逐 token decode 耗时。

正常完成返回码为 `0`；参数错误返回 `2`；音频、模型加载或推理失败返回 `1`。

## 当前输入约束

- 单条本地 WAV；
- mono 16 kHz；
- PCM16 或 float32；
- CPU FP32；
- greedy decode；
- 无 batch、流式、时间戳、重采样和 Vulkan。

原始 Qwen3-ASR 对非整 hop 音频会产生 `floor(N/160)` 个 Log-Mel 帧和
`ceil(N/160)` 长度的 feature attention mask，随后可能在 Audio Encoder 内触发
`split_with_sizes`。ncnn-omni 定义了明确的兼容规范：在归一化 PCM 尾部只补足
最后一个 160-sample hop，并要求 PyTorch parity 路径使用同一份 canonical PCM。
从 centered STFT 开始完全复用 Qwen 的参数和数值语义；这一步不截断或改动任何
原始样本，只增加最多 159 个尾部零。

长音频还有两个不能省略的执行约束：

- Audio Conv 的最后一个 Mel chunk 必须按同批最大宽度显式补零到 100 帧，完成
  三层带 bias 的 Conv/GELU 后，再裁剪到真实输出长度；直接用短 tensor 推理会
  改变尾 token。
- 当前第一版对齐 Transformers CPU/SDPA 路径，因此完整 audio token 序列执行
  全局 Attention。FlashAttention varlen 的 104-token 分窗是另一种模型执行策略，
  不能作为性能优化静默替换。

## 对齐验证

单元测试包含 Whisper Log-Mel 数值采样点和 Qwen2 tokenizer token ID 对齐：

```bash
QWEN3_ASR_ASSETS=/path/to/Qwen3-ASR-0.6B \
  ./build/ncnn-omni-unit-tests
```

完整归一化 PCM、feature mask 和 Log-Mel tensor 对齐：

```bash
python tools/parity/qwen3_asr_frontend_parity.py \
  --dump-binary build/ncnn-omni-qwen3-asr-frontend-dump \
  --assets /path/to/Qwen3-ASR-0.6B \
  --audio /path/to/16k-mono.wav \
  --report /tmp/qwen3_asr_frontend_parity.json
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

当前固定回归覆盖 4.20 s、8.41 s、15.05 s 和 19.26 s 四组音频，对应生成
10、17、47 和 61 个 token，均与 Transformers 逐项相等。长音频问题的定位过程、
反证实验和修复数据见
[长音频 token 分叉诊断](../diagnostics/qwen3-asr-long-prefill-divergence.md)。
