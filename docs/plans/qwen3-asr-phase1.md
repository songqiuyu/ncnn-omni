# 第一阶段：Qwen3-ASR-0.6B NCNN 推理实现与状态

状态：第一版 CPU FP32 纵向链路已实现，四组长短音频均完成 token 级对齐

更新日期：2026-07-15

## 1. 阶段目标

第一阶段的目标不是继续转换模型，而是把已经转换并独立验证的 5 个 NCNN
子模型组成一个可以实际识别音频的 Qwen3-ASR-0.6B 推理程序。

第一阶段完成时，应当能够执行：

```bash
ncnn-omni-asr \
  --model /path/to/Qwen3-ASR-0.6B-ncnn \
  --audio /path/to/16k-mono.wav \
  --language Chinese \
  --max-new-tokens 256
```

并得到与 Transformers 参考实现相同的生成 token 序列、语言和最终文本。

本阶段以**桌面 CPU FP32 正确性**为唯一主目标。只有端到端结果对齐以后，才进入
FP16、Vulkan、移动端、内存和速度优化。

## 2. 参考行为

参考入口为外层仓库：

```text
examples/example_qwen3_asr_transformers.py
```

真正的 Transformers 调用链是：

```text
Qwen3ASRModel.transcribe
  -> 音频归一化 / 16 kHz
  -> 构造 chat prompt
  -> Qwen3ASRProcessor
       -> WhisperFeatureExtractor
       -> 展开 audio placeholder
       -> Qwen2Tokenizer
  -> Qwen3ASRForConditionalGeneration.generate
       -> Audio Encoder
       -> audio embedding 替换
       -> Text Decoder prefill
       -> greedy autoregressive decode
  -> tokenizer decode
  -> parse_asr_output
```

示例还展示了 URL、Base64、批量、上下文、指定语言和 Forced Aligner 时间戳。
这些功能不全部进入第一条纵向切片，范围见下文。

## 3. 第一版范围

### 3.1 必须支持

- Qwen3-ASR-0.6B；
- 单条音频推理；
- 本地 PCM WAV 文件；
- C++ API 直接接收 mono float32 PCM 和 sample rate；
- 第一版 CLI 限制为 16 kHz mono，非法输入明确报错；
- 空 context 或自定义 context；
- 自动判断语言；
- 可选强制语言，例如 `Chinese`、`English`；
- greedy decode，`do_sample=false`；
- `max_new_tokens`；
- EOS token `151643` 和 `151645`；
- 输出 raw decoded text、language 和 normalized transcription；
- CPU FP32；
- 分阶段耗时和 token 数量统计。

### 3.2 明确不进入第一版

- Forced Aligner 和时间戳；
- batch；
- 流式 ASR；
- URL、Base64 和任意音频容器；
- 非 16 kHz 重采样；
- Vulkan、FP16、BF16、INT8、INT4；
- Android/iOS API；
- 长音频跨 1200 秒分段与结果合并；
- continuous batching、prefix cache 和 speculative decode。

这些能力不能阻塞第一条正确性链路，但当前 API 不能设计成以后无法扩展它们。

## 4. 已有实现情况

### 4.1 已完成

| 模块 | 文件 | 当前验证情况 |
|---|---|---|
| Audio Conv | `audio_conv.ncnn.param/bin` | CPU FP32 动态宽度 1–100 已与 PyTorch 对齐 |
| Audio Transformer | `audio_transformer.ncnn.param/bin` | CPU FP32 已验证边界长度 1–104，以及 dense `L=196/251` |
| Token Embedding | `embed_token.ncnn.param/bin` | Token ID 边界和随机输入已对齐 |
| Text Decoder | `decoder.ncnn.param/bin` | 28 层、56 路 KV；prefill、decode、chunked 和 prefill→decode 已通过差分测试 |
| LM Head | `lm_head.ncnn.param/bin` | 1024→151936 logits 已对齐 |

已确认的文本模型参数：

```text
vocab_size          151936
hidden_size         1024
intermediate_size   3072
layers              28
attention_heads     16
kv_heads            8
head_dim             128
rope_theta          1000000
```

除子图差分外，当前还完成了四组真实音频的逐 token 端到端回归；长英文和中英
拼接样例分别覆盖 Decoder prompt 211 和 266。详细证据见
[长音频 token 分叉诊断](../diagnostics/qwen3-asr-long-prefill-divergence.md)。

### 4.2 第一版已实现的纵向能力

| 能力 | 状态 |
|---|---|
| ncnn-omni CMake 构建和 CLI | 已实现 |
| WAV/PCM 输入和 Whisper 128-bin Log-Mel | 已实现 |
| Audio Conv 分块、尾块 padding 和位置编码 | 已实现 |
| CPU/SDPA 全局 Audio Attention 调度 | 已实现 |
| Qwen Byte-level BPE tokenizer 和 prompt | 已实现 |
| text/audio embedding 融合 | 已实现 |
| RoPE、causal mask 和 KV Session | 已实现 |
| greedy generation、detokenize 和结果解析 | 已实现 |
| 四组真实音频端到端逐 token 对齐 | 已通过 |

版本化 manifest、打包自包含 tokenizer 资源、非 CPU 后端和第 3.2 节列出的扩展
能力仍未实现。当前整体状态是：**模型转换和第一版 CPU FP32 完整推理均已完成，
公开 API 与跨后端执行策略仍处于实验阶段**。

## 5. 五个子模型的真实组合方式

五个模型不是简单的一条直线。Audio Conv 和 Audio Transformer 之间、音频和
LLM 之间、Decoder 的每一次调用之间都有必须在 C++ 中实现的图外逻辑。

```text
16 kHz PCM
  -> Whisper Log-Mel [128, T]
  -> 按 100 个 Mel 帧切块
  -> Audio Conv [1, 128, W] -> [ceil(W/8), 896]
  -> 每个 100 帧块独立添加正弦位置编码
  -> CPU/SDPA 对齐模式下整段送入 Audio Transformer
  -> Audio Transformer [L, 896] -> [L, 1024]
  -> 得到 audio embeddings

context + audio placeholder + assistant prompt
  -> Qwen tokenizer
  -> Token Embedding [S, 1024]
  -> 用 audio embeddings 替换 audio-pad token 对应行
  -> Decoder prefill + KV cache
  -> LM Head(last hidden)
  -> argmax token
  -> Token Embedding(one token)
  -> Decoder single-token decode + KV cache
  -> 重复直到 EOS / max_new_tokens
  -> detokenize
  -> parse language / transcription
```

## 6. Audio Encoder 图外逻辑

### 6.1 Whisper Log-Mel

必须与 `WhisperFeatureExtractor` 的配置和数值行为对齐：

```text
sampling_rate       16000
feature_size        128
n_fft               400
hop_length          160
padding_value       0.0
dither              0.0
return_attention_mask true
```

当前实现：

1. 公共 API 先接收 float32 PCM；CLI 提供最小 WAV reader。
2. 使用确定性的 400 点直接 DFT 实现 correctness-first STFT；不依赖平台 FFT
   或音频框架。
3. 按 Transformers 相同公式生成 periodic Hann window 和 Slaney-normalized
   `128 x 201` Mel filter matrix。
4. 实现 Whisper 的 power spectrum、Mel 投影、log10/clamp 和归一化。
5. 对 partial final hop 显式执行 compatibility zero pad；原始 Qwen processor
   在该边界存在 Mel `floor` 与 mask `ceil` 的已复现不一致。
6. 使用实际 C++ tensor dump 对四条真实音频和 float32 边界输入完成全 PCM、mask
   和完整 Log-Mel 验证，详见
   [音频前处理对齐报告](../diagnostics/qwen3-asr-audio-frontend-parity.md)。

第一版拒绝非 16 kHz 输入。重采样器在端到端正确以后添加。

### 6.2 Audio Conv 调度

`audio_conv` 只包含三层二维卷积和 `conv_out`，不包含分块、mask 或位置编码。

输入：

```text
in0 float32 [1, 128, W], 1 <= W <= 100
```

输出：

```text
out0 float32 [ceil(W/8), 896]
```

调用方需要：

1. 将 `[128, T]` 按时间轴切成最多 100 帧的块；
2. 若长音频尾块不足 100 帧，按上游 `pad_sequence` 语义显式补零到 100，运行
   Audio Conv 后再裁到真实下采样长度；短于 100 帧且只有一个 chunk 时不补到 100；
3. 对每块输出添加从位置 0 重新开始的 896 维正弦位置编码；
4. 只保留该块的真实输出长度；
5. 按原始顺序拼接。

位置编码不能对整条音频连续累加；上游实现对每个 100 帧块重新从 0 开始。

### 6.3 Audio Transformer 调度

`audio_transformer` 包含 18 层 Audio Transformer、后置 LayerNorm 和投影，但
不包含窗口 mask 或调度。

输入输出：

```text
in0  float32 [L, 896]
out0 float32 [L, 1024]
1 <= L（当前已额外验证 196 和 251）
```

上游的有效窗口来自：

```text
n_window       = 50
n_window_infer = 800
100 Mel 帧 -> 13 CNN tokens
800 Mel 帧 -> 104 CNN tokens
```

必须根据参考后端选择 Attention policy。第一版对齐的是 CPU/SDPA：上游没有把
block mask 传给 SDPA，`cu_seqlens` 也不会被 SDPA 使用，因此实际语义是所有
audio token 的全局 Attention。C++ 必须将拼接后的 Conv features 一次送入 dense
Audio Transformer。

FlashAttention varlen 路径可能按 104 token window 执行，不能与 CPU/SDPA
基准混为一谈。未来若支持该策略，需要 manifest 显式声明并建立独立 parity。

## 7. Prompt、Tokenizer 和 Embedding 融合

### 7.1 Prompt 语义

无强制语言时：

```text
<|im_start|>system\n{context}<|im_end|>\n
<|im_start|>user\n
<|audio_start|>{N 个 <|audio_pad|>}<|audio_end|><|im_end|>\n
<|im_start|>assistant\n
```

其中 `N` 必须等于 Audio Transformer 最终输出 token 数量。

指定语言时，在 generation prompt 后追加：

```text
language {Language}<asr_text>
```

重要 token：

```text
audio_start  151669
audio_end    151670
audio_pad    151676
asr_text     151704
eos          151643, 151645
```

### 7.2 Tokenizer

模型包当前没有 tokenizer 和 processor 资源，必须补充：

```text
vocab.json
merges.txt
tokenizer_config.json
chat_template.json
preprocessor_config.json
generation_config.json
config.json 或精简后的 manifest 字段
Mel filter/window 参考资源
```

第一版实现可复用 `ncnn_llm` 中经过 Qwen 验证的 byte-level BPE 思路或经许可的
底层代码，但不能依赖 `ncnn_llm_gpt`，也不能把它的模型类复制进来。Tokenizer
应落在 `processors/text`，并通过以下数据与 Hugging Face tokenizer 做精确验证：

- 完整默认 prompt；
- 带中文 context；
- 带英文 context；
- 指定语言 prompt；
- audio-pad 连续重复；
- 中文、英文、数字、标点的 encode/decode；
- special token 是否在 decode 时跳过。

### 7.3 Embedding 融合

1. Tokenizer 产生完整 prompt token IDs；
2. 检查 ID `151676` 的数量严格等于 audio embedding 长度；
3. `embed_token` 对所有 prompt token 查表；
4. 将所有 audio-pad 行替换成 `[N, 1024]` audio embeddings；
5. 融合后的 `[S, 1024]` 送入 Decoder prefill。

数量不一致必须报错，不能截断或静默补零。

## 8. Decoder、RoPE 和 KV Cache

### 8.1 Decoder NCNN 接口

```text
in0      hidden states       [S, 1024]
in1      attention mask      [S, P+S]
in2      RoPE cos            [S, 64]
in3      RoPE sin            [S, 64]
in4..59  28 层 K/V cache     每路 [8, P, 128]

out0      normalized hidden  [S, 1024]
out1..56  updated K/V cache  每路 [8, P+S, 128]
```

`P` 是物理 KV cache 长度，`S` 是本次输入长度。

### 8.2 初始 sentinel cache

NCNN 动态图不能依赖长度为 0 的 cache，因此初次 prefill 使用物理长度为 1 的
全零 sentinel cache，并将 attention mask 的第 0 列设为一个足够小的负值，使
所有真实 token 都看不到 sentinel。

实现时必须区分：

```text
physical_cache_length = logical_token_count + 1
logical_position      = real token position, starts from 0
```

现有转换测试主要验证了 Decoder 子图自身以及 sentinel 链接，但第一阶段还必须
增加一项与**原始 Transformers 模型**的对比，确认 sentinel 不会让 RoPE 位置整体
偏移 1。正确目标是：

- prefill 的真实 token RoPE 位置为 `0..S-1`；
- 第一个 decode token 的 RoPE 位置为 `S`；
- attention mask 始终屏蔽物理 cache 的 sentinel 第 0 列；
- cache 输入输出仍保留 sentinel 槽位。

这是端到端实现中优先级最高的风险项。

### 8.3 Prefill 和 decode

Prefill：

1. 输入融合后的全部 prompt embeddings；
2. 创建 causal mask，并额外屏蔽 sentinel；
3. 生成真实位置 0 开始的 RoPE cos/sin；
4. 运行 Decoder；
5. 仅取最后一个真实 token 的 hidden state；
6. 运行 LM Head 并 argmax，得到第一个生成 token。

Decode：

1. 对新 token 运行 Token Embedding；
2. 创建 `[1, physical_past + 1]` mask，屏蔽 sentinel；
3. 使用 `logical_token_count` 生成 RoPE；
4. 输入上一轮 56 路 cache；
5. 运行 Decoder、LM Head 和 argmax；
6. 遇到 EOS 或上限时停止。

每次 decode 都要检查 56 路 cache 的层顺序、K/V 顺序、shape 和有限值。

## 9. 输出解析

默认模式可能生成：

```text
language Chinese<asr_text>识别文本
```

实现与上游一致的解析规则：

- 有 `<asr_text>` 时拆分 metadata 和正文；
- `language None` 且无正文表示静音；
- 有 `language X` 时规范化语言名；
- 强制语言模式把全部生成结果视为纯文本；
- 保留原始 decoded string 供差分诊断；
- 第一版先实现必要解析，不在推理层加入不可追踪的文本“修正”。

## 10. 当前精简代码布局

```text
include/ncnn_omni/
├── qwen3_asr.h
└── status.h

src/
├── runtime/ncnn/ncnn_module.h/.cpp
├── processors/qwen2_tokenizer.h/.cpp
├── processors/whisper_log_mel.h/.cpp
├── processors/wav.cpp
└── models/qwen3_asr.cpp

examples/asr/
└── qwen3_asr_cli.cpp

tools/parity/
├── qwen3_asr_frontend_dump.cpp
├── qwen3_asr_frontend_parity.py
└── qwen3_asr_e2e.py

tests/unit/
```

当前只有 `runtime/ncnn` 和模型适配器直接接触 NCNN；通用调用错误检查集中在
`NcnnModule`，Qwen3-ASR 特有的调度、prompt、RoPE 和 KV 规则留在模型适配器中。
未实现的 VLM、TTS、bindings 和通用 pipeline 不创建空目录，等真实代码验证边界
后再增加。

## 11. 实施里程碑

### M1.0 模型包和参考数据（部分完成）

- 为 5 个 `.param/.bin` 增加 manifest；
- 补齐 tokenizer、processor、generation 和精简模型配置；
- 增加文件 checksum；
- 编写 Python reference exporter；
- 导出真实音频各关键中间结果。

输出 fixture 至少包括：

```text
PCM
Log-Mel + valid frame count
每个 Audio Conv chunk 输出
加位置编码并拼接后的 features
完整 Audio Transformer 输出及不同 Attention policy 的对照输出
完整 audio embeddings
prompt string 和 token IDs
融合后的 prefill embeddings
position IDs / cos / sin / mask
prefill hidden、KV 摘要、logits top-k
前若干 decode step 的 token/logits/cache 摘要
最终 raw text、language、text
```

### M1.1 最小 C++ 工程和 NcnnModule（完成）

- CMake 静态库和 CLI target；
- `find_package(ncnn CONFIG REQUIRED)`；
- checked load/input/extract；
- CPU FP32 固定选项；
- 五个模型 load/smoke；
- 统一 Status 和阶段计时。

### M1.2 Audio Processor 和 Audio Encoder（完成）

- WAV/PCM；
- Whisper Log-Mel；
- 100 帧 Audio Conv 调度；
- 正弦位置编码；
- CPU/SDPA 全局 Audio Transformer 调度，并覆盖 L=196/251；
- 与 reference fixture 逐阶段对齐。

### M1.3 Tokenizer、Prompt 和 Embedding Mixer（完成）

- Qwen byte-level BPE；
- special tokens；
- 默认/强制语言 prompt；
- placeholder 展开；
- Token Embedding；
- audio embedding 替换；
- 与 Transformers token IDs 和 fused embeddings 对齐。

### M1.4 Text Decoder Engine（完成）

- RoPE；
- causal/sentinel mask；
- 28 层 56 路 KV Session；
- prefill；
- LM Head + argmax；
- single-token decode；
- EOS 和 max token；
- tokenizer decode。

### M1.5 端到端 CLI（完成）

- 本地音频输入；
- context；
- optional language；
- raw/normalized 输出；
- TTFT、prefill、decode、RTF 和峰值近似统计；
- 明确的错误和非零退出码。

### M1.6 精度验收和回归（进行中：四组长短音频已通过）

- 短中文；
- 短英文；
- 中英/数字/标点；
- 空音频或近静音；
- 指定语言；
- 至少一条较长音频；
- 连续重复运行；
- 不同 CPU 线程数。

## 12. 验收标准

### 12.1 必须通过

1. Prompt token IDs 与 Hugging Face 完全一致。
2. Audio placeholder 数量与 audio embedding 长度完全一致。
3. Log-Mel、Audio Conv、位置编码、Audio Transformer 和融合 embedding 分别
   达到预先记录的数值容差。
4. Prefill 的 argmax token 与 Transformers 一致。
5. 固定 fixture 的每一步 greedy token 与 Transformers 一致。
6. 最终 raw decoded text、language、normalized text 一致。
7. 所有 NCNN return code 都被检查；所有中间 tensor 均为有限值。
8. 多次创建/销毁 Session 不污染下一次结果。

### 12.2 如果 token 分叉

不能只报告最终文本不同。差分报告必须包含：

- 首个分叉 step；
- PyTorch/NCNN top-10 token 和 logits；
- top-1/top-2 margin；
- 该步输入 hidden、RoPE、mask 和 cache 摘要；
- 使用 PyTorch cache 与 NCNN 自生成 cache 的两组结果；
- 分叉来自固有数值误差还是前序 cache 累积。

### 12.3 第一阶段完成定义

第一阶段只有在以下条件同时满足时才完成：

- 一个纯 C++ `ncnn-omni-asr` 可以使用这 5 个 NCNN 模型完成真实音频识别；
- 至少三类真实 fixture 的 token 序列和最终文本与 Transformers 一致；
- 端到端测试不依赖 Python；
- Python 只负责生成参考 fixture；
- 已记录 CPU FP32 的耗时和内存基线；
- README 中不再描述为只有转换模型，而是提供可复现的运行命令。

## 13. 资源和性能现实

当前 FP32 权重近似为：

```text
Audio Conv          42 MiB
Audio Transformer  669 MiB
Token Embedding    594 MiB
Text Decoder       1.6 GiB
LM Head            594 MiB
```

Embedding 和 LM Head 来源于 tied weights，但当前被转换成两个独立 NCNN 权重，
因此磁盘和内存存在约 594 MiB 重复。第一阶段接受这一点以确保正确性，但它不是
手机端可接受的最终形态。

FP32 KV cache 每个真实 token 约为：

```text
28 layers * 2(K/V) * 8 heads * 128 * 4 bytes ~= 224 KiB/token
```

另外，当前 Decoder 图每一步输出完整增长后的 56 路 cache，会产生随上下文增长
的复制和分配成本。因此：

- 第一阶段默认限制短音频，例如 30 秒；
- Audio Encoder 完成后应允许卸载其两个网络，再加载 LLM 网络，降低峰值；
- 正确性完成后应重新设计 tied weight 共享、KV storage 和 cache 输出策略；
- 在解决这些问题前，不能宣称已达到手机端生产性能。

## 14. 主要风险及处理顺序

| 优先级 | 风险 | 处理方法 |
|---|---|---|
| P0 | sentinel 导致 RoPE 位置整体偏移 | 对原始 Transformers 做 prefill+decode 层级对比，区分物理 cache 与逻辑位置 |
| P0 | Whisper Log-Mel 实现不一致 | 导出 Mel filter/window 和逐阶段 fixture，不用最终文本掩盖误差 |
| P0 | 尾 Conv chunk padding 或 Audio Attention policy 错误 | 测试 99/100/101 Mel 帧、L=104/196/251 和真实长音频 |
| P0 | Tokenizer special token 行为错误 | prompt token IDs 必须逐 ID 相等 |
| P0 | 56 路 cache 顺序或 NCNN Mat 维度错误 | 使用命名映射和 shape assertion，不用裸下标散落在代码中 |
| P1 | 长 prefill 数值误差导致 token 分叉 | 保存 top-k/margin，定位第一分叉 step |
| P1 | 权重与 KV 内存过高 | 第一阶段 PC CPU；音频/LLM 分阶段驻留；后续量化与共享 |
| P1 | 每步完整 cache 输出过慢 | 第一阶段保真；后续调整 Decoder 导出和预分配策略 |

## 15. 推荐执行顺序

严格按以下顺序推进：

```text
reference fixtures
  -> NcnnModule smoke
  -> Log-Mel parity
  -> complete Audio Encoder parity
  -> tokenizer/prompt parity
  -> fused embedding parity
  -> Decoder prefill parity
  -> one-token decode parity
  -> multi-token greedy parity
  -> final text parity
  -> performance and mobile optimization
```

不建议一开始直接运行完整音频并只观察最终文本。只要最终结果不同，就会同时面对
音频、tokenizer、embedding、RoPE、cache 和 decode 六类变量，定位成本会非常高。
