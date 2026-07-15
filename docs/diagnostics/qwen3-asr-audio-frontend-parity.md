# Qwen3-ASR 音频前处理对齐报告

状态：CPU FP32 支持范围内通过

日期：2026-07-15

## 1. 验收目标

证明 ncnn-omni 实际 C++ 路径送入 Audio Encoder 的完整 `input_features` 与
Qwen3-ASR Transformers 路径一致，而不是只比较少量采样点或依赖最终文本。

当前支持范围是 mono 16 kHz PCM16/float32 WAV。URL/Base64、立体声转 mono 和
重采样仍由上层负责，不在第一版 C++ 前处理范围内；不支持的输入会明确拒绝。

## 2. 实际参考路径

参考端直接调用 Qwen3-ASR 使用的代码：

```text
SoundFile float32 decode
-> qwen_asr.inference.utils.normalize_audio_input
-> final-hop compatibility canonicalization
-> Transformers WhisperFeatureExtractor (torch STFT path)
```

ncnn 端比较的是诊断二进制从实际 C++ 类导出的：

```text
load_pcm_wav normalized PCM
-> prepare_qwen3_asr_frontend_samples
-> WhisperLogMel::compute
```

逐阶段检查 sample rate、channels、PCM dtype/shape/value、canonical PCM、feature
attention mask、Log-Mel dtype/shape 和完整 tensor 数值。

## 3. 上游 partial-hop 不一致

原始 `WhisperFeatureExtractor` 对未对齐到 160 samples 的音频执行 centered STFT，
删除最后一帧后得到 `floor(N/160)` 个 Mel 帧；attention mask 以步长 160 切片后
长度却是 `ceil(N/160)`。原始 Qwen3-ASR 将后者作为 `feature_lens`，会在 Audio
Encoder 的 `split_with_sizes` 中失败。

官方短中文的未修改路径已经实际复现：

```text
normalized samples       67263
input_features           [1,128,420]
feature_attention_mask   [1,421], sum=421
generate                 RuntimeError: split sizes sum to 421, input length is 420
```

四条测试音频均不是整 hop。为了让参考端和部署端具有同一份有效输入，定义以下
兼容规范：归一化后只在右侧补 `0..159` 个零，使长度成为 160 的整数倍，然后两端
从同一 canonical PCM 开始执行 Qwen Whisper 前端。该步骤不截断或修改任何原始
样本。parity 报告同时保留未补零的上游 shape/mask，不能隐藏这项兼容差异。

## 4. 参数对齐

| 参数 | 值 |
|---|---:|
| sampling rate | 16000 |
| channels | 1 |
| dtype | float32 |
| normalize | Qwen `float_range_normalize` |
| padding | batch longest；额外 final-hop compatibility pad |
| truncation | false |
| n_fft | 400 |
| hop length | 160 |
| window | periodic Hann |
| center padding | reflect, 200 samples |
| power | 2.0 |
| Mel bins | 128, Slaney normalized |
| dither | 0.0 |
| log | log10, clamp 1e-10 |
| dynamic range | max - 8.0 |
| output scale | `(x + 4.0) / 4.0` |

C++ 使用确定性的 double 累加直接 DFT，参考使用 PyTorch FP32 STFT，所以要求数值
容差而不是 bit-exact Log-Mel。归一化 PCM 本身要求逐样本相等。

## 5. 完整 tensor 结果

| 音频 | 原始 samples | 补零 | Log-Mel shape | PCM max abs | Mel max abs | Mel mean abs | Cosine |
|---|---:|---:|---:|---:|---:|---:|---:|
| 短中文 | 67263 | 97 | `[128,421]` | `0` | `1.955e-5` | `1.854e-7` | `0.999999999999654` |
| 中文重复 | 134526 | 34 | `[128,841]` | `0` | `1.955e-5` | `1.866e-7` | `0.999999999999661` |
| 长英文 | 240820 | 140 | `[128,1506]` | `0` | `2.229e-5` | `1.278e-7` | `0.999999999999674` |
| 中英拼接 | 308083 | 77 | `[128,1926]` | `0` | `1.884e-5` | `1.117e-7` | `0.999999999999722` |
| float32 越界正弦 | 16001 | 159 | `[128,101]` | `0` | `2.122e-5` | `2.436e-8` | `0.999999999999769` |

验收阈值：PCM max abs `<=1e-7`；Mel max abs `<=5e-5`、mean abs `<=5e-6`、
cosine `>=0.99999999`。四组真实 PCM16 音频和一组峰值大于 1 的 float32 边界
输入全部通过；后者还验证了 Qwen `float_range_normalize` 的缩放和裁剪语义。

## 6. 复现命令

先构建 `ncnn-omni-qwen3-asr-frontend-dump`，再运行：

```bash
python tools/parity/qwen3_asr_frontend_parity.py \
  --dump-binary build/ncnn-omni-qwen3-asr-frontend-dump \
  --assets /path/to/Qwen3-ASR-0.6B \
  --audio /path/to/audio.wav \
  --report /tmp/qwen3_asr_frontend_parity.json
```

`--audio` 可重复传入。工具会以非零退出码报告 shape、PCM、mask 或完整 Log-Mel
超出容差，并在 JSON 中记录未修改 Qwen processor 与 canonical processor 两套
长度信息。

## 7. 结论

在第一版明确支持的 mono 16 kHz WAV 范围内，Qwen 音频归一化、兼容 padding、
Whisper 参数、完整 Audio Encoder 输入 shape 和数值均已对齐。兼容 padding 是对
已复现上游 partial-hop shape/mask 不一致的显式处理，不应被描述为原始 processor
本身的步骤；PyTorch parity 必须应用同一规则。
