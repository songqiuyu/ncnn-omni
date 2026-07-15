# Qwen3-ASR 长音频 token 分叉诊断

状态：根因已确认，修复已完成，四组端到端 token 回归通过

日期：2026-07-15

## 1. 问题定义

Qwen3-ASR-0.6B 的五个 ncnn 子图均通过 PyTorch 差分测试，短音频也能做到
greedy generation token 完全相等；但较长音频会在生成早期出现 token 分叉。

严格验收要求比较生成 token，而不是只判断转写内容是否“看起来正确”。一旦
greedy argmax 的第一个 token 不同，后续 KV cache 接收的输入也不同，自回归
序列将自然分叉。因此需要定位**第一次分叉前**的数值来源。

## 2. 已观察到的边界

所有端到端对比均使用相同的 16 kHz mono PCM、相同尾部 hop padding、CPU FP32
和 8 个线程。

| 样例 | 音频时长 | Audio tokens | Decoder prompt | 结果 |
|---|---:|---:|---:|---|
| Qwen 官方短中文 | 4.20 s | 55 | 70 | 10 个生成 token 完全相等 |
| 短中文重复两次 | 8.41 s | 110 | 125 | 17 个生成 token 完全相等 |
| Qwen 官方长英文 | 15.05 s | 196 | 211 | 第 7 个生成 token 分叉 |
| 中文后拼接英文 | 19.26 s | 251 | 266 | 第 7 个生成 token 分叉 |

现有 Decoder 常规回归的最大 prefill 为 `S=128`。通过样例的 prompt 为 70/125，
失败样例为 211/266，这个边界使长 Decoder prefill 成为首要怀疑对象。

## 3. 已排除的结构性问题

现有差分测试结果：

- Audio Conv：7 组全部通过，最差 max abs `8.106e-6`；
- Audio Transformer：7 组全部通过，最差 max abs `1.249e-5`；
- Token Embedding：3 组 bit-exact；
- LM Head：4 组全部通过，最差 max abs `1.192e-6`；
- Decoder：decode、prefill、chunked、sentinel 和 prefill→decode 共 12 组通过。

额外运行的合成 prefill：

| Sequence | Final max abs | Relative L2 | Cosine |
|---:|---:|---:|---:|
| 211 | `0.004673` | `3.045e-5` | `0.999999999536` |
| 266 | `0.004593` | `3.156e-5` | `0.999999999502` |

这说明模型没有 shape、mask、RoPE、cache 或算子级崩坏。误差很小，但仍可能改变
两个接近 logit 的排名。

## 4. 初始原因排序（实验前）

### 4.1 初始首要假设：长 dense prefill 的并行归约误差

当前 C++ 一次把完整 prompt 送入 Decoder。Attention 的点积、softmax 和 value
归约顺序会随序列长度改变。ncnn 和 PyTorch 的并行实现不保证使用相同的浮点
累加顺序；层数达到 28 后，小误差传递到最后 hidden state 和 logits。

这不是“误差随 decode 次数累计”造成的第一次分叉：失败发生前，两个后端收到的
生成 token 仍相同。真正的触发点是某一步 top-1/top-2 logit margin 小于后端数值
差异，argmax 才变成不同 token。分叉之后才进入自回归累计。

### 4.2 次要原因：音频前端和 Audio Encoder 的微小差异

C++ Log-Mel 精确复现 Hugging Face NumPy STFT 数学定义，而端到端 PyTorch 参考在
安装了 torch 时使用 `torch.stft`。两条官方实现本身可有约 `2e-5` 的 Log-Mel
差异。实验前还假设 Audio Transformer 的 104-token 分窗与上游 block mask 数学
等价、只存在浮点归约顺序差异；后续源码和 A/B 实验证明 CPU/SDPA 根本未应用
该 block mask，因此实际是执行语义不同，详见 7.5 节。

这些差异可能传入 Decoder。目前真实音频链路没有导出每阶段完整 tensor，因此
不能把全部责任只归给 Decoder。

### 4.3 放大机制：greedy argmax 不连续

Embedding 和 LM Head 自身几乎无误差，也不能保证 token 一致。只要最终 hidden
使两个接近的 logits 交换顺序，argmax 就会离散跳变。长英文样例第一次差异只是
逗号与 `yeah` 的选择；中英拼接样例随后出现更大文本差异，是第一次分叉后的正常
自回归结果，不代表后续每一步都发生了新的模型转换错误。

## 5. 第一项实验：可配置 chunked prefill

对 causal Decoder，把 prompt 按顺序分块并使用前块 KV cache，在数学上等价于
一次 dense prefill。首轮计划使用 64-token chunk：

```text
[0,64) -> cache
[64,128) + cache -> cache
[128,192) + cache -> cache
...
```

目的：

1. 让每次 Decoder 调用落在已经充分验证的短序列范围；
2. 减小单次 Attention 并行归约规模；
3. 降低 prefill 临时内存；
4. 保持真实 position、sentinel mask 和最终 KV 语义不变。

判定实验：

- `chunk=0` 保留原 dense prefill，作为对照；
- `chunk=64` 对四个现有样例重新做逐 token 对比；
- 若长样例恢复一致，说明 dense prefill 数值归约是主因；
- 若仍分叉，则下一步必须比较真实音频的 Log-Mel、Audio Conv 输出、Audio
  Transformer embedding、首轮 Decoder hidden 和 logits margin。

## 6. 当前不能过度声明的结论

- 不能把“所有子图通过”直接等价成“所有端到端 token 必然相同”；
- 不能把中英拼接后的大文本差异误判为 Audio Encoder 完全错误；
- 不能在没有真实中间 tensor 对比前宣称问题只来自 Decoder；
- 不能用最终文本语义接近替代严格 token parity。

## 7. 实验结果与根因收敛

### 7.1 Chunked Decoder prefill：否定

长英文使用 `chunk=64` 后，第 7 个生成 token 仍与 dense prefill 完全相同地
分叉；prefill 耗时由约 0.32 秒增加到 1.35 秒。因此没有采用该修改，并撤回了
相关 API。Decoder 长 prefill 不是本次问题的主因。

### 7.2 NumPy 与 torch STFT：否定

强制 Transformers 使用 NumPy Whisper Log-Mel 后，仍生成原始正确序列，没有
转向 ncnn 的分叉序列。因此约 `2e-5` 的前端实现差异不是主因。

### 7.3 Hybrid audio embedding：确认问题在 Audio Encoder

对同一份 NumPy Log-Mel 比较完整 Audio Encoder：

```text
PyTorch vs ncnn audio embedding
shape      [196,1024]
max_abs    0.130849
mean_abs   0.002524
cosine     0.970578
```

再把 ncnn audio embedding 注入原始 PyTorch Decoder，PyTorch 生成了与 ncnn
完全相同的分叉序列。这一实验排除了 Text Decoder、LM Head 和 tokenizer，确认
错误已经存在于 Audio Encoder 输出。

### 7.4 根因一：尾 Mel chunk 的卷积 padding 语义错误

长英文共有 1506 个 Mel 帧，最后一个 chunk 只有 6 帧。上游先把所有 chunk
用 `pad_sequence` 补到该音频的最大 chunk 宽度 100，再运行三层 Conv，最后按
真实下采样长度裁剪。旧 C++ 直接以宽度 6 运行尾块。

两者不等价：显式输入零经过第一层 Conv bias 和 GELU 后会产生非零激活，并参与
后续卷积；缩短 tensor 后，每层边界 padding 都是纯零。逐行对比发现：

```text
Audio Conv + position rows 0..194: 约 1e-6 误差
row 195 before fix: max_abs=1.292632, cosine=0.940151
row 195 after fix:  全 tensor max_abs=1.001e-5, cosine≈1
```

修复：当音频含有完整 100 帧 chunk 时，将尾 chunk 显式补到 100，运行 Conv 后
只保留 `ceil(real_width/8)` 个输出。

### 7.5 根因二：CPU/SDPA Audio Attention 被错误切窗

旧 C++ 根据 `n_window_infer` 把 Audio Transformer 输入切成最多 104 token 的
独立窗口。这是上游 FlashAttention varlen 路径的意图，但当前对齐参考是 CPU
SDPA。上游 CPU 路径把 `cu_seqlens` 传入 SDPA 接口，却没有传 block attention
mask；SDPA 不使用 `cu_seqlens`，所以全部 196 token 实际进行全局 Attention。

用完全相同的 PyTorch 层做 A/B：

```text
上游一次全局执行 vs PyTorch 104/92 分窗执行
max_abs    0.130849
mean_abs   0.002537
cosine     0.970789
```

这与 ncnn 旧输出的差异一致，证明是调度语义错误，不是 ncnn Attention 算子精度
问题。将同一 ncnn Audio Transformer 直接运行 `L=196`：

```text
PyTorch CPU global vs ncnn global
max_abs    1.173e-6
mean_abs   3.884e-8
cosine     0.999999999996
```

修复：CPU FP32 对齐模式下将完整 Audio Conv token 序列一次送入 dense Audio
Transformer。新增 `L=196`、`L=251` 模块回归，均通过。

## 8. 修复后端到端结果

| 样例 | Decoder prompt | 修复前 | 修复后 |
|---|---:|---|---|
| 官方短中文 4.20 s | 70 | PASS 10 tokens | PASS 10 tokens |
| 中文重复 8.41 s | 125 | PASS 17 tokens | PASS 17 tokens |
| 官方英文 15.05 s | 211 | FAIL at token 7 | PASS 47 tokens |
| 中英拼接 19.26 s | 266 | FAIL at token 7 | PASS 61 tokens |

四组均为生成 token ID 逐项相等，而非只比较最终文本。

## 9. 适用范围和后续设计约束

- 当前目标是复现 `example_qwen3_asr_transformers.py` 在 CPU/SDPA 下的行为，
  因此必须使用全局 Audio Attention；
- FlashAttention varlen 路径可能使用 104-token window，它与当前 CPU 参考不是
  相同数值语义，未来必须作为独立执行策略和独立 parity 基准；
- 全局 Attention 的时间/内存复杂度为 `O(L²)`，超长音频不能直接依赖当前策略；
- 模型包 manifest 后续应声明 Audio Attention policy，运行时不能根据性能需要
  静默切窗，否则会改变模型输出；
- 尾 chunk padding 属于模型处理器契约，需要进入长期集成回归。

这次问题说明：单个动态 shape 子图通过，不足以证明图外 batching、padding、
window scheduling 与上游一致。完整部署验收必须保留真实长音频端到端测试。
