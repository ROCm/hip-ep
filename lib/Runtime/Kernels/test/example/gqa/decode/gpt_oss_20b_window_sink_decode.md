# gpt-oss-20b decode（sliding window + head sink）：改动说明与 8060S 验证指引

> **本文适用对象**：在 AMD Radeon 8060S（Ryzen AI Max+ 395, Strix Halo, gfx1151）上做验证的同学。
>
> **开发机**：gfx1150，`multiProcessorCount = 8`。所有本机数据都在这台上采集，**它只有 8060S 五分之二的
> SIMD 数量**，且实测过程中出现明显降频，因此本机的绝对数字只能当量级参考，结论以 8060S 为准。

---

## 0. 给 8060S 的待办清单

按顺序做这四步，每步都有明确的预期值和判定标准，详见[第 6 节](#6-8060s-上的验证步骤)。

| # | 做什么 | 判定 |
|---|--------|------|
| 1 | 确认派发路径已切到 v2 | debug 日志出现 `flash GQA decode (fp16)`，且 **不再**出现 `flash GQA decode (legacy)` |
| 2 | 看 autotune 给全注意力层选了什么 | `skv >= 4096` 时预期选 `scalar`，`splits` 落在 32–64 |
| 3 | 反算全注意力层的有效带宽 | 应显著高于本机的 ~65 GB/s；到 150 GB/s 以上说明占用率问题已解决 |
| 4 | gpt-oss-20b 端到端 TPS + 输出质量 | 相对撤销 WMMA 前应有提升，长上下文提升更明显 |

---

## 1. 起点：为什么先撤销了 `use_wmma`

gpt-oss-20b 在 40 CU 机器上 decode 变慢。定位到 PR #438（`9fc5085` / `c7c407f`，*[feat]:add opt for normal GQA*）给
`hip_gqa_flash_decode` 加了一个 WMMA split-K kernel，并在 `d == 64` 时默认走它。

问题出在这个 launcher 被**写死** `K_SPLITS = 8`：

- WMMA kernel 的 grid 是 `(G, splits, B)`，**每个 block 只有 1 个 wave**（32 线程）。
- 对 gpt-oss-20b（`G=8, B=1`）就是 `8 x 8 x 1 = 64` 个 block，也就是 **64 个 wave**，**与 GPU 多宽无关**。
- 在 gfx1150（32 个 SIMD）上，64 个 wave 每个 SIMD 分到 2 个，够用，所以本机测出来 WMMA 赢。
- 在 8060S（80 个 SIMD）上，64 个 wave **连每个 SIMD 铺一个都不够**，五分之一的 SIMD 完全空转。

**`b9cc93b` — `revert(gqa): stop dispatching the WMMA kernel on the legacy decode path`**

把 `hip_gqa_flash_decode` 恢复成 #438 之前的函数体（无条件走 scalar split-K，去掉 `HIPDNN_GQA_DECODE_SCALAR/_WMMA`
覆盖）。`legacy_gqa_flash_decode_wmma_kernel` 作为**未实例化的模板**保留（不产生任何代码），并在注释里写明重新
启用前必须先解决的问题。已验证：恢复后该函数与 `204d799` 逐字节一致（仅差强制加的 `legacy_` 前缀）。

---

## 2. 真正的发现：#438 动的那条路径，正是 gpt-oss-20b 唯一走的路径

撤销之后继续查，发现问题比"WMMA kernel 选错了"更深一层——**gpt-oss-20b 的 24 层 GQA 根本就没走到优化过的
`hip_gqa_flash_decode_v2` 上**。

绕路是这样发生的：

1. `fused_supported` 谓词拒绝任何 `local_window_size > 0`、`head_sink` 或 `smooth_softmax`。
   gpt-oss-20b 全部 24 层都带 head sink，其中 12 层还带 `window = 128`，**全部被拒**。
2. 于是全部掉进 `gqa_forward_hipblaslt`。在那里 `fused_predicate` 又把 decode 情况抓回来，
   调用 **legacy** 的 `hip_gqa_flash_decode`——也就是 #438 改动的那个函数：8 splits 写死、
   没有 autotune、不支持 INT8、只支持 HpG ∈ {4,8}。
3. 如果 `skv` 低于 flash-decode 的最小阈值（256），连这道门都过不去，decode 直接退化成
   decomposed GEMM 流水线（要物化 `[H, skv]` 的完整 score 矩阵）。
4. 与此同时，`hip_gqa_flash_decode_v2` 被硬编码传入
   `local_window_size = 0, head_sink = nullptr, smooth_softmax = 0`——
   **v2 的运行时可调 split 数和逐 shape 的 (impl, split) autotune，恰好永远够不到最需要它们的那些层。**

这解释了为什么 #438 的改动会被 gpt-oss-20b 精确命中：它改的正是这个模型唯一能走到的那条 decode 路径。

---

## 3. `6d04f74` — 把 window/sink decode 接到 v2 上

**没有写任何新 kernel。** v2 早就完整实现了这两个特性：split kernel 里的 `kv_lo` 窗口裁剪、
reduce kernel 里把 sink 折进 softmax 分母；而且它本来就用**窗口长度**而不是全上下文长度作为
autotune 的 key（滑窗层和全注意力层因此会各自独立调优）。缺的只是把参数传进去。

改动：

- `gqa_forward_fused` 新增 `local_window_size` / `head_sink` / `use_smooth_softmax` 三个参数，
  透传给 `hip_gqa_flash_decode_v2`，不再置零。
- `fused_supported` 里引入 `window_sink_ok`，**只对 decode** 放开 window/sink 限制。
- **prefill 维持原状**（WMMA prefill 不支持这两个特性），并加了防御性断言，一旦收到就报错。

副作用：`gqa_forward_hipblaslt` 里那个 legacy `hip_gqa_flash_decode` 调用变成**不可达**了——
一个 decode 要同时走到那个函数并通过 `fused_predicate`，它的 HpG 必须落在 {1,2,3,4,5,8,16} 之外，
而 `legacy_flash_decode_geometry_ok`（只认 HpG ∈ {4,8}）永远不会接受这种情况。
**代码暂时保留一个周期**，这样只要翻转 `window_sink_ok` 一个布尔量就能整体回滚；kernel 本体等
8060S 验证通过后再单独删。

### 本机（gfx1150）实测

| 项目 | 结果 |
|------|------|
| legacy vs v2 数值一致性 | rel-L2 ~3e-4，max-abs ~6e-5，`total_seq` 128–8192 全覆盖（差异来自 split 数不同导致的 fp16 求和顺序） |
| 24 层单 token decode | 1.33x（skv=8192）→ 4.38x（skv=128） |
| 其中滑窗层 | 3.6–4.7x |
| 其中全注意力层 | 1.3–2.6x |

---

## 4. `956cdd0` — 让 decode 的配置选择能随设备变宽

### 4.1 autotune 的测量可信度

原来每个候选固定测 2 次预热 + 10 次迭代。滑窗层单次约 3µs，10 次才 30µs，**启动抖动完全盖过了信号**：
在**完全相同**的配置上反复采样，测出来的值有 1.4–2.3x 的离散度。后果是 tuner 有约 6% 的概率把一个
慢 2.4x 的配置写进缓存，然后**整个进程生命周期**里所有滑窗层都用它。

改成 `timeDecodeCandidate()`：先探测 3 次估算单次耗时，再按 **~1ms 总 GPU 工作量**反推迭代次数
（1–4096 自适应），跑 3 轮取**最快的一轮**（最小值是受系统干扰最少的样本，正是排序配置时想要的）。

### 4.2 放宽候选集：`kMaxKeysPerSplit` 256 → 1024

这个下界防的是"split 太少导致每个 block 串行扫描过多 key"的悬崖（skv=8192 时 splits=2 要 1.13ms，
splits=4 要 0.59ms，而 splits>=8 只要 <=0.36ms）。但 256 这个值把 `splits=16` 也挡掉了
（skv=4096 时下界是 17，skv=8192 时是 33），而 **16 恰好是这两个尺寸下 scalar 和 WMMA 的共同最优解**。
原来收得这么紧是因为旧的 10 次迭代 tuner 可能噪声性地选到灾难配置；现在测量可信了，可以放宽。
1024 仍然挡得住悬崖（skv=8192 时下界还有 8）。

### 4.3 split 数兜底改为按设备推导

`hip_gqa_flash_decode_v2` 里原来的启发式默认是写死的 `{WMMA, splits = 8}`——正是[第 1 节](#1-起点为什么先撤销了-use_wmma)
那个 64-wave 问题。新增两个函数：

```cpp
flashDecodeTargetWaves()    // 目标 wave 数 = 设备 wave 槽位总数 / 4（约 4 waves/SIMD）
flashDecodeDefaultSplits()  // 从候选集里取「第一个能达到目标」的 split 数
```

> **踩过的坑：`multiProcessorCount` 在 RDNA 上返回的是 WGP，不是 CU。**
> 8060S 物理 40 CU，但这个字段返回 **20**；开发机返回 8，实际是 16 CU。
> 佐证是 `maxThreadsPerMultiProcessor = 2048`，即 64 个 wave32 槽位 = 4 个 SIMD32 × 16 waves，
> 这正是一个 RDNA3 WGP 的规格（单个 CU 只有 1024）。
> 所以代码**不去数 SIMD**，直接用 `multiProcessorCount * (maxThreadsPerMultiProcessor / 32)`
> 得到设备的 wave 槽位总数，绕开 WGP/CU 的歧义。

候选来自 `flashDecodeCandidateSplits()` 而不是现算，这样兜底值会自动继承它的 `min_splits` 下界——
否则光看占用率的话，scalar 在 skv=8192 时会选 `splits=1`（已经有 64 个 wave 了），正好踩进 4.5x 的串行扫描悬崖。

实际取值（把 `flashDecodeCandidateSplits()` + `flashDecodeDefaultSplits()` 按两台设备的
`multiProcessorCount` 复算得出；在机器上可以用 `HIPDNN_GQA_DECODE_NOAUTOTUNE=1` 加调试日志直接核对）：

| 设备 | wave 槽位 | 目标 | kv_len=128 (WMMA) | kv_len=8192 (WMMA) | kv_len=8192 (scalar) |
|------|-----------|------|-------------------|--------------------|-----------------------|
| gfx1150 开发机（8 WGP / 32 SIMD） | 512 | 128 | 8 → 64 waves, 2.0/SIMD | 16 → 128 waves, 4.0/SIMD | 8 → 512 waves, 16/SIMD |
| **8060S（20 WGP / 80 SIMD）** | 1280 | 320 | 8 → 64 waves, **0.8/SIMD** | **48 → 384 waves, 4.8/SIMD** | 8 → 512 waves, 6.4/SIMD |

注意最后一行第一列：`kv_len=128` 的滑窗层在 8060S 上，WMMA 无论怎么选 split 都填不满设备
（候选集被 `max_useful` 限制在 {2,4,8}）。这是**固有的**——128 个 key 的窗口本来就没有足够并行度，
而且滑窗层只占单 token 时间的 0.34%（见下节），不值得为此改 kernel。

**这个兜底只在 autotune 被关掉或被环境变量覆盖时生效**；正常路径上 autotune 的结果会覆盖它。

---

## 5. 理论分析：为什么 WMMA 在 decode 上是错的工具

本机降频导致实测不可信之后，改用解析模型。以下量都可以从 kernel 的 grid 几何和 KV cache 布局直接算出。

### 5.1 decode 是纯带宽瓶颈，算力过剩 25 倍

一个全注意力层：

```
读取字节 = 2(K,V) x G x skv x d x 2 = 2048 x skv
浮点运算 = 2(QK,PV) x H x skv x d x 2 = 16384 x skv
算术强度 = 8 FLOP/byte   （与 skv 无关）
```

8060S 的 machine balance 约 50 TFLOPS ÷ 250 GB/s ≈ **200 FLOP/byte**。也就是说这个 kernel
只需要 **4% 的峰值算力**就能吃满带宽。WMMA 唯一的优势是矩阵吞吐，而这里根本不缺算力。

### 5.2 而 WMMA 的 grid 几何反而砍掉了访存并行度

| 实现 | grid | block | 每 block wave 数 | 启动 wave 数（B=1,H=64,G=8） |
|------|------|-------|------------------|------------------------------|
| scalar | `(B*G, splits)` | `HpG*32` | **8** | `64 x splits` |
| WMMA | `(G, splits, B)` | `32` | **1** | `8 x splits` |

同样的 split 数，WMMA 启动的 wave 数少 8 倍，访存并行度（MLP）也就少 8 倍。
在 8060S 的 80 个 SIMD 上：

| splits | scalar waves/SIMD | WMMA waves/SIMD |
|--------|-------------------|-----------------|
| 8 | 6.4 | **0.8** |
| 16 | 12.8 | 1.6 |
| 32 | 25.6 | 3.2 |
| 48 | 38.4 | 4.8 |
| 64 | 51.2 | 6.4 |

WMMA 要靠堆 split 才能填满设备，而 split 超过 48 之后 partials 流量会涨到 KV 流量的 13% 以上，开始反噬。

### 5.3 时间预算：滑窗层已经不值得再优化了

按 200 GB/s 有效带宽、gpt-oss-20b（3.6B 激活参数 @ MXFP4 ≈ 1.91 GB 权重读取）估算单 token：

| 上下文 | 权重读取 | 全注意力 ×12 | 滑窗 ×12 | 合计 | 滑窗占比 |
|--------|----------|--------------|----------|------|----------|
| 1024 | 9.60 ms | 0.126 ms | 0.036 ms | 9.76 ms | 0.37% |
| 4096 | 9.60 ms | 0.504 ms | 0.036 ms | 10.14 ms | 0.36% |
| 8192 | 9.60 ms | 1.008 ms | 0.036 ms | 10.64 ms | **0.34%** |

滑窗层（window=128）每层只读 262 KB，12 层合计 3.1 MB，完全落在 32 MB MALL 里，
真实访存约 0.4µs——实测的 3µs 基本全是两次 kernel 启动加中间那道依赖 barrier。
**把它优化到 0，TPS 也只涨 0.3%。** 收益早在 `6d04f74` 把它从 decomposed 流水线搬走时就拿到了
（单层访存从约 24 MB 降到 262 KB）。

**剩下唯一还能动到主导项的杠杆是 INT8 KV cache**：全注意力层 8k 下每层读 16.78 MB，
INT8 直接减半，12 层省约 0.50 ms/token，约 +4.7% TPS。kernel 侧已完整支持
（`HIP_KV_DTYPE_INT8`），且 `6d04f74` 放开的派发谓词已允许 quantized KV 走 window+sink 路径——
缺的是导出侧要让模型带上 KV 量化属性。

---

## 6. 8060S 上的验证步骤

### 前置：打开调试日志

```powershell
$env:HIPDNN_EP_DEBUG = "1"
```

相关日志行（`stderr`）：

```
[REAL] flash GQA decode (fp16): B=1 skv=... H=64 G=8 d=64 max_splits=64 window=128 sink=1 smooth=1
[custom_kernels]   flash_decode cand impl=wmma   splits= 8 : 0.0271 ms
[custom_kernels] flash_decode autotune: key=0x... -> impl=scalar splits=32 (cache size=2)
```

> `max_splits=64` 是 workspace 容量上限，**不是** autotune 选中的值；实际选择看
> `flash_decode autotune: ... -> impl=... splits=...` 那行。
> autotune 的 key 含窗口裁剪后的长度，所以滑窗层和全注意力层会产生**两条不同的** autotune 记录。

### 步骤 1：确认派发路径已切换

跑一次 gpt-oss-20b decode，检查日志：

- 应出现 `[REAL] flash GQA decode (fp16)`，且 `window=128 sink=1` 的行确实存在（说明滑窗层进了 fused 路径）。
- **不应**再出现 `[REAL] flash GQA decode (legacy)`。

如果还看到 legacy，说明派发没切过去，先查 `fused_supported` 有没有被别的条件挡住（例如 packed-QKV、dtype）。

### 步骤 2：看 autotune 选了什么

抓 `flash_decode autotune` 那几行。**模型预测**：

| 层类型 | 预期 impl | 预期 splits |
|--------|-----------|-------------|
| 全注意力（skv >= 4096） | `scalar` | 32–64 |
| 全注意力（skv ~1024） | 任意 | >= 16 |
| 滑窗（window=128） | 任意 | 8（候选集上限就是 8） |

关键是**全注意力层不应该选到 WMMA + 低 split**。若想看每个候选的具体耗时，
`flash_decode cand` 那些行已经全部打出来了，可以直接确认排序是否合理。

### 步骤 3：反算全注意力层的有效带宽

取 `skv = 8192` 的全注意力层单层耗时 `t`（ms），带入：

```
有效带宽 (GB/s) = 16.78 / t
```

（16.78 MB = `2 x 8 x 8192 x 64 x 2` 字节）

| 结果 | 含义 |
|------|------|
| ~65 GB/s | 和开发机一样，**占用率问题没解决**，回到步骤 2 看选了什么配置 |
| 150–200 GB/s | 符合预期，已接近带宽下限，GQA 这块没有更多空间了 |
| > 200 GB/s | 超出建模假设，说明 MALL 命中率比预期高，很好 |

对照组：手工强制几个配置看差异

```powershell
$env:HIPDNN_GQA_DECODE_WMMA = "1";   $env:HIPDNN_GQA_DECODE_SPLITS = "8"
$env:HIPDNN_GQA_DECODE_SCALAR = "1"; $env:HIPDNN_GQA_DECODE_SPLITS = "32"
```

模型预测在 8060S 上后者明显更快；**如果不是，见[第 7 节](#7-如果预测错了)。**

### 步骤 4：端到端

跑 gpt-oss-20b 完整推理，记录 TPS 和输出质量。长上下文（4k/8k）的提升应该比短上下文明显，
因为滑窗层相对 decomposed 路径省下的绝对时间随上下文线性增长。

### 相关环境变量速查

| 变量 | 作用 |
|------|------|
| `HIPDNN_EP_DEBUG=1` | 打开全部 runtime / kernel 调试日志 |
| `HIPDNN_GQA_DECODE_SCALAR=1` | 强制 scalar 实现（跳过 autotune） |
| `HIPDNN_GQA_DECODE_WMMA=1` | 强制 WMMA 实现（跳过 autotune） |
| `HIPDNN_GQA_DECODE_SPLITS=N` | 强制 split 数（跳过 autotune） |
| `HIPDNN_GQA_DECODE_NOAUTOTUNE=1` | 跳过 autotune，使用[第 4.3 节](#43-split-数兜底改为按设备推导)的设备推导兜底 |
| `HIPDNN_EP_GQA_FLASH_DECODE=0` | 整个关掉 flash decode（退回 decomposed 路径，用于对照） |
| `HIPDNN_EP_GQA_FLASH_DECODE_MIN_SKV=N` | flash decode 的最小 skv 阈值（默认 256） |

---

## 7. 如果预测错了

**情况 A：WMMA 在 8060S 上仍然赢过 scalar。**
说明瓶颈不在访存并行度，而在别处——优先怀疑 scalar kernel 的 LDS staging bank conflict，
或者 `__syncthreads()` 造成的 8 个 wave 串行化。这种情况下**正确的改法是给 WMMA kernel
每个 block 配多个 wave**（让它也能靠 block 内并行度扩展），而不是继续堆 split：
split 超过 48 之后 partials 流量占比会超过 13%，边际收益转负。

**情况 B：全注意力层有效带宽仍停在 ~65 GB/s，且 autotune 确实选了高 split 的 scalar。**
说明模型里"wave 数 → MLP → 有效带宽"这条链断了，瓶颈可能在 L2/MALL 的 miss 处理能力
或 LPDDR5X 的实际可达带宽远低于 200 GB/s。建议先用 rocprof 抓一次
`FETCH_SIZE` / `L2CacheHit` 确认，再决定要不要继续优化 GQA。

**情况 C：滑窗层耗时随 skv 增长。**
说明窗口裁剪没生效，`kv_lo` 的计算或 `local_window_size` 的透传有问题。
这是功能性 bug，优先级高于所有性能问题。

---

## 8. 相关提交

| commit | 说明 |
|--------|------|
| `9fc5085` / `c7c407f` | PR #438，引入 legacy WMMA decode kernel（问题来源） |
| `b9cc93b` | 撤销 legacy decode 路径上的 WMMA 派发 |
| `6d04f74` | 把 window/sink decode 接到 `hip_gqa_flash_decode_v2` |
| `956cdd0` | autotune 测量可信度、候选集放宽、按设备推导的 split 兜底 |

**验证通过后的清理项**：删掉 `gqa_forward_hipblaslt` 里已不可达的 legacy flash decode 分支，
以及 `legacy_gqa_flash_decode_kernel` / `legacy_gqa_flash_decode_wmma_kernel` /
`legacy_gqa_flash_decode_reduce_kernel`（约 700 行）。
