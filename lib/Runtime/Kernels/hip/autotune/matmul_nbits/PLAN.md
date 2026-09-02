# MatMulNBits 离线 autotune 表：现状与后续计划

本文记录截至目前的改动、实测数据，以及三项已定方向的执行计划。
后续工作基于本文继续。

---

## 一、已完成：三级分桶 → 两级最近邻

### 改了什么

原设计是三级：Exact（逐 shape 精确）→ Fuzzy（N/K 塌缩成八度桶，桶内多数票）→
Fallback。问题出在 Fuzzy：多数票会**丢掉离查询最近的那次实测**，改用"桶里赢得
最多的那个 config"。桶跨度是每维一倍，所以一个 shape 可能拿到比表里已有的更远的
点测出来的 config。数据里就有受害者——`N=10240 K=5120` 实测最优 `thr=32 tile_n=2`，
被同桶多数票改成 `thr=64 tile_n=4`。

现在改成 Tensile 的做法：

```
类别键精确匹配 (phase, bits, group_size, zero_point, row_stride)
    └─> 组内在 (M, N, K) 的对数空间取加权最近邻
            距离 0 ─ 即该 shape 被实测过 ─ 就是精确命中
    └─> 组不存在或点全被拒 → 该 phase 的 fallback
```

Exact 不再是独立一级，而是最近邻距离为 0 的退化情形。因此原来纠结的
"剪枝"问题自动消失：没有冗余层可剪。

距离度量：

```
d = sqrt( (wm·log2(M/Mp))² + (wn·log2(N/Np))² + (wk·log2(K/Kp))² )
```

权重**存在表里**（`weight_m/n/k`），不写死在代码里。这样离线拟合和运行时度量
不会走偏——重新拟合权重只是重新生成表，不需要代码跟着改。

类别键为什么必须精确匹配、不能算距离：group size、zero-point 布局改变 kernel
每步读什么；padded row stride 会直接改变谁是最优 tile（实测 `N=5120 K=4096`
从 cfg11 变 cfg3）。这些维度上做插值没有意义。

### 表的变化

| | 改之前 | 改之后 |
|---|---|---|
| 行数 | 3219（1963 Exact + 1206 Fuzzy + 3 Fallback） | 2010 点 + 3 fallback |
| 分组 | — | 16 个类别组 |
| config 池 | 93 | 92 |
| `.fb` | 64.1 KB | **40.5 KB** |
| JSON | 0.66 MB | 0.31 MB |

### 改动的文件

| 文件 | 内容 |
|---|---|
| `matmul_nbits_autotune.fbs` | schema v3：`MatmulNbitsTunePoint` / `MatmulNbitsFallback` / `weight_*`；删掉 `MnTier` `MnMBucket` `MnDimBucket` |
| `matmul_nbits_autotune.h` | `Source::Fuzzy` → `Source::Nearest`；`Result` 增加 `distance`；`Stats` 字段改名 |
| `matmul_nbits_autotune.cpp` | 重写 `resolve()` 为最近邻；载入时预算 log 并按类别键分组 |
| `scripts/update_lut.py` | build 产出点集；新增 `fit` 子命令；解析每 config 耗时；净化假读数 |
| `tools/matmul_nbits_lut_test.cpp` | 距离断言取代分层断言 |
| `tools/matmul_nbits_lut_query.cpp` | 输出距离直方图 |
| `matmul_nbits_kernel.hip` | 只改了注释和日志（`src=%d d=%.3f`），逻辑未动 |

### 尚未验证

**本地不能构建 hip-ep**，所以 C++ 查找路径只做了代码审查，没有编译验证。
见第五节。

---

## 二、发现的 bug：在线 autotuner 会选中"幽灵读数"

### 现象

sweep 日志里带有每个 config 的完整耗时表。统计 2018 个实测点：

- **21 个点**存在耗时低于 1 µs 的 config（5 次 launch 合计不到 1 µs，
  比发起这些 launch 本身还快）
- 这 21 个点，**tuner 全部选中了它**（100%）

100% 的命中率就是结论：所有 tuner 都取最小值，假读数一旦出现必然获胜。
真正快的 config 不可能每次都恰好是最小值。

典型例子：

```
prefill M=512 N=512  K=1536  gs=32   → 选中 0.00020 ms，次优 0.02940 ms（147x）
prefill M=8   N=1024 K=2560  gs=128  → 选中 0.00020 ms，次优 0.03540 ms（177x）
dp4a    M=1   N=768  K=768   gs=32   → 选中 0.00020 ms，次优 0.00180 ms
```

### 根因

所有 tuner 的计时循环都没有任何 plausibility 检查：

```c++
hipEventRecord(ev0, stream);
for (int i = 0; i < ITERS; i++)
    launchWmmaConfig(...);
hipEventRecord(ev1, stream);
hipEventSynchronize(ev1);

float ms = 0.0f;
hipEventElapsedTime(&ms, ev0, ev1);
if (ms < cfg_ms[cid]) cfg_ms[cid] = ms;   // ← 取 min，假读数必胜
```

WMMA 还额外取两遍的 min，等于给假读数两次机会。

同一 config 在相邻 M 上不复现（`N=1024 K=2560` 的 M=8 和 M=256 命中的是不同
config），说明是**瞬时计时抖动**，不是某个 config 恒定失效。

### 影响范围

不只是 LUT。**生产环境在线 autotune 遇到这些 shape 时会缓存同样的错误 config**，
并写进磁盘缓存长期生效。

### 目前的处置

生成侧已加净化（`update_lut.py` 的 `drop_implausible`）：丢弃低于该点中位数
20 倍的读数，再重算 winner。效果：

| | 净化前 | 净化后 |
|---|---|---|
| p99 | 11.08x | 3.51x |
| max | 177x | 14x |

实际丢弃 24 条读数、修复 31 个点。

**内核侧未动。** 那是生产代码，改动方案见第四节。

### 一个附带结论

顺带验证了 Phase 1 的 tie-break 优化（偏向大 warp tile）是干净的：
tuner 自己选的 winner 只比该点最快耗时慢 **0.2%**（geomean 1.0021，
p90 1.0083，100% 在 5% 以内）。所以 `update_lut.py` 保留日志里的 winner，
而不是自己重算 argmin——重算会把这个优化抹掉。只有被幽灵读数污染的点才重算。

---

## 三、实测数据：为什么"加密网格"不管用

原本的想法是补合成 shape 填满对数空间，让最近邻更准。数据不支持这个方向。

### 覆盖度收益曲线（留一整个 shape）

| 覆盖度 | 几何均值 | 中位 | p90 | 5% 以内 |
|---|---|---|---|---|
| 20% | 1.2410 | 1.1016 | 1.789 | 38.8% |
| 40% | 1.2007 | 1.0749 | 1.583 | 42.7% |
| 60% | 1.1819 | 1.0691 | 1.511 | 44.5% |
| 80% | 1.1646 | 1.0585 | 1.464 | 47.3% |
| 100% | 1.1563 | 1.0549 | 1.436 | 48.7% |

shape 数翻 2.5 倍，几何均值只从 1.20 降到 1.16。

### 距离与代价的相关性

| 最近邻距离（八度） | 点数 | 几何均值代价 |
|---|---|---|
| ≈0 | 1344 | **1.1415** |
| 0.5 | 536 | 1.1719 |
| 1.0 | 57 | 1.2995 |
| 3.0 | 54 | 1.2151 |

**最近邻就在隔壁的那 1344 个点，代价已经是 1.14。** 距离拉远 3 个八度也才 1.22。
说明代价的主体不是"距离远"，加密网格补不上。

### 排除了噪声的可能

| 指标 | 结果 |
|---|---|
| 同一 (点, config) 重复测量 max/min | 中位 1.0105，p90 1.0310，p99 1.0790 |
| 每点第二名 / 第一名 | 中位 1.0152，p90 1.1339 |
| 每点 2% 以内的 config 数 | 中位 2 个 |
| 每点 5% 以内的 config 数 | 中位 4 个 |

噪声很小，而 config 分布很尖锐。所以 1.15x 是**真实损失**：最优 config 对 shape
是不连续的，隔壁 0.25 个八度的 shape 最优解就已经不同。

### 结论

最近邻的天花板大约就是 **1.15x**，靠加密网格突破不了。
补 shape 该补的是**真实模型 shape，让它们直接精确命中**（精确命中 ≈ 1.00x）。

表才 40 KB、每点 20 字节，加到 10000 点也只有 200 KB，容量完全不是约束。

---

## 四、oga_models 当前覆盖情况

**回答"oga_models 是否都能精确命中"：不是 100%，有 37 个 shape/path 组合缺失。**

按实际可达路径统计（padrow 门槛 `shouldPadRow(K)`：`K/2 % 128 == 0` 且
`gcd(K/2/128, 64) >= 8`，成立走 Padded，否则走 Arrival，两者只有一个会被走到）：

| phase / stride | 查询数 | 精确命中 | 覆盖率 |
|---|---|---|---|
| Decode / Any | 140 | 139 | 99.3% |
| DecodeDp4a / Any | 140 | 139 | 99.3% |
| Prefill / Arrival | 870 | 770 | 88.5% |
| Prefill / Padded | 530 | 481 | 90.8% |

M 阶梯：prefill 是 `8,16,32,64,128,256,512,1024,2048,4096`（10 档），
decode/dp4a 固定 M=1。

### 缺口的三类成因

**(a) 词表投影（lm_head）全 M 缺失** — 主 sweep 有 10 GB 输出缓冲上限，
补测只覆盖了 4 个模型。完全没有数据的：

| N | K | gs | zp | 模型 |
|---|---|---|---|---|
| 100352 | 5120 | 32 | Sym | phi4-14B |
| 128256 | 4096 | 128 | Asym | Llama-3.1-8B |
| 128256 | 8192 | 128 | Sym | DeepSeek-R1-Distill-Llama-70B |
| 152064 | 5120 | 128 | Asym | Qwen2.5-14B / Qwen2.5-Coder-14B |
| 201088 | 2880 | 32 | Sym | gpt-oss-20b |
| 248320 | 2048 | 32 | Sym | Qwen3.5-35B-A3B / Qwen3.6-35B-A3B |
| 248320 | 4096 | 32 | Sym | Qwen3.5-9B |
| 248320 | 5120 | 32 | Sym | Qwen3.6-27B |
| 248320 | 5120 | 128 | Asym | Qwen3.8-27B |
| 262144 | 2816 | 32 | Asym | gemma-4-26B-A4B |
| 262208 | 2560 | 128 | Sym | gemma3-4b |

**(b) 大 N 的高 M 档缺失** — 同样是输出缓冲上限，`M=2048/4096` 被跳过。
涉及 20 个 shape，例如 `N=17408 K=5120`（缺 M=2048,4096）、
`N=28672 K=8192`（缺 M=2048,4096）、`N=13824 K=5120`（缺 M=4096）。

**(c) `N=1152 K=4304` 完全缺失（含 decode）** — `K % 32 != 0`
（4304/32 = 134.5），被 sweep 的 CSV 加载器过滤掉了。
ONNX MatMulNBits 允许最后一个 block 不满，所以这是合法 shape，
过滤规则需要放宽。来自 `google-gemma-4-26B-A4B-it-FP16W4-qmoe`。

---

## 五、后续计划

三项方向已定。因为改 tuner 必须重跑全量 sweep，**补 shape 要排在重跑之前**，
一遍 sweep 覆盖所有事。

### 步骤 1：修内核 tuner 的计时 — 作废（不动内核）

原计划把内核 tuner 的"取最小值"改成"多遍取中位数"。**已放弃**：任何在内核
tuner 里加遍数的做法都会让在线 autotune 变慢，而 LUT 的全部意义就是绕开慢的
在线 sweep——本末倒置。

结论：**内核 tuner 逻辑保持原样，一行不改。** 幽灵读数只在离线侧处理，而且
已经在做——`update_lut.py` 的 `drop_implausible`（建表时丢弃低于该 shape 中位
数/20 的读数并重算 winner）。sweep 工具只是触发内核 autotune 并读日志，过滤发生
在建表阶段，不碰生产代码、不影响任何在线路径。

### 步骤 2：补齐 shape

**2a. 修 sweep 的两个限制**（在 `tools/matmul_nbits_autotune_sweep.cpp`）— 已完成

已落地的改动：

- loader 不再整条丢弃 K%32≠0，只要求 K/N/gs>0；
- `max_out_elems` 默认 32Mi→256Mi：旧值把 N<65536 的真实 prefill 高 M 档
  （如 N=17408 K=5120 的 M=2048/4096）误杀了；新值放行到 M=4096，
  同时 lm_head（N≥big_n）仍由 big_n 强制 M=1；
- K%32≠0 的处理跟随下面的"选项 B"：decode 用真实 K，prefill 用 k_pad
  （dispatch 内部把 K 补到 32 的倍数走 WMMA），dp4a 仍跳过（内核要求
  K%32==0）。prefill 的 `#SHAPE` 标记打印 k_pad，与运行时 LUT 查询的 K 一致。

本地模拟确认 `max_out_elems` 修复关掉 20 个 prefill 高 M 缺口；`N=1152 K=4304`
的 decode 由"丢弃"恢复，prefill 由选项 B 走 WMMA。

剩余的 lm_head M=1 缺口（N≥65536）工具本就支持 M=1，之前只是没跑到——
步骤 3 重跑整份 CSV 即可补上。4 个非 last-token lm_head 模型
（gemma-4-12B/E2B/E4B、Nemotron）的宽 M 档需 `--big-n 0` 或单独 CSV，
工具已支持。

**选项 B：K%32≠0 的 int4 prefill 走 WMMA（K 补零）— 已完成并本地验证**

问题：某些模型的 intermediate_size 不是 32 的倍数（如 gemma-4-26B-A4B 的
down_proj，K=4304=16×269），WMMA 要求 K%32==0，这些 shape 的 prefill 会掉到
naive，性能很差，且每层每次 prefill 都踩。

做法（`matmul_nbits_kernel.hip` 的 host dispatch，**不改 WMMA kernel**）：
K%32≠0 且 M≥kWmmaMinM 时，把 K 补到 32 的倍数 K_pad，用一个补零的激活拷贝
（`ensureAPadBuffer` + `hipMemcpy2DAsync` 拷真实列 + `hipMemset2DAsync` 清尾部）
走原封不动的 WMMA。权重行距本就按满组补齐到 K_pad，无需改。补的激活是 0，对
点积贡献 0，输出 [M,N] 不变，无需裁切。K%32==0 的 shape 不进这个分支，逐字节
不受影响。

本地单 kernel 验证（gfx1151，对 CPU fp32 参考）：

| 路径 | shape | max_abs | rel |
|---|---|---|---|
| 新 K-pad | M=32/128/17, N=1152, K=4304, gs=32 | 0.0004 | 0.05% |
| 控制（未改动） | K=4096；N=14336 | 0.0003–0.0007 | 0.06–0.10% |

测试在 `tools/matmul_nbits_kpad_test.cpp`。构建要点（避免重蹈覆辙）：
- flatc 默认生成 C 风格 enum（`MatmulNbitsBits_B4`），代码用 scoped → 必须
  `flatc --cpp --scoped-enums`；
- 这个 Windows flatc 的 `-o` 是坏的（报成功不写文件）→ cd 进目标目录不带 `-o`；
- 用 therock 自带 clang（`therock-dist/lib/llvm/bin/clang++`），和它的 device
  libs 配套；链接 `matmul_nbits_autotune.cpp` + `tools/empty_lut_data.cpp`。

**2b. 扩充业界模型 shape — 已放弃（超出当前目标）**

当前目标就是"oga_models 全覆盖、每个 shape 命中自己实测的最佳 config"，
不需要扩充别的模型。手推业界模型的 (N,K) 还有导出融合方式不确定的风险
（QKV 是否合并、gate/up 是否合并、GQA kv 维度、head_dim），容易产出与真实
导出对不上的死点。若将来要做，应沿用 `extract_shapes.py` 从真实 ONNX 图抽取，
而不是手算。

### 步骤 3：重跑全量 sweep 并重建表

```powershell
$d = "lib/Runtime/Kernels/hip/autotune/matmul_nbits"
python $d/scripts/update_lut.py measure --sweep <mn_sweep.exe> `
       --shapes $d/shapes/all_models_bits4.csv
python $d/scripts/update_lut.py fit --verbose        # 看权重和精度
python $d/scripts/update_lut.py build --fit-weights
python $d/scripts/update_lut.py compile --flatc <flatc.exe>
```

验收指标：

- 目标 shape 集精确命中率 **100%**（用第四节的脚本重新核对）
- 幽灵读数 0 条（`drop_implausible` 应报告丢弃 0）
- 留一 shape 的 geomean 仍在 1.15x 附近（这是最近邻的天花板，
  不会因为加 shape 而改善，也不应该变差）

### 步骤 4：验证（本地不能构建 hip-ep）

用 Python 实现一份等价查找，直接对 `.fb` 跑全量查询做交叉验证：

- 解析 `lut/gfx1151.fb`（或直接读 `gfx1151.json`，两者由 flatc 保证一致）
- 复刻 `resolve()`：类别键精确匹配 → 组内加权对数距离最近邻 → fallback
- 对全部目标 shape × M × phase 跑查询，断言：
  - 每个实测点返回自身的 config，距离 0，`Source::Exact`
  - 没有查询落到 `Source::None`
  - 类别键不串组（Decode 和 DecodeDp4a 各自独立）
- 与 `update_lut.py` 内部的 `nearest()` 对拍，确保
  Python 参考实现和 C++ 的 `resolve()` 语义一致
  （两者必须一致，否则 `fit` 拟合的是运行时不使用的度量）

`tools/matmul_nbits_lut_test.cpp` 和 `tools/matmul_nbits_lut_query.cpp` 保留，交给能构建的机器
或 CI 跑。

### 步骤 5：收尾文档

更新 `README.md`：两级设计的取舍、与 Tensile/CUTLASS/llama.cpp 的对比、
实测精度（1.15x 天花板及其成因）、幽灵读数问题及处置。

---

## 六、可复现的分析命令

本文所有数据都来自这两条命令：

```powershell
$d = "lib/Runtime/Kernels/hip/autotune/matmul_nbits"

# 权重拟合 + 留一 shape 精度 + 最差点清单
python $d/scripts/update_lut.py fit --verbose

# 留一 point（只挖掉一个 M）的对照
python $d/scripts/update_lut.py fit --holdout point
```

覆盖度曲线、距离-代价相关性、噪声估计这三项分析是临时脚本跑的，
若要沉淀下来，建议加成 `update_lut.py` 的 `analyze` 子命令。
