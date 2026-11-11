<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# GEMM autotune / LUT

Offline-tuning home for the custom dense GEMM kernel (`lib/Runtime/Kernels/hip/gemm_kernel.hip`).
Mirrors the `matmul_nbits` autotune design (`../matmul_nbits/`), and as of
schema v2 shares its dtype vocabulary with every other op via `../common/`.
The installed format is v3: point and fallback config indices are `uint8`,
with fewer than 256 distinct configs; the current table contains 62.

## Schema v3: explicit dtypes, compact config indices, four phases

v1 keyed a point on `type_bytes` (2/4/8) -- an element-size proxy that cannot
tell fp16 from bf16 (both 2 bytes), cannot express activation != weights, and
is meaningless for the sub-byte quantized weights `matmul_nbits` needs next
(see `plan.md` §1 D2 for the full design review). v2 replaces it with three
explicit fields from the shared `HipdnnDType` vocabulary
(`../common/hipdnn_dtype.fbs`): `act_dtype` / `wts_dtype` / `out_dtype`.
`out_dtype` is stored on every point but is **not** part of the grouping key
-- every point measured so far has `out == act`, and `out` only changes the
epilogue store, never the tile choice (promote it into the key later if a
measured `out != act` point ever changes a winner; that's a data change, not
a re-measurement).

The grouping key is `(phase, act_dtype, wts_dtype, trans_b)`, packed as an
8-bit-per-lane `uint64_t` (`groupKey()` / `fallbackKey()` in
`gemm_autotune.cpp`) so a phase or dtype enum growing past its old 2-bit field
can never silently collide with another group (v1's D3 bug).

The schema has four measured phases:

| phase | serves | config kind |
|---|---|---|
| `Wmma` | prefill tiles, fp16/bf16 only | `Wmma` (bm16/bn16/swizzle/wt_m/wt_n/bk/split_k) |
| `GemvNt` | decode, N-major (the original GEMV path) | `Gemv` (threads/tile_n) |
| `GemvNn` | decode, N-parallel (previously a single hardcoded `BLOCK=256`, no table at all) | `Gemv` (threads only, `tile_n` unused/0) |
| `TiledFma` | the plain-FMA tiled path: fp32 (`gemm_f32_tile_kernel` for transA=0 on wave32, `gemm_tiled_fma_kernel` otherwise), fp64, and wave64 fp16/bf16 (no WMMA on wave64 hardware) | `TiledFma` (bm16/bn16/bk/wt_m/wt_n=TM/TN/threads) |

Eligible fp32 wave32 tiles use `gemm_f32_tile_kernel` (`kF32Tile`, tuner
`tuneF32Tile`, forced index `HIPDNN_EP_GEMM_F32_TILE_CFG`). Its geometries
share the `TiledFma` kind; the validator accepts only a geometry supported by
the selected kernel family. The installed gfx1151 table has **zero fp32
points**, so every fp32 request misses and retains its existing custom default,
including the skinny, GEMV and tiled-FMA routes. No fp32 table refresh is part
of this integration.

`phase -> config kind` is an explicit table (`phaseConfigKind()`, fixes v1's
D4 bug where a third phase would have been silently accepted as a `Gemv`
config through a `wmma ? Wmma : Gemv` boolean guess).

**`fallbacks` is always empty this round, on purpose** (D5 note in
`plan.md` §1): an empty fallback table plus exact-match-only groups is what
gives a dtype with no measured points a clean **miss** back to the static
default, instead of silently borrowing another dtype's (or another phase's)
point. `fallbackKey()` shares `groupKey()`'s bit layout and *does* carry
`trans_b` (v1 dropped it, conflating NN/NT winners), so filling `fallbacks`
in later is a pure data change whenever that's wanted.

## Config selection order (dispatch path)

`hip_gemm` checks the LUT before the existing measured/skinny dispatch ladder
in default `lookup` mode. It caches hits and misses by phase, dtype, transpose,
M/N/K and fp32 tile family. FP16 and BF16 have distinct cache keys.

1. **Offline LUT preflight** resolves the request's phase/dtypes/transB and
   nearest numeric dimensions. A legal hit selects an existing custom launcher;
   WMMA keeps all existing launch remaps below. Stored values are tile
   *geometries*, validated against the live config table.
2. **Explicit config pins** retain their original priority. Setting an existing
   `HIPDNN_EP_GEMM_*_CFG` or `*_TCFG` makes the preflight hand control to the
   original dispatch. Raster/stagger and launch controls retain their behavior.
3. **Miss** enters the existing custom measured/skinny/default ladder. Its phase
   resolvers retain the cached static heuristic/default in `lookup` mode.
   There is no vendor GEMM retry and no automatic on-device sweep on a miss.
4. **Online mode** bypasses preflight and preserves the original dispatch,
   including candidate sweeps only where that family supports online selection
   and the sweep is safe. Transposed-A and several small-M families retain their
   static custom default.

The existing table cannot express every family. Multi-row small-M and
transposed-A requests use the host-only `Unconfigured` phase, query the loader
as a miss (`reason=unconfigured-phase`) and continue to their custom default.
They do not borrow a WMMA/GEMV point for a different operation. Degenerate
M/N/K requests retain the existing zero/epilogue handling before tile lookup.

## Mode switch

`HIPDNN_GEMM_AUTOTUNE_MODE` (read once, latched), same semantics as
`HIPDNN_MATMUL_AUTOTUNE_MODE`:

| value | behaviour |
|---|---|
| `lookup` (default) | resolve from the offline LUT; on a miss use the static occupancy heuristic (`pickWmmaHeuristic` / GEMV/TiledFma defaults), NOT the on-device sweep |
| `online` | bypass the LUT; use the existing family's sweep where supported, otherwise its static custom route |

The `lookup`-miss path deliberately avoids the runtime sweep: on the gfx1151
APU compute-bound tiles throttle hard and the throttling correlates with tile
size, so an on-device sweep systematically mis-ranks configs (measured
picking 128x64 over the ~2x-faster 128x128 WT2x4 on large-K shapes). The
heuristic is robust; the sweep exists to populate the durable LUT offline.

`HIPDNN_EP_GEMM_AUTOTUNE=0` disables the sweep entirely and uses the static
occupancy heuristic, for perf isolation.

## Production runtime entries

Dense GEMM entry points always call custom `hip_gemm` for fp16, bf16 and fp32.
The runtime returns unsupported requests and launch errors to the caller.

| Entry | Layout and semantics |
|---|---|
| ONNX `Gemm` / `wrap_gemm` | Four transpose states, alpha/beta and scalar/row/column/matrix C broadcast. If C aliases output, session workspace preserves C for online selection and broadcast tiles. |
| ONNX `MatMul` / retained `wrap_hipblasLtMatmul` | Tight flat batches, shared B or explicit B stride. The element-kind ABI is 2=f16, 4=f32, 8=f64, -2=bf16. |
| `wrap_matmul_broadcast` | Independent leading-axis broadcasts, unbatched A with batched B, cross-axis and dynamic batches; vectors use the existing M/N=1 promotion. Host dimension arrays select the right tight matrix without tensor expansion. |
| Retained `wrap_hipblasLtGemm` | Public column-major fp32 ABI; swaps operands and M/N. C==Y disables unsafe online candidate sweeps in the kernel resolver. |
| Decomposed GQA / MultiHeadAttention | Existing column-major score/value views map to tight row-major calls per batch/head/chunk. FP16 Q/K cast exactly to fp32 before score GEMM so softmax still reads fp32 scores; values use the existing uniform dtype. Explicit page/query/output strides and session scratch are retained. |
| MatMulNBits dequantized prefill | Existing wave64 int4/fp16 prefill dequantizes B as before, then calls custom NT GEMM. Quantized RDNA kernels and their LUT are unchanged. |

Static dense MatMul/Gemm anchors no longer outline into rocMLIR; Conv outlining
remains available. MatMulBnb4 dequantization, projection MatMuls, Einsum
contractions and patch-embedding Gemm therefore converge on the same dense
runtime. Integer QGemm/MatMul paths remain their existing custom integer kernels.

The custom kernel accepts uniform A/B/C/Y dtype and tight matrices with checked
32-bit M/N/K. General leading-axis broadcasts use checked 64-bit extents. Mixed
attention score precision is provided by the runtime cast adapter. Unsupported
layouts/dtypes fail explicitly. FP64 and wave64 routes remain the existing
custom fallback families; this integration's GPU validation targets gfx1151.

`HIPDNN_GEMM_LUT_LOG=1` reports table loading, exact/nearest selection and misses;
`HIPDNN_EP_DEBUG=1` adds the runtime entry, LUT preflight and final WMMA geometry.
Use kernel tracing to identify the actual launch after remapping. Performance
observations must run with debug and per-op tracing disabled.

Removing vendor GEMM does not remove vendor-library lifecycle symbols. The
default session still creates/destroys a hipBLASLt handle; the optional
`HIPDNN_EP_DISABLE_VENDOR_BLAS` path retains error-returning ABI shims. Packaging
continues to stage vendor dependencies until their remaining load requirements
are separately reviewed.

## Launch-side rules (applied after the config is resolved, not in the LUT)

The LUT stores a tile geometry; `launchWmma` may still remap it at launch. These
remaps remain part of the frozen custom kernel and are preserved by LUT-first
dispatch. The `bench3` measurements below describe their original qualification
on gfx1151 (halo13), with idle checks, alternating rounds and the slower round.
The installed table's current qualification is recorded in the gfx1151 table
section below.

With `HIPDNN_EP_DEBUG=1`, `launchWmma` logs the tile it actually launches,
after every rule below: `[gemm] wmma final M= N= K= -> BMxBN wt=WMxWN bk=
pgr2= lb= be= bl= sw= stag= wgs=` (`pgr2` = 1 only on a tile with a PGR2
instance, `lb` / `be` = line-wide B / bias epilogue, `bl` = the buffer-load
read path variant, 0 = none, `wgs` = workgroups in the grid). The `[gemm]
wmma LUT hit` / `heuristic` lines print the config before these rules, e.g.
NN 16x4096^2 logs 64x128 WT2x4 BK64 and runs 64x64 WT2x2 BK32 (grid fill).

- **K%1024 alias rule** (NT fp16/bf16, both row strides a multiple of 1024
  elements): M-first raster (8 M-tiles per group; 16 on the 128-row BK=64
  tile when there are >= 16 M-tiles and 16 A panels fit 16 MB) + StaggerU +
  PGR2 on BK=32 tiles, with remaps WT4x4 BK32 -> WT4x2, BK16 -> BK64,
  128x256 -> 256x128 (M >= 256), non-PGR2 BK32 -> 128x128 WT4x2 (M >= 128),
  M >= 1024 -> 128x128 WT4x2 BK64. The raster + StaggerU + PGR2 part is what
  hipBLASLt does on these shapes (104 of its 129 picks on the 34-shape table
  are PGR2). fp16 1024x4096^2 1.007 ms, 2048x4096^2 1.853, 4096x4096^2 3.85
  (before: 1.83 / 2.47 / 4.37 with the per-call padrow copy; hipBLASLt
  0.89-1.00 / 1.87 / 3.74).
- **StaggerU** runs only when K is a multiple of 1024 (the aliasing rows the
  rule above targets); on other K it cost 2-3% on short-K
  shapes (f16 729x4304x1152 0.296 -> 0.268 ms, 512x1536x384 0.0319 -> 0.0297).
- **No row-padded copy of B.** The copy-free rule above handles the aliasing
  K. The per-call padded copy (formerly `HIPDNN_EP_GEMM_PADROW` and the
  large-M `HIPDNN_EP_GEMM_BIGM_PAD` route) was slower when forced
  (4096x4096^2 fp16 4.19 vs 3.86 ms) and only won at M >= 4096 with
  B <= 32 MB (4096^3 fp16 3.82 vs 3.90, bf16 3.86 vs 3.93), at the cost of a
  persistent device scratch, so it was removed.
- **Ragged M (NT)**: when only M (and/or N) is not a tile multiple and
  K%BK == 0, the launch stays on the fast BC=false kernel with full tiles:
  A rows past M re-read the last row, the epilogue masks them. Vision
  1576x3072x768 fp16 0.250 -> 0.216 ms, 729x4304x1152 0.258 -> 0.215
  (hipBLASLt 0.201 / 0.195). `HIPDNN_EP_GEMM_MRAGGED=0` disables.
- **Grid fill** (outside the K%1024 rule): a tile that puts fewer than 40
  workgroups on the 40-CU part steps down 128x128 WT4x2 -> 64x128 -> 128x64
  -> 64x64 (BK=16 128x128 / 64x128 when K%32 == 16). 729x1152x4304 0.295 ->
  0.206 ms, 512x1536x384 0.029 -> 0.021, 512x384x1536 0.034 -> 0.023.
  `HIPDNN_EP_GEMM_GRIDFILL=0` disables.
- **Mid-size NT shapes** (after grid fill, 256 <= M <= 2048, no C term):
  the power-of-two tiles the LUT and the ladder offer leave these grids
  unevenly spread over the 20 WGPs (729x1152x4304: 54 workgroups of 8
  waves, 2.7 per WGP; hipBLASLt runs 72 of 4 waves) or re-read A too often.
  A forced sweep of every existing tile (bench3 g1_force) and a screening
  harness over new ones picked, per shape class (bench3 g2_c1 / g4, 2-4
  ABBA rounds, slower round, fp16 / bf16 ms, old -> new; hipBLASLt):
  - K % 32 == 16, M 512..1024, N <= 2048 -> 96x128 WT3x4 BK16 (72 WGs):
    729x1152x4304 0.213 / 0.206 -> 0.197 / 0.196 (0.186 / 0.184);
  - K % 32 == 0, K <= 2048, M 512..1024, N 2048..8192 -> 128x128 WT4x4 BK32
    (4 waves): 729x4304x1152 0.223 / 0.210 -> 0.205 / 0.205 (0.197 /
    0.157);
  - K % 32 == 0, K <= 1024, M 1025..2048, N 2048..4096 -> 96x96 WT3x3 BK32:
    1576x3072x768 0.216 / 0.229 -> 0.205 / 0.207 (0.198 / 0.219);
  - K%1024 alias, M 1024..2048, N <= 1024 -> 128x256 WT4x2 BK64 (A read
    N/256 times instead of N/128): 1576x768x3072 0.208 / 0.212 -> 0.187 /
    0.184 (0.192 / 0.191);
  - M 256..512, N 256..512, K >= 1024 -> 64x64 WT2x2 BK64: 512x384x1536
    0.0237 / 0.0234 -> 0.0228 / 0.0220 (0.0156 / 0.0154);
  - M 256..512, N 1024..4096, K <= 512 -> 128x128 WT4x4 BK32: 512x1536x384
    0.0215 / 0.0208 -> 0.0199 / 0.0201 (0.0204 / 0.0152).
  The two new tiles (96x128 WT3x4 BK16, 96x96 WT3x3 BK32, 4 waves) are not
  in `kWmma`: only this rule reaches them (not the LUT, the online sweep or
  `HIPDNN_EP_GEMM_WMMA_CFG`), and they are built for NT only;
  `GemmWmmaImpl` stages B like A (B_CH halfs per lane, BK / B_CH lanes per
  row) when BN is not a power of two. minilm stays 0.6-0.8x hipBLASLt: its 20 us
  kernels spend 22-28% of the sampled wave time in WMMA against 35% for
  hipBLASLt (RGP); the K loop's address arithmetic and same-iteration load
  waits, not the grid, are what is left. `HIPDNN_EP_GEMM_MIDTILE=0`
  disables.
- **Line-wide B loads on the alias rule's 128x128 WT4x2 BK64 tile** (NT,
  K%1024): the default B staging gives each lane 64 contiguous bytes of one
  row, so one load instruction touches 16 rows x 32 B; this instance (PGR
  bit 4) stages B like A, 8 lanes per 128 B row, 4 rows x 128 B per
  instruction. The big K%1024 shapes are bound by L2-miss traffic, not by
  the WMMA loop: RGP counters on fp16 2048x4096x14336 show the memory unit
  stalled 98.7% of the time at 354 GB/s of L2-miss traffic and a 20% L2 hit
  rate (hipBLASLt: 69% stalled, 432 GB/s, 24%). Gains are small and within
  the run-to-run noise on single shapes; over 2 bench3 sessions (g2_c1,
  g3) the geomean of with / without over the six K%1024 shapes is 0.987.
  `HIPDNN_EP_GEMM_LINEB=0` disables.
- **Bias epilogue** (NT fp16/bf16, C a row-broadcast vector, cDim0 == 1,
  even N): the default epilogue with a C term reads C through `gemmStoreY`
  for every output and writes 2-byte stores (fp16 1576x3072x768 on 96x96:
  0.207 ms without C, 0.249 with). The PGR bit 8 instances read C once per
  fragment column and keep the paired 4-byte stores of the plain epilogue.
  Built for the alias 128x128 WT4x2 BK64 line-wide-B tile and 96x96 WT3x3
  BK32, which the mid-shape rule then also takes with a bias. bench3 g3,
  slower of 2 ABBA rounds, fp16 / bf16 ms without -> with (hipBLASLt):
  2048x4096^2 + bias 1.988 / 2.075 -> 1.869 / 1.963 (1.887 / 1.894),
  1576x3072x768 + bias 0.260 / 0.243 -> 0.235 / 0.217 (0.207 / 0.211).
  `HIPDNN_EP_GEMM_BIASEPI=0` disables.
- **Buffer-load read path** (`gemm_wmma_buf_kernel`; NT fp16/bf16,
  K % 128 == 0, even N): the same LDS tile and WMMA step as `GemmWmmaImpl`,
  a different K loop. A and B are read with `buffer_load_b128` through one
  buffer resource per operand (base = the workgroup's first row, word3
  0x31004000 as in hipBLASLt's gfx11 kernels); each lane's offset is set once
  and the K position is one SGPR offset per tile, so the loop has no address
  VALU (the replaced minilm loops: 17 VALU per tile, 10 of them 64-bit
  pointer adds, and 14 SALU; now 0 VALU and 2 SALU). The loop is unrolled
  by two so the LDS buffer and the register slot are immediates; StaggerU's
  wrap is a second copy of the loop with the resource moved back by K, not a
  select per tile. Template parameter RP: `& 3` = pipeline (1: the next tile is
  issued during the step and stored after it; 2: two register slots, tile
  t+2 issued while tile t computes), bit 4 / 16 = `sched_group_barrier`
  order spreading the step's global loads (and with 16 the next k-step's
  fragment loads) between the WMMAs, bit 8 = the bias epilogue. On the
  gfx12-style WMMA encoding (gfx1170) the same loop uses plain global loads
  (the buffer resource format there is not verified) and per-element stores.
  Replaces two mid-shape picks (no C term, alpha 1). bench3 c1_scan, 2 ABBA
  rounds, slower round, fp16 / bf16 ms, old -> new (hipBLASLt fastest):
  - M 256..512, N 256..512, K 1024..2048, at most 48 WGs of 64x64 -> the
    same 64x64 WT2x2 BK64 tile, loads spread between the WMMAs (RP 5):
    512x384x1536 0.0242 / 0.0227 -> 0.0199 / 0.0199 (0.0193 / 0.0148),
    256x384x1536 0.0149 / 0.0150 -> 0.0138 / 0.0142, 512x256x1152 0.0115 /
    0.0116 -> 0.0105 / 0.0105. Not at 64 WGs (512x512x4352: 1.09 / 1.03 of
    the old time);
  - M 256..512, N 1024..4096, K <= 512 -> hipBLASLt's 128x80 tile (WT2x5
    BK64, single-buffered 30 KB so 4 WGs fit per WGP; 80 WGs on 512x1536)
    with two tiles of loads in flight (RP 2), instead of 128x128 WT4x4 BK32:
    512x1536x384 0.0203 / 0.0199 -> 0.0179 / 0.0171 (0.0163 / 0.0144),
    256x1536x384 0.0132 / 0.0136 -> 0.0095 / 0.0100, 384x2048x512 0.0322 /
    0.0327 -> 0.0291 / 0.0294.
  These 20 us kernels run up to ~30% faster still once the GPU has been
  busy for more than ~20 ms without a break (stage5 screen: 512x384x1536
  0.014 ms; the old tile 0.0155-0.021); bench3's ~10 ms blocks keep both
  sides out of that state. `bl=` in the final line (1 / 2).
  `HIPDNN_EP_GEMM_BUFLOAD=0` disables.
- **Buffer-load read path on the alias rule's tile** (`bl=3` / `4`): the
  K%1024 rule's 128x128 WT4x2 BK64 line-wide-B tile (M >= 1024, Y = A*B or
  the bias epilogue) becomes the buffer-path 128x128 WT2x4 BK64,
  single-buffered, two tiles of loads in flight (RP 2; RP 10 with the bias
  epilogue). For K >= 12288 (no bias) the next k-step's fragment loads are
  also spread one share per WMMA (RP 18; stage5 screen on 2048x4096x14336:
  0.916 / 0.911 of the old tile, RP 2 0.953 / 0.950). The WT4x2 tile with
  the same loop was 1-4% behind WT2x4. The old tile's loop had 16 VALU per
  tile, all 64-bit pointer adds, and 10 SALU; the new one 0 and 2, no
  `vmcnt(0)`. bench3 c2_scan, 2 ABBA rounds, slower round, fp16 / bf16 ms,
  old -> new (hipBLASLt fastest):
  2048x4096x14336 8.201 / 8.249 -> 7.417 / 7.540 (8.014 / 7.631),
  1024x16384x3072 2.611 / 2.598 -> 2.470 / 2.544 (2.494 / 2.483),
  2048x4096^2 1.823 / 1.816 -> 1.791 / 1.774, with a bias 1.860 / 1.876 ->
  1.805 / 1.822, 2048x14336x4096 5.996 / 5.960 -> 5.861 / 5.824, 4096^3
  3.800 / 3.759 -> 3.729 / 3.804 (3.693 / 4.087); over 15 K%1024 shapes
  (M 1024-4096, K 3072-14336) the geomean is 0.961 / 0.968 of the old
  time. `HIPDNN_EP_GEMM_LINEB` / `_BIASEPI` only matter with
  `HIPDNN_EP_GEMM_BUFLOAD=0`.
- **Paired epilogue stores**: the BC=false fp16/bf16 epilogue (no bias / alpha
  / beta, even N) swaps one value between neighbouring lanes so each lane
  writes two columns with one 4-byte store.
- **transA=1 WMMA** (`kWmmaTa`, `pickWmmaTa`): A is read as [K, M] straight
  into LDS, no transpose copy. 512x1024x1024 TN fp16 29.6 -> 0.043 ms
  (hipBLASLt 0.052). transA with B = [N, K] and 2 <= M <= 128 takes the
  skinny kernel where `pickSkinnyT` routes it (below).
- **Skinny-M WMMA** (`gemm_wmma_skinny_kernel`, NT fp16/bf16, 2 <= M <=
  128): each lane loads its WMMA fragments straight from global memory (or B
  goes through a wave-private LDS row copy), the waves of a workgroup split K
  with no barrier in the K loop and meet in LDS at the end; rows past M are
  padding. Route (`pickSkinny`): B <= 32 MB and M <= 32, or B > 32 MB and
  M <= 32 / M <= 64 / M <= 128 with N >= K; everything else stays on the
  WMMA tiles. It replaces the multi-row GEMV for NT M = 2..15 and the split-K
  / 64x128 tiles for M = 16..32 (fp16 / bf16 vs hipBLASLt): 8x4096^2 0.040 /
  0.039 ms (hb 0.050 / 0.038), 16x14336x4096 0.62 / 0.61 (hb 0.75 / 0.59).
- **NN and transA (B = [N, K]), 2 <= M <= 128** also take the skinny kernel
  (`GEMM_SKINNY_TCFGS`, `pickSkinnyT`): the operand whose k is not
  contiguous (NN's B, transA's A) is read as k-row runs and transposed through
  the wave's LDS tile. More rows than one tile (16*MT) run as several M-tile
  workgroups, adjacent in dispatch order so they share the B panel in L2.
  The skinny kernel uses StaggerU like the WMMA tiles: on for B = [N, K],
  off for NN (on NN it spreads one k-row's N-tiles over different rows:
  32x4096^2 0.070 -> 0.057 ms). transA with B = [K, N] (both operands
  transposed) stays at M < 16.
  - M < 16: NN with M <= 4 on a B > 32 MB keeps the [K, N] GEMV. fp16 /
    bf16 vs hipBLASLt: transA 8x4096^2 0.043 / 0.043 ms (hb 0.081 / 0.082),
    NN 8x4096^2 0.042 / 0.041 (hb 0.045 / 0.048), transA 8x14336x4096 0.67 /
    0.67 (hb 0.87 / 0.88).
  - transA, M = 16..128: B <= 32 MB -> MT 2 NTL 2 KW 4; B > 32 MB -> NTL 1
    with MT 2 up to M = 32, else MT 4, and KW 8 when K > N (up to M = 64),
    else KW 4. fp16 / bf16, slower of 2 ABBA rounds, old route -> new
    (hipBLASLt): 16x4096^2 0.205 / 0.204 -> 0.048 / 0.048 ms (0.084 /
    0.082), 16x4096x14336 1.456 / 1.445 -> 0.628 / 0.628 (0.986 / 1.010),
    64x14336x4096 1.694 / 1.743 -> 0.661 / 0.659 (1.105 / 1.133),
    128x4096x14336 1.922 / 1.930 -> 1.002 / 1.010. 159-194 GB/s of weight
    at M = 16..64 on the 90-117 MB weights.
  - NN, M = 16..128, B <= 32 MB: M <= 16 -> the M < 16 config, M <= 64 ->
    MT 2 NTL 2 KW 4 KU 1; larger M stays on the tiles. 16x4096^2 0.102 /
    0.102 -> 0.045 / 0.043 ms (hipBLASLt 0.048 / 0.048), 32x4096^2 0.111 /
    0.111 -> 0.058 / 0.057.
  - NN, B > 32 MB and N >= 2K (short K, many N-tiles): one wave per
    workgroup (KW = 1) streaming the whole K, NTL 4 up to M = 32, then
    MT 4 / 2 / 3 with NTL 2 up to M = 48 / 64 / 96; M > 96 stays on the
    tiles. 16x14336x4096 0.769 / 0.778 -> 0.549 / 0.550 ms (hipBLASLt 0.598
    / 0.559), 48x14336x4096 0.786 / 0.795 -> 0.568 / 0.567 (0.600 / 0.591),
    64x14336x4096 0.758 / 0.758 -> 0.597 / 0.600 (0.602 / 0.649),
    32x11008x4096 0.458 / 0.467 -> 0.402 / 0.402 (0.414 / 0.407).
    Splitting K over 2-4 waves was 0.61-0.68 ms on 14336x4096. In some
    processes these KW = 1 configs ran at 0.72-0.83 ms on 14336x4096
    outside the M range they are routed to.
  - NN, B > 32 MB otherwise: M <= 16 -> MT 1 NTL 4 KW 2 KU 1, then MT 2
    NTL 4 KW 2 KU 1 up to M = 96 (128 when K > N). 16x4096x14336 0.560 /
    0.560 -> 0.522 / 0.521 ms (hipBLASLt 0.572 / 0.536), 64x4096x14336
    0.654 / 0.655 -> 0.583 / 0.579 (0.602 / 0.572).
- **Small M under the K%1024 rule** (M <= 128, not taken by the skinny
  kernel): M <= 64 -> 64x128 WT2x2 BK64 (its split-K-2 WT2x4 twin when
  K >= 2N and B > 32 MB), M <= 128 -> 128x64 WT4x2 BK64, whatever the LUT
  says: 64x4096^2 fp16 0.113 -> 0.080 ms (hb 0.101), 128x4096^2 0.220 ->
  0.139 (hb 0.169).
- **StaggerU keyed by N-tile** when there are fewer M-tile rows than one
  M-first raster group (8): keyed by M-tile every workgroup of such a grid
  started at the same K offset. 256x4096^2 fp16 0.353 -> 0.277 ms,
  512x4096^2 0.573 -> 0.516 (hb 0.310 / 0.537).
- **NN** (B = [K, N]) has no LUT points: M = 1 (and M <= 4 on a B > 32 MB)
  takes the B = [K, N] GEMV (or the ring GEMV below), other M <= 128 the
  skinny kernel where `pickSkinnyT` routes it,
  M >= 128 with K%64 == 0 the 128x64 WT4x2 BK=64 tile instead of
  `pickWmmaHeuristic`: fp16 1x4096^2 0.203 -> 0.051 ms, 2048x4096^2
  3.65 -> 2.05.
- **NN mid-M** (`gemm_wmma_nnmid_kernel`, `GEMM_NNMID_CFGS`, `pickNnMid`,
  `launchNnMid`; fp16 / bf16, NN, transA = 0, 33 <= M <= 128, N % 8 == 0, K a
  multiple of the config's KC, 16-byte aligned A and B; tried before the
  skinny kernel). One workgroup owns all M rows (NWM waves x MW 16-row tiles)
  and 16 * NTL columns: B is read once and transposed into LDS once per
  workgroup, shared by every wave, and A -- k-contiguous -- goes straight from
  global memory (L2) into the WMMA fragments. Per KC-row chunk the threads read
  B as 16-byte row pieces and store 4-k-row x 8-column pieces transposed
  (packed 8-byte stores) into one of two LDS buffers, one barrier per chunk; B
  runs PF chunks ahead in registers. The skinny kernel instead transposes B per
  wave and re-reads it per 16 * MT-row workgroup (and spends 110-176 64-bit
  address VALU per 8-16 WMMA), and the tiles stage A through LDS too and run
  the bounds-checked K loop on a ragged M (64x128 BC=1: 68 ds_store, 239 VALU
  and 5 vmcnt(0) per BK = 64 step against 40 / 21 / 1 at BC=0). Route:
  B <= 32 MB -> nnmid configs by M (32 columns, KC 64), except the 128x64 WT4x2
  BK 64 tile at M > 96 on a 32 MB B; B > 32 MB -> by N vs K and M: KC 32 nnmid
  configs, the existing NN 128x64 WT4x2 BK 32 tile as a full-tile launch (rows
  past M masked), or the old route (`pickNnMid` lists the measurements).
  `[gemm] wmma-nnmid` in the debug log shows the pick.
- **[K, N] GEMV with a register ring** (`gemm_gemv_kn_ring_kernel`,
  `GEMM_KN_RING_CFGS`, `pickKnRing`; NN, transA = 0, N % (16 / element
  size) == 0, K a multiple of the config's row group, N * K >= 64K, 16-byte
  aligned A and B). `gemm_gemv_kn_kernel` waits for each 16-byte B load right
  after issuing it (`vmcnt(0)` per load: one load in flight per wave) and on
  N = 4096 its 32-column blocks read 64 bytes of a k-row. The ring kernel:
  a block is KW = 8 waves over LPR * 16 bytes of each k-row (LPR = 8 lanes
  -- one 128-byte line -- at 16 bits, LPR = 32 -- 512 bytes -- at fp32), the
  32 / LPR lane rows of a wave read different k-rows, a lane owns U
  consecutive k-rows (its A values are one vector load), k-row groups go
  round-robin to the waves, and each wave keeps NS = 2 groups in flight in a
  statically indexed register ring (loop unrolled by NS, the consumed slot
  refilled, the prologue and last refill load unconditionally: a load that
  may or may not have been issued made the compiler wait for all loads at
  the loop head). One lane computes MR rows from the same B registers; the
  waves meet in LDS once. Route: fp16 / bf16 M = 1 on a B <= 32 MB; fp16 /
  bf16 M = 2..4 on a B <= 32 MB or with N < 2K (also taken out of the skinny
  kernel's M < 16 route); fp32 M = 1 on a B > 32 MB with N <= K; fp32
  M = 2..8 on a B <= 32 MB (out of the fp32 NN skinny kernel). The other NN
  M = 1 shapes on a B > 32 MB take `gemm_gemv_kn_kernel`'s 16-wave
  128-column entry (kn-cfg 0) whatever the block count: it streams DRAM at
  216-226 GB/s, the ladder's 32 / 64-column picks on N = 4096 / 8192 did not.
  Screening harness (slower of 2 ABBA rounds) and bench3: see the commit.
- **fp32 skinny-M** (`gemm_f32_skinny_kernel`, `GEMM_F32_SKINNY_CFGS`; fp32
  NT, transA = 0, K % 4 == 0, 16-byte aligned A and B, wave32): the skinny
  structure on VALU FMAs, checked before the multi-row GEMV and
  `gemm_f32_tile_kernel`. A workgroup owns MT rows x 4*CL columns and its KW
  waves split K into 32-wide chunks with no barrier in the K loop, each wave
  keeping the next chunk's loads in flight; 8 adjacent lanes read a whole
  128-byte line of one B row per load (4 rows per instruction), A goes
  through a wave-private LDS tile, and the KW partial sums meet in LDS once.
  The 8-lane line was the decisive choice: one lane per B row (32 rows x
  16 B per load, #1120's mapping) reached 150-165 GB/s at M = 16, 2 / 4
  lanes 185-202, 8 lanes 210-224. Route (`pickF32Skinny`): M <= 4 keeps the
  GEMV; 5 <= M <= 64 takes MT 8 / 16, CL 1 / 2 by M (and B <= 48 MB, the
  partly Infinity-Cache-resident 3072^2), unless the A panel M*K*4 exceeds
  3 MB (64x4096x14336 stays on the tile: 1.77 vs 2.06 ms) or K < 256 or
  N*K < 64K. bench3, 2 rounds ABBA, slower round vs the old route (hipBLASLt
  fastest): 16x4096^2 0.420 -> 0.320 ms (0.356), 16x14336x4096 1.520 ->
  1.103 (1.164), 16x4096x14336 1.515 -> 1.135 (1.220), 8x14336x4096 1.161
  -> 1.097, 32x14336x4096 1.462 -> 1.215, 48x14336x4096 1.574 -> 1.240,
  64x14336x4096 1.605 -> 1.475. Weight GB/s on the 64-224 MB shapes, new
  (old): M 5-16 207-214 (154-211), 17-32 181-207 (149-162), 33-64 150-189
  (134-149). A 512-thread (KW = 16) variant was 3-5% faster on an idle GPU
  but slowed 4-40x (hipBLASLt 2x) while another process held the GPU, so
  none is routed.
- **fp32 skinny-M NN** (`gemm_f32_skinny_nn_kernel`,
  `GEMM_F32_SKINNY_NN_CFGS`; fp32 NN, B = [K, N], transA = 0, N % 4 == 0,
  K % 4 == 0, 16-byte aligned A and B, wave32): the same K split for a weight
  whose k-rows run along N. Lane l reads VW consecutive columns of one k-row
  (VW = 4: a wave reads 512 contiguous bytes of the row per load), so no lanes
  share a column and there is no shuffle reduction; a workgroup owns MT rows x
  32*VW*CL columns, its KW waves split K into KC-row chunks with no barrier in
  the K loop (DB-deep register ring), A goes through a wave-private LDS tile
  read back as A[m][k..k+3] (an LDS broadcast), and the KW partial tiles meet
  in LDS once. All M rows of a column strip stay in one workgroup: M-tiles in
  separate workgroups did not share the B panel through L2 on a long K
  (screening harness, 16x4096x14336: two MT 8 tiles 1.4-1.6 ms, one MT 16 tile
  1.06-1.19). Route (`pickF32SkinnyNn`, after `launchF32Skinny`, before the
  [K, N] GEMV and `gemm_f32_tile_kernel`): M <= 8 -> MT 8 VW 4 CL 2 when
  B <= 48 MB or N >= 2K, else MT 16 VW 2 CL 2; M <= 16 -> MT 16 VW 2 CL 2;
  M <= 32 -> MT 32 VW 2 CL 2, which also takes 33-64 on B <= 48 MB; not when
  the A panel M*K*4 exceeds 3 MB, K < 256 or N*K < 64K. bench3, 2 rounds ABBA,
  slower round, old route -> new (hipBLASLt fastest): 12x4096x14336
  2.474 -> 1.034 ms (1.056), 12x14336x4096 2.084 -> 1.128 (1.221), 16x4096^2
  0.384 -> 0.300 (0.323), 8x4096x14336 1.399 -> 1.030 (1.192), 32x14336x4096
  1.358 -> 1.110 (1.120). 208-237 GB/s of weight at M = 2-32 on the 180-235 MB
  weights. Above 32 rows the tile stays: on the 235 MB weights MT 64 (VW 2) was
  1.32-1.80 ms and two MT 32 tiles up to 2.2 (tile 1.24-1.39).

| env | default | effect |
|---|---|---|
| `HIPDNN_EP_GEMM_ALIASK` | 1 | 0 disables the K%1024 alias rule (A/B only) |
| `HIPDNN_EP_GEMM_SWIZZLE` | 0 | non-zero overrides the raster width; < 0 = M-first groups of -N M-tiles |
| `HIPDNN_EP_GEMM_STAGGER` | NT 1, NN 0 | StaggerU on/off (start-K offset per tile; WMMA tiles and skinny kernel, only when K%1024 == 0; transA follows its B layout) |
| `HIPDNN_EP_GEMM_STAGGER_U` / `_BYTES` | 32 / 256 | number of offsets / step in bytes |
| `HIPDNN_EP_GEMM_MRAGGED` | 1 | 0 sends a ragged-M NT launch back to the bounds-checked kernel |
| `HIPDNN_EP_GEMM_GRIDFILL` | 1 | 0 disables the grid-fill tile step-down |
| `HIPDNN_EP_GEMM_BIASEPI` | 1 | 0 sends a row-broadcast C (bias) back to the per-element epilogue |
| `HIPDNN_EP_GEMM_LINEB` | 1 | 0 keeps the default B staging on the alias 128x128 WT4x2 BK64 tile |
| `HIPDNN_EP_GEMM_MIDTILE` | 1 | 0 disables the mid-size NT tile rule (96x128 / 96x96 / 128x128 WT4x4 / 128x256 BK64 / 64x64 BK64 picks) |
| `HIPDNN_EP_GEMM_BUFLOAD` | 1 | 0 keeps the minilm-class mid-shape tiles and the alias rule's 128x128 BK64 tile on the `GemmWmmaImpl` loop instead of the buffer-load read path |
| `HIPDNN_EP_GEMM_WMMA_TA_CFG` | -1 | pin a `kWmmaTa` index (transA WMMA) |
| `HIPDNN_EP_GEMM_GEMV_KN_CFG` | -1 | pin the B = [K, N] GEMV column tile: 0 = 128, 1 = 64, 2 = 32 |
| `HIPDNN_EP_GEMM_NNMID` | 1 | 0 disables the NN mid-M route (`gemm_wmma_nnmid_kernel` and its full-tile 128x64 launches) |
| `HIPDNN_EP_GEMM_NNMID_CFG` | -1 | pin a `GEMM_NNMID_CFGS` index for every eligible NN fp16 / bf16 shape with 2 <= M <= its rows |
| `HIPDNN_EP_GEMM_KN_RING` | 1 | 0 disables the ring [K, N] GEMV and the M = 1 DRAM column-tile pick (back to the ladder, the skinny kernel for NN M = 2..4 and the fp32 NN skinny kernel for M = 2..8) |
| `HIPDNN_EP_GEMM_KN_RING_CFG` | -1 | pin a `GEMM_KN_RING_CFGS` index for every eligible NN shape of its dtype with M <= its MR |
| `HIPDNN_EP_GEMM_GEMV_MR` | 1 | 0 sends NT M = 2..15 back to WMMA and NN M < 16 back to the one-row NN GEMV (when the skinny kernel does not take them) |
| `HIPDNN_EP_GEMM_SKINNY` | 1 | 0 disables the skinny-M WMMA kernel |
| `HIPDNN_EP_GEMM_SKINNY_CFG` | -1 | pin a `GEMM_SKINNY_CFGS` index (NT) |
| `HIPDNN_EP_GEMM_SKINNY_TCFG` | -1 | pin a `GEMM_SKINNY_TCFGS` index (NN / transA; 0-4 are the M < 16 configs, 5-16 the M = 16..128 ones; an index not built for the layout keeps the existing path) |
| `HIPDNN_EP_GEMM_STAGGER_MAP` | unset | pins the StaggerU key (0 M-tile, 1 N-tile, 2 serial); unset = N-tile below 8 M-tile rows, else M-tile |
| `HIPDNN_EP_GEMM_F32_SKINNY` | 1 | 0 sends fp32 NT 5 <= M <= 64 and fp32 NN 2 <= M <= 32 (64 on B <= 48 MB) back to the GEMVs / `gemm_f32_tile_kernel` |
| `HIPDNN_EP_GEMM_F32_SKINNY_CFG` | -1 | pin a `GEMM_F32_SKINNY_CFGS` index for every fp32 NT shape with transA = 0, M >= 2, K % 4 == 0 and aligned A / B |
| `HIPDNN_EP_GEMM_F32_SKINNY_NN_CFG` | -1 | pin a `GEMM_F32_SKINNY_NN_CFGS` index for every fp32 NN shape with transA = 0, M >= 2, N % 4 == 0, K % 4 == 0 and 16-byte aligned A / B |
| `HIPDNN_EP_GEMM_F32_STAGGER` | 1 | 0 turns StaggerU off in the fp32 NT tile and the fp32 NT / NN skinny kernels |

## Measurement protocol (plan.md §4 / `hip-kernel-perf-measurement` skill)

Every tuner (not just `tuneWmma`) now shares two helpers in `gemm_kernel.hip`:

- `gemmSettleClock()` -- drives a representative config until per-iter latency
  plateaus (>=60 ms total, then 3 consecutive reps within 2%) *before* the
  candidate loop starts. `tuneGemvNt` did not have this before this round
  (WARMUP=1, no settle, averaged, no negative-value guard) -- exactly the
  shape of the gfx1152 decode-tuner incident the skill documents (a table
  shipped from an unsettled sweep was slower than the tuner it replaced).
- `gemmTimeConfigPeak()` -- per-candidate WARMUP launches (untimed,
  re-equilibrate) then ITERS timed launches, keeping the **min** (peak, least
  throttled) with a negative-value guard (`hipEventElapsedTime` occasionally
  returns a small negative on sub-0.1 ms kernels, and an ungated min always
  "wins" with it).

The **offline tie-break** (near-ties within 3% preferring the larger warp
tile) lives in `scripts/update_lut.py`, not in the kernel -- the runtime
tuner stays a plain single pass; the more careful choice is made once,
offline, from the full per-config timings the sweep logs record (mirrors
`../matmul_nbits/scripts/update_lut.py`).

## gfx1151 table (2026-11-10)

Rebuilt from scratch on the frozen kernels with the unit-test timer
(`HIPDNN_UT_TIMING=screen`, one pass over every storable candidate, deduplicated
by final launch). The baseline is the no-LUT default route: a point is stored only
when a candidate beats it by more than 2%, so most shapes keep the default and
have no point. Points from earlier tables are not inherited.

- fp16 1146 points, bf16 108 points, 62 configs. fp32 has no points; every fp32
  request falls back to the resolver default.
- Checked: CSV/JSON/FB consistency, final routes of all 43,695 design requests,
  fp32 empty-default routes, original GEMM leaves at `COVERAGE=2`.
- Held-out sample: 144 of 228 completed; 53 picks are more than 3% slower than the
  best measured candidate (geomean pick/best 1.107). Real-leaf sample of 178
  changed requests: initial geomean new/no-LUT 0.888; the 35 points that were
  slower than the default were reverted to it.
- Deferred: third-layer densification (1,374 requests), 9 requests stopped by the
  measurement cutoff, 21 fp16 and 1 bf16 requests rejected by interference,
  independent bf16 groups, and all fp32 points.

## Generating the table

The 2026-10-23 gfx1151 WMMA refresh used checked REALTIME medians of the final product
launches. The existing `update_lut.py measure`/`build` pipeline remains available
for online-tuner logs; its minimum/peak readings and 3% larger-tile tie-break do
not implement this refresh protocol.

1. Build the scratch measurement runner with the product target
   `--offload-arch=gfx11-generic -O2 -ffp-contract=fast`. Keep the product
   source unchanged and audit the device instructions against the product
   object. Probe every applicable live `kWmma` config through all launch
   remaps and deduplicate by final kernel, grid, block and scalar arguments.
   Include the existing LUT selection as the control. At HEAD5315bf8d the
   ladder has34 configs; independent PGR and signed raster fields are not
   representable in the current LUT schema.
2. Settle the GPU after a host/buffer gap and record a candidate shader
   plateau. Measure opposing candidate orders using at least5 accepted
   coherent-held100MHz REALTIME blocks per candidate/process; corroborate
   each block with CPU release-to-completion time within2%. Use per-process
   medians and a clean third-process confirmation. A difference<=2%
   retains the old selection;
   this is an operational tie policy, not a certified error bound.
   The screening sweep sizes blocks separately in each process; its25ms
   target and the50ms third confirmation do not guarantee identical counts
   across processes. Preserve those counts as a limitation of ranking evidence.
   This refresh follows the user-approved usability gate: no additional
   old/new LUT timing is required before shipment.
3. Merge confirmed improvements into the complete previous JSON. Preserve
   unmeasured phases/points, config indices, dtype groups, ABI/schema and
   weights1/1/1. Do not rebuild only the measured subset or fit imaginary
   weights: current `update_lut.py` has neither a CSV build flag nor
   `--fit-weights`. `lut/gfx1151_points.csv` now records all preserved and
   refreshed points with explicit activation/weight/output dtypes and
   complete geometry. The checked JSON retains the canonical config pool
   and metadata; the complete CSV must reproduce its point geometry.
4. Compile the checked JSON with the unchanged product command:

   ```text
   python scripts/update_lut.py compile --arch gfx1151 --flatc <flatc>
   ```

5. Query all points and model shapes, require0 rejected points, and compare
   actual product routes/validators, then execute every changed final
   request once and compare every output element with an independent CPU
   reference at the original leaf tolerances (fp16 .06, bf16 .3, fp32 .002). Include nearest-neighbour effects and both bias states of every
   WMMA request in the verification universe: bias is absent from the LUT
   key but affects launch remaps.
   Remove any point whose actual route or numerical validation fails and
   regenerate the payload. Confirm all selected geometries exist in the
   product, with zero scratch and spills; retain the unchanged device object. Run the three original GEMM leaves in lookup and autotune,
   `COVERAGE=2`; verify gfx1151 loading and exact hits with
   `HIPDNN_GEMM_LUT_LOG=1` before the scoped data commit.

On the shared GPU, each foreground process runs the idle check, exclusively
creates `GPU_IN_USE.txt` with a `gemm_lut-*` owner, runs for about5min or less,
then releases the lock and yields>=60s. A foreign lock/activity waits until
completion plus10min idle. External GPU counters>=2% reject the whole process
and opposite-order partner, including PID4Copy. This refresh applies that rule
without an own-paging exception.

The source CSV/JSON and compiled `.fb` are committed; scratch runners, embeds
and probes are not. The execution archive is
`AgentStation/handoffs/gemm_lut_refresh_gfx1151/work/` (merge adapter,
CSV reconstruction, device/route audits, accepted blocks and exclusions).
The per-shape invariant-remap list is in that archive. Removing a launch rule
requires a separate route and performance proof.

**After `compile`, re-configure when building the full product.** CMake embeds
`.fb` bytes at configure time (`file(READ ... HEX)`); a build-only refresh can
leave the DLL on old bytes. On the GPU boxes use the original single-kernel
leaf Makefiles rather than a full CMake build. Confirm the payload actually
loaded with `HIPDNN_GEMM_LUT_LOG=1`.

## bf16 vs fp16: merge decision (plan.md §5 Stage C)

**Measured 2026-09-19, verdict: do not merge (`bf16_aliases_f16 = false`).**
Every Wmma shape with both an f16 and a bf16 measured point (2968 shapes)
was compared by winning config index (same global config pool, so an equal
index means an identical tile geometry):

- **Only 62.84% (1865/2968) picked the identical winner config** -- far
  below the >=95% bar plan.md §5 Stage C sets for merging.
- For the 1103 shapes that picked a *different* config, the cost of forcing
  the other dtype's winner (measured from the same shape's own candidate
  timings, not estimated): median ~2-3% either direction, but the tails are
  large -- up to **457% slower** forcing bf16's winner onto f16, and up to
  **2000% slower** forcing f16's winner onto bf16 (n=293 shapes where both
  candidates' timings were available in the raw sweep log; the coarse
  autotune pass does not try every config on every shape, so a cross cost
  could not be computed for every differing shape, but the sample is large
  enough that the tails are not a fluke).

**The old undocumented assumption ("f16 and bf16 share tile geometry" --
`gemm_autotune.fbs:26`'s original comment) was wrong.** bf16 gets its own
measured group in the table, exactly like fp16 and fp32; `resolve()` never
aliases a BF16 query to an F16 point. Analysis script and raw output:
`handoffs/gemm_lut_dtype_gfx1151/artifacts/stage_c_analysis.py` /
`stageC_bf16_vs_f16_analysis.txt` on 171.

## Distance weights (plan.md §1 D8)

`weight_m` / `weight_n` / `weight_k` stay fixed at `1.0` this round -- a
single weight triple for the whole table, same as v1. **Enabling weight
fitting requires first changing this to per-group weights**: fp16 prefill
(all three of M/N/K vary) and fp32 decode (M is pinned at 1, so `log2(M)` is
a constant) do not belong in the same distance metric. `GemmAutotuneLut`
adding a `[GemmGroupWeights]` table is a flatbuffers field-add (backward
compatible, no `schema_version` bump needed) -- do that before fitting
weights, not after.

## Status

Infrastructure is **wired into the DLL** (mirrors `../matmul_nbits/`):

- `gemm_autotune.h/.cpp/.fbs` -- API, flatbuffer-backed nearest-neighbour
  resolver (arch/schema/ABI gated), schema v3 (`schema_version=3`,
  `kernel_abi="gemm-v2"`). Earlier schema versions are rejected at load and
  lookup misses use the static custom default.
- `../common/hipdnn_dtype.{fbs,h}` + `dtype.py` -- the shared dtype
  vocabulary (one enum, three consumers: this schema, the C++ loader, and
  `update_lut.py`; append-only, never renumber).
- `gemm_kernel.hip` includes the header and aliases `gemm_lut =
  hipdnn_ep::gemm_autotune`; the map -> LUT -> sweep dispatch and the
  `wmmaConfigFromGeom` / `gemvNtConfigFromGeom` / `gemvNnConfigFromGeom` /
  `tiledFmaConfigFromGeom` validators call `resolve()`.
- `schemas/CMakeLists.txt` generates `gemm_autotune_generated.h` (flatc), and
  `lib/Runtime/Kernels/CMakeLists.txt` links `gemm_autotune.cpp` and embeds the
  checked-in `lut/<arch>.fb` bytes into each `custom_kernels_<arch>` with pure
  CMake (`file(READ ... HEX)`) -- no generated `.cpp` is committed. Mirrors
  `../matmul_nbits/` and `../gqa/`.

Only the `.fb` (plus its `.json` source) ships. An arch with **no** measured
table still builds: `resolve()` reports a miss, and every shape falls through
to the static custom resolver/default in `lookup` mode. Only explicit
`online` mode runs the supported candidate sweeps. Landing or growing a table is a
pure data addition (a `.fb`), never a code or build change.

## Layout

```
gemm/
  gemm_autotune.h        # LUT lookup API (Request / WmmaAnswer / GemvAnswer / TiledFmaAnswer / resolve)
  gemm_autotune.cpp      # flatbuffer-backed resolver, arch/schema/ABI gated, uint64 group keys
  gemm_autotune.fbs      # schema v2 (includes ../common/hipdnn_dtype.fbs)
  shapes/gemm_shapes.csv       # 317 (N,K,category,priority,source,m_list) rows, derived from HF configs
  shapes/hf_model_configs.json # provenance for gemm_shapes.csv
  scripts/derive_shapes.py     # regenerates gemm_shapes.csv from hf_model_configs.json
  scripts/update_lut.py  # measure (drive the sweep) + build (logs -> json) + compile (json -> fb)
  scripts/data/          # gitignored raw sweep logs -- one file per measure/tag, never committed
  lut/gfx1151.json       # human-readable source (built from scripts/data logs)
  lut/gfx1151.fb         # committed table; CMake embeds its bytes into the DLL
  tools/gemm_autotune_sweep.cpp # offline sweep driver: touches (shape,dtype,phase) once, lets the
                                # kernel's own tuner log its winner (one implementation of "who wins")
  tools/gemm_autotune_stub.cpp  # flatbuffers-free resolve() for the standalone Makefile / the sweep tool
```

## Long-K NT down projections (2026-10-21)

`launchWmma` applies a measured down-projection rule to plain fp16/bf16 NT calls
(`Y=A*B`), with BUFLOAD enabled, even N, and K divisible by 128. It reaches only
the M/N/K pairs below; the tile instances stay outside `kWmma`.

| M | N | K | LONGK_CFG | final instance |
|---|---|---|---|---|
| 512 | 4096 | 14336 | 0 | rolling 128x64 WT4x2 BK64, double LDS buffer, RP161 |
| 512 | 896 | 4864 | 1 | buffer 128x192 WT2x6 BK64, single LDS buffer, RP2 |
| 512 | 2048 | 11008 | 2 | buffer 256x128 WT4x2 BK64, single LDS buffer, RP2 |
| 512 or 2048 | 5120 | 13824 | 2 | buffer 256x128 WT4x2 BK64, single LDS buffer, RP2 |
| 2048 | 1536 | 8960 | 3 | existing buffer 128x128 WT2x4 BK64, single LDS buffer, RP2 |
| 2048 | 2048 | 11008 | 3 | existing buffer 128x128 WT2x4 BK64, single LDS buffer, RP2 |
| 2048 | 3584 | 18944 | 4 | rolling 128x80 WT2x5 BK64, double LDS buffer, RP161 |

The independent `gemm_wmma_bufroll_kernel` stores each tile t+1 row group into
the alternate LDS slot before reusing its register slot to issue tile t+2. RP161
combines mode 1 with bit 32 (rolling stores) and bit 128 (groups spread uniformly
across four WMMA K steps). Address offsets advance in SGPRs; the hot loop has no
address VALU or `vmcnt(0)`. All loads remain in flight, with one WG barrier per
BK64 step. The two-loop StaggerU wrap and straight even tail keep the original
K order; no device workspace is needed. The original templates are preserved.

`HIPDNN_EP_GEMM_LONGK=0` disables the rule.
`HIPDNN_EP_GEMM_LONGK_CFG=0..4` forces an entry only within this measured set.
The explicit `HIPDNN_EP_GEMM_SWIZZLE` override still applies. Default raster is
M-first group 8 on aliasing K, N-first group 8 otherwise, and group 32 for cfg3.

The halo13 gfx1151 two-round headline measurements below compare the baseline
011d10bd against this candidate. Ours uses the slower round; hipBLASLt retains
its fastest physically valid observation across baseline, smoke and comparison.
Cold B above 230 GB/s and dtype peak violations are excluded.

| target MxNxK | dtype | old ms | new ms | fastest valid hipBLASLt ms | hb/new |
|---|---|---:|---:|---:|---:|
| 512x4096x14336 | fp16 | 2.0562 | 1.6967 | 1.8760 | 1.106 |
| 512x4096x14336 | bf16 | 2.0817 | 1.6788 | 1.7348 | 1.033 |
| 2048x3584x18944 | fp16 | 7.1644 | 6.6653 | 6.7271 | 1.009 |
| 2048x3584x18944 | bf16 | 7.0622 | 6.3886 | 6.5152 | 1.020 |

Baseline 929 device instances are instruction-identical; the 937-instance
candidate adds 8 with scratch 0. Rolling 128x64/128x80 use 211/246 VGPR and
55296/59904 B LDS (compiler occupancy 2); buffer 128x192/256x128 use 244/188 VGPR,
46080/55296 B LDS (occupancy 4/8). gfx1170 and gfx11-generic device builds return
rc 0, with all 8 new instances scratch 0.

The complete 450-cell route probe changes 16 cells. Twelve use exactly the same
route and device ISA as the previous lk1 draft and retain its two-round evidence;
four rolling cells are remeasured with 22 required sentinels in 104 processes.
All16 affected cells have new/old0.709–0.997. Required22 sentinels pass the
3% gate after focused paired rechecks; original three GEMM leaves in lookup
and autotune modes (COVERAGE=2,20/30 cases each) and two custom pre512 route
checks pass. One fp32 lookup run with a System copy monitor record is replaced
by an isolated recheck. The actual compiled code objects also preserve all
23,713,036 instruction bytes across929 baseline functions.

## IC-resident NN projections on gfx1151 (2026-10-21)

Two independent FP16 `Y=A*B` routes precede `launchNnMid`, only on runtime
`gfx1151` devices with 16-byte aligned A/B. Both inputs are NN, alpha is 1,
and beta is 0. The exact measured shapes are 128x3072x3072 (route 0) and
80x4096x4096 (route 1); the instances remain outside `kWmma`.

Route 0 is a 128x96 WT4x3 BK64 four-wave tile. A and transposed B share
HB-relative LDS addresses, two slots separated by XOR 32768. The FIFO loop
stores each completed vector into the next slot before reusing its registers
for the next global read. Route 1 is 80x128 WT5x2 BK64, with manual buffer
loads and `slc` on both A and B reads. Repeated product preheat is required:
the same unhinted route was slower after its own preheat than after nnmid
preheat. The cache hint fixes the demonstrated route-1 sensitivity.

`HIPDNN_EP_GEMM_NNWIDE=0` disables both routes.
`HIPDNN_EP_GEMM_NNWIDE_CFG=0|1` restricts dispatch to that entry within the
measured set. No weight repack, workspace, or device allocation is added.

Halo13 System32 HIP 7.2.60201 measurements use GPU 100 MHz REALTIME block
boundaries with coherent stream holds, CPU-completion corroboration,
five samples per side, each product's own 512-call preheat, and adjacent
opposite-order processes. Ours is the slower round's peak; hipBLASLt keeps
the fastest physically valid observation from every retained run and is
pinned by its full kernel name. Continuous blocks and a second block length
are reported separately because a 100 ms gap improves the 128-row result.

| FP16 target | launches/block | old Long-K ms | new ms | retained HB ms | HB/new |
|---|---:|---:|---:|---:|---:|
| 128x3072x3072 | 96 | 0.081018021 | 0.069723958 | 0.064388906 | 0.9235 |
| 128x3072x3072 | 64 | 0.078901094 | 0.070779063 | 0.064388906 | 0.9097 |
| 80x4096x4096 | 96 | 0.119702292 | 0.078114271 | 0.099176563 | 1.2696 |
| 80x4096x4096 | 64 | 0.116048438 | 0.081431562 | 0.099176563 | 1.2179 |

BF16 48x3072x3072 retains its original nnmid route: REALTIME slower-round
peaks 0.036270521/0.035163750 ms at 96/64 launches, retained HB 0.042822708 ms,
ratios 1.1806/1.2178. The NN-wide three-cell 0.95 gate remains FAIL because
the continuous 128-row measurements are below target. Its 100 ms-gap result
(0.065229063 ms, 0.9871) does not replace the continuous failures.

The 450-cell route probe changes exactly the two FP16 cells above. Actual
compiled instruction bytes retain all 929 baseline functions and all
23,713,036 instruction bytes, as well as the eight Long-K instances. The
939-function code object adds two independent NN instances with scratch 0:
248/250 VGPR, 29/30 SGPR, 63616/55936 B LDS. Both use four waves per WG and
one WG per CU at these LDS sizes. Main steady loops retain 13/12 outstanding
buffer reads respectively; draining waits belong to prologue/tail.

Scope validation uses adjacent old/new bench3 processes in two opposite-order
rounds: exactly two changed cells plus 22 sentinels,96 processes. The changed
cells have new/old0.9024/0.6954. Initial FP16 siglip-down and BF16 decode
suspects resolve in eight focused processes (0.9939/0.9947). All24 cells pass
the3% gate and accuracy checks. These scoped event results compare old/new
only; the absolute targets above use separate checked REALTIME measurements.
Six original GEMM leaf/mode tests pass COVERAGE=2 (20/20 each), as do four
focused NNwide correctness checks.114 scope/UT/retest monitor files have no
foreign GPU records. gfx1170 and gfx11-generic device builds both return rc0.

## Streaming NN 8192-square projections on gfx1151 (2026-10-21)

Two independent `gemm_nnstream_kernel` instances precede `launchNnMid` for
runtime gfx1151, plain NN, alpha 1 / beta 0 and 16-byte aligned A/B. Dispatch
is limited to the measured FP16 64x8192x8192 (cfg 0) and BF16 48x8192x8192
(cfg 1) cells. These instances are outside `kWmma`; no device allocation,
weight transform or workspace is added.

The BK32 loop uses natural row-major B in two LDS slots, with eight elements
of padding per 16 K rows. A loads directly into WMMA fragments. One rolling
register slot stores tile t+1 before reusing each group to prefetch t+2 while
tile t computes. FP16 uses 64x128 WT1x8 / 64 workgroups; BF16 uses 48x64
WT3x1 / 128 workgroups. Only B has the measured `slc` cache hint.

`HIPDNN_EP_GEMM_NNSTREAM=0` disables both routes.
`HIPDNN_EP_GEMM_NNSTREAM_CFG=0|1` restricts the route to that measured entry.

Pure-read probes compare N=8192 and N=8256 on the same halo13. The old
nnmid global read map alone reaches 214/204 GB/s at8192 (FP16/BF16), while
buffer variants reach188/202; ratios against8256 depend on instruction,
barrier and map. This does not establish a universal fixed4KB bandwidth
ceiling. Controlled GEMM screens show a large gain from rolling scheduling
(FP16) and direct A fragments (BF16); changing only global/buffer/slc on the
old nnmid map is insufficient.

Product comparisons use checked GPU100MHz REALTIME, coherent stream holds,
CPU completion, full-name pinned fastest historical hipBLASLt kernels,
512 calls of each product's own preheat, five blocks per side and adjacent
opposite-order rounds. The table keeps every earlier valid HB minimum and
uses ours slower round. A System-copy-marked FP16 process is excluded and
its complete four-process ABBA pair replaced. Both32/64 launch lengths pass
HB/new>=0.95, including the slower round's block median.

| dtype | launches/block | old a5768ffa ms | new ms | retained valid HB ms | HB/new | new GB/s |
|---|---:|---:|---:|---:|---:|---:|
| f16 | 32 | 0.785200000 | 0.609192500 | 0.689064219 | 1.1311 | 220.32 |
| f16 | 64 | 0.787478125 | 0.607150625 | 0.689064219 | 1.1349 | 221.06 |
| bf16 | 32 | 0.747660000 | 0.578235625 | 0.632737187 | 1.0943 | 232.12 |
| bf16 | 64 | 0.745926719 | 0.575821875 | 0.632737187 | 1.0988 | 233.09 |

All929 original functions retain their23,713,036 instruction bytes; all939
preceding functions also remain byte-identical. Only two new functions are
added (941 total): FP16 VGPR211 / SGPR26 / LDS16448B and BF16 VGPR91 /
SGPR28 / LDS8256B, both scratch0, four waves per WG. The unfolded two-BK32
steady loops contain32/12 WMMA and16/28 buffer-b128 loads. B store/refill
waits retain nonzero vmcnt; A fragment dependency waits drain before WMMA,
so the whole loop is not described as free of vmcnt(0).

gfx1170 and gfx11-generic device builds return rc0 and both new instances
have scratch0 (gfx1170 VGPR159/110; generic187/83). The full450-cell route
probe changes exactly these two target cells. Required validation covers
two changed routes plus22 sentinels, adjacent old/new opposite-order rounds,
and all six original dtype/mode UTs with COVERAGE=2 plus four focused route
checks. One BF16 NT16 scope cell passes a focused opposite-order repeat;
two System-copy-marked fp32 UT processes pass individual clean repeats.
Complete scope/UT evidence and monitor audit accompany the handoff.

## Next-tile load overlap under generic compilation

The independent `gemm_wmma_overlap_kernel` instances preserve the existing
WMMA kernels and leave the online tile table unchanged. In the main loop,
the next tile's global reads issue into staging registers before the
current tile's WMMA work. The staging values reach LDS after that work.
Compiler scheduling builtins retain this order under `gfx11-generic -O2`.
Qwen-up and SigLIP-down use a selective VMEM/WMMA fence that allows
address ALU and LDS reads to move across it, plus a B-fragment fence that
orders those reads before WMMA while allowing address ALU. ViT-down uses a full scheduling fence. LDS synchronization uses
compiler builtins and the wait encoding for 16-half fragments.

Only the measured FP16/BF16 plain NT requests take the new route:

| Config | M x N x K | Tile / wave tile / BK | Buffers | Raster / stagger |
|---|---|---|---|---|
| 0 | 2048 x 18944 x 3584 | 256x128 / 4x2 / 32 | single | 8 / 0 |
| 1 | 729 x 1152 x 4304 | 96x128 / 3x4 / 16 | double | 8 / 0 |
| 2 | 1576 x 768 x 3072 | 128x256 / 4x2 / 64 | single | -8 / 0x21f |

The guard requires alpha=1, beta=0, no C input, aligned A/B, original
contiguous strides, the final measured tile, one BC=false launch and the
measured raster/stagger. Buffer reads, line-wide B and PGR2 retain their
existing dispatch. Other shapes and the gfx1170 fragment encoding use the
existing route. No workspace allocation is added.

`HIPDNN_EP_GEMM_OVERLAP=0` disables the route (default 1).
`HIPDNN_EP_GEMM_OVERLAP_CFG=0|1|2` restricts the route to that measured
request; it does not force an unmeasured tile. Default -1 permits all three.

The Qwen and SigLIP kernels increase register use and reduce theoretical
residency on devices with 1024 VGPRs per SIMD. Measure these exact requests
on gfx1150/gfx1152 before merging this experimental branch. The occupancy
and scope tables in the accompanying result are the evidence for this
restriction.

Formal validation uses five serial processes and five accepted
REALTIME/CPU blocks per object per process, with matching launch counts,
coherent hold, loaded-state settle and symmetric medians. P and the new
product are generic/O2; A is the original gfx1151/O3 reference.
The installed hipBLASLt is gfx1151-specific compilation and serves as a
reference; a ready generic device library was not found.

| Request | P ms | New ms | New/P | New/A |
|---|---:|---:|---:|---:|
| bf16_2048_18944_3584_0_1_0 | 7.336336 | 6.370082 | 0.8683 | 0.9882 |
| f16_2048_18944_3584_0_1_0 | 7.543051 | 6.653214 | 0.8820 | 0.9994 |
| bf16_729_1152_4304_0_1_0 | 0.218603 | 0.199486 | 0.9125 | 0.9940 |
| bf16_1576_768_3072_0_1_0 | 0.205982 | 0.189963 | 0.9222 | 1.0039 |
| f16_729_1152_4304_0_1_0 | 0.225260 | 0.205356 | 0.9116 | 0.9997 |
| f16_1576_768_3072_0_1_0 | 0.210481 | 0.194570 | 0.9244 | 0.9969 |

All six requests improve by at least 5% and tie the architecture-specific
self reference within the 2% comparison policy. The 941 preceding kernels
retain 23,847,700 instruction bytes; the six new instances are scratch/spill
zero. The gfx1170 build retains 941 instances; generic has 947. Separate
generic device compilation matches all product instructions.

The 450-request route union changes exactly six target launches and
retains the other 444 kernel identities and launch geometry. The six
changed requests are already in the 22 sentinel set, so the measured
scope union contains 22 requests. Bias uses its original route.

The scope comparison uses five clean processes and five accepted blocks
per side. Two unchanged 128x4096x4096 sentinels initially read 6.16% slower
(FP16) and 3.97% slower (BF16). A separate five-process diagnostic keeps
those different-output comparisons and adds old/new calls sharing one
output address. The shared-output new/old ratios are 1.0032 and 1.0015;
the old/old controls also tie. All four sides launch the same existing
kernel with the same grid, block and dynamic LDS. The difference follows
the output-address condition; its physical cache/address cause is not
established. The original valid readings remain in the scope report.
No code regression is confirmed by the controlled comparison.

All six original FP32/FP16/BF16 lookup/autotune tests pass COVERAGE=2
(20 cases each, 120 total), with clean GPU monitors. Original Makefiles
and test sources are unchanged; the scratch builds reuse the audited
product object. The six new routes pass separate formal/scope correctness
and actual-launch checks.

## Measured small-M NT rolling routes

Seven measured FP16/BF16 requests use small rolling WMMA tiles before the
existing skinny/WMMA route ladder. The rolling loop stores a group from
tile t+1 into alternate LDS, then reuses its staging registers to read
tile t+2 while tile t computes. Load groups are spread over four K steps;
the K wrap is split into two loops. The independent M16 implementation
owns LDS within one wave and omits workgroup barriers. The other three
geometries use new instances of the existing rolling template.

| Config | M x N x K | Dtype | Tile / wave tile / BK | Threads / WGs |
|---|---|---|---|---|
| 0 | 32 x 4096 x 4096 | FP16, BF16 | 32x64 / 2x2 / 64 | 64 / 64 |
| 1 | 64 x 14336 x 4096 | FP16, BF16 | 16x16 / 1x1 / 64 | 32 / 3584 |
| 2 | 16 x 4096 x 4096 | FP16 only | 16x32 / 1x2 / 64 | 32 / 128 |
| 3 | 128 x 4096 x 14336 | FP16, BF16 | 64x16 / 1x1 / 64 | 128 / 512 |

Only these exact requests use the entry: NT layout, alpha=1, beta=0,
16-byte aligned A/B/Y, 16-element WMMA fragments and runtime gfx1151.
The M16 BF16 request retains its preceding route. New instances stay out
of kWmma, and the original online sweep/remaps remain in use.

`HIPDNN_EP_GEMM_SMALLM=0` disables the entry (default 1).
`HIPDNN_EP_GEMM_SMALLM_CFG=0|1|2|3` restricts it to the matching measured
request; default -1 permits all four. It yields to explicit SKINNY, WMMA,
GEMV, TILEDFMA and LONGK config pins, and to SKINNY=0 or BUFLOAD=0.
Explicit SWIZZLE and stagger settings apply. The measured defaults are
M-first raster -8 and StaggerU 0x10021f, with both operand strides equal K.

The product comparison uses gfx11-generic -O2 and the fixed complete
hipBLASLt kernel+solution identities from the preceding LUT refresh.
The installed vendor device kernels target gfx1151. Five serial processes
use five accepted REALTIME/CPU blocks per side, matched launch counts,
coherent hold, loaded settle and symmetric medians. Correctness has
independent outputs; every timed side shares one physical output address.
The gfx1151 -O3 columns compile the same old/new sources as a reference.

| Request | Old product ms | New product ms | HB ms | HB/new product | HB/new gfx1151 |
|---|---:|---:|---:|---:|---:|
| f16_32_4096_4096_0_1_0 | 0.070910750 | 0.045061145 | 0.048276965 | 1.0714 | 1.0715 |
| bf16_32_4096_4096_0_1_0 | 0.069770590 | 0.045029880 | 0.048316235 | 1.0730 | 1.0729 |
| f16_64_14336_4096_0_1_0 | 0.697516800 | 0.561981400 | 0.581422400 | 1.0346 | 1.0465 |
| bf16_64_14336_4096_0_1_0 | 0.697816550 | 0.561265450 | 0.584601650 | 1.0416 | 1.0500 |
| f16_16_4096_4096_0_1_0 | 0.041132035 | 0.038962247 | 0.040320795 | 1.0349 | 1.0273 |
| f16_128_4096_14336_0_1_0 | 1.045464200 | 0.799707700 | 0.851759450 | 1.0651 | 1.0668 |
| bf16_128_4096_14336_0_1_0 | 1.052394700 | 0.791655750 | 0.845186900 | 1.0676 | 1.0702 |

All seven product ratios meet the 1.00 gate. The preceding 947 kernels
retain all 24,003,336 instruction bytes; seven scratch/spill-zero additions
bring the generic product to 954. Their instructions exactly match the
selected screening instances. Separate generic device compilation matches
all 954 product payloads; gfx1170 retains all 941 preceding payloads.

| New tile | Used VGPR / descriptor | LDS bytes | Threads | WG/WGP at 1024 / 1536 VGPRs | Wave/SIMD at 1024 / 1536 |
|---|---|---:|---:|---|---|
| 32x64 | 162 / 200 | 27648 | 64 | 4 / 4 | 2 / 2 |
| 16x16 | 87 / 200 | 9216 | 32 | 14 / 14 | 3.5 / 3.5 |
| 16x32 | 131 / 200 | 13824 | 32 | 9 / 9 | 2.25 / 2.25 |
| 64x16 | 79 / 168 | 23040 | 128 | 5 / 5 | 5 / 5 |

These are complete-WG theoretical bounds in WGP mode: four SIMD and
128 KiB physical LDS, with a 64 KiB per-WG addressability limit.
Descriptor VGPR counts round further to the physical 16/24-register
granules on 1024/1536-VGPR devices. Used VGPR counts alone do not give
residency. The first three geometries increase WG capacity while reducing
wave capacity versus the replaced kernels; the down-projection increases
both. Only gfx1151 performance was measured; the 1024-VGPR columns are
theory, and extending runtime enablement needs separate measurements.

The 450-request route union changes exactly these seven cells and retains
443 final templates, launch geometries and runtime parameters. Scope
validation covers the seven changes, 22 preceding sentinels and two
baseline-passing NN M128 cells, with shared timed output and preserved
confirmations for any greater-than-2-percent suspect.

All six original dtype/mode leaf UTs pass COVERAGE=2 (120 cases).
Four original-CLI checks additionally execute FP16 M16 and BF16 M32 in
lookup/autotune modes and confirm the new entry. Test sources, Makefiles
and LUTs retain their original content. This is the Stage 1 experiment;
the remaining decode, NN and vision work is outside this change.

## Measured FP16 NT decode route

The preceding FP16 1x4096x4096 NT entry selected GEMV cfg0: 32 threads,
one output column and 4096 workgroups. Each thread traversed 16 vector
iterations with two b128 loads and eight dependent mixed FP32 FMAs per
iteration. FP16 widening folds into mixed FMA instructions; independent
conversion instructions do not explain the difference from BF16, whose
preceding route used 256 threads, four columns and two vector iterations.

An independent gemm_gemv_decode_nt_kernel<half,512,4> instance covers four
output columns per workgroup. Four accumulators share A, and 512 threads
cover K4096 in one vector iteration, reducing the grid to 1024 workgroups.
The vector body has five b128 loads and 32 mixed FMAs with progressive
VMEM waits. The K loop has no workgroup barrier; one post-loop barrier
reduces the 16 waves through 256 bytes of LDS. Geometry, reuse and loop
length change together, so their individual performance contributions
have not been isolated.

Only FP16 M=1,N=4096,K=4096 NT, alpha=1,beta=0, 16-byte aligned A/B/Y,
runtime gfx1151 and 16-element fragments use this entry. It precedes
the measured small-M ladder and stays outside kWmma. BF16 and other
shapes retain their preceding routes. HIPDNN_EP_GEMM_DECODENT=0 disables
it (default 1). HIPDNN_EP_GEMM_DECODENT_CFG=0 selects only this measured
request; default -1 permits it, and other nonnegative values fall through.
Explicit GEMV, WMMA, SKINNY and TILEDFMA config pins take priority.
The disable and GEMV_CFG=0 controls were checked through actual launches.

Both product sides use gfx11-generic -O2 on halo13; hipBLASLt uses its
fixed complete LUT-refresh kernel and solution identity, compiled for
gfx1151. Five serial processes take five accepted REALTIME/CPU blocks
per side, with matched 2000 launches, coherent hold, loaded settle of
at least 250 ms and three stable transitions, rotating order and
symmetric medians. Timed sides share a physical output; correctness uses
independent outputs. B is resident at the 32 MiB MALL boundary.

| Shape | Old product ms | New product ms | Speed improvement % (old/new - 1) | Time decrease % | HB ms | Before HB/ours | After HB/ours | Launches/block |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FP16 1x4096x4096 NT | 0.042052675 | 0.035614040 | 18.079 | 15.311 | 0.036851035 | 0.8763 | 1.0347 | 2000 |

The HB/new gate is >=1.00. Effective useful B rates old/new/HB are
797.9/942.2/910.5 GB/s under this resident protocol;
they are not DRAM bandwidth measurements. BF16 new/old is
0.999654, with the identical preceding 256x4 route.

All 954 preceding generic device kernels preserve 24,046,320 instruction
bytes. The sole new kernel is 2816 bytes and matches screened FP16 cfg9
byte for byte. Scratch and both spill counts are zero. Same-compiler
assembly reassembly matches all 955 actual object payloads. Independent
device compilation of the installed final source matches all 955 generic
payloads and preserves all 941 preceding gfx1170 payloads; gfx1151 device
compilation also succeeds. The new route is not instantiated for the
8-element fragment build.

| Instance | Used / descriptor VGPR | LDS B | Threads | Physical VGPR 1024 / 1536 | WG/WGP 1024 / 1536 | Waves/SIMD 1024 / 1536 | Decrease |
|---|---|---:|---:|---|---|---|---|
| Old FP16 cfg0 | 15 / 16 | 0 | 32 | 16 / 24 | 16 / 16 | 4 / 4 | -- |
| New FP16 decode | 36 / 40 | 256 | 512 | 48 / 48 | 4 / 4 | 16 / 16 | WG capacity 16 to 4 on both budgets |

These complete-WG theoretical bounds use four SIMD per WGP, a 16-WG cap,
16 waves per SIMD and 128 KiB physical LDS, with 512-byte LDS allocation
and 16/24-register physical VGPR granules. WG capacity decreases 16 to 4
while wave capacity increases 4 to 16 on both budgets. Actual dynamic
residency and performance on other devices were not measured.

The 450-shape union changes exactly this FP16 cell; 449 final templates,
launch geometries and uniform parameters remain identical. The 31-cell
scope includes 22 sentinels, all seven Stage 1 cells and two NN M128
baseline-passing cells, with five checked processes and shared timed Y.
No clean initial scope cell exceeds 2%.
All six original dtype/mode leaves pass COVERAGE=2 (120 cases), plus two
original positional-CLI checks for this route in lookup/autotune modes.
Test sources, Makefiles and LUTs retain their original content. This
completes Stage 2; the remaining NN and vision stages are separate work.
The original FP32 and FP16 mode pairs were retained and excluded for
unowned System Copy samples at or above 2%; complete clean replacement
pairs were accepted under the unchanged original test protocol.

<!-- gemm_lag22_s3_gfx1151:start -->
## Stage3 exact NN2048 (2026-10-27; exp/gemm-lag22 after e0dd8e24)

Stage3精确NN 2048×4096×4096 fp16/bf16，旧版e0dd8e24、新版gfx11-generic -O2，halo13/gfx1151。
统一独立96×128 WT6×2 BK32、单LDS、PGR2两寄存器槽、staged A、自然B[k][n] LDS、swizzle=32，grid32×22=704WG、128threads/4waves。
本格swizzle32遍历整行32个N tiles；M2048有22个M tiles与两维面板复用，不能把这个结论外推到旧small-M失败试验。
旧FP16 lookup miss走128×64，BF16 exact LUT config57走128×128；旧B逐半元素转置LDS。
BK32两步(K64)有48WMMA、176LDS读/1024B每lane、14b128全局读、14b128 LDS写、4barrier。
收益来自自然LDS/计算复用/两槽流水和遍历取舍；没有消除同步，HB K64只有2barrier。

| shape | 旧产品 ms | 新产品 ms | 提升 % (旧/新−1) | 耗时下降 % | hipBLASLt ms | 优化前 HB/我们 | 优化后 HB/我们 | launch 数 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| f16 2048×4096×4096 NN | 1.992764375 | 1.775942656 | 12.209% | 10.880% | 1.857459844 | 0.9321× | 1.0459× | 64 |
| bf16 2048×4096×4096 NN | 2.069461094 | 1.777708906 | 16.412% | 14.098% | 1.892774531 | 0.9146× | 1.0647× | 64 |
五进程、每边五采纳块、相同64launch/块、REALTIME/CPU≤2%、hold、≥250ms+3stable、块/进程顺序轮换、common physicalY、独立正确性Y与完整固定vendor身份。B32MiB、A16MiB；不把有用字节率称为DRAM带宽，不声称持续频率或动态驻留已测。

NNLARGE=0关闭；NNLARGE_CFG=0只限制到已测请求，其余非负CFG回退。
全名为HIPDNN_EP_GEMM_NNLARGE、HIPDNN_EP_GEMM_NNLARGE_CFG。
仅NN、alpha1/beta0、aligned16 A/B/Y、fragment16、runtimegfx1151；显式WMMA/SKINNY/GEMV/TILEDFMA cfg pin优先。新实例不进kWmma，无新设备分配/inline asm，无AMD_DIRECT_DISPATCH。

原955设备payload/24,049,136字节与原955 ABI/resource metadata全部不变，新增2个scratch/spill0；957个同编译器assembly重组payload全等。
筛选与产品新payload逐字节相等：True。20次正式实际launch绑定通过。
450路由仅改变2格、448不变；scope33五进程每边五块，新/旧最大1.004417，2%不退化通过。
原六leaf lookup/autotune/COVERAGE2共120case与四次原CLI NN路由检查通过；generic957/gfx1170-941编译和payload检查通过，gfx1151编译通过。

| 实例 | used/descriptor VGPR | LDS B | threads/WG | 1024 physical VGPR | 1024 WG/WGP | 1024 waves/SIMD | 1536 physical VGPR | 1536 WG/WGP | 1536 waves/SIMD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 旧 f16 | 251/256 | 27648 | 128 | 256 | 4 | 4.0 | 264 | 4 | 4.0 |
| 旧 bf16 | 160/160 | 36864 | 256 | 160 | 3 | 6.0 | 168 | 3 | 6.0 |
| 新 bf16 | 234/240 | 15904 | 128 | 240 | 4 | 4.0 | 240 | 6 | 6.0 |
| 新 f16 | 234/240 | 15904 | 128 | 240 | 4 | 4.0 | 240 | 6 | 6.0 |

新BF16在1024下wave容量6→4降低，完整WG容量3→4增加；只路由gfx1151，未在gfx1150/1152运行。表为完整WG理论上界，不是动态实测驻留。

Evidence: AgentStation `handoffs/gemm_lag22_s3_gfx1151/work/` (diagnose, screen_report, finalists_audit, formal, scope, occupancy and gate audits).
Stage3 ends here; Stage4 requires a new handoff.
<!-- gemm_lag22_s3_gfx1151:end -->

<!-- gemm_lag22_s4a_gfx1151:start -->
## Stage4a exact NT ViT-up (2026-10-27; exp/gemm-lag22 after a62a4a81)

The five 1576×3072×768 NT requests now use independent gfx1151 routes. The old half route was 96×96 WT3×3 BK32 double-LDS/PGR1, 544 WG; the old FP32 bias route was 128×64 BK32 TM8×8 with 51,200 B LDS. Separate K scans estimate fixed costs of 10.41–11.20% for half and 15.78% for FP32. These are fitted intercepts, not dynamic cycle fractions.

Half uses 80×128 WT5×2 BK32, double LDS/PGR2 and swizzle 8, with exact M/N/K constants. Aligned K8-chunk XOR removes LDS row padding while preserving b128 reads/writes. Full tiles avoid per-output mask branches; plain output consumes accumulators directly and bias keeps paired four-byte stores. FP32 uses 128×64 BK16 TM8×8, bounded K-unroll8, bias reads outside the output row loop and b128 Y stores. LDS drops 51,200→25,600 B and used VGPR 173→131 for FP32.

|shape|old product ms|new product ms|speed improvement %|time decrease %|hipBLASLt ms|HB/old|HB/new|launches|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|f16 1576×3072×768 NT|0.204144063|0.183081016|11.505%|10.318%|0.186566484|0.9139×|1.0190×|128|
|bf16 1576×3072×768 NT|0.202432969|0.182626563|10.845%|9.784%|0.189030000|0.9338×|1.0351×|128|
|f16 1576×3072×768 NT +bias|0.207104219|0.189211094|9.457%|8.640%|0.195381484|0.9434×|1.0326×|128|
|bf16 1576×3072×768 NT +bias|0.206848828|0.187961563|10.048%|9.131%|0.191415859|0.9254×|1.0184×|128|
|f32 1576×3072×768 NT +bias|0.862758906|0.691354531|24.793%|19.867%|0.807158594|0.9356×|1.1675×|64|

Speed improvement % =100*(old/new−1); time decrease % =100*(1−new/old). Product gfx11-generic -O2, halo13/gfx1151, five clean serial processes with five accepted REALTIME/CPU blocks per side. Launch counts match across old/new/vendor, with hold, loaded settle ≥250 ms plus three stable transitions, alternating order and common physical timed Y. Vendor uses verified complete fixed LUT-refresh identities. The strict HB/new ≥1.00 gate is PASS; all clean observations are retained. Screening/finalist values are separate.

`HIPDNN_EP_GEMM_VITUP=0` disables these routes. `HIPDNN_EP_GEMM_VITUP_CFG=0..4` restricts selection to FP16 plain, BF16 plain, FP16 bias, BF16 bias or FP32 bias respectively. The default −1 permits all five; other nonnegative values fall through. Eligibility is exact NT shape, alpha1/beta0 plain or alpha1/beta1 row-broadcast C, aligned16 A/B/Y and aligned16 C for bias. Explicit WMMA/SKINNY/GEMV/TILEDFMA/F32_TILE cfg pins and nonzero raster pins have priority. New entries stay outside kWmma and the LUT. No device allocation, inline assembly or direct-dispatch setting is added.

|instance|used / descriptor VGPR|LDS B|threads|physical VGPR 1024 /1536|WG/WGP 1024 /1536|waves/SIMD 1024 /1536|
|---|---:|---:|---:|---|---|---|
|old f16_1576_3072_768_0_1_0|223 / 224|30720|128|224 / 240|4 / 4|4.0 / 4.0|
|new f16_1576_3072_768_0_1_0|235 / 240|26624|128|240 / 240|4 / 4|4.0 / 4.0|
|old bf16_1576_3072_768_0_1_0|223 / 224|30720|128|224 / 240|4 / 4|4.0 / 4.0|
|new bf16_1576_3072_768_0_1_0|235 / 240|26624|128|240 / 240|4 / 4|4.0 / 4.0|
|old f16_1576_3072_768_0_1_1|223 / 224|30720|128|224 / 240|4 / 4|4.0 / 4.0|
|new f16_1576_3072_768_0_1_1|236 / 240|26624|128|240 / 240|4 / 4|4.0 / 4.0|
|old bf16_1576_3072_768_0_1_1|223 / 224|30720|128|224 / 240|4 / 4|4.0 / 4.0|
|new bf16_1576_3072_768_0_1_1|236 / 240|26624|128|240 / 240|4 / 4|4.0 / 4.0|
|old f32_1576_3072_768_0_1_1|173 / 200|51200|128|208 / 216|2 / 2|2.0 / 2.0|
|new f32_1576_3072_768_0_1_1|131 / 168|25600|128|176 / 168|5 / 5|5.0 / 5.0|

Complete-WG theory uses four SIMD/WGP, 128 KiB LDS with 512-byte allocation, 16 waves/SIMD, 16 WG/WGP and physical VGPR granules 16/24 for budgets 1024/1536. Half stays 4 WG/WGP; FP32 rises 2→5. No new instance reduces theoretical WG or wave capacity. Dynamic residency and performance on 1024-VGPR devices were not measured; runtime routing is gfx1151 only.

All 957 original instruction payloads (24,081,240 bytes) and their ABI/resource metadata remain unchanged. Exactly five scratch/spill 0 kernels are added. All five product payloads equal the selected screening payloads; all 962 equal same-compiler assembly reassembly. Generic final device code equals the measured product; gfx1170 retains 941 old payloads and does not instantiate these additions. gfx1151 compilation succeeds.

The 450-request final-route union changes exactly five and preserves 445 templates/geometries/uniforms. Scope 38 includes Stage3’s 33 requests plus these five; maximum initial new/old is 1.007016, with 0 initial >2% suspects. Any independent confirmations remain in the scope audit; no clean tie was resampled to force a favorable outcome.

All six original dtype/mode leaves pass COVERAGE2 (120 cases). Four unchanged positional-CLI calls check the FP16/BF16 plain routes in both modes. That CLI fixes beta0, so two unchanged five-cell validator runs cover plain and bias in lookup/online modes (10 focused cells), including actual new-route capture and independent correctness outputs. Original test sources, Makefiles and LUTs are unchanged.

Evidence: AgentStation `handoffs/gemm_lag22_s4a_gfx1151/work/` (diagnose,screen_report,finalhalf/finalf32 audits,formal,scope,occupancy, product/arch/launch/UT audits). Stage4a ends here; Stage4b is separate work.
<!-- gemm_lag22_s4a_gfx1151:end -->

<!-- gemm_lag22_s4b_gfx1151:start -->
## Stage4b exact NT SigLIP and MiniLM (2026-11-02; exp/gemm-lag22 after a387b9c0)

The old SigLIP-down route used 96x128 WT3x4 BK16 with 72 WG. Its main
K64 work required four shared barrier rounds, while the pinned vendor needed two.
SigLIP-up used 128x128 WT4x4 and 204 WG, against the vendor's 128x96 and 270 WG.
MiniLM-down used 64x64 BK64 buffer staging and 48 WG; the fitted fixed-cost
fractions were 13.44% FP16 and 13.61% BF16. Fitted intercepts include launch,
prologue, epilogue, tail and state costs; they do not isolate writeback cycles.

SigLIP-down uses independent exact NT instances. The regular pipeline
computes K4288, followed by a bounded K16 suffix in the original order.
FP16: 96x96 WT3x3 BK32, 4 waves/WG, 96 WG, sw8, double LDS/PGR2, CF0/SF2.
Per-row-chunk B scalar buffer descriptors use one common vector
row/K-chunk offset. Descriptors are initialized before K, avoiding
per-chunk scalar additions in the main loop.
A keeps its original last-row clamps; N is an exact multiple of
the tile, so B needs no column clamp.
BF16: 96x96 WT3x3 BK32, 4 waves/WG, 96 WG, sw8, double LDS/PGR2, CF0/SF2.
Per-row-chunk B scalar buffer descriptors use one common vector
row/K-chunk offset. Descriptors are initialized before K, avoiding
per-chunk scalar additions in the main loop.
A keeps its original last-row clamps; N is an exact multiple of
the tile, so B needs no column clamp.
Compact aligned K8-XOR staging and compiler lifetime fences are retained.
Full output tiles avoid element mask branches; the final M tile keeps
its row checks. These mechanisms are measured together; static counts
and fitted intercepts do not isolate the contribution of each change.

SigLIP-up uses 128x96 WT4x3 BK32 double-LDS/PGR2, sw8 and a before-store fence,
with 270 WG and unchanged boundary checks.
Each exact entry owns a verbatim fixed-raster helper copy so whole-product
compilation retains the selected screening payload. The legacy shared raster
helper and all previous kernel definitions remain unchanged.

MiniLM uses one 64x96 output tile per eight-wave WG, grid4x8=32 WG. Two
four-wave groups compute disjoint K768 halves with separate double-LDS staging.
After both groups finish, group 1 writes 6144 FP32 partials (24576 B) into the
existing B LDS allocation. A second uniform barrier publishes those values;
group 0 adds them and uses the unchanged SDK output conversion. Group 1 returns
after the final shared barrier. FP16 uses CF0/SF2 with rolling store/refill;
BF16 uses CF1/SF2 with two register slots. There is no cross-WG reduction,
atomics or new device allocation. The FP32 sum order changes and passes the
unchanged accuracy thresholds.

The MiniLM backedge advances K64 per group, K128 across the WG. Compared with
the unsuccessful four-wave 64x96 BK32 alternative, main staging barrier rounds
fall from 48 to 24, plus two reduction rounds. The parent already used BK64 and
about 24 main rounds, so the parent barrier count was not halved. Parallel K
work, operand reuse, grid, staging and output are a measured combined tradeoff.

|shape|old product ms|new product ms|speed improvement %|time decrease %|hipBLASLt ms|HB/old|HB/new|launches|gate|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
|f16 729x1152x4304 NT|0.198167500|0.175116719|13.163%|11.632%|0.178029063|0.898377x|1.016631x|128|PASS|
|bf16 729x1152x4304 NT|0.193618750|0.170862188|13.319%|11.753%|0.175863281|0.908297x|1.029270x|128|PASS|
|f16 729x4304x1152 NT|0.202587500|0.194442266|4.189%|4.021%|0.195381875|0.964432x|1.004832x|128|PASS|
|f16 512x384x1536 NT|0.021362415|0.019048510|12.147%|10.832%|0.019709470|0.922624x|1.034699x|2000|PASS|
|bf16 512x384x1536 NT|0.021084955|0.018927420|11.399%|10.233%|0.019519240|0.925743x|1.031268x|2000|PASS|

Speed improvement %=100*(old/new-1); time decrease %=100*(1-new/old).
Product gfx11-generic -O2 on halo13/gfx1151. Five clean serial processes, five
accepted held REALTIME/CPU blocks per side, matched 128 launches for SigLIP and
2000 for MiniLM, loaded preheat >=250 ms plus three stable transitions,
alternating order and one common physical timed Y. Vendor identities are the
verified fixed complete kernel+solution names selected by the prior LUT refresh.
The statistic is the median of process medians; ranges remain in formal_v1.md.
All clean failures, ties and unselected candidates are retained. Screening and
finalist values are separate from this actual-product table.

Strict HB/new>=1.00 gate: PASS.

`HIPDNN_EP_GEMM_MID4B=0` disables the new routes. `MID4B_CFG=0..4` (full
name `HIPDNN_EP_GEMM_MID4B_CFG`) restricts eligibility to FP16 down, BF16 down,
FP16 up, FP16 MiniLM or BF16 MiniLM. Default `-1` allows all five; other nonnegative
values fall through. Eligibility is the exact measured NT request, alpha=1/beta=0,
16-byte aligned A/B/Y, fragment16 and runtime gfx1151. Explicit old WMMA/SKINNY/GEMV/
TILEDFMA/F32_TILE config pins and nonzero raster pins have priority. Disable,
CFG0, wrong-CFG, WMMA pin and raster pin controls pass. New entries remain
outside kWmma and the LUT; no inline assembly or direct-dispatch setting is added.

|instance|used / descriptor VGPR|LDS B|threads/WG|physical VGPR 1024 /1536|WG/WGP 1024 /1536|waves/SIMD 1024 /1536|lower WG capacity|
|---|---:|---:|---:|---|---|---|---|
|old f16_729_1152_4304_0_1_0|190 / 192|21504|128|192 / 192|5 / 6|5.0 / 6.0|none|
|new f16_729_1152_4304_0_1_0|183 / 184|24576|128|192 / 192|5 / 5|5.0 / 5.0|1536|
|old bf16_729_1152_4304_0_1_0|190 / 192|21504|128|192 / 192|5 / 6|5.0 / 6.0|none|
|new bf16_729_1152_4304_0_1_0|183 / 184|24576|128|192 / 192|5 / 5|5.0 / 5.0|1536|
|old f16_729_4304_1152_0_1_0|254 / 256|40960|128|256 / 264|3 / 3|3.0 / 3.0|none|
|new f16_729_4304_1152_0_1_0|247 / 248|28672|128|256 / 264|4 / 4|4.0 / 4.0|none|
|old f16_512_384_1536_0_1_0|124 / 200|36864|128|208 / 216|3 / 3|3.0 / 3.0|none|
|new f16_512_384_1536_0_1_0|133 / 152|40960|256|160 / 168|3 / 3|6.0 / 6.0|none|
|old bf16_512_384_1536_0_1_0|124 / 200|36864|128|208 / 216|3 / 3|3.0 / 3.0|none|
|new bf16_512_384_1536_0_1_0|145 / 152|40960|256|160 / 168|3 / 3|6.0 / 6.0|none|

Complete-WG theory uses 20 WGP, four SIMD/WGP, 128 KiB LDS rounded to 512 B,
16 waves/SIMD and 16 WG/WGP. Descriptor VGPR rounds to 16/24 physical granules
for 1024/1536 budgets. Lowered capacities are marked; dynamic residency,
sustained shader frequency and gfx1150/gfx1152 performance were not measured.
Runtime routing remains gfx1151 only.

All 962 old instruction payloads (24,158,884 bytes) and ABI/resource metadata
remain identical. Five new scratch/spill0 instances match their selected
screening payloads; all 967 equal same-SDK assembly reassembly. LDS completion
waits immediately precede every new shared barrier. gfx11-generic final device
code equals the measured product; gfx1170 retains 941 old payloads and omits the
new fragment16 instances. gfx1151 compilation succeeds.

The 450-request route union changes exactly five requests and preserves 445.
Scope41 inherits Stage4a's 38 cells and the additional changed requests, five processes
and five accepted blocks per side on the same timed Y. Initial maximum new/old
is 1.014265; initial >2% suspects=0, separately
confirmed=0. The 2% regression gate passes with all
initial clean points retained. All six original dtype/mode COVERAGE2 leaves
pass 120 cases; ten original positional CLI calls check every new route in both
lookup and autotune modes. Test sources, Makefiles and LUTs are unchanged.

Evidence: AgentStation `handoffs/gemm_lag22_s4b_gfx1151/work/` (diagnose,
screen_report, finalists, formal, mechanisms, scope, occupancy and product/arch/
launch/UT/controls audits). The 22-cell table is a mixed-stage per-cell validated
history, not a new same-session vendor sweep. Merge prerequisites remain
gfx1150/gfx1152 matched product measurements and targeted LUT coverage review.
<!-- gemm_lag22_s4b_gfx1151:end -->

## 第二轮阶段 1：小 M 精确路由（2026-11-03）

本段仅处理第二轮同会话基线与阶段 1。阶段 2–4 和 LUT 更新留待后续任务。79 个冻结目标中，最终 26 格满足严格 hipBLASLt/new ≥1.00，本次改变 55 个精确路由。0.98–1.00 仍未达标；原 33 格基线中的8 个缺数据格在规定整对补测耗尽后仍不可用，因此任务总体状态为 FAIL。

`HIPDNN_EP_GEMM_LAGR2=0` 关闭本段；`HIPDNN_EP_GEMM_LAGR2_CFG=<route>` 只在下表对应的精确 shape 上启用指定路由。默认开。运行时只命中 gfx1151，要求 alpha=1、beta=0、C=null、A/B/Y 16 字节对齐；其他请求走原路由。已有族开关与强制 cfg/raster/stagger 控制保留优先权，新实例不进入 kWmma 或 LUT。

下表来自实际安装源码的 gfx11-generic -O2 产品 DLL。旧 P、新 A、固定完整 hipBLASLt 身份在同一组五进程中轮换；每边每进程五个采纳 REALTIME/CPU 块，共用物理计时输出和匹配 launch 数，取进程中位数的中位数。筛选数据另存，不用于拼接绝对耗时或替代五进程结果。

cell 编码为 dtype_M_N_K_ta_tb_bias；ta/tb=0/0 为 NN，0/1 为 NT，1/1 为 transA。

54/55 个改变路由的格子快于旧产品；f32_4_4096_4096_0_0_0 耗时增加 0.338%。完整数值保留如下。

|route|cell|template|旧产品 ms|新产品 ms|提升 %|耗时下降 %|HB ms|前 HB/P|后 HB/new|launch 数|
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
|0|bf16_128_11008_4096_0_0_0|gemm_wmma_nnlarge_kernel<bf16, 64, 64, 4, 2, 32, 0, 2, 0, 1>|0.472652701|0.453168851|4.299|4.122|0.440666552|0.932326|0.972411|174|
|1|bf16_128_4096_14336_0_0_0|gemm_wmma_nnlarge_kernel<bf16, 128, 128, 4, 2, 32, 0, 2, 1, 1>|0.721722564|0.709124444|1.777|1.746|0.646008205|0.895092|0.910994|117|
|2|bf16_16_14336_4096_0_1_0|gemm_wmma_skinny_kernel<bf16, 1, 1, 4, 4, 0, 1, 1, 16, 14336, 4096, 543>|0.612987109|0.598282656|2.458|2.399|0.586691094|0.957102|0.980625|128|
|3|bf16_16_4096_4096_0_1_0|gemm_wmma_smallroll_kernel<16, 32, 1, 2, bf16, 64, 1, 1281>|0.041685560|0.039997625|4.220|4.049|0.039379745|0.944686|0.984552|2000|
|4|bf16_24_11008_4096_0_0_0|gemm_wmma_nnlarge_kernel<bf16, 32, 64, 2, 1, 32, 0, 2, 0, 1>|0.400839067|0.388935803|3.060|2.970|0.393581710|0.981895|1.011945|193|
|5|bf16_24_4096_14336_0_0_0|gemm_wmma_nn_direct_r2_kernel<bf16, 24, 4096, 14336, 4, 2, 1>|0.537903846|0.517693974|3.904|3.757|0.504256795|0.937448|0.974044|156|
|6|bf16_32_11008_4096_0_0_0|gemm_wmma_nnlarge_kernel<bf16, 32, 64, 2, 1, 32, 0, 2, 0, 1>|0.402737409|0.391149067|2.963|2.877|0.395724560|0.982587|1.011698|193|
|7|bf16_32_4096_14336_0_0_0|gemm_wmma_nn_direct_r2_kernel<bf16, 32, 4096, 14336, 4, 2, 1>|0.542507417|0.522034702|3.922|3.774|0.510872781|0.941688|0.978618|151|
|8|bf16_48_11008_4096_0_0_0|gemm_wmma_nnlarge_kernel<bf16, 48, 96, 3, 3, 32, 1, 1, 1, 1>|0.444594579|0.403930684|10.067|9.146|0.398792526|0.896980|0.987280|190|
|9|bf16_48_14336_4096_0_0_0|gemm_wmma_nn_direct_r2_kernel<bf16, 48, 14336, 4096, 4, 1, 1, 0>|0.795433609|0.577956992|37.629|27.341|0.585260075|0.735775|1.012636|133|
|10|bf16_4_4096_4096_0_1_0|gemm_wmma_smallroll_kernel<16, 32, 1, 2, bf16, 64, 1, 1185>|0.038058240|0.037418480|1.710|1.681|0.037791000|0.992978|1.009956|2000|
|11|bf16_64_11008_4096_0_0_0|gemm_wmma_nnlarge_kernel<bf16, 64, 96, 4, 3, 32, 0, 1, 1, 1>|0.421634479|0.411931615|2.355|2.301|0.402474583|0.954558|0.977042|192|
|12|bf16_64_14336_4096_0_0_0|gemm_wmma_nn_direct_r2_kernel<bf16, 64, 14336, 4096, 4, 1, 1, 0>|0.734372366|0.593894122|23.654|19.129|0.596941832|0.812860|1.005132|131|
|13|bf16_8_14336_4096_0_1_0|gemm_wmma_skinny_kernel<bf16, 1, 1, 4, 4, 0, 1, 1, 8, 14336, 4096, 543>|0.605676875|0.591964922|2.316|2.264|0.579434844|0.956673|0.978833|128|
|14|bf16_8_4096_4096_0_1_0|gemm_wmma_smallroll_kernel<16, 32, 1, 2, bf16, 64, 1, 1185>|0.039682365|0.038383885|3.383|3.272|0.038596935|0.972647|1.005551|2000|
|15|bf16_96_11008_4096_0_0_0|gemm_wmma_nnlarge_kernel<bf16, 96, 64, 3, 2, 32, 0, 2, 1, 1>|0.446277898|0.438820341|1.699|1.671|0.441327045|0.988906|1.005712|176|
|16|bf16_96_14336_4096_0_0_0|gemm_wmma_nnlarge_kernel<bf16, 96, 96, 3, 3, 32, 0, 1, 1, 1>|0.676394228|0.655679187|3.159|3.063|0.626908211|0.926839|0.956120|123|
|17|bf16_96_3072_3072_1_1_0|gemm_wmma_ta6_r2_kernel<96, 96, 3, 3, 1, bf16, 32, 0, 1, 2, 1>|0.093984900|0.073546874|27.789|21.746|0.083474773|0.888172|1.134987|947|
|18|bf16_96_4096_4096_0_0_0|gemm_wmma_nnlarge_kernel<bf16, 96, 64, 3, 2, 32, 1, 2, 1, 1>|0.110000959|0.108261059|1.607|1.582|0.110012060|1.000101|1.016174|699|
|19|f16_128_11008_4096_0_0_0|gemm_wmma_nnlarge_kernel<half, 64, 64, 4, 2, 32, 0, 2, 0, 1>|0.471420395|0.452967458|4.074|3.914|0.439641299|0.932589|0.970580|177|
|20|f16_128_4096_14336_0_0_0|gemm_wmma_nnlarge_kernel<half, 128, 128, 4, 2, 32, 0, 2, 1, 1>|0.719844215|0.710054298|1.379|1.360|0.649450083|0.902209|0.914648|121|
|21|f16_16_11008_4096_0_0_0|gemm_wmma_kernel<16, 128, 1, 2, 0, half, 32, 0, 1, 1, 0>|0.398319901|0.383668762|3.819|3.678|0.385513465|0.967849|1.004808|202|
|22|f16_16_14336_4096_0_1_0|gemm_wmma_skinny_kernel<half, 1, 1, 4, 4, 0, 1, 1, 16, 14336, 4096, 543>|0.613564375|0.598111094|2.584|2.519|0.584938047|0.953344|0.977976|128|
|23|f16_24_11008_4096_0_0_0|gemm_wmma_nnlarge_kernel<half, 32, 64, 2, 1, 32, 0, 2, 0, 1>|0.401824205|0.388160821|3.520|3.400|0.393139538|0.978387|1.012826|195|
|24|f16_24_4096_14336_0_0_0|gemm_wmma_nn_direct_r2_kernel<half, 24, 4096, 14336, 4, 2, 1>|0.533289026|0.517228117|3.105|3.012|0.504057727|0.945187|0.974537|154|
|25|f16_32_4096_14336_0_0_0|gemm_wmma_nn_direct_r2_kernel<half, 32, 4096, 14336, 4, 2, 1>|0.539809200|0.520821000|3.646|3.518|0.511320667|0.947225|0.981759|150|
|26|f16_48_11008_4096_0_0_0|gemm_wmma_nnlarge_kernel<half, 48, 96, 3, 3, 32, 1, 1, 1, 1>|0.445848497|0.403009637|10.630|9.608|0.397813886|0.892262|0.987108|193|
|27|f16_48_14336_4096_0_0_0|gemm_wmma_nn_direct_r2_kernel<half, 48, 14336, 4096, 4, 1, 1, 0>|0.805069699|0.579435338|38.940|28.027|0.585023308|0.726674|1.009644|133|
|28|f16_4_4096_4096_0_1_0|gemm_wmma_smallroll_kernel<16, 32, 1, 2, half, 64, 1, 1185>|0.039130890|0.037597725|4.078|3.918|0.037613115|0.961213|1.000409|2000|
|29|f16_64_11008_4096_0_0_0|gemm_wmma_nnlarge_kernel<half, 64, 96, 4, 3, 32, 0, 1, 1, 1>|0.421567842|0.412536421|2.189|2.142|0.401672632|0.952807|0.973666|190|
|30|f16_64_14336_4096_0_0_0|gemm_wmma_nn_direct_r2_kernel<half, 64, 14336, 4096, 4, 2, 1, 0>|0.734964394|0.628566439|16.927|14.477|0.594802576|0.809294|0.946284|132|
|31|f16_64_4096_4096_0_0_0|gemm_wmma_nnlarge_kernel<half, 64, 64, 4, 2, 64, 0, 2, 1, 1>|0.088670189|0.083178158|6.603|6.194|0.089055083|1.004341|1.070655|901|
|32|f16_8_14336_4096_0_1_0|gemm_wmma_skinny_kernel<half, 1, 1, 4, 4, 0, 1, 1, 8, 14336, 4096, 543>|0.605364688|0.592737422|2.130|2.086|0.579118672|0.956644|0.977024|128|
|33|f16_8_4096_4096_0_1_0|gemm_wmma_smallroll_kernel<16, 32, 1, 2, half, 64, 1, 1281>|0.040397065|0.039224855|2.988|2.902|0.039533125|0.978614|1.007859|2000|
|34|f16_96_11008_4096_0_0_0|gemm_wmma_nnlarge_kernel<half, 64, 64, 4, 2, 32, 0, 2, 0, 1>|0.446616780|0.440090621|1.483|1.461|0.438336045|0.981459|0.996013|177|
|35|f16_96_14336_4096_0_0_0|gemm_wmma_nnlarge_kernel<half, 96, 96, 3, 3, 32, 0, 1, 1, 1>|0.677701905|0.647568016|4.653|4.446|0.626976984|0.925152|0.968203|126|
|36|f16_96_3072_3072_1_1_0|gemm_wmma_ta6_r2_kernel<96, 96, 3, 3, 1, half, 32, 0, 1, 2, 1>|0.095704511|0.074865142|27.836|21.775|0.083873764|0.876383|1.120331|951|
|37|f16_96_4096_4096_0_0_0|gemm_wmma_nnlarge_kernel<half, 96, 64, 3, 2, 32, 1, 2, 1, 1>|0.113243761|0.110945292|2.072|2.030|0.113735437|1.004342|1.025149|686|
|38|f32_12_14336_4096_0_0_0|gemm_f32_skinny_nn_kernel<12, 4, 2, 4, 4, 2>|1.128908169|1.011619859|11.594|10.390|1.079201831|0.955970|1.066806|71|
|39|f32_16_14336_4096_0_0_0|gemm_f32_skinny_nn_kernel<16, 4, 2, 4, 4, 2>|1.130471918|1.024973288|10.293|9.332|1.081387671|0.956581|1.055040|73|
|40|f32_2_4096_14336_0_0_0|gemm_gemv_kn_raw_r2_kernel<float, 2, 32, 4, 2, 2, 4096, 14336>|1.034552000|0.998749625|3.585|3.461|0.983400125|0.950556|0.984631|80|
|41|f32_2_4096_4096_0_0_0|gemm_gemv_kn_raw_r2_kernel<float, 2, 32, 4, 2, 2, 4096, 4096>|0.295594029|0.292076557|1.204|1.190|0.285075092|0.964414|0.976029|273|
|42|f32_32_4096_14336_0_0_0|gemm_f32_skinny_nn_kernel<32, 4, 1, 4, 4, 2>|1.080547067|1.048677333|3.039|2.949|1.041467867|0.963834|0.993125|75|
|43|f32_48_11008_4096_0_0_0|gemm_f32_tile_kernel<0, 64, 192, 16, 8, 12, 128>|0.852898247|0.781916082|9.078|8.322|0.811821443|0.951839|1.038246|97|
|44|f32_48_14336_4096_0_0_0|gemm_f32_tile_kernel<0, 64, 256, 16, 8, 8, 256>|1.322934531|1.150702656|14.968|13.019|1.257827500|0.950786|1.093095|64|
|45|f32_48_4096_4096_0_0_0|gemm_f32_tile_kernel<0, 32, 32, 16, 4, 4, 64>|0.359030135|0.357922117|0.310|0.309|0.353243423|0.983882|0.986928|222|
|46|f32_4_4096_4096_0_0_0|gemm_gemv_kn_raw_r2_kernel<float, 4, 32, 4, 2, 2, 4096, 4096>|0.295485370|0.296483148|-0.337|-0.338|0.286792296|0.970580|0.967314|270|
|47|f32_4_4096_4096_0_1_0|gemm_gemv_mr_nt_r2_kernel<float, 0, 4, 4, 256>|0.323000135|0.312690280|3.297|3.192|0.321268015|0.994637|1.027432|2000|
|48|f32_64_11008_4096_0_0_0|gemm_f32_tile_kernel<0, 64, 192, 16, 8, 12, 128>|0.817715773|0.784653918|4.214|4.043|0.813455876|0.994790|1.036707|97|
|49|f32_64_14336_4096_0_0_0|gemm_f32_tile_kernel<0, 64, 256, 16, 8, 8, 256>|1.370601406|1.159935156|18.162|15.370|1.271945156|0.928020|1.096566|64|
|50|f32_64_4096_14336_0_0_0|gemm_f32_tile_kernel<0, 32, 32, 16, 4, 4, 64>|1.301857188|1.259064219|3.399|3.287|1.234957969|0.948612|0.980854|64|
|51|f32_64_4096_4096_0_0_0|gemm_f32_tile_kernel<0, 32, 32, 16, 4, 4, 64>|0.376569636|0.356867955|5.521|5.232|0.355288955|0.943488|0.995575|220|
|52|f32_6_4096_4096_0_0_0|gemm_gemv_kn_raw_r2_kernel<float, 6, 32, 4, 2, 2, 4096, 4096>|0.296729665|0.289430260|2.522|2.460|0.291194052|0.981345|1.006094|269|
|53|f32_8_14336_4096_0_1_0|gemm_f32_skinny_kernel<8, 1, 8, 16, 32, 3>|1.096466250|1.039274063|5.503|5.216|1.090062578|0.994160|1.048869|128|
|54|f32_8_4096_4096_0_0_0|gemm_gemv_kn_raw_r2_kernel<float, 8, 32, 4, 2, 2, 4096, 4096>|0.296453577|0.288449818|2.775|2.700|0.284774453|0.960604|0.987258|274|

原有 967 个设备 payload 字节保持，metadata 与 descriptor 语义一致；新增选中实例 scratch=0、无 VGPR/SGPR spill，实际产品与筛选实例完全一致，DLL 嵌入对象和 同 SDK 汇编重组核对通过。450 请求实际路由并集只有表中精确目标变化；变化格复用产品正式配对结果，另测 10 个指定哨兵；65/65 通过。本次没有 >2% 疑点，因此没有启动独立确认进程。

1024/1536 VGPR 完整 WG 理论容量对比已记录，23 格至少在一种预算下容量降低。这些是资源上界，不是实测驻留；本次没有在其他 GPU 上测性能。gfx1170、gfx11-generic、gfx1151 单文件设备编译通过。

三个原 gemm leaf × lookup/autotune × COVERAGE=2 共 120 case 通过，另用原 CLI 检查 110 次未纳入 COVERAGE2 的精确 shape。原单测源码、Makefile、表/LUT 均保持。

静态循环、指令和资源记录只支持编译结构分析；本次没有 RGP，不能据此认定运行时 stall。half NT 固定维度 stag0 与原 StaggerU543 同时不同；保留543的独立对照用于区分这一混杂。FP32 32×32 tile 与原64×64 tile 的 BK 都为16，不能将收益解释为 K迭代/barrier频率减半。

独立 GEMM DLL 载体相对25c8974d增加 1171456 B；它不是完整 custom_kernels DLL。完整 DLL 实物基准不可用，没有全量构建，不能据此判定完整 DLL 的1 MiB CI规则。

完整逐格结果、未达标差距、静态/资源审计及限制：AgentStation `handoffs/gemm_lag_r2_gfx1151/work/`；任务完成报告在相邻 `RESULT.md`。

## perf(gemm): 优化 pre2048 精确路由 under gfx11-generic

pre2048-up FP16/BF16 的 generic -O2 原产品落后于固定 hipBLASLt，而 gfx1151 -O3 参照已过线；旧 generic 实例分配250/256 VGPR。Qwen-up FP16 候选虽然在产品02三格正式测量中达到1.001181，但未取得完整干净的原CLI lookup/autotune模式对，不能安装其精确路由。

仅pre2048两dtype改用独立固定维度128×128 WT2×4 BK64/PGR1/SCH7、256threads、SW−16/STAG543。两个精确NT请求在运行时gfx1151、alpha1/beta0/Cnull、16字节对齐条件下启用，尊重ALIASK/BUFLOAD和已有config/raster/stagger控制；新实例不进入kWmma，调度仅用builtin/属性。Qwen保留原路由。

halo13 / gfx1151；旧产品 `e0ec4c49d41d99faca6ff1c51b6c0bc774df1b03`，新产品为实际仓库单对象 `gfx11-generic -O2`。固定完整 hipBLASLt kernel+solution；各方每进程五个 REALTIME/CPU核验块中位数，再取五个进程中位数；相同launch数、hold、至少250ms负载settle和三个稳定shader转换，顺序轮换、公共物理Y。

|cell|旧产品 ms|新产品 ms|提升 %|耗时下降 %|HB ms|HB/旧|HB/新|launch数|严格达标|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
|f16_2048_14336_4096_0_1_0|6.282102187|5.806484063|8.191|7.571|6.145960312|0.978329|1.058465|32|PASS|
|bf16_2048_14336_4096_0_1_0|6.279929062|5.797501875|8.321|7.682|6.114063125|0.973588|1.054603|32|PASS|
|f16_2048_18944_3584_0_1_0|6.706945313|6.711685313|-0.071|-0.071|6.551860625|0.976877|0.976187|32|FAIL|

### 有效吞吐与名义峰值

|cell|B MiB|新/HB有效B GB/s|新/HB TFLOPS|新/名义算力峰值 %|
|---|---:|---|---|---:|
|f16_2048_14336_4096_0_1_0|112.000|20.226/19.109|41.422/39.134|69.73|
|bf16_2048_14336_4096_0_1_0|112.000|20.257/19.208|41.487/39.339|69.84|
|f16_2048_18944_3584_0_1_0|129.500|20.232/20.726|41.435/42.446|69.76|

有效B率是有用权重字节除以时间，未测实际DRAM事务；峰值取2.9GHz名义half59.4/FP3229.7TFLOPS，持续频率未测。

### 资源与设备代码

|cell / side|threads|used/descriptor VGPR|LDS B|1024 physical / WG/WGP / waves/SIMD|1536 physical / WG/WGP / waves/SIMD|
|---|---:|---|---:|---|---|
|f16_2048_14336_4096_0_1_0 / 旧|256|250/256|36864|256/2/4.0|264/2/4.0|
|f16_2048_14336_4096_0_1_0 / 新|256|159/160|36864|160/3/6.0|168/3/6.0|
|bf16_2048_14336_4096_0_1_0 / 旧|256|250/256|36864|256/2/4.0|264/2/4.0|
|bf16_2048_14336_4096_0_1_0 / 新|256|159/160|36864|160/3/6.0|168/3/6.0|

原1015个payload、metadata与descriptor语义不变，全部同SDK assembly/reassembly相等，实际DLL嵌入精确对象。选中实例与筛选的payload/metadata/descriptor完整相等，scratch/spill为0。

本阶段新增2个实例。单GEMM对象DLL相对`25c8974d`增加1203200字节，相对`e0ec4c49`增加31744字节；相对本阶段上一个产品增加31744字节。

VGPR/LDS是实际编译资源；occupancy为完整WG理论上界，不能当动态驻留。新路由只在运行时gfx1151启用，不进入kWmma。gfx11-generic/gfx1151/gfx1170设备编译全部通过；generic独立设备对象与实际产品相等，既有gfx1170设备代码保持一致。

### 不退化、路由与单测

450请求路由并集仅改变2个精确目标，保留448个请求。8个host-only完整并集控制通过。

|scope cell|new/old|2%不退化|
|---|---:|---|
|f16_2048_14336_4096_0_1_0|0.934422|PASS|
|bf16_2048_14336_4096_0_1_0|0.923656|PASS|
|bf16_2048_18944_3584_0_1_0|1.000793|PASS|
|f16_2048_4096_4096_0_0_0|1.000497|PASS|
|bf16_2048_4096_4096_0_0_0|1.001022|PASS|
|f16_128_4096_4096_0_1_0|0.996731|PASS|
|bf16_128_4096_4096_0_1_0|1.001460|PASS|
|f32_128_4096_14336_0_1_0|1.003146|PASS|

原三dtype×lookup/autotune、COVERAGE2共120case通过；缺少的目标另有4次采纳的原CLI准确性和实际路由检查通过。

|UT完整模式对|采纳进程|保留的无效/污染进程|
|---|---|---|
|llm_product_03_ut_Xfp32_Wfp32_Yfp32|llm_product_03_ut_Xfp32_Wfp32_Yfp32_lookup, llm_product_03_ut_Xfp32_Wfp32_Yfp32_autotune|无|
|llm_product_03_ut_Xfp16_Wfp16_Yfp16|llm_product_03_ut_Xfp16_Wfp16_Yfp16_lookup, llm_product_03_ut_Xfp16_Wfp16_Yfp16_autotune|无|
|llm_product_03_ut_Xbf16_Wbf16_Ybf16|llm_product_03_ut_Xbf16_Wbf16_Ybf16_lookup, llm_product_03_ut_Xbf16_Wbf16_Ybf16_autotune|无|
|llm_product_03_ut_route_000|llm_product_03_ut_route_000_lookup, llm_product_03_ut_route_000_autotune|无|
|llm_product_03_ut_route_001|llm_product_03_ut_route_001_lookup, llm_product_03_ut_route_001_autotune|无|

无效或污染的一方使整个lookup/autotune模式对作废；仅该完整模式对按原声明最多补测三轮。原始用例、日志和clean进程均保留，UT耗时不作为性能结论。

本阶段新增指令字节19732，设备实例总数1015→1017；单对象DLL不含全产品其他kernel。

### 实例与选择依据

- `f16_2048_14336_4096_0_1_0`：`gemm_wmma_lagr2b_llm_kernel<128, 128, 2, 4, half, 64, 0, 1, 7, 2048, 14336, 4096, -16, 543>`；grid=112,16,1，block=256,1,1；来源llm_screen_01/llm_screen_01pre cfg1。
- `bf16_2048_14336_4096_0_1_0`：`gemm_wmma_lagr2b_llm_kernel<128, 128, 2, 4, bf16, 64, 0, 1, 7, 2048, 14336, 4096, -16, 543>`；grid=112,16,1，block=256,1,1；来源llm_screen_01/llm_screen_01pre cfg1。

### 开关、限制与后续

`HIPDNN_EP_GEMM_LAGR2B_LLM=0`关闭本阶段；`HIPDNN_EP_GEMM_LAGR2B_LLM_CFG`只限制已测精确请求。已有config pin和显式raster/stagger保留优先级。

pre首轮PGR1比可复用RP18快超过1%，两个独立实例有完整五进程依据。本表来自重新编译并验证的实际较小产品03；产品02和其三格formal、scope、controls、devices及全部原始/补测记录均保留。Qwen的原始轮和最多三次整对补测共八次准确性检查全部通过，但四轮都至少一方含PID4 System Copy≥2%样本（2.01295、2.48832、2.05428、2.60231%），完整模式对0/4，因而撤下尚未提交的Qwen候选；没有第四次补测、跨轮拼接或未经占用响应证明的Copy豁免。该候选的1.001181仅属未安装版本证据，Qwen最终比值以本次未改路由的同会话正式表为准。静态ISA位置、VGPR/LDS和理论驻留不能证明实际stall/DRAM事务；NN RGP仍在后续阶段执行。产品01 native stderr失败和所有clean loss/tie均保留。

实际FLOP/耗时是有效计算率；没有把有用B字节率当成DRAM事务。1024/1536驻留表不是其他设备实测。完整custom_kernels DLL相对main大小、gfx1150/gfx1152实测、统一补LUT和模型端到端验证仍属合入前工作。

阶段严格HB门槛：FAIL；未达标：f16_2048_18944_3584_0_1_0。其他提交检查均通过。

## perf(gemm): improve exact vision and MiniLM routes under gfx11-generic

阶段3的七个视觉/MiniLM plain NT 请求在继承基线中落后固定完整hipBLASLt身份。MiniLM短K固定成本和tile资源、SigLIP的N tile数量，以及ViT-down的流水/几何需要分别处理；未把静态指令或有效字节率当作运行时stall。

只在runtime gfx1151安装七条已测精确路由。MiniLM-up half使用固定128x80/BK64单LDS，两dtype共用结构；SigLIP-up BF16使用128x96/BK32单LDS和270WG。FP32 MiniLM-up使用128x64/BK16普通FMA，down使用64x96/BK16、WG内两个128线程K组和LDS归约。ViT FP16使用独立四wave mid11特化，BF16使用128x256/BK32双LDS mid05特化。所有新实例独立于旧模板及kWmma，保留已有显式控制优先级。

halo13 / gfx1151；旧产品 `c5d9afec8163026b9188a10682f5c93a56f53d6b`，新产品为实际仓库单对象 `gfx11-generic -O2`。固定完整 hipBLASLt kernel+solution；各方每进程五个 REALTIME/CPU核验块中位数，再取五个进程中位数；相同launch数、hold、至少250ms负载settle和三个稳定shader转换，顺序轮换、公共物理Y。

|cell|旧产品 ms|新产品 ms|提升 %|耗时下降 %|HB ms|HB/旧|HB/新|launch数|严格达标|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
|bf16_1576_768_3072_0_1_0|0.179254688|0.173539063|3.294|3.189|0.172200938|0.960650|0.992289|32|FAIL|
|bf16_512_1536_384_0_1_0|0.019342010|0.018733415|3.249|3.146|0.019700510|1.018535|1.051624|2000|PASS|
|bf16_729_4304_1152_0_1_0|0.198112344|0.186514531|6.218|5.854|0.194766562|0.983112|1.044243|64|PASS|
|f16_1576_768_3072_0_1_0|0.183058438|0.173186875|5.700|5.393|0.176302812|0.963096|1.017992|32|PASS|
|f16_512_1536_384_0_1_0|0.019794740|0.018858600|4.964|4.729|0.019812060|1.000875|1.050558|2000|PASS|
|f32_512_1536_384_0_1_0|0.076440825|0.058818915|29.960|23.053|0.072328080|0.946197|1.229674|2000|PASS|
|f32_512_384_1536_0_1_0|0.089881500|0.080126695|12.174|10.853|0.081167735|0.903053|1.012992|2000|PASS|

### 有效吞吐与名义峰值

|cell|B MiB|新/HB有效B GB/s|新/HB TFLOPS|新/名义算力峰值 %|
|---|---:|---|---|---:|
|bf16_1576_768_3072_0_1_0|4.500|27.190/27.402|42.852/43.185|72.14|
|bf16_512_1536_384_0_1_0|1.125|62.970/59.879|32.241/30.658|54.28|
|bf16_729_4304_1152_0_1_0|9.457|53.167/50.914|38.759/37.117|65.25|
|f16_1576_768_3072_0_1_0|4.500|27.246/26.764|42.939/42.180|72.29|
|f16_512_1536_384_0_1_0|1.125|62.552/59.542|32.027/30.485|53.92|
|f32_512_1536_384_0_1_0|2.250|40.111/32.619|10.268/8.351|34.57|
|f32_512_384_1536_0_1_0|2.250|29.445/29.067|7.538/7.441|25.38|

有效B率是有用权重字节除以时间，未测实际DRAM事务；峰值取2.9GHz名义half59.4/FP3229.7TFLOPS，持续频率未测。

### 资源与设备代码

|cell / side|threads|used/descriptor VGPR|LDS B|1024 physical / WG/WGP / waves/SIMD|1536 physical / WG/WGP / waves/SIMD|
|---|---:|---|---:|---|---|
|f16_512_1536_384_0_1_0 / 旧|128|241/248|29952|256/4/4.0|264/4/4.0|
|f16_512_1536_384_0_1_0 / 新|128|241/248|29952|256/4/4.0|264/4/4.0|
|bf16_512_1536_384_0_1_0 / 旧|128|241/248|29952|256/4/4.0|264/4/4.0|
|bf16_512_1536_384_0_1_0 / 新|128|241/248|29952|256/4/4.0|264/4/4.0|
|bf16_729_4304_1152_0_1_0 / 旧|128|254/256|40960|256/3/3.0|264/3/3.0|
|bf16_729_4304_1152_0_1_0 / 新|128|247/248|14336|256/4/4.0|264/5/5.0|
|f32_512_1536_384_0_1_0 / 旧|128|173/200|51200|208/2/2.0|216/2/2.0|
|f32_512_1536_384_0_1_0 / 新|128|129/168|25600|176/5/5.0|168/5/5.0|
|f32_512_384_1536_0_1_0 / 旧|128|81/136|17408|144/7/7.0|144/7/7.0|
|f32_512_384_1536_0_1_0 / 新|256|111/200|49152|208/2/4.0|216/2/4.0|
|f16_1576_768_3072_0_1_0 / 旧|512|146/152|55296|160/1/4.0|168/2/8.0|
|f16_1576_768_3072_0_1_0 / 新|128|254/256|32768|256/4/4.0|264/4/4.0|
|bf16_1576_768_3072_0_1_0 / 旧|512|146/152|55296|160/1/4.0|168/2/8.0|
|bf16_1576_768_3072_0_1_0 / 新|256|254/256|49152|256/2/4.0|264/2/4.0|

原1017个payload、metadata与descriptor语义不变，全部同SDK assembly/reassembly相等，实际DLL嵌入精确对象。选中实例与筛选的payload/metadata/descriptor完整相等，scratch/spill为0。

本阶段新增7个实例。单GEMM对象DLL相对`25c8974d`增加1350144字节，相对`e0ec4c49`增加178688字节；相对本阶段上一个产品增加146944字节。

VGPR/LDS是实际编译资源；occupancy为完整WG理论上界，不能当动态驻留。新路由只在运行时gfx1151启用，不进入kWmma。gfx11-generic/gfx1151/gfx1170设备编译全部通过；generic独立设备对象与实际产品相等，既有gfx1170设备代码保持一致。

### 不退化、路由与单测

450请求路由并集仅改变7个精确目标，保留443个请求。10个host-only完整并集控制通过。

|scope cell|new/old|2%不退化|
|---|---:|---|
|bf16_1576_3072_768_0_1_0|1.000819|PASS|
|bf16_1576_3072_768_0_1_1|0.997497|PASS|
|bf16_1576_768_3072_0_1_0|0.977615|PASS|
|bf16_512_1536_384_0_1_0|0.959874|PASS|
|bf16_512_384_1536_0_1_0|1.001100|PASS|
|bf16_729_1152_4304_0_1_0|1.011618|PASS|
|bf16_729_4304_1152_0_1_0|0.939095|PASS|
|f16_1576_3072_768_0_1_0|0.992687|PASS|
|f16_1576_3072_768_0_1_1|0.998711|PASS|
|f16_1576_768_3072_0_1_0|0.953884|PASS|
|f16_512_1536_384_0_1_0|0.971398|PASS|
|f16_512_384_1536_0_1_0|1.003422|PASS|
|f16_729_1152_4304_0_1_0|0.996305|PASS|
|f16_729_4304_1152_0_1_0|1.004964|PASS|
|f32_1576_3072_768_0_1_1|0.995954|PASS|
|f32_512_1536_384_0_1_0|0.772407|PASS|
|f32_512_384_1536_0_1_0|0.906278|PASS|

原三dtype×lookup/autotune、COVERAGE2共120case通过；缺少的目标另有14次采纳的原CLI准确性和实际路由检查通过。

|UT完整模式对|采纳进程|无效进程|外部GPU活动（保留记录）|
|---|---|---|---|
|mid_product_01_ut_Xfp32_Wfp32_Yfp32|mid_product_01_ut_Xfp32_Wfp32_Yfp32_lookup, mid_product_01_ut_Xfp32_Wfp32_Yfp32_autotune|无|无|
|mid_product_01_ut_Xfp16_Wfp16_Yfp16|mid_product_01_ut_Xfp16_Wfp16_Yfp16_lookup, mid_product_01_ut_Xfp16_Wfp16_Yfp16_autotune|无|无|
|mid_product_01_ut_Xbf16_Wbf16_Ybf16|mid_product_01_ut_Xbf16_Wbf16_Ybf16_lookup, mid_product_01_ut_Xbf16_Wbf16_Ybf16_autotune|无|无|
|mid_product_01_ut_route_000|mid_product_01_ut_route_000_lookup, mid_product_01_ut_route_000_autotune|无|无|
|mid_product_01_ut_route_001|mid_product_01_ut_route_001_lookup, mid_product_01_ut_route_001_autotune|无|无|
|mid_product_01_ut_route_002|mid_product_01_ut_route_002_lookup, mid_product_01_ut_route_002_autotune|无|无|
|mid_product_01_ut_route_003|mid_product_01_ut_route_003_lookup, mid_product_01_ut_route_003_autotune|无|无|
|mid_product_01_ut_route_004|mid_product_01_ut_route_004_lookup, mid_product_01_ut_route_004_autotune|无|无|
|mid_product_01_ut_route_005|mid_product_01_ut_route_005_lookup, mid_product_01_ut_route_005_autotune|无|无|
|mid_product_01_ut_route_006|mid_product_01_ut_route_006_lookup, mid_product_01_ut_route_006_autotune|无|无|

按2026-11-05用户规则，正确性单测以正常退出码、原用例结果和精确路由判定；外部GPU活动完整记录，不因此作废或补测。计时过程仍按原污染策略严格审计。原始用例与全部日志均保留，UT耗时不作为性能结论。

本阶段新增指令字节119916，设备实例总数1017→1024；单对象DLL不含全产品其他kernel。

### 实例与选择依据

- `f16_512_1536_384_0_1_0`：`gemm_lagr2b_mid_half_kernel<half, 512, 1536, 384, 128, 80, 2, 5, 64, 0, 2, 8, 0, 8, 0>`；grid=20,4,1，block=128,1,1；来源mid_screen_01a/mid_screen_measure_01a cfg0。
- `bf16_512_1536_384_0_1_0`：`gemm_lagr2b_mid_half_kernel<bf16, 512, 1536, 384, 128, 80, 2, 5, 64, 0, 2, 8, 0, 8, 0>`；grid=20,4,1，block=128,1,1；来源mid_screen_01a/mid_screen_measure_01a cfg0。
- `bf16_729_4304_1152_0_1_0`：`gemm_lagr2b_mid_half_kernel<bf16, 729, 4304, 1152, 128, 96, 4, 3, 32, 0, 2, 0, 1, 8, 0>`；grid=45,6,1，block=128,1,1；来源mid_screen_01a/mid_screen_measure_01a cfg2。
- `f32_512_1536_384_0_1_0`：`gemm_lagr2b_mid_f32_kernel<512, 0, 0, 1, 128, 64, 16, 8, 8, 128, 0, 0>`；grid=24,4,1，block=128,1,1；来源mid_screen_01a/mid_screen_measure_01a cfg2。
- `f32_512_384_1536_0_1_0`：`gemm_lagr2b_mid02a_f32_kernel<512, 384, 1536, 1, 64, 96, 16, 4, 12, 128, 0, 0, 2, 1>`；grid=4,8,1，block=256,1,1；来源mid_screen_02a/mid_screen_measure_02a_c012 cfg1。
- `f16_1576_768_3072_0_1_0`：`gemm_lagr2b_mid11_vit_kernel<half, 0, 3>`；grid=8,10,1，block=128,1,1；来源mid_screen_11/mid_screen_measure_11_c02 cfg0。
- `bf16_1576_768_3072_0_1_0`：`gemm_lagr2b_mid05_vit_kernel<bf16, 1576, 768, 3072, 128, 256, 4, 4, 32, 1, 1055>`；grid=3,13,1，block=256,1,1；来源mid_screen_05/mid_screen_measure_05_c01 cfg0。

### 开关、限制与后续

`HIPDNN_EP_GEMM_LAGR2B_MID=0`关闭本阶段；`HIPDNN_EP_GEMM_LAGR2B_MID_CFG`只限制已测精确请求。已有config pin和显式raster/stagger保留优先级。

正式七格6/7达到HB/new≥1.00，BF16 ViT-down为0.992289200，仍差0.771080%；它相对同轮旧产品耗时下降3.18855%，因此按时间盒规则保留已证实改进并如实记FAIL。十五批筛选的75个原clean进程、全部loss/tie与83个观察保留，不为跨线重采clean结果。阶段3原始六COVERAGE2和九CLI结果在用户排队后复用，只补五个未启动CLI；正确性进程保留外部GPU遥测、判定只看原用例/退出码/路由，所有计时仍严格按污染规则。继续Qwen精确补装、NN代表RGP和优化、PID4b来源诊断、大B优化与最终完整回归。

实际FLOP/耗时是有效计算率；没有把有用B字节率当成DRAM事务。1024/1536驻留表不是其他设备实测。完整custom_kernels DLL相对main大小、gfx1150/gfx1152实测、统一补LUT和模型端到端验证仍属合入前工作。

阶段严格HB门槛：FAIL；未达标：bf16_1576_768_3072_0_1_0。其他提交检查均通过。

## Qwen FP16精确路由补装

Qwen FP16精确路由此前只因正确性进程的System Copy活动而撤回；原四轮八次CLI正确性全通过。用户更新正确性规则并明确要求补装。

在完成MID的实际本地提交上恢复产品02的原包装实例、精确谓词、route2和OVERLAP开关；不改此前的设备载荷和其余请求。

当前父提交`e95c0f2c3f73d9bf866865aa74f404f3a11c2317`，实际产品`gfx11-generic -O2`。以下计时来自用户指定的产品02正式五进程结果，未重跑计时。每方五个REALTIME/CPU核验块取进程中位数，再取五进程中位数；完整固定HB身份、相同launch数、hold与负载settle。

|格子|历史旧产品 ms|已复用的新产品 ms|HB ms|提升 %|耗时下降 %|HB/旧|HB/新|launch数|严格≥1|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
|f16_2048_18944_3584_0_1_0|6.707922812|6.543164062|6.550889062|2.518|2.456|0.976589810|1.001180621|32|PASS|

进程中位范围：旧[6.624660625, 6.7507665625]、新[6.4630371875, 6.5708334375]、HB[6.503530625, 6.58149875] ms。本轮没有把补装构建、单测或RGP耗时当成性能证据。

新450请求host路由证明只改变该格，其他449格完整模板/grid/block/shared/uniform/resources签名保持。Qwen old/new签名分别与产品02正式对的old/new相等，满足复用条件；没有额外变化需要补测。产品02scope与controls的原审计完整保留。

新增1实例、8204指令字节；used/descriptor VGPR 159/160，LDS 36864 B，scratch/spill0。原1024个载荷/metadata/descriptor保持，全部同SDK重组相等，DLL嵌入精确对象；新Qwen与产品02机器码/metadata/descriptor完整相等。

原实例`gemm_wmma_overlap_kernel<256, 128, 4, 2, half, 32, 0, 7>` → 新实例`gemm_wmma_lagr2b_qwen_kernel<128, 128, 2, 4, half, 64, 0, 1, 7, 2048, 18944, 3584, 8, 0>`；原/new grid 148,8,1/148,16,1，block 512,1,1/256,1,1。旧普通路径的资源和调度在固定Qwen格被独立固定尺寸包装实例替代，使用既有编译器builtin调度控制；新结构的具体作用以实际ISA为证据，不把名义occupancy当动态驻留。

|side / VGPR模型|used/descriptor VGPR|threads|LDS B|physical VGPR|理论WG/WGP|理论waves/SIMD|
|---|---|---:|---:|---:|---:|---:|
|旧/1024|130/136|512|30720|144|1|4.0|
|旧/1536|130/136|512|30720|144|2|8.0|
|新/1024|159/160|256|36864|160|3|6.0|
|新/1536|159/160|256|36864|168|3|6.0|

新/HB有效计算率 42.502/42.452 TFLOPS（新/59.4名义峰值 71.55%）；有用B字节率 20.753/20.729 GB/s，不是实际DRAM事务。持续shader频率未测，occupancy为理论上界。

单GEMM对象DLL相对25c8974d增加1361408 B，相对e0ec4c49增加189952 B，相对父产品增加11264 B。完整custom_kernels DLL相对main仍需合入前检查。

gfx11-generic、gfx1151、gfx1170三个设备编译通过；generic独立设备对象与实际产品相等，既有gfx1170载荷保持。原三个leaf×lookup/autotune COVERAGE2共120case新单测通过。只复用原Qwen八次CLI的正确性，不重复已有测量。

|复用的原CLI过程|正常退出/正确性|外部GPU≥2%样本数|max util %|
|---|---|---:|---:|
|llm_product_02_ut_route_002_lookup|PASS|1|2.01295|
|llm_product_02_ut_route_002_autotune|PASS|0|0.00000|
|llm_product_02_ut_route_002_lookup_r1|PASS|0|0.00000|
|llm_product_02_ut_route_002_autotune_r1|PASS|1|2.48832|
|llm_product_02_ut_route_002_lookup_r2|PASS|0|0.00000|
|llm_product_02_ut_route_002_autotune_r2|PASS|1|2.05428|
|llm_product_02_ut_route_002_lookup_r3|PASS|1|2.60231|
|llm_product_02_ut_route_002_autotune_r3|PASS|0|0.00000|

以上Copy记录与原严格UT撤回审计仍保存。用户2026-11-05更新后，正确性-only过程只按退出码和用例结果判定；外部活动继续记录。所有计时过程的污染策略保持。

|最终COVERAGE2模式对|采纳过程|外部GPU≥2%样本数|
|---|---|---:|
|llm_product_04_ut_Xfp32_Wfp32_Yfp32|llm_product_04_ut_Xfp32_Wfp32_Yfp32_lookup, llm_product_04_ut_Xfp32_Wfp32_Yfp32_autotune|0|
|llm_product_04_ut_Xfp16_Wfp16_Yfp16|llm_product_04_ut_Xfp16_Wfp16_Yfp16_lookup, llm_product_04_ut_Xfp16_Wfp16_Yfp16_autotune|0|
|llm_product_04_ut_Xbf16_Wbf16_Ybf16|llm_product_04_ut_Xbf16_Wbf16_Ybf16_lookup, llm_product_04_ut_Xbf16_Wbf16_Ybf16_autotune|0|

`HIPDNN_EP_GEMM_LAGR2B_LLM=0`关闭本阶段；`_CFG=2`为补装Qwen精确路由。原OVERLAP=0、旧config/raster/stagger控制继续优先。只在运行时gfx1151、alpha1/beta0/Cnull/aligned16条件下启用，不进kWmma。

复用用户指定的历史正式性能/scope/controls与原CLI正确性证据。新450路由证明与产品02的Qwen old/new完整签名相等，机器码/metadata/descriptor相等；新六个COVERAGE2过程通过。补装不占NN优化时间盒。最终完整回归仍按任务顺序执行。

## perf(gemm): retain measured full-M NN gains under gfx11-generic

NN 4096x14336 FP16/BF16 remains below the fixed hipBLASLt reference. The closed M16/M48/M128 profiles locate global waits and address arithmetic, with M128 additionally showing ISA barriers. These profiles explain candidate directions; they do not measure physical DRAM reads or establish isolated causes of a speedup.

Add six exact runtime-gfx1151 routes from closed NN05/NN06 evidence. M48/M64 use a full-M N32 single-K-wave direct global-read kernel with legal 64-bit B reads and no workgroup barrier in the main loop. M96 uses a full-M N64 kernel with shared transposed B, direct A, one-chunk prefetch and KC32. The M96 prefetch choice is within 1% of the measured deeper-prefetch alternative and uses fewer instruction bytes and registers. Keep all preceding payloads and descriptors unchanged, instantiate additions outside kWmma, and preserve the SKINNY/NNMID switches and existing forced controls.

halo13 / gfx1151；旧产品 `e7e44e2df7360088eb7b1bcbeebed57223802a08`，新产品为实际仓库单对象 `gfx11-generic -O2`。固定完整 hipBLASLt kernel+solution；各方每进程五个 REALTIME/CPU核验块中位数，再取五个进程中位数；相同launch数、hold、至少250ms负载settle和三个稳定shader转换，顺序轮换、公共物理Y。

|cell|旧产品 ms|新产品 ms|提升 %|耗时下降 %|HB ms|HB/旧|HB/新|launch数|严格达标|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
|bf16_128_4096_14336_0_0_0|0.708326068|0.707685556|0.091|0.090|0.644611026|0.910048|0.910872|117|FAIL|
|bf16_16_4096_14336_0_0_0|0.521946346|0.521049936|0.172|0.172|0.497910641|0.953950|0.955591|156|FAIL|
|bf16_24_4096_14336_0_0_0|0.518313141|0.517993269|0.062|0.062|0.504711282|0.973757|0.974359|156|FAIL|
|bf16_32_4096_14336_0_0_0|0.521795629|0.521616821|0.034|0.034|0.510505563|0.978363|0.978698|151|FAIL|
|bf16_48_4096_14336_0_0_0|0.575070685|0.558031712|3.053|2.963|0.538925548|0.937147|0.965762|146|FAIL|
|bf16_64_4096_14336_0_0_0|0.582111560|0.565892057|2.866|2.786|0.550233688|0.945238|0.972330|141|FAIL|
|bf16_96_4096_14336_0_0_0|0.663128915|0.649874884|2.039|1.999|0.602130775|0.908015|0.926533|129|FAIL|
|f16_128_4096_14336_0_0_0|0.707043140|0.707314545|-0.038|-0.038|0.646974793|0.915043|0.914692|121|FAIL|
|f16_16_4096_14336_0_0_0|0.522205256|0.521682179|0.100|0.100|0.498153846|0.953943|0.954899|156|FAIL|
|f16_24_4096_14336_0_0_0|0.518049221|0.518510519|-0.089|-0.089|0.504931558|0.974679|0.973812|154|FAIL|
|f16_32_4096_14336_0_0_0|0.521739133|0.521984200|-0.047|-0.047|0.510671867|0.978788|0.978328|150|FAIL|
|f16_48_4096_14336_0_0_0|0.574528819|0.557054583|3.137|3.041|0.538400486|0.937117|0.966513|144|FAIL|
|f16_64_4096_14336_0_0_0|0.581994043|0.564409645|3.116|3.021|0.549447447|0.944077|0.973491|141|FAIL|
|f16_96_4096_14336_0_0_0|0.662417364|0.649620698|1.970|1.932|0.607472713|0.917054|0.935119|129|FAIL|

### 有效吞吐与名义峰值

|cell|B MiB|新/HB有效B GB/s|新/HB TFLOPS|新/名义算力峰值 %|
|---|---:|---|---|---:|
|bf16_128_4096_14336_0_0_0|112.000|165.950/182.188|21.242/23.320|35.76|
|bf16_16_4096_14336_0_0_0|112.000|225.392/235.867|3.606/3.774|6.07|
|bf16_24_4096_14336_0_0_0|112.000|226.722/232.689|5.441/5.585|9.16|
|bf16_32_4096_14336_0_0_0|112.000|225.147/230.047|7.205/7.362|12.13|
|bf16_48_4096_14336_0_0_0|112.000|210.455/217.916|10.102/10.460|17.01|
|bf16_64_4096_14336_0_0_0|112.000|207.532/213.438|13.282/13.660|22.36|
|bf16_96_4096_14336_0_0_0|112.000|180.712/195.042|17.348/18.724|29.21|
|f16_128_4096_14336_0_0_0|112.000|166.037/181.523|21.253/23.235|35.78|
|f16_16_4096_14336_0_0_0|112.000|225.119/235.751|3.602/3.772|6.06|
|f16_24_4096_14336_0_0_0|112.000|226.496/232.587|5.436/5.582|9.15|
|f16_32_4096_14336_0_0_0|112.000|224.989/229.973|7.200/7.359|12.12|
|f16_48_4096_14336_0_0_0|112.000|210.824/218.129|10.120/10.470|17.04|
|f16_64_4096_14336_0_0_0|112.000|208.077/213.743|13.317/13.680|22.42|
|f16_96_4096_14336_0_0_0|112.000|180.783/193.326|17.355/18.559|29.22|

有效B率是有用权重字节除以时间，未测实际DRAM事务；峰值取2.9GHz名义half59.4/FP3229.7TFLOPS，持续频率未测。

### 资源与设备代码

|cell / side|threads|used/descriptor VGPR|LDS B|1024 physical / WG/WGP / waves/SIMD|1536 physical / WG/WGP / waves/SIMD|
|---|---:|---|---:|---|---|
|bf16_48_4096_14336_0_0_0 / 旧|64|198/200|16400|208/7/3.5|216/7/3.5|
|bf16_48_4096_14336_0_0_0 / 新|32|130/136|1536|144/16/4.0|144/16/4.0|
|bf16_64_4096_14336_0_0_0 / 旧|64|198/200|16400|208/7/3.5|216/7/3.5|
|bf16_64_4096_14336_0_0_0 / 新|32|162/168|1536|176/16/4.0|168/16/4.0|
|bf16_96_4096_14336_0_0_0 / 旧|192|108/112|10240|112/4/6.0|120/6/9.0|
|bf16_96_4096_14336_0_0_0 / 新|192|109/112|10240|112/4/6.0|120/6/9.0|
|f16_48_4096_14336_0_0_0 / 旧|64|198/200|16400|208/7/3.5|216/7/3.5|
|f16_48_4096_14336_0_0_0 / 新|32|130/136|1536|144/16/4.0|144/16/4.0|
|f16_64_4096_14336_0_0_0 / 旧|64|198/200|16400|208/7/3.5|216/7/3.5|
|f16_64_4096_14336_0_0_0 / 新|32|162/168|1536|176/16/4.0|168/16/4.0|
|f16_96_4096_14336_0_0_0 / 旧|192|108/112|10240|112/4/6.0|120/6/9.0|
|f16_96_4096_14336_0_0_0 / 新|192|109/112|10240|112/4/6.0|120/6/9.0|

原1025个payload、metadata与descriptor语义不变，全部同SDK assembly/reassembly相等，实际DLL嵌入精确对象。选中实例与筛选的payload/metadata/descriptor完整相等，scratch/spill为0。

本阶段新增6个实例。单GEMM对象DLL相对`25c8974d`增加1411072字节，相对`e0ec4c49`增加239616字节；相对本阶段上一个产品增加49664字节。

VGPR/LDS是实际编译资源；occupancy为完整WG理论上界，不能当动态驻留。新路由只在运行时gfx1151启用，不进入kWmma。gfx11-generic/gfx1151/gfx1170设备编译全部通过；generic独立设备对象与实际产品相等，既有gfx1170设备代码保持一致。

### 不退化、路由与单测

450请求路由并集仅改变6个精确目标，保留444个请求。8个host-only完整并集控制通过。

|scope cell|new/old|2%不退化|
|---|---:|---|
|bf16_24_11008_4096_0_0_0|1.000528|PASS|
|bf16_32_11008_4096_0_0_0|1.002332|PASS|
|bf16_48_14336_4096_0_0_0|0.999340|PASS|
|bf16_48_4096_14336_0_0_0|0.936543|PASS|
|bf16_64_14336_4096_0_0_0|1.001285|PASS|
|bf16_64_4096_14336_0_0_0|0.945458|PASS|
|bf16_96_11008_4096_0_0_0|1.000406|PASS|
|bf16_96_4096_14336_0_0_0|0.982607|PASS|
|bf16_96_4096_4096_0_0_0|1.001703|PASS|
|f16_16_11008_4096_0_0_0|1.000513|PASS|
|f16_24_11008_4096_0_0_0|0.998000|PASS|
|f16_48_14336_4096_0_0_0|0.996914|PASS|
|f16_48_4096_14336_0_0_0|0.937326|PASS|
|f16_64_4096_14336_0_0_0|0.945373|PASS|
|f16_64_4096_4096_0_0_0|1.000334|PASS|
|f16_96_4096_14336_0_0_0|0.984707|PASS|
|f16_96_4096_4096_0_0_0|0.995205|PASS|

原三dtype×lookup/autotune、COVERAGE2共120case通过；缺少的目标另有12次采纳的原CLI准确性和实际路由检查通过。

|UT完整模式对|采纳进程|无效进程|外部GPU活动（保留记录）|
|---|---|---|---|
|nn_product_01_ut_Xfp32_Wfp32_Yfp32|nn_product_01_ut_Xfp32_Wfp32_Yfp32_lookup, nn_product_01_ut_Xfp32_Wfp32_Yfp32_autotune|无|无|
|nn_product_01_ut_Xfp16_Wfp16_Yfp16|nn_product_01_ut_Xfp16_Wfp16_Yfp16_lookup, nn_product_01_ut_Xfp16_Wfp16_Yfp16_autotune|无|无|
|nn_product_01_ut_Xbf16_Wbf16_Ybf16|nn_product_01_ut_Xbf16_Wbf16_Ybf16_lookup, nn_product_01_ut_Xbf16_Wbf16_Ybf16_autotune|无|无|
|nn_product_01_ut_route_000|nn_product_01_ut_route_000_lookup, nn_product_01_ut_route_000_autotune|无|无|
|nn_product_01_ut_route_001|nn_product_01_ut_route_001_lookup, nn_product_01_ut_route_001_autotune|无|无|
|nn_product_01_ut_route_002|nn_product_01_ut_route_002_lookup, nn_product_01_ut_route_002_autotune|无|无|
|nn_product_01_ut_route_003|nn_product_01_ut_route_003_lookup, nn_product_01_ut_route_003_autotune|无|无|
|nn_product_01_ut_route_004|nn_product_01_ut_route_004_lookup, nn_product_01_ut_route_004_autotune|无|无|
|nn_product_01_ut_route_005|nn_product_01_ut_route_005_lookup, nn_product_01_ut_route_005_autotune|无|无|

按2026-11-05用户规则，正确性单测以正常退出码、原用例结果和精确路由判定；外部GPU活动完整记录，不因此作废或补测。计时过程仍按原污染策略严格审计。原始用例与全部日志均保留，UT耗时不作为性能结论。

本阶段新增指令字节29140，设备实例总数1025→1031；单对象DLL不含全产品其他kernel。

### 实例与选择依据

- `bf16_48_4096_14336_0_0_0`：`gemm_wmma_nn_direct_r2_kernel<bf16, 48, 4096, 14336, 3, 2, 1, 1, 2, 0, 3, 2, 0, 6, 1>`；grid=128,1,1，block=32,1,1；来源nn_screen_06/nn_screen_measure_06_c012 cfg2。
- `bf16_64_4096_14336_0_0_0`：`gemm_wmma_nn_direct_r2_kernel<bf16, 64, 4096, 14336, 4, 2, 1, 1, 2, 0, 3, 2, 0, 6, 1>`；grid=128,1,1，block=32,1,1；来源nn_screen_06/nn_screen_measure_06_c012 cfg2。
- `bf16_96_4096_14336_0_0_0`：`gemm_wmma_nn_direct_r2_kernel<bf16, 96, 4096, 14336, 96, 1, 4, 6, 32, 1, 1, 0, 8>`；grid=64,1,1，block=192,1,1；来源nn_screen_05/nn_screen_measure_05_c012 cfg0。
- `f16_48_4096_14336_0_0_0`：`gemm_wmma_nn_direct_r2_kernel<half, 48, 4096, 14336, 3, 2, 1, 1, 2, 0, 3, 2, 0, 6, 1>`；grid=128,1,1，block=32,1,1；来源nn_screen_06/nn_screen_measure_06_c012 cfg2。
- `f16_64_4096_14336_0_0_0`：`gemm_wmma_nn_direct_r2_kernel<half, 64, 4096, 14336, 4, 2, 1, 1, 2, 0, 3, 2, 0, 6, 1>`；grid=128,1,1，block=32,1,1；来源nn_screen_06/nn_screen_measure_06_c012 cfg2。
- `f16_96_4096_14336_0_0_0`：`gemm_wmma_nn_direct_r2_kernel<half, 96, 4096, 14336, 96, 1, 4, 6, 32, 1, 1, 0, 8>`；grid=64,1,1，block=192,1,1；来源nn_screen_05/nn_screen_measure_05_c012 cfg0。

### 开关、限制与后续

`HIPDNN_EP_GEMM_LAGR2B_NN=0`关闭本阶段；`HIPDNN_EP_GEMM_LAGR2B_NN_CFG`只限制已测精确请求。已有config pin和显式raster/stagger保留优先级。

These are supported partial gains, not strict hipBLASLt passes. Per the user close instruction, only routes whose actual-product five-process old/new exceeds 1.01 and whose scope and original unit tests pass may be retained. All 14 primary NN rows remain in formal reporting. No screen is declared after NN06; the unbuilt NN09 declaration is withdrawn and the 24 secondary targets are deferred. Clean losses and ties remain preserved. Physical DRAM traffic and repeat-capture stability were not established; instruction-trace busy values are sample counts. Narrow-N KW1 accumulation changes floating-point order, so the original FP16/BF16 precision thresholds remain mandatory. The recorded occupancy is a compiler-resource upper bound, not dynamic occupancy. Broader device measurements, full-DLL size relative to main, a unified LUT update and end-to-end model validation remain pre-merge work.

实际FLOP/耗时是有效计算率；没有把有用B字节率当成DRAM事务。1024/1536驻留表不是其他设备实测。完整custom_kernels DLL相对main大小、gfx1150/gfx1152实测、统一补LUT和模型端到端验证仍属合入前工作。

阶段严格HB门槛：FAIL；未达标：bf16_128_4096_14336_0_0_0, bf16_16_4096_14336_0_0_0, bf16_24_4096_14336_0_0_0, bf16_32_4096_14336_0_0_0, bf16_48_4096_14336_0_0_0, bf16_64_4096_14336_0_0_0, bf16_96_4096_14336_0_0_0, f16_128_4096_14336_0_0_0, f16_16_4096_14336_0_0_0, f16_24_4096_14336_0_0_0, f16_32_4096_14336_0_0_0, f16_48_4096_14336_0_0_0, f16_64_4096_14336_0_0_0, f16_96_4096_14336_0_0_0。其他提交检查均通过。


## Exact BF16 LM head partial gain (2026-11-07)

The BF16 NT 2048x128256x8192 request uses one independent fixed-dimension
128x128 WT2x4 BK64 single-LDS PGR1/SCH7 instance under gfx11-generic.
The prior buffer/RP2 instance already issued VMEM ahead of selected WMMA steps;
the measurements compare complete dimension, pipeline, scheduling and lifetime
choices and do not isolate one cause of the gain. The selected trial01/cfg1
instance uses SW=-8/STAG543, grid 1002x16 and 256 threads.

| Shape | Old product ms | New product ms | old/new | Time reduction | hipBLASLt ms | HB/old | HB/new | Launches/block |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BF16 NT 2048x128256x8192 |107.973043333|102.896226667|1.049339192|4.701929768%|100.924810000|0.934722287|0.980840729|3|

This is a 4.933919% partial improvement; the strict HB/new>=1.00 target
remains FAIL (1.915927% below the target). These five-process actual-product
results are reused from the exact archived candidate under the explicitly
authorized per-key r2b Copy source policy. Newly launched sentinel timing
uses the r2c PID4 System Copy window rule: >=2% inside any attempted gate's
release/completion interval with one monitor sampling interval at each
endpoint rejects its complete opposite-order unit; outside Copy is recorded.
Other external GPU engines/processes retain whole-process rejection.

The route requires runtime gfx1151, 16-half WMMA fragments, plain NT,
alpha1/beta0/Cnull and 16-byte aligned A/B/Y. HIPDNN_EP_GEMM_LAGR2B_BIGB=0 disables
it; HIPDNN_EP_GEMM_LAGR2B_BIGB_CFG=0 restricts the measured route (the archived
screen called this trial01/cfg1). Existing forced geometry/raster/stagger
controls take priority; ALIASK=0 or BUFLOAD=0 falls through. The instance stays
outside kWmma and makes no LUT or device allocation change.

Used/descriptor VGPR 250/256 becomes 159/160; LDS remains 36864 bytes and
scratch/spill is 0. Static bounds on the compiled budget of 1024 VGPRs improve
physical allocation 256->160, groups/WGP 2->3 and waves/SIMD 4->6. With a 1536
budget and allocation granularity 24, physical allocation 264->168 gives the same
group/wave bounds. These are theoretical resource bounds; captured runtime
occupancy remains one group/MP. Existing payloads, metadata and descriptor
semantics are unchanged.

The measured single-GEMM product DLL grows 1,428,480 bytes (1.362 MiB) versus
feat/custom-gemm. Full custom_kernels DLL size versus main, gfx1150/gfx1152
measurements, unified LUT and model end-to-end validation remain before merge.

Validation for this installation:

| Original sentinel | New/old |
|---|---:|
| bf16_2048_14336_4096_0_1_0 | 0.993726635 |
| bf16_2048_4096_4096_0_0_0 | 1.006447950 |
| f16_2048_14336_4096_0_1_0 | 1.003959812 |
| f16_2048_18944_3584_0_1_0 | 1.000630224 |
| f16_2048_4096_4096_0_0_0 | 1.012862196 |

All five accepted opposite-order processes passed every original sentinel (maximum new/old=1.012862196). Eight full-union host controls each cover 450 requests: only CFG0 changes the exact BF16 head; disabled/invalid/legacy/ALIASK/BUFLOAD controls preserve the parent route. Fresh gfx11-generic device bytes match the measured candidate, and fresh gfx1170/gfx1151 builds preserve every old payload, metadata entry and descriptor semantic. All six original leaf COVERAGE 2 calls (120 CPU cases) and both original BF16 head CLI lookup/autotune calls pass with natural exits. External observations are recorded for correctness calls and do not reject them.

Before adding this head route, the parent d3a7fdbe passed the complete 105-request comparison against 0697a193: maximum new/old=1.012473572, no >2% suspects. This is parent regression evidence plus a qualified one-route addition, with its own unchanged-device/route and sentinel evidence.
