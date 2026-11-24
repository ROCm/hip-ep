# hip_gemm Xfp16_Wfp16_Yfp32

This leaf compiles and runs only `Xfp16_Wfp16_Yfp32`. Its self-contained
`test_gemm.cpp` uses a leaf-local fixed `hip_gemm` dtype code and host/device
storage: fp16=0, bf16=3, fp32=1. It has no runtime dtype selector.

Timing uses [`unit_test/common`](../../common/README.md) adaptive complete-call
policy (`HIPDNN_UT_TIMING=screen|standard|confirm`). The default `standard`
races a 1,000-call goal against one second of GPU time under the shared policy; pass `TIMING=confirm` for
formal performance numbers or `TIMING=screen` for short LUT ranking.

```
make test HIP_SDK=<HIP SDK root> OFFLOAD=--offload-arch=gfx11-generic MODE=lookup COVERAGE=1 TIMING=screen
make test_custom HIP_SDK=<HIP SDK root> M=32 N=512 K=512 TA=0 TB=1
make clean
```

`MODE=lookup` is the default. Coverage tiers thin the typical-shape list the
same way as the other unit_test leaves. Results go to `out/results.csv` with
`UT_TIMING` JSON evidence on stdout.

Build native ranking with `make build RANK=1 BUILD_DIR=out_rank HIP_SDK=<sdk>`.
`HIPDNN_UT_RANK=<csv>` enables default + supplied configs;
`HIPDNN_UT_RANK_CATALOG=1` enumerates production descriptors, and
`HIPDNN_UT_RANK_REFERENCE=1` runs independent sequential confirmation.
The CSV header is `leaf,shape,priority,config`; shapes use
`M16_N512_K256_ta0_tb1`, with optional `_bias` or
`_bias_crows16_ccols1`. `default` is required; catalog IDs or exact table
geometry strings select candidates. Ordinary tests keep their usual coverage
and CSV output. Shared timing checks every deduplicated candidate against the
CPU reference before screen/standard/confirm ranking. Optional
`ADLX_INCLUDE`, `GPU_PHASE_HEADER` and `HIPDNN_UT_GPU_ADMISSION` follow the
[common UT contract](../../common/README.md).

Use the GEMM owning [JSON table refresh commands](../../../hip/autotune/gemm/README.md#refresh-through-native-ut-ranking)
for `rank-seed --from-lut` through `rank-build` and private FB compilation.
`RANK_KERNEL_OBJ` accepts an existing adapter object only for a matching
source/include/compiler/flag closure. Production DLL builds expose no ranking
controls.

The single-case interface is `--m M --n N --k K --ta 0|1 --tb 0|1`,
with defaults `M=128 N=512 K=256 TA=0 TB=1`. Supplying any shape or
transpose flag selects one custom case; dimensions must be positive. The
legacy positional `M N K TA TB` form remains supported. `make test_custom`
uses the same named flags and defaults. [`ut_interface.json`](../../ut_interface.json)
records the parameters, constraints and runnable examples.

Caller-owned batched workspace is allocated before correctness and timing.
The measured entry is `hip_gemm_strided_batched`, including split reduction.
`--batch`, `--a-group`, `--b-group` describe logical batches and consecutive
batches sharing one physical matrix. Element strides `--stride-a`,
`--stride-b`, `--stride-c`, `--stride-y` accept -1 for tight storage or 0 for
input/C broadcast; output batches must not overlap. `--alpha`, `--beta`,
`--crows`, `--ccols` expose the complete epilogue; CCOLS=-1 selects N.
K=0 applies beta*C. Every tier crosses transpose, C broadcast, tails,
zero/padded strides, shared inputs and workspace/output guards.
The full CPU reference uses the exact represented input/C values.
RANK=1 accepts `_B<batch>_ag<ag>_bg<bg>` and optional `_sa..._sb..._sc..._sy...`
shape fields. Explicit-stride diagnostics do not add tight-layout LUT points.
