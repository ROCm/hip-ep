<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# PR #259 Split Plan

Plan for breaking the WIP mega-PR **#259** (`release/msbuild` → `main`, 153
files vs `main`) into small, reviewable PRs.

## Context / decisions already made

- **PR #284** (`feat/runtime-custom-kernels-split`) already split out the
  **runtime + custom-kernel** slice (32 files: `lib/Runtime/real/*`,
  `lib/Runtime/mock/*`, `3rd-party/custom_kernels/*`, `hipdnn_ep_runtime.h`,
  `runtime_tensor.cpp`). All op-lowering PRs that emit `wrap_*` calls base on
  PR #284 because the runtime ABI lives there.
- **PRs #260–#265 are MERGED to main** — a full HIP-dialect shape-inference
  cascade (`--hip-infer-shapes` + per-op `inferReturnTypes` / `reify`).
  `main`'s pipeline runs `hip::createInferShapesPass()` first thing in
  `buildOnnxToHipPipelineTail`. This **supersedes** PR #259's ONNX-level
  forward shape inference.
- **Output-shape / DimSource is NOT split out.** The entire mechanism
  (compute scheme + metadata recording + runtime marshal) will be replaced by
  a new reify-based approach in a separate PR landing directly on `main`.
  Keep `main`'s current DimSource behaviour untouched. Verified zero coupling:
  DimSource consumers are a closed cluster (`InferOnnxShapes.cpp`,
  `pass_main.cpp`, `MlirCompiler.*`, `MlirCustomOp.cpp`, `compiler_api/types.h`,
  morphizen ort-bridge) — no op conversion / lowering / pass reads it.

## EXCLUDED — not ported (stays on main / future PR / obsolete)

| File / item | Reason |
|---|---|
| `lib/Conversion/OnnxToHip/InferOnnxShapes.cpp` (forward + `traceDimOrigin`) | forward → superseded by #260–#265; backward trace → superseded by new output-shape PR |
| `lib/Conversion/OnnxToHip/RefineLoopBodyTypes.cpp` | replaced by PR #265 (HIP `InferTypeOpInterface` on `Hip_LoopOp`) |
| DimSource cluster: `metadata.proto` DimSource, `pass_main.cpp` build, `MlirCompiler.cpp/.h` C-ABI getters, `compiler_api.h`, `compiler_types.h`, `CompilerDriver.*` output-shape stash, `lib/CInterface/CompilerAPI.cpp`, `MlirCustomOp.cpp` resolution | whole output-shape mechanism replaced by new scheme PR on main |
| `docs/design/dynamic-shapes.md` | documents the obsolete ONNX-level scheme |
| lit: `test_infer_onnx_shapes.mlir`, `test_infer_onnx_shapes_trace_mult.mlir`, `test_refine_loop_body_types.mlir`, `test_expand_shape_trace.mlir` | tests for the excluded passes |
| `test_qwen_vision_patch_merger_dynshape` (in `test_qwen3_5_9b.py`) | exercises DimSource `mult=0.25` — deferred to new output-shape PR |

## PR breakdown

Two roots: **group A** bases on **PR #284** (op conversions + HIP lowerings,
runtime already in #284); **group B** bases on **main** (compiler passes;
Loop + host-scalar carry their own new runtime). **Group C** (vision) is the
convergence of A + B. PR-5 is a self-contained correctness fold set, submitted
LAST.

### Group A — ONNX op conversions + HIP lowerings (base: PR #284)

#### PR-1 — elementwise / binary / util ops  ★ most-base of group A
Introduces the shared `OnnxToHipUtils.h` helpers (`createBroadcastEmptyTensor`,
`getInlineScalarFromOnnxConstant`, `materializeScalarFromDenseAttr`) that PR-2 /
PR-3 / PR-9 depend on.
- Conversions: `AndConversion`, `DivConversion`, `EqualConversion`,
  `LessConversion`, `MinConversion`, `ModConversion`, `ElementwiseConversion`,
  `ShapeConversion`
- Lowerings: `DivLowering`, `ElementwiseLowering`, `EqualLowering`
- Shared (partial slice): `HipOps.td`, `OnnxToHip.cpp` (registration lines),
  `OnnxToHipUtils.h` (the 3 helpers), `HipToLLVM.cpp`, `HipToLLVMUtils.h`,
  `HipDialect.cpp`, `OnnxToHip/CMakeLists.txt`, `HipToLLVM/CMakeLists.txt`
- lit: `onnx-to-hip/test_add`, `test_div`, `test_sub`; `hip-to-llvm/test_div`,
  `test_sub`, `test_equal`
- deps: PR #284

#### PR-2 — shape-producing / data-movement ops  (deps: PR-1)
- Conversions: `ConstantOfShapeConversion`, `ExpandConversion`,
  `RangeConversion`, `TileConversion`, `WhereConversion`
- Pipeline (for ConstantOfShape/Where → linalg): `InitAllPasses.h`
  (linalg dialect + arith buffer-dealloc registration), `Pipelines.cpp`
  (`ConvertLinalgToLoops` / `SCFToControlFlow` / `ReconcileUnrealizedCasts`)
- lit: `onnx-to-hip/test_split`, `test_split_invalid` (if Split touched here)
- deps: PR-1 (uses `createBroadcastEmptyTensor` + scalar helpers)

#### PR-3 — indexing / embedding-merge ops  (deps: PR-1)
- Conversions: `NonZeroConversion`, `ScatterNDConversion`
- Lowerings: `NonZeroLowering`, `ScatterNDLowering`, `GatherLowering`
- lit: `onnx-to-hip/test_nonzero`, `test_scatter_nd`,
  `test_nonzero_scatter_backtrace`; `hip-to-llvm/test_nonzero`,
  `test_scatter_nd`, `test_gather`
- deps: PR-1

#### PR-4 — LpNormalization  (independent — base PR #284, no PR-1 dep)
- `LpNormalizationConversion` + `populateLpNormalizationConversionPatterns`
- lit: `onnx-to-hip/test_lpnormalization`, `e2e/test_lpnormalization_model`
- deps: PR #284 only

### Group B — compiler passes / structural (base: main)

#### PR-6 — onnx.Loop support  (carries own runtime)
- `LoopOutline` (OnnxToHip), `LoopLowering` (HipToLLVM)
- runtime: `hipdnn_ep_runtime_loop.cpp` (new), `runtime_state.cpp` nested-pool
  slice, `runtime_state_internal.h` slice, `hipdnn_ep_runtime.h` loop decls
- `HipOps.td` loop op (if not already in main)
- deps: main
- ⚠️ do NOT port `RefineLoopBodyTypes` (PR #265 covers it); verify the
  outlined-body type refinement works against main's PR #265 mechanism

#### PR-7 — host-scalar SEGV (MaterializeHostScalars)  (carries own runtime)
- `lib/Dialect/Transforms/MaterializeHostScalars.cpp`
- `HipOps.td` `get_host_scratch` op (partial); `MemoryLowering.cpp`
  `GetHostScratchOpLowering` (partial — shares file with PR-8)
- runtime: `runtime_state.cpp` host_scratch slice, `runtime_state_internal.h`
  slice, `hipdnn_ep_runtime.h` decls
- `Passes.td` (partial), `Pipelines.cpp` (partial), `Dialect/Transforms/CMakeLists.txt`
- lit: `Dialect/hip-materialize-host-scalars.mlir`
- deps: main

### Group C — vision / projector (convergence of A + B)

#### PR-8 — vision accuracy  (deps: PR-6 + PR #284)
- `SimplifyOnnx.cpp` (PrecisionFreeCast), `SliceConversion.cpp` (empty-extent)
- `lib/Dialect/Transforms/FixLoopAccumulatorOffset.cpp` (needs Loop = PR-6)
- `MemoryLowering.cpp` memref-copy stride verification (partial — shares file
  with PR-7); runtime `memrefCopy` strides already in PR #284 (`hip.cpp`)
- `Pipelines.cpp` FixLoopAccum wiring (partial), `Passes.td` (partial)
- lit: `onnx-to-hip/test_simplify_onnx_precision_free_cast`, `test_slice`;
  `hip-to-llvm/test_memref_copy`
- doc: `docs/design/qwen-vision-ops.md`
- deps: PR-6, PR #284

#### PR-9 — vision / projector compute ops  (deps: PR-1)
- `FastGeluFusion.cpp` (exact-Gelu Erf + vision variants),
  `ProjectorOpsRewrites.cpp` (Conv→Gemm + AveragePool/Pow/ReduceMean — keep all),
  `OnnxResultTypeInference.cpp/.h` (now only consumed by these vision rewrites)
- Lowerings: `ConvLowering.cpp` (dtype+bias), `MatmulLowering.cpp` (b_batched)
- lit: `onnx-to-hip/test_projector_rewrites`
- deps: PR-1 (OnnxToHipUtils helpers)

### Group D — last

#### PR-5 — dynamic reshape / memref support  ★ submit LAST (self-contained)
No other PR's code calls these — pure pass/fold set. Submit last; re-diff vs
260–265 main first (HIP reify may have made parts redundant).
- `ReshapeConversion.cpp` (same-rank dynamic branch), `ReshapeShapeFold.cpp`
- `lib/Dialect/Transforms/RelaxMultiDynExpandShape.cpp`, `PoolAllocs.cpp`
  (transitive hoist)
- `Pipelines.cpp` relax-multi-dyn wiring (partial), `Passes.td` (partial)
- lit: `onnx-to-hip/test_reshape`, `Dialect/hip-relax-multi-dyn-expand-shape`
- deps: main
- ⚠️ **MUST re-diff vs 260–265 main before sizing** — may shrink substantially

#### PR-10 — build / CI / docs / tests / tools  (deps: all)
- `build.py`, `environment.yml`, `.github/workflows/*`, `README.md`,
  `CLAUDE.md`, `3rd-party/morphizen` (submodule bump),
  `msvc_hip_cmath_workaround.h`, `tools/hip-mlir-opt/*`, `tools/hip-onnx-runner.cpp`
- tests: `test_qwen3_5_9b.py` (minus the excluded patch-merger dynshape test),
  `test_gemma3_4b.py`, `conftest.py`, `images/tower.jpg`, `test/qwen3_5_9b/*`,
  `test/numeric/*`
- vision end-to-end accuracy is gated on the new output-shape PR

## Dependency DAG

```
                  ┌─ PR-2 (shape/data ops)
PR #284 ─► PR-1 ──┼─ PR-3 (embedding ops)
           │      └─ PR-9 (vision compute) ─┐
           └─ PR-4 (LpNorm, independent)    │
                                            ├─► PR-10 (build/docs/tests)
main ─► PR-6 (loop) ──► PR-8 (vision acc) ──┘
     └─ PR-7 (host-scalar)
     └─ PR-5 (dyn reshape/memref) ──────────── submit LAST
```

## merge order

1. Parallel leaves: **PR-1**, **PR-4**, **PR-6**, **PR-7**
2. On PR-1: **PR-2**, **PR-3**, **PR-9**
3. Convergence: **PR-8** (needs PR-6 + PR #284)
4. **PR-10** (build/docs/tests)
5. **PR-5** last (after the 260–265 re-diff)
