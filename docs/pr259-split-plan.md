<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# PR #259 Split Plan

Plan for breaking the WIP mega-PR **#259** (`release/msbuild` → `main`, 153
files vs `main`) into small, reviewable PRs.

## Status (updated 2026-06-08)

### Progress snapshot — PRs cut so far

| PR | What | Base | PR # | State |
|---|---|---|---|---|
| PR-5  | RelaxMultiDynExpandShape + ReshapeShapeFold | `main` | **#303** | pushed, build+lit green |
| PR-3a | `GatherLowering` 9→11 arg ABI | `#284` (`feat/runtime-custom-kernels-split`) | **#304** | pushed, build+lit green |
| PR-3b | NonZero/ScatterND `count_buf`/`valid_count` GPU-count | `#284` | **#305** | pushed, build+lit 100% |

`#304` and `#305` are **parallel siblings on #284** (disjoint files → either merge
order). Earlier `#305` was mistakenly stacked on `#304`; rebased onto `#284`
2026-06-08 so the two are independent.

**Remaining to cut:** PR-1/#299 (div/sub/equal lowerings — verify pushed), PR-9
(conv/matmul lowerings + vision rewrites), PR-4 (LpNorm, off PR-9), PR-8 (vision
accuracy, off PR-1), PR-10 (build/tests/docs, last).

- **PR #284** (`feat/runtime-custom-kernels-split`) and **PR-1 / #299**
  (`pr259-split/01-elementwise-ops`) are both already based on the current
  `syncorigin/main` (verified: `syncorigin/main` is an ancestor of both) —
  **no rebase needed**.
- **KEY FINDING — `syncorigin/main` has moved well past #259's base.** The
  #260–#265 shape-inference cascade landed a large set of op conversions +
  lowerings + op defs (and much of the runtime) on `main` already. So several
  "group A" op-conversion PRs no longer have net-new conversions — the op
  bodies are on `main`, and the only residual #259 delta is
  accuracy/feature fixes. **Every remaining op PR MUST be re-diffed vs
  `syncorigin/main` before cutting** (same supersession effect the plan
  already flagged for PR-5).
- **PR-2 / PR-6 / PR-7 are DISSOLVED** — their conversions / passes / runtime
  are already on `syncorigin/main` (PR-2 shape ops; PR-6 Loop = `LoopOutline` +
  `LoopLowering` + `hipdnn_ep_runtime_loop.cpp`; PR-7 `MaterializeHostScalars`).
  PR-2's residual accuracy fixes move to **PR-8**.

### FINALIZED decisions (2026-06-08)

- **Q1 ABI coupling (IMPORTANT).** PR #284 changed the parameter signatures of
  **8 `wrap_*` runtime functions** (`wrap_div`, `wrap_elementwise_sub`,
  `wrap_equal`, `wrap_gather`, `wrap_nonzero`, `wrap_scatter_nd`,
  `wrap_miopenConvolutionForward`, `wrap_hipblasLtMatmul`). **`main`'s current
  lowerings still emit the OLD argument lists** (verified: `ConvLowering` emits
  23 args, runtime wants 24; `DivLowering` emits 6, runtime wants 17). C linkage
  matches by symbol name only — the arg-count mismatch is **silent runtime UB**,
  no compile/link error.   The matching lowering-half updates are split across
  **PR-1 (#299)** (div/sub/equal — verified new 17-arg `DivLowering`),
  **PR-3a (#304) + PR-3b (#305)** (gather / nonzero / scatter_nd lowerings),
  **PR-9** (conv/matmul lowerings — verified #259 `ConvLowering` is 24-arg).
  Therefore PR #284 and these lowering PRs are a **correctness-coupled set**.
- **Merge strategy = MERGE TRAIN (parallel siblings on #284).** PR #284 must NOT
  land on `main` before its lowering halves. The lowering PRs (PR-1, PR-3a,
  PR-3b, PR-9) are **all cut off #284 as parallel siblings** (disjoint op files →
  any internal merge order). `main` is not built/validated until the whole set
  lands. See "Merge train" below.
- **PR-3 is REINSTATED as a required lowering-half PR** (NOT folded into PR-10),
  and **SPLIT into 3a/3b** (both DONE). Its conversions are already on `main`;
  PR-3 ports only the LOWERINGS to PR #284's new signatures:
  - **PR-3a (#304)** — `GatherLowering` 9→11 arg. Self-contained, no op-def
    change. Base #284. **DONE.**
  - **PR-3b (#305)** — `NonZeroLowering` / `ScatterNDLowering` + the `count_buf`
    2nd-DPS-out / `valid_count` operand op-def surgery + backtrace conversion +
    `test_nonzero_scatter_backtrace.mlir`. Base #284, parallel to #304. **DONE.**
- **PR-4 and PR-5 are NOT combined** — separate PRs. But **PR-4 is NOT
  standalone after all**: its LpNorm decomposition emits a broadcasting
  `onnx.Div`, which #259 handles via `BroadcastDivToMulReciprocal` — a pattern
  defined+registered in `ProjectorOpsRewrites.cpp` (= PR-9, net-new, not on
  main). main's `DivLowering`/`wrap_div` is the old non-broadcasting path, so a
  broadcasting Div is broken on bare main. **Decision: PR-4 stays a separate PR
  but DEPENDS ON PR-9** (cut off PR-9, merge after it). Only **PR-5** is truly
  straight-to-main independent (verified: no PR-1 helper, no `wrap_`, dialect
  passes only).

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

#### PR-2 — shape-producing / data-movement ops  — DISSOLVED (already on main)
**Verified 2026-06-08 against `syncorigin/main`:** the ConstantOfShape / Expand
/ Range / Tile / Where conversions AND their lowerings AND op defs are ALREADY
on `syncorigin/main` (landed via #260–#265), and the runtime (`wrap_expand` /
`wrap_tile`) is on main too. PR-2 as originally conceived (introduce these
conversions) has no net-new content.

The ONLY remaining #259 delta in these files is a set of **shape-op
correctness / Qwen-vision accuracy fixes** (commit `1c438063`
"fix(qwen-vision): three accuracy fixes"), **moved to PR-8** for triage. They
are NOT new conversions and do NOT base on PR #284 / PR-1 — base is
`syncorigin/main`. Each must be ported onto main's inferred-type op API
(drop the explicit-resultType `Op::create`):
  - `ConstantOfShape`: `ConstantOfShapeAsScalar` (benefit=3) fold for
    ConstantOfShape→Where + `buildRank0ScalarAttr` — fixes HIP-719 device
    fault on multimodal mask-fill.
  - `Expand`: shape-vector traceback (pool-allocs dominance) +
    `max(input_dim, shape_value)` ONNX-correct sizing — fixes Expand silently
    shrinking inputs.
  - `Range`: rank-0 / rank-1-size-1 operand acceptance + `extractScalar` /
    `peekScalarProducer` (keeps the Range count free of `memref.load` for pool
    hoist). Needs `getInlineScalarFromOnnxConstant` /
    `materializeScalarFromDenseAttr` added to `OnnxToHipUtils.h`.
  - `Tile`: init size = `input.dim[i] * repeats[i]` — fixes tile under-size.
  - `Where`: SKIP — `syncorigin/main` is already functionally identical.
  - Pipeline: `Pipelines.cpp` linalg/scf passes (`ConvertLinalgToLoops` /
    `SCFToControlFlow` / `ReconcileUnrealizedCasts`) are NOT on main and are
    needed by the ConstantOfShape linalg.fill path.

#### PR-3 — gather / nonzero / scatter_nd LOWERING half  (deps: PR #284) — **SPLIT 3a/3b**
> **CORRECTION (verified against #284 head):** NOT just "3 lowering files".
> `count_buf` makes `hip.nonzero` 2-result and `valid_count` adds an operand to
> `hip.scatter_nd` — and those op defs are now `Hip_DpsOp` with the
> reify/InferType infra from **#260–#265 (post-#259)**. Wholesale-porting the IR
> files (`HipOps.td`/`HipDialect.cpp`/`HipReifyResultShapesImpl.cpp`/
> `HipShapeUtils.cpp`) is **+244/−1822** and would DELETE that infra. So PR-3 is
> split:
> - **PR-3a — Gather lowering ABI fix = PR #304 (base #284).** Self-contained:
>   `GatherLowering` 9→11 arg (`axis_size`,`inner_size` from the data operand).
>   No op-def/conversion change. `hip-to-llvm/test_gather` → 11 params. Built +
>   lit green. **DONE.**
> - **PR-3b — NonZero/ScatterND GPU-count = PR #305 (base #284, PARALLEL to
>   #304 — disjoint files, either merge order).** The surgical half, completed
>   without deleting #260–#265 infra:
>   - `hip.nonzero`: 2nd DPS out `count_buf:memref<1xi32>`. `getDpsInitsMutable`
>     spans `{y,count_buf}` (shared `HipDpsOp::reifyResultShapes` reifies both);
>     `autoInfer=0` + hand-written 2-result `inferReturnTypes`. 9-arg lowering
>     (`count_ptr` + stack `input_dims`).
>   - `hip.scatter_nd`: `valid_count` ins + `has_valid_count` attr.
>     `ScatterNDConversion` backtraces `indices` → upstream `hip.nonzero`
>     `count_buf` (defers on unconverted `onnx.NonZero`). 16-arg lowering
>     (`count_ptr` = valid_count, or null). Operand-only → auto-infer untouched.
>   - lit: hip-to-llvm + onnx-to-hip `test_nonzero`/`test_scatter_nd` updated,
>     new `NonZero→Transpose→ScatterND` backtrace regression, `hip-infer-shapes`
>     2-out. ctest 100%. **DONE.**

#### PR-4 — LpNormalization  (deps: PR-9 — base on PR-9)
- `LpNormalizationConversion` + `populateLpNormalizationConversionPatterns`
  (decomposes to onnx.Mul/ReduceSum/Sqrt/Div; no new runtime)
- lit: `onnx-to-hip/test_lpnormalization`, `e2e/test_lpnormalization_model`
- deps: **PR-9**. The decomposition emits a broadcasting `onnx.Div` that relies
  on PR-9's `BroadcastDivToMulReciprocal` (in `ProjectorOpsRewrites.cpp`) to
  rewrite into `Mul(x, Reciprocal(norm))` (broadcasting Mul IS supported on
  main via `miopenOpTensor`). Register LpNorm in the SAME pre-lowering block as
  ProjectorOps, exactly as #259 does. Cut off the PR-9 branch, merge after PR-9.

### Group B — DISSOLVED (already on main)

#### PR-6 / PR-7 — DISSOLVED
Both fully landed on `syncorigin/main` via the cascade:
- **PR-6 (onnx.Loop)**: `LoopOutline`, `LoopLowering`,
  `hipdnn_ep_runtime_loop.cpp`, nested-pool slice, loop op — all on main.
- **PR-7 (MaterializeHostScalars)**: `MaterializeHostScalars.cpp`,
  `get_host_scratch` op, host-scratch runtime — all on main.

### Group C — vision / projector (real remaining work)

#### PR-5 — dynamic reshape / memref residue  (independent — base main) — **PR #303 (pushed)**
> **CORRECTION (verified against current main):** the earlier "only 2 net-new
> files; do NOT port ReshapeConversion/PoolAllocs" note was WRONG. main is the
> OLD ReshapeConversion (hard-fails on dynamic reshape; no `tensor.reshape`
> fallback), so `ReshapeShapeFold` is inert without the conversion delta. main's
> `resolveDimAtSource` also lacks reshape/cast/view/subview handling. PR #303
> ships the full functional set (4 pieces below). The ReshapeConversion +
> PoolAllocs ports are strict supersets of main (only refactor-wrapping
> deletions), so no regression. Build + lit + pre-commit all green.
Most of #259's reshape work is ALREADY on main (`ReshapeConversion` same-rank
dynamic branch with `arith.divui`; `PoolAllocs` `resolveDimAtSource` /
`foldDimOfReshape` transitive hoist — verified present). The ONLY net-new
residue is two passes:
- `lib/Dialect/Transforms/RelaxMultiDynExpandShape.cpp` (net-new)
- `lib/Conversion/OnnxToHip/ReshapeShapeFold.cpp` (net-new)
- `Pipelines.cpp` relax-multi-dyn wiring (partial), `Passes.td` (partial)
- lit: `Dialect/hip-relax-multi-dyn-expand-shape`
- deps: **none** — straight to `main`. Verified: no PR-1 helper, no `wrap_`.
- ⚠️ do NOT port `ReshapeConversion` / `PoolAllocs` deltas — main's versions
  supersede #259's (divergent re-impl; porting would regress main).

#### PR-8 — vision accuracy  (deps: PR-1)
- `SimplifyOnnx.cpp` (PrecisionFreeCast — NOT on main, +263), `SliceConversion.cpp`
  (empty-extent / `logical_extent` — NOT on main)
- `lib/Dialect/Transforms/FixLoopAccumulatorOffset.cpp` (net-new; Loop already
  on main)
- `MemoryLowering.cpp` memref-copy stride verification; runtime `memrefCopy`
  strides already in PR #284 (`hip.cpp`)
- `Pipelines.cpp` FixLoopAccum wiring (partial), `Passes.td` (partial)
- lit: `onnx-to-hip/test_simplify_onnx_precision_free_cast`, `test_slice`;
  `hip-to-llvm/test_memref_copy`
- doc: `docs/design/qwen-vision-ops.md`
- **Absorbs the dissolved PR-2 shape-op accuracy fixes** (ConstantOfShape
  AsScalar / Expand max-sizing / Tile under-size / Range scalar-extract + the
  2 `OnnxToHipUtils.h` scalar helpers + linalg/scf pipeline). **Triage at
  PR-8 time:** re-diff each vs `syncorigin/main` and keep only the ones still
  needed (some may already be superseded).
- deps: **PR-1** (Range fix uses `getInlineScalarFromOnnxConstant` /
  `materializeScalarFromDenseAttr` — verified). Loop & slice runtime already on
  main / in PR #284.

#### PR-9 — vision / projector compute ops  (deps: PR #284)
**Lowering-half + net-new rewrites.** ConvLowering / MatmulLowering pair with
PR #284's `wrap_miopenConvolutionForward` (+`data_type`) / `wrap_hipblasLtMatmul`
(+`b_batch_stride`) — part of the ABI train.
- `FastGeluFusion.cpp` (exact-Gelu Erf + vision variants, +350 — NOT on main),
  `ProjectorOpsRewrites.cpp` (net-new: Conv→Gemm + AveragePool/Pow/ReduceMean),
  `OnnxResultTypeInference.cpp/.h` (net-new; consumed only by these rewrites)
- Lowerings: `ConvLowering.cpp` (dtype+bias, 24-arg), `MatmulLowering.cpp`
  (b_batch_stride)
- lit: `onnx-to-hip/test_projector_rewrites`
- deps: **PR #284** (conv/matmul runtime). ProjectorOps does NOT need PR-1
  (verified). ⚠️ verify `wrap_hipblasLtMatmul` `b_batch_stride` runtime is in
  PR #284 (not found in matmul.cpp grep — may need a small runtime hunk here).

#### PR-10 — build / CI / docs / tests / tools  (deps: all)
- `build.py`, `environment.yml`, `.github/workflows/*`, `README.md`,
  `CLAUDE.md`, `3rd-party/morphizen` (submodule bump),
  `msvc_hip_cmath_workaround.h`, `tools/hip-mlir-opt/*`, `tools/hip-onnx-runner.cpp`
- tests: `test_qwen3_5_9b.py` (minus the excluded patch-merger dynshape test),
  `test_gemma3_4b.py`, `conftest.py`, `images/tower.jpg`, `test/qwen3_5_9b/*`,
  `test/numeric/*`
- vision end-to-end accuracy is gated on the new output-shape PR

## Merge train (ABI-coupled — parallel siblings on #284)

PR #284 (runtime ABI) + its lowering halves are one logical atomic change split
for review. `main` is broken for the 8 coupled ops in any intermediate state, so
the lowering PRs all base on #284 as **parallel siblings** (disjoint op files →
any internal merge order) and `main` is validated only after the whole set lands:

```
#284 (runtime: 8 wrap_ new sigs)
  ├─► PR-1/#299  (div / sub / equal lowerings)
  ├─► PR-3a/#304 (gather lowering 9→11 arg)               DONE
  ├─► PR-3b/#305 (nonzero/scatter_nd count_buf/valid_count) DONE
  └─► PR-9       (conv/matmul lowerings + vision rewrites)  ← train tail
```

After all four land, `main` is ABI-consistent again. Then PR-4 (LpNorm) merges
after PR-9, and PR-8 (vision accuracy) stacks on PR-1.

## Dependency DAG

```
TRAIN (parallel siblings on #284, validate main only after all land):
  #284 ─┬─► PR-1/#299  (div/sub/equal lowerings)
        ├─► PR-3a/#304 (gather lowering)                 DONE
        ├─► PR-3b/#305 (nonzero/scatter_nd GPU-count)    DONE
        └─► PR-9       (conv/matmul lowerings + vision rewrites)
                          │
                          └─► PR-4 (LpNorm; needs PR-9 broadcast-Div)

  PR-1/#299 ─► PR-8 (vision accuracy; needs PR-1 helpers)

INDEPENDENT (straight to main, any time, no coupling):
  main ─► PR-5/#303 (RelaxMultiDynExpandShape + ReshapeShapeFold)   DONE/pushed

LAST:
  PR-10 (tests/assets/build/docs) ── after everything
```

## merge order

1. **PR-5** (= **#303**, pushed) — the only truly independent PR, straight to
   `main` (in review now, parallel to the train).
2. **Train** (all base #284; merge back-to-back, no main validation in between):
   **#284 first**, then its lowering siblings **PR-1/#299, PR-3a/#304,
   PR-3b/#305, PR-9** (any internal order), then **PR-4** (after PR-9).
3. **PR-8** — stacks on PR-1 (vision accuracy + absorbed PR-2 fixes).
4. **PR-10** — tests/assets/build/docs, last (after all above).
