<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Pipeline Pass Menu

A version-pinned reference of the pass and pipeline **names** that resolve in
the compiler's pass registry, the **order** the built-in pipeline runs them in,
and the **anchor op** each one expects. Use it when composing a custom pipeline
through the `HIPDNN_EP_PIPELINE` override or when deciding which
[plugin pipeline slot](design/plugin-interface.md) to target.

> **Stability / version pin.** The names below are the stable textual
> identifiers (`getArgument()` / pipeline registration names) as of the commit
> that ships this document. They are part of the compiler's public surface: a
> rename is a breaking change. Pin your override / plugin against a tagged
> release and re-check this table when you upgrade. The single source of truth
> for *which* names are registered is
> `include/hip/InitAllPasses.h::registerAllPasses()`; the order is
> `lib/Dialect/Transforms/Pipelines.cpp`.

---

## The override in one paragraph

Set `HIPDNN_EP_PIPELINE` to replace the built-in ONNX→HIP→LLVM pass order. The
value is parsed by MLIR's textual pass-pipeline parser, so it must name passes
that are registered in the compiler's registry (the tables below). Two forms
are accepted:

```bash
# Bare inner list (composed at module scope):
export HIPDNN_EP_PIPELINE="simplify-onnx, hip-add-context-arg, ..."

# Explicit module wrapper (identical effect; the leading `builtin.module(...)`
# is stripped before parsing):
export HIPDNN_EP_PIPELINE="builtin.module(simplify-onnx, func.func(canonicalize), ...)"
```

When `HIPDNN_EP_PIPELINE` is **unset**, the built-in pipeline runs unchanged.
When it is set, **you** own the load-bearing ordering invariants and the
required terminal stages. After the pipeline runs, the driver hard-fails if the
module lacks an `inference_compute` entry point:

```
HIPDNN_EP_PIPELINE: the custom pipeline did not produce an 'inference_compute'
entry point (did it omit generate-interface?)
```

A pass anchored on `func.func` must be nested under `func.func(...)` in the
textual form (e.g. `func.func(hip-pool-allocs)`); a pass anchored on
`builtin.module` is named at the top level. The **Anchor** column below tells
you which.

See [`docs/design/plugin-interface.md`](design/plugin-interface.md) §"Pipeline
composition" for the design and rationale.

---

## Recommended composition model

Reconstructing the whole pipeline by hand is brittle and, today, impossible by
text alone (a few load-bearing passes are not individually name-registerable —
see [below](#passes-that-are-not-individually-nameable)). Prefer, in order:

1. **Compose the registered _pipeline_ names.** `onnx-to-hip-pipeline` and
   `hip-to-llvm-pipeline` (or the combined `hipdnn-pipeline`) bake in the full,
   correct stage order **including** the passes you cannot name individually
   (`generate-interface`, the `expand-strided-metadata` / `lower-affine`
   utility passes, etc.). This is the only override form guaranteed to clear
   the `inference_compute` guardrail:

   ```bash
   export HIPDNN_EP_PIPELINE="onnx-to-hip-pipeline, hip-to-llvm-pipeline"
   ```

2. **Insert vendor passes at a [plugin slot](#plugin-pipeline-slots)** instead
   of via the env override. The slots sit at semantically meaningful points and
   keep the surrounding default order intact — the robust path for a shipping
   plugin.

3. **Use the textual override for boundary reordering / single-pass insertion**
   — e.g. running an extra canonicalize, or prepending a vendor pass before the
   pipeline names. Reserve full hand-composition for experiments.

---

## Composable pipeline names

| Name | Options | What it builds |
|---|---|---|
| `onnx-to-hip-pipeline` | `externalize-output-dir`, `externalize-min-num-elements`, `skip-constant-data` | ONNX dialect → fully bufferized, pooled HIP memref IR with resolved extern constants. |
| `hip-to-llvm-pipeline` | `constants-file` | HIP memref IR → LLVM dialect + the C-ABI interface (`inference_init/compute/cleanup`, metadata). |
| `hipdnn-pipeline` | `constants-file`, `constants-dir`, `externalize-min-num-elements` | The full ONNX→HIP→LLVM→interface flow (chains the two above). |

Options use MLIR's pipeline-option syntax:
`onnx-to-hip-pipeline{skip-constant-data=true}`.

> These pipeline names reproduce the same flow the EP / `hip-compiler`
> front-end runs, so composing them is the way to match the default compile
> exactly. Graph outputs are always allocated in-graph via `hip.alloc_output`
> (the output-allocator ABI); there is no mode toggle.

---

## Individually nameable passes

### HIP transform & conversion passes

| Name | Anchor | One-liner |
|---|---|---|
| `simplify-onnx` | module | Pre-lowering ONNX cleanups (`CastLike`→`Cast`, drop dead type-donor args). |
| `hip-add-context-arg` | module | Inject the `!hip.context` argument into functions. |
| `onnx-loop-outline` | module | Outline `onnx.Loop` bodies into separate `func.func` ops. |
| `hip-infer-loop-body-shapes` | module | Rank-establish unranked tensors inside outlined loop bodies. |
| `outline-onnx-to-hipdnn` | module | Outline subgraphs targeted at hipDNN graph compilation. |
| `convert-onnx-to-hip` | module | Pattern-match ONNX ops by name → HIP dialect; externalize large constants. |
| `hip-infer-shapes` | module | Refine `?` dims on HIP DPS result types via `ReifyRankedShapedTypeOpInterface`. |
| `hip-split-duplicate-dps-inits` | func.func | De-alias DPS init operands that CSE merged onto one `tensor.empty`, so an op that reads back its own outputs (e.g. `hip.gqa` present K/V) does not share a buffer (pre-bufferize). |
| `hip-resolve-tensor-dims` | func.func | Fold `tensor.dim` of reshape chains into root-dim arithmetic (pre-bufferize). |
| `hip-loop-body-to-out-params` | module | Promote outlined loop bodies to the out-param ABI. |
| `hip-use-output-allocator` | func.func | Rewrite returned `memref.alloc` → `hip.alloc_output` (graph outputs become EP/runtime-owned, not pooled or deallocated). |
| `hip-fix-loop-accumulator-offset` | func.func | Rewrite frozen Concat-accumulator offsets in loop bodies to iter-driven offsets. |
| `hip-promote-strided-operands` | func.func | Materialize contiguous temporaries for strided memref operands of `hip.*` ops. |
| `hip-materialize-host-scalars` | func.func | Redirect tiny host-fed scalar allocs to runtime-owned host-mapped scratch (away from the GPU pool). |
| `hip-hoist-alloc-size-arith` | func.func | Hoist speculatable size arithmetic above the earliest dynamic alloc (PoolAllocs precondition). |
| `hip-pool-allocs` | func.func | Pack allocations into a single grow-on-demand GPU pool; stamp `hipdnn.pool_size`. |
| `hip-lower-allocs` | func.func | Replace `memref.alloc`/`dealloc` with `hip.alloc`/`hip.free`. |
| `hip-relax-multi-dyn-expand-shape` | func.func | Rewrite multi-dynamic-per-group `memref.expand_shape` → `reinterpret_cast` before strided-metadata expansion. |
| `hip-resolve-extern-constants` | module | Resolve extern constants → `memref.view` into the constants-blob argument. |
| `assign-op-state-slots` | module | Assign one op-state slot per stateful op instance (op-state-slots design). No-op without stateful ops. |
| `generate-op-state-init` | module | Emit `@hipdnn_ep_op_states_init_fn` from each stateful op's `generateOpStateInit`. No-op without stateful ops. |
| `convert-hip-to-llvm` | module | Lower HIP ops to runtime C-API calls / LLVM dialect. |

### Standard MLIR passes the pipeline interleaves

| Name | Anchor | One-liner |
|---|---|---|
| `canonicalize` | any | Standard canonicalization. |
| `cse` | any | Common-subexpression elimination. |
| `one-shot-bufferize` | module | Tensor → memref bufferization (function boundaries, identity layout). |
| `buffer-deallocation-pipeline` | module | Ownership-based buffer deallocation (a sub-pipeline). |
| `convert-bufferization-to-memref` | any | Lower residual `bufferization.*` ops to memref. |
| `convert-scf-to-cf` | any | Lower `scf.for`/`scf.if` to unstructured control flow. |
| `reconcile-unrealized-casts` | any | Drop leftover `unrealized_conversion_cast`s. |
| `resolve-shaped-type-result-dims` | any | Fold `tensor.dim`/`memref.dim` of shaped-type op results through reify patterns. |

---

## Passes that are NOT individually nameable

These run inside the built-in pipeline but cannot be named in a textual
override (so a hand-listed pipeline omitting them will trip the
`inference_compute` guardrail or fail translation). Use the
[pipeline names](#composable-pipeline-names), which include them:

| Pass | Why it is not nameable |
|---|---|
| `generate-interface` | Requires a `CompilationOptionsT` (not a no-arg pass). Emitted by `hip-to-llvm-pipeline`. |
| `compile-hipdnn-graphs` | Requires a live runtime handle. Only present on the handle-bearing pipeline overload. |
| `expand-strided-metadata` | MLIR utility pass added directly inside `hip-to-llvm-pipeline`. |
| `lower-affine` | Added directly inside `hip-to-llvm-pipeline` (lowers `affine.apply` from strided-metadata expansion). |
| `convert-linalg-to-loops` | Added directly inside the ONNX→HIP tail. |

---

## Canonical built-in order (reference)

What the default pipeline runs, with the plugin slots interleaved. Source of
truth: `lib/Dialect/Transforms/Pipelines.cpp`.

```
ONNX → HIP  (buildOnnxToHipPipeline)
  simplify-onnx
  «slot: AfterSimplifyOnnx»
  hip-add-context-arg
  onnx-loop-outline
  hip-infer-loop-body-shapes
  «slot: AfterOnnxLoopOutline»
  [outline-onnx-to-hipdnn + compile-hipdnn-graphs]   (handle overload only)
  convert-onnx-to-hip
  «slot: AfterConvertOnnxToHip»
  hip-infer-shapes
  canonicalize ; cse
  func.func(hip-split-duplicate-dps-inits)
  func.func(hip-resolve-tensor-dims)
  «slot: BeforeBufferization»
  one-shot-bufferize
  hip-loop-body-to-out-params
  buffer-deallocation-pipeline
  func.func(hip-use-output-allocator)  (slot 4.5)
  func.func(hip-fix-loop-accumulator-offset)
  cse ; canonicalize
  func.func(convert-linalg-to-loops)
  func.func(hip-promote-strided-operands)
  func.func(hip-materialize-host-scalars)
  func.func(hip-hoist-alloc-size-arith)
  func.func(hip-pool-allocs)
  «slot: AfterPoolAllocs»
  convert-bufferization-to-memref
  cse ; canonicalize
  func.func(hip-lower-allocs)
  hip-resolve-extern-constants
  cse ; canonicalize

HIP → LLVM  (buildHipToLLVMPipeline)
  func.func(hip-relax-multi-dyn-expand-shape)
  «slot: BeforeConvertHipToLLVM»
  expand-strided-metadata
  lower-affine
  assign-op-state-slots
  generate-op-state-init
  convert-scf-to-cf
  reconcile-unrealized-casts
  convert-hip-to-llvm
  generate-interface
  «slot: AfterGenerateInterface»
```

---

## Plugin pipeline slots

A plugin's `requestPipelineSlot(slot, "pass-name")` inserts a registered pass at
a fixed point without rewriting the order. Slots (see
`include/hip/Compiler/PluginRegistry.h::PipelineSlot`):

| Slot | Sits after / before | Typical use |
|---|---|---|
| `AfterSimplifyOnnx` | after `simplify-onnx` | Canonical ONNX dialect, no HIP context arg yet. |
| `AfterOnnxLoopOutline` | after loop outlining + body shape inference | Operate on outlined ONNX loop bodies. |
| `AfterConvertOnnxToHip` | after `convert-onnx-to-hip` | Most common slot — lower `onnx.Custom`, vendor `hip.*` canonicalizations. |
| `BeforeBufferization` | before `one-shot-bufferize` | Lower/canonicalize `hip.*` while still in tensor type-system. |
| `AfterPoolAllocs` | after `hip-pool-allocs` | Analyze/transform pooled allocations (allocator tagging, etc.). |
| `BeforeConvertHipToLLVM` | before `convert-hip-to-llvm` | Last chance on `hip.*`/memref IR before LLVM lowering erases it. |
| `AfterGenerateInterface` | after `generate-interface` | Stamp metadata / add LLVM-dialect ops alongside the C interface. |

See [`docs/plugin_authoring.md`](plugin_authoring.md) for the registration
mechanics.
