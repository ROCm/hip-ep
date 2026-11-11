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
| `rocmlir-pipeline` | — | Prepare HIP tensor IR for rocMLIR kernel offload: split `hip.conv_transpose` into plain convolutions, outline the rocMLIR-eligible ops into `rock.kernel` funcs, then lower those bodies to TOSA. Not part of the default compile — `hip-rocmlir-compiler` runs it. |

Options use MLIR's pipeline-option syntax:
`onnx-to-hip-pipeline{skip-constant-data=true}`.

`hip-rocmlir-compiler` defaults to quick autotune: it enumerates the rocMLIR
quick space, compiles every applicable candidate, benchmarks them on the
current HIP device, and embeds the fastest binary. Every search config in the
model is compiled in one pool (default: one job per hardware thread;
`HIP_ROCMLIR_COMPILE_JOBS` overrides the width). GPU benchmarking stays serial,
and it starts only after that pool finishes, so candidate timings do not
overlap and a slow config does not stall the next kernel's compiles. Within
one compile, the first kernel for a rocMLIR problem key (`getTuningProblemStr`,
plus the selected space) is searched; later kernels with that key compile only
the winning perfConfig, together in a second pool, and are not benchmarked
again. The compiled binary is not reused, because each outlined kernel has its
own symbol. `--autotune[=quick|full|exhaustive]`
overrides the search space. Set `HIP_ROCMLIR_SKIP_BENCHMARKING=1` (any value
other than `0`) to skip the GPU search and affix the first enumerated
perfConfig. The log names the space (`quick`, `full`, or `exhaustive`) and the
number of configs in each pool. At the end it prints three wall times:
perfConfig compile (pool wait, summed across the two pools), serial GPU
benchmark, and HIP-to-LLVM lowering plus bitcode emit. Per-config timing and
the selected perfConfig string are printed only with `--verbose`.

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
| `onnx-scan-outline` | module | Outline supported forward, axis-0 `onnx.Scan` bodies and materialize their state and stacked outputs. |
| `onnx-if-outline` | module | Outline `onnx.If` branches before ONNX-to-HIP conversion. |
| `hip-infer-loop-body-shapes` | module | Rank-establish unranked tensors inside outlined loop bodies. |
| `outline-onnx-to-hipdnn` | module | Outline subgraphs targeted at hipDNN graph compilation. |
| `convert-onnx-to-hip` | module | Pattern-match ONNX ops by name → HIP dialect; lower both constant sweeps to policy-neutral `hip.constant`; legalize GatherBlockQuantized (INT4 packing, unsigned, quantize_axis). Pre-lowering folds run first on ONNX IR, among them the `onnx.Attention` mask fold (sliding window and key-padding bias, `AttentionWindowFold.cpp`) and the sigmoid + correction-bias QMoE routing fold (`QMoEConversion.cpp`). |
| `hip-fold-batchnorm` | func.func | Fold an inference `hip.batch_norm` with constant parameters into the filter and bias of the `hip.conv` producing it, directly or through a `hip.add` of two convs (the shift goes into one bias). Every intermediate result must have the batch norm as its only consumer. Reads inline and file-backed constants only; a batch norm over memory-address-backed weights is left in place. |
| `hip-externalize-constants` | module | Validate explicit serialization order, append unordered carriers deterministically, plan/serialize, then replace `hip.constant` using the existing inline/full/streaming/hybrid metadata contract. Direct threshold default: 0. |
| `hip-fold-constant-extracts` | func.func | Resolve constant-index `tensor.extract` of vectors assembled with `tensor.insert_slice` (ONNX shape vectors) to the `from_elements` operand or integer constant element they read. |
| `hip-infer-shapes` | module | Refine `?` dims on HIP DPS result types via `ReifyRankedShapedTypeOpInterface`. |
| `hip-fusion-transform` | module | HIP-to-HIP rewriting at unrestricted arity (one op → many, many → one, or subgraph → subgraph). The patterns live in `lib/Dialect/Transforms/fusion_pattern/` — PDLL files compiled into the library, plus native C++ for rewrites whose shape depends on the matched op's rank — and share one `RewritePatternSet`. |
| `hip-split-duplicate-dps-inits` | func.func | De-alias DPS init operands that CSE merged onto one `tensor.empty`, so an op that reads back its own outputs (e.g. `hip.gqa` present K/V) does not share a buffer (pre-bufferize). |
| `hip-resolve-tensor-dims` | func.func | Fold `tensor.dim` of reshape chains into root-dim arithmetic (pre-bufferize). |
| `hip-loop-body-to-out-params` | module | Promote outlined loop bodies to the out-param ABI. |
| `hip-use-output-allocator` | func.func | Rewrite returned `memref.alloc` → `hip.alloc_output` (graph outputs become EP/runtime-owned, not pooled or deallocated). |
| `hip-fix-loop-accumulator-offset` | func.func | Rewrite frozen Concat-accumulator offsets in loop bodies to iter-driven offsets. |
| `hip-optimize-memrefs` | func.func | HIP-specific buffer reuse / subview folding. |
| `hip-promote-strided-operands` | func.func | Materialize identity-layout temporaries for non-identity-layout DPS-input memrefs. |
| `hip-materialize-host-scalars` | func.func | Redirect tiny host-fed scalar allocs to runtime-owned host-mapped scratch (away from the GPU pool). |
| `hip-resolve-memref-dims` | func.func | Fold post-bufferization `memref.dim` through view/reshape chains before pool planning. |
| `hip-hoist-alloc-size-arith` | func.func | Hoist pure size producers before the earliest allocation their dependencies permit (PoolAllocs precondition). |
| `hip-pool-allocs` | func.func | Pack allocations into one or more grow-on-demand GPU pools; stamp pool-planning module attributes. |
| `hip-lower-allocs` | func.func | Replace `memref.alloc`/`dealloc` with `hip.alloc`/`hip.free`. |
| `hip-relax-multi-dyn-expand-shape` | func.func | Rewrite multi-dynamic-per-group `memref.expand_shape` → `reinterpret_cast` before strided-metadata expansion. |
| `hip-resolve-extern-constants` | module | Resolve extern constants → `memref.view` into the constants-blob argument. |
| `assign-op-state-slots` | module | Assign one op-state slot per stateful op instance (op-state-slots design). No-op without stateful ops. |
| `generate-op-state-init` | module | Emit `@hipdnn_ep_op_states_init_fn` from each stateful op's `generateOpStateInit`. No-op without stateful ops. |
| `convert-hip-to-llvm` | module | Lower HIP ops to runtime C-API calls / LLVM dialect. |

### rocMLIR offload passes

Anchored on `func.func`, so the textual form needs `func.func(...)`. These run
against HIP **tensor** IR (before bufferization) and are reached through
`rocmlir-pipeline`. `hip-decompose-conv-transpose` also runs in the default
ONNX-to-HIP tail; the others do not. The fusion design is in
[docs/design/rocmlir-fusion.md](design/rocmlir-fusion.md).

| Name | Anchor | One-liner |
|---|---|---|
| `hip-decompose-conv-transpose` | func.func | Rewrite `hip.conv_transpose` into stride-1 `hip.conv`s, one per stride residue (MIOpen backward-data v4r1). Residues with equal tap counts, or unequal ones whose zero-padded filters cost at most twice the real taps, stack on the output-channel axis into one convolution with the constant bias tiled into it; its result is interleaved by an expand, a transpose (DepthToSpace) and a collapse, then cropped. Other residues are one convolution each, with the bias, reassembled with `tensor.insert_slice`. Reads the filter from an inline or file-backed `hip.constant`. Rank-3 inputs expand to a unit-height rank-4 conv so dynamic batch and width stay legal. Leaves unsupported cases (grouped, dilated, rank-4 dynamic, non-constant, memory-address-backed or non-f16/bf16/f32 weights, stride > kernel, > 64 residues) for the MIOpen path. Also runs in the ONNX-to-HIP tail, before shape inference. |
| `hip-hoist-pointwise-over-views` | func.func | Move a pointwise op whose other inputs are constant scalars above a chain of single-use element-move views over a single-use conv, matmul or gemm result, so the anchor fusion takes it into the epilogue. The anchor result may have at most 50% more elements than the op's input. |
| `hip-rocmlir-prepare-quant` | func.func | Sink each per-tensor `hip.quantize_linear` back through element-move ops to its dequantizes, split each QLinearConv / QLinearMatMul with ui8 weights or a non-zero weight zero point into integer ops on `w - offset` and over the shifted zero point, then rebias each ui8 tree that reaches an integer anchor to i8. |
| `hip-fuse-rocmlir-elementwise` | func.func | Group the pointwise ops, transposes, static reshapes and per-tensor quantizes and dequantizes into `hip.rocmlir_elementwise` modules that an anchor can take whole. |
| `hip-fuse-rocmlir-anchor` | func.func | Plan an epilogue and a prologue of modules for each conv, matmul and gemm anchor, and outline each plan into a `rock.kernel` function (stamped with `rock.arch` from the `arch` option when set) behind `hip.rocmlir`. A module that no plan takes and that saves a launch becomes a `rock.elementwise` kernel when `standalone-elementwise` is on (the default; `HIPDNN_EP_ROCMLIR_NO_STANDALONE_ELEMENTWISE=1` turns it off in the EP), after its boundary transposes that move the innermost dimension are peeled into HIP transposes and the slice chains of its inputs are copied into it; the other modules are inlined. Decomposes the QLinear ops and dequantized convs that become integer anchors, then merges gemm-elementwise-gemm pairs. |
| `hip-inline-rocmlir-elementwise` | func.func | Put the body of each `hip.rocmlir_elementwise` back in its block. For tests and replays. |
| `hip-rocmlir-fusion-report` | module | Append one JSON line for each kernel and host function to the `report` file. |
| `convert-hip-to-tosa` | func.func | Lower the outlined kernel bodies from HIP ops to TOSA. |

Order matters: `hip-decompose-conv-transpose` must run **before** the fusion,
because rocMLIR has no transposed-convolution anchor and would otherwise refuse
to outline the op at all. `hip-hoist-pointwise-over-views` then moves the
activation after the interleave and crop onto the convolution's result.
`hip-rocmlir-prepare-quant` fixes the quantize trees
before the elementwise pass groups them, and the anchor pass plans its kernels
on those modules. The registered pipeline runs

```
func.func(hip-decompose-conv-transpose)
func.func(hip-hoist-pointwise-over-views)
func.func(hip-rocmlir-prepare-quant)
func.func(hip-fuse-rocmlir-elementwise)
func.func(hip-fuse-rocmlir-anchor)
duplicate-function-elimination
func.func(convert-hip-to-tosa)
canonicalize
```

The decomposition emits its residues already folded — the sub-filters are
computed at pass time — so it needs no canonicalize of its own before the fusion
passes, and none runs there.

> The `hipsr-*` family and `convert-onnx-to-hipsr` are registered but not yet
> tabulated here.

### Standard MLIR passes and sub-pipelines

| Name | Anchor | One-liner |
|---|---|---|
| `canonicalize` | any | Standard canonicalization. The `hip.cast` patterns also fold an exact float widening, a chain of single-use element-move ops and the narrowing back into the chain on the narrow type (see [docs/design/element-move-interface.md](design/element-move-interface.md)). |
| `cse` | any | Common-subexpression elimination. |
| `one-shot-bufferize` | module | Tensor → memref bufferization (function boundaries, identity layout); `HIPDNN_EP_BUFFERIZE_COPY_BEFORE_WRITE=1` enables the huge-graph copy-before-write escape hatch. |
| `buffer-deallocation-pipeline` | module | Ownership-based buffer deallocation; available to custom pipelines and characterization tests, but omitted from the built-in pipeline. |
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
| `compile-hipdnn-graphs` | Requires a live hipDNN handle, which exists only when built with `BUILD_HIPDNN_GRAPH=ON` (default OFF). |
| `expand-strided-metadata` | MLIR utility pass added directly inside `hip-to-llvm-pipeline`. |
| `lower-affine` | Added directly before buffer-reuse normalization in `onnx-to-hip-pipeline`, and after strided-metadata expansion in `hip-to-llvm-pipeline`. |
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
  onnx-scan-outline
  onnx-if-outline
  hip-infer-loop-body-shapes
  «slot: AfterOnnxLoopOutline»  (after Loop / Scan / If outlining)
  [outline-onnx-to-hipdnn + compile-hipdnn-graphs]   (BUILD_HIPDNN_GRAPH=ON only)
  convert-onnx-to-hip
  func.func(hip-fold-batchnorm)      (shared with the rocMLIR head, so it runs
                                      before any conv rewrite or kernel outlining)
  «slot: AfterConvertOnnxToHip»  (supported hip.constant producer boundary)
  func.func(hip-decompose-conv-transpose)  (rank-3 -> rank-4 hip.conv before
                                           the MIOpen lowering; rocmlir-pipeline
                                           runs the same pass in its own head)
  func.func(hip-fold-constant-extracts)
  hip-infer-shapes
  hip-fusion-transform               (must precede externalization: its patterns
                                      read hip.constant payloads)
  hip-externalize-constants
  canonicalize ; cse
  func.func(hip-split-duplicate-dps-inits)
  func.func(hip-resolve-tensor-dims)
  «slot: BeforeBufferization»
  verify no hip.constant carriers survive
  one-shot-bufferize
  hip-loop-body-to-out-params
  func.func(hip-use-output-allocator)  (slot 4.5)
  func.func(hip-fix-loop-accumulator-offset)
  lower-affine
  cse ; canonicalize
  func.func(convert-linalg-to-loops)
  func.func(hip-materialize-host-scalars)
  func.func(hip-optimize-memrefs)
  func.func(hip-promote-strided-operands)
  func.func(hip-resolve-memref-dims)
  func.func(cse)
  func.func(canonicalize)
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
| `AfterOnnxLoopOutline` | after loop/if outlining + loop-body shape inference | Operate on outlined ONNX control-flow bodies. |
| `AfterConvertOnnxToHip` | after `convert-onnx-to-hip` | Most common slot — lower `onnx.Custom`, vendor `hip.*` canonicalizations. |
| `BeforeBufferization` | before `one-shot-bufferize` | Lower/canonicalize `hip.*` while still in tensor type-system. |
| `AfterPoolAllocs` | after `hip-pool-allocs` | Analyze/transform pooled allocations (allocator tagging, etc.). |
| `BeforeConvertHipToLLVM` | before `convert-hip-to-llvm` | Last chance on `hip.*`/memref IR before LLVM lowering erases it. |
| `AfterGenerateInterface` | after `generate-interface` | Stamp metadata / add LLVM-dialect ops alongside the C interface. |

See [`docs/plugin_authoring.md`](plugin_authoring.md) for the registration
mechanics.
