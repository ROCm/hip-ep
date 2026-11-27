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
current HIP device, and embeds the fastest binary. The ONNX Runtime provider
does the same whenever `HIPDNN_EP_ROCMLIR` is on. Both honor
`HIP_ROCMLIR_SKIP_BENCHMARKING`: unset or `0` benchmarks. A present empty
value skips, and so does any other value, keeping rocMLIR's default perfConfig.
Unset `HIPDNN_EP_ROCMLIR_AUTOTUNE`
selects quick. `quick`, `full`, and `exhaustive` are the only accepted
values; anything else warns and stays on quick. Every search config in the
model is compiled in one pool (default: one job per hardware thread;
`HIP_ROCMLIR_COMPILE_JOBS` overrides the width). GPU benchmarking stays serial,
and it starts only after that pool finishes, so candidate timings do not
overlap and a slow config does not stall the next kernel's compiles. Within
one compile, the first kernel for a rocMLIR problem key (`getTuningProblemStr`,
plus the selected space) is searched; later kernels with that key compile only
the winning perfConfig, together in a second pool, and are not benchmarked
again. The compiled binary is not reused, because each outlined kernel has its
own symbol. `--autotune[=quick|full|exhaustive]`
overrides the search space. A given tuning type skips the shipped problem-cache
LUT; with none given, the shipped LUT is consulted before the search. A table
answers only for an identical problem string; any other problem is searched.
Nearest-config lookup, which lets a miss borrow the tile of the nearest
measured problem of the same kernel kind, is off by default. It is turned on by
`--enable-nearest-config` for one `hip-rocmlir-compiler` run, or by
`HIPDNN_EP_ROCMLIR_NEAREST_CONFIG=1`, which both binaries read, for a whole
shell. It applies to the shipped LUT and to an external file alike, and changes
nothing when neither is in use. `--enable-problem-cache-rw <file.json>` still
takes priority over the search and replaces the shipped table. The flag names a
LUT JSON file, such as one it wrote. It also writes back: after a successful
compile the winners the search picked are merged into that file, so a later
compile reads them. Only winners of a search are written. A tile taken from the
file, even a nearest neighbour's, is not a measurement and is not stored as
one. Points already in the file are kept in their order, including ones this
build cannot use, and so are fallbacks and the configs they name; new problems
are appended. With no new winners the file is not rewritten. A missing or empty
file is created. A file that is not a LUT, is for another `gpu_arch`, has a
malformed `configs`, `points` or `fallbacks` section, has a value the table
format cannot hold (`rocm_version` beyond 32 bits, `m`, `n` or `k` outside 0 to
2^32-1, more than 65536 configs), is read-only, is not a regular file, sits in
a directory the rewrite cannot write to (or create), or is a `.fb` is refused
before the compile starts, with the file untouched; a write that fails after
the compile is a warning, and the output is still written. The file is replaced
atomically and flushed to disk before the rename. It is written owner-only
(mode 0600, or on Windows an access list that names only you), so no one else
can read it halfway. On POSIX a new file stays owner-only and a replaced one
gets its old mode and group back, but not its owner, which becomes the user who
ran the compile; on Windows the file stays private to you, and anyone else who
needs it must be granted access afterwards. A symbolic link, in the file name
or in any directory of the path, is followed to its target, which is created if
the link is dangling. A dangling link is followed only by the rule the kernel
uses in a shared directory like `/tmp`: its owner must be you or the
directory's owner. On Windows the directory that must accept the new file (the
nearest one that exists) is checked against its access list before the compile.
The rename replaces the file, so ACLs, extended attributes and hard links of
the old one are not carried over. `schema_version` is always written out.
Compiles in one process take turns, but two processes writing it at once can
lose one set of winners. The provider does the same as the flag through
`HIPDNN_EP_ROCMLIR_PROBLEM_CACHE_RW=<file.json>`, except that a file it cannot
use or update is reported and the compile still succeeds. A repeated problem in
the same compile reuses the in-session winner, then a miss searches the
selected space. A provider artifact records `+autotune=<space>` only. A table
starts from a path that does not exist yet. The file has one point per unique
problem and one config per distinct perfConfig, and `problem_cache_to_fb.py`
compiles it to a `.fb` to ship it as a LUT. The flag requires autotune;
`HIP_ROCMLIR_SKIP_BENCHMARKING` writes nothing. The provider's hip and TOSA
dumps, `HIPDNN_EP_ROCMLIR_DUMP_HIP` and `_DUMP_TOSA`, are described in
[rocmlir-fusion.md](design/rocmlir-fusion.md). Set
`HIP_ROCMLIR_SKIP_BENCHMARKING=1` (any present value other than `0`, including
empty) to skip the GPU search and affix the first enumerated perfConfig.
Neither the shipped LUT nor an external problem cache is consulted in that
mode. When benchmarking is enabled, a provided tuning type skips the shipped
LUT, but an external `--enable-problem-cache-rw` file is consulted before the
search. The log names the space (`quick`, `full`, or `exhaustive`) and the
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
| `hip-fusion-transform` | module | HIP-to-HIP rewriting at unrestricted arity (one op → many, many → one, or subgraph → subgraph). The patterns live in `lib/Dialect/Transforms/fusion_pattern/` — PDLL files compiled into the library, plus native C++ for rewrites whose shape depends on the matched op's rank — and share one `RewritePatternSet`. Includes the GELU fusion: collapses primitive `hip.mul`/`add`/`tanh`/`erf`/`div`/`pow` chains to one `hip.gelu` when the coefficients are statically known scalar or one-element floating-point constants, with supported casts evaluated. |
| `hip-split-duplicate-dps-inits` | func.func | De-alias DPS init operands that CSE merged onto one `tensor.empty`, so an op that reads back its own outputs (e.g. `hip.gqa` present K/V) does not share a buffer (pre-bufferize). |
| `hip-resolve-tensor-dims` | func.func | Fold `tensor.dim` of reshape chains into root-dim arithmetic (pre-bufferize). |
| `hip-loop-body-to-out-params` | module | Promote outlined loop bodies to the out-param ABI. |
| `hip-use-output-allocator` | func.func | Rewrite returned `memref.alloc` → `hip.alloc_output` (graph outputs become EP/runtime-owned, not pooled or deallocated). |
| `hip-fix-loop-accumulator-offset` | func.func | Rewrite frozen Concat-accumulator offsets in loop bodies to iter-driven offsets. |
| `hip-forward-concat-slots` | func.func | Retarget the writer and every reader of a private buffer at an identity view of its packed concat slot, and drop that input's `memref.copy`. A chained concat buffer becomes a prefix view of the outer buffer, and a dense block's concats share one buffer. A slot off a 16-byte boundary keeps its copy. Runs after `hip-materialize-host-scalars`. |
| `hip-optimize-memrefs` | func.func | HIP-specific buffer reuse / subview folding. |
| `hip-promote-strided-operands` | func.func | Materialize identity-layout temporaries for non-identity-layout DPS-input memrefs. `hip.reverse` reads strided inputs directly and is skipped. `hip.concat` reads contiguous views in place, and becomes per-input slot copies when its strided inputs would otherwise need as many copies. |
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
| `hip-decompose-conv-transpose` | func.func | Rewrite `hip.conv_transpose` into stride-1 `hip.conv`s, one per stride residue (MIOpen backward-data v4r1). Residues with equal tap counts, or unequal ones whose zero-padded filters cost at most twice the real taps, stack on the output-channel axis into one convolution with the constant bias tiled into it (a single-use `hip.add` of a per-channel constant after the op folds into that bias too); its result is interleaved by an expand, a transpose (DepthToSpace) and a collapse, then cropped. Other residues are one convolution each, with the bias, reassembled with `tensor.insert_slice`. Reads the filter from an inline or file-backed `hip.constant`. Rank-3 inputs expand to a unit-height rank-4 conv so dynamic batch and width stay legal. Leaves unsupported cases (grouped, dilated, rank-4 dynamic, non-constant, memory-address-backed or non-f16/bf16/f32 weights, stride > kernel, > 64 residues) for the MIOpen path. Also runs in the ONNX-to-HIP tail, before shape inference. |
| `hip-decompose-multi-head-attention` | func.func | Rewrite a static f32 `hip.multi_head_attention`, or a static f16 one that is decode (query sequence 1) or whose padded head dim is above 256, with separate query, key, and value into two `hip.matmul`s, a scale `hip.mul`, and `hip.miopen.softmax`. f16 prefill with a padded head dim of at most 256 stays on the runtime kernel so wave32 can take the flash path. Packed QKV, bias, masks, past key or value, causal masking, dynamic shapes, and any other float type stay there too. bf16 is an error: the runtime kernel would read it as f16. |
| `hip-winograd-conv` | func.func | Rewrite an eligible fp16 3x3 stride-1 pad-1 group-1 `hip.conv` with a constant filter, and the LeakyRelu or ReLU that alone consumes it, into `hip.winograd_conv` when the shape gate admits it. Runs first in the rocMLIR fusion passes with `HIPDNN_EP_WINOGRAD=1`; see [winograd-conv.md](design/winograd-conv.md). |
| `hip-hoist-pointwise-over-views` | func.func | Move a pointwise op whose other inputs are constant scalars above a chain of single-use element-move views over a single-use conv, matmul or gemm result, so the anchor fusion takes it into the epilogue. The anchor result may have at most 50% more elements than the op's input. |
| `hip-rocmlir-prepare-quant` | func.func | Split each static 4-bit `hip.matmul_nbits` into unpack, blocked dequantize and `hip.matmul`, then sink each per-tensor `hip.quantize_linear` back through element-move ops to its dequantizes, replace each QLinearConv / QLinearMatMul whose channels all have a requantize multiplier below 2^-33 with a constant of its output zero point, split each QLinearConv / QLinearMatMul with ui8 weights or a non-zero weight zero point into integer ops on `w - offset` and over the shifted zero point, then rebias each ui8 tree that reaches an integer anchor to i8. |
| `hip-fuse-rocmlir-elementwise` | func.func | Group the pointwise ops, transposes, static reshapes and per-tensor quantizes and dequantizes into `hip.rocmlir_elementwise` modules that an anchor can take whole. |
| `hip-rocmlir-pre-fusion` | func.func | Apply the pre-fusion patterns to main_graph's top-level block, without folding. A `hip.gather` of one constant index on static float data becomes a size-1 `tensor.extract_slice` (plus a `tensor.collapse_shape` for a scalar index) when its one reader will likely run in a rocMLIR kernel: an anchor, a module next to an anchor, a module the anchor pass makes into a kernel of its own (its `outlinesAlone` test), or a fusable view with such a reader; inside a module, one op must read it. A gather with an identical sibling is skipped. |
| `hip-fuse-rocmlir-anchor` | func.func | Plan an epilogue and a prologue of modules for each conv, matmul and gemm anchor, and outline each plan into a `rock.kernel` function (stamped with `rock.arch` from the `arch` option when set) behind `hip.rocmlir`. A module that no plan takes and that saves a launch becomes a `rock.elementwise` kernel when `standalone-elementwise` is on (the default; `HIPDNN_EP_ROCMLIR_NO_STANDALONE_ELEMENTWISE=1` turns it off in the EP), after its boundary transposes that move the innermost dimension are peeled into HIP transposes and the slice chains of its inputs are copied into it; the other modules are inlined. Decomposes the QLinear ops and dequantized convs that become integer anchors, then merges gemm-elementwise-gemm pairs. |
| `hip-inline-rocmlir-elementwise` | func.func | Put the body of each `hip.rocmlir_elementwise` back in its block. For tests and replays. |
| `hip-rocmlir-fusion-report` | module | Append one JSON line for each kernel and host function to the `report` file. |
| `convert-hip-to-tosa` | func.func | Lower the outlined kernel bodies from HIP ops to TOSA. |

Order matters: `hip-decompose-conv-transpose` and
`hip-decompose-multi-head-attention` must run **before** the fusion: rocMLIR
has no transposed-convolution anchor, and `hip.multi_head_attention` is not an
outlining anchor, so both would otherwise stay outside a `rock.kernel`.
`hip-hoist-pointwise-over-views` then moves the activation after the
interleave and crop onto the convolution's result.
`hip-rocmlir-prepare-quant` fixes the quantize trees
before the elementwise pass groups them, and the anchor pass plans its kernels
on those modules. `hip-rocmlir-pre-fusion` runs between the two, so its
patterns can tell a pointwise reader inside a module from one that stays a HIP
op. The registered pipeline runs

```
func.func(hip-decompose-conv-transpose)
func.func(hip-decompose-multi-head-attention)
func.func(hip-hoist-pointwise-over-views)
func.func(hip-rocmlir-prepare-quant)
func.func(hip-fuse-rocmlir-elementwise)
func.func(hip-rocmlir-pre-fusion)
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
