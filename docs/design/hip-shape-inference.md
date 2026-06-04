<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# HIP dialect shape inference

How HIP DPS (destination-passing-style) ops express their shape
contract — verifier, `ReifyRankedShapedTypeOpInterface` impl,
`--hip-infer-shapes` pass — and how to wire a new op into the same
machinery.

## Goals

1. **Result types tracked through the converter.** A HIP DPS op that
   declares `InferTypeOpInterface` lets the converter construct it
   without restating the result type at every callsite — the
   auto-generated `Op::create` overload reads the result type(s) from
   the `outs` operand types. This keeps the DPS contract
   `result_type == outs_operand_type` closed by construction. `hip.matmul`
   is the worked example on #260 (`autoInfer=1, declareInfer=1`); other
   DPS ops migrate in #262 when the parameterized base flips its
   defaults to opt every single-result op in.
2. **Reify dynamic dims.** When a result dim is `?` (`kDynamic`) at the
   type level but is computable in terms of operand dims, the op exposes
   that knowledge through
   `ReifyRankedShapedTypeOpInterface::reifyResultShapes`. Downstream
   `--resolve-shaped-type-result-dims` then folds `tensor.dim` of the
   result into either an `arith.constant` (static dims) or a
   `tensor.dim` of the relevant input (dynamic dims). The shared default
   on `HipDpsOpInterface` (lifts from `getDpsInits()`) covers every
   `Hip_DpsOp` whose result shape == outs shape; ops with tighter
   contracts opt out via `autoReify=0` and ship a hand-written body in
   `HipReifyResultShapesImpl.cpp`.
3. **Module-level refinement.** The `--hip-infer-shapes` pass walks the
   module, calls `reifyResultShapes` on every HIP-dialect op carrying
   the interface, and refines `?` dims in result types in place —
   including the DPS `outs` operand's `tensor.empty` producer, which is
   necessary to keep the IR well-formed. Wired into the production
   pipeline immediately after `convert-onnx-to-hip` (see
   [Pipeline placement](#pipeline-placement)).
4. **Static verification where it pays rent.** Ops with a non-trivial
   static shape contract (canonical: `hip.matmul`'s
   `[..., M, K] @ [..., K, N] -> [..., M, N]`) carry `let hasVerifier = 1`
   and a per-op `verify()` driving the same shape helper used by Reify.
   The default for new DPS ops is no verifier — the InferType-driven
   `outs` typing already closes the DPS contract, and a Reify impl that
   returns the wrong shape is caught by LIT cases under
   `--hip-infer-shapes`.

`hip.matmul` is the worked example that exercises all four layers.
Other ops compose the layers from a small helper menu chosen by shape
contract — `reifyElementwiseSameShape` (default), `reifyBroadcastShape`
(NumPy broadcast), dedicated helpers for permutation / reduction /
gather, fold-or-bail Tier-1 helpers for shape-arithmetic ops
(`reifyPadShape` etc.). The "result shape == outs shape" case is the
shared default on `HipDpsOpInterface`; no per-op helper is needed for
it. See [How to add a new op](#how-to-add-a-new-op) for the table.

## Design rationale

### Why `InferTypeOpInterface` + `ReifyRankedShapedTypeOpInterface`?

Three MLIR interfaces are relevant here:

| Interface | Static shape | Dynamic shape | We use? |
|---|---|---|---|
| `InferTypeOpInterface` | yes (`inferReturnTypes`) | no | **YES** |
| `InferShapedTypeOpInterface` | yes (`inferReturnTypeComponents`) | partial | no |
| `ReifyRankedShapedTypeOpInterface` | implicit (IntegerAttr) | yes (Value) | **YES** |

For DPS ops the result type is **tautologically tied** to the `outs`
operand type — the result type IS the outs type, so an interface has
nothing to "infer" from the inputs in the classical sense. The job
`InferTypeOpInterface` does for us is different and concrete: the
**converter** still has to construct each op, and on the legacy
`Op::create` builder path every callsite has to spell out the result
type list explicitly even though it is just the (already typed)
`outs` operand type echoed back. `inferReturnTypes` moves that
boilerplate into the op once — the auto-generated `Op::create`
overload reads the outs operand types from the operation state and
produces the result types. The auto-emitted body reads the outs SSA
value via the op's typed adaptor (named by `outsAccessor`, default
`"Output"`); if the value is a `RankedTensorType` it is pushed as the
single result type, otherwise (memref mode) zero result types are
returned — matching the rule that memref-mode DPS ops have no SSA
results because the destination operand carries the writes through
its memref descriptor. Every `OnnxToHip*Conversion.cpp` callsite then
drops its explicit type list and relies on the inference; the matmul
converter is the worked example on #260 (see
`lib/Conversion/OnnxToHip/MatMulConversion.cpp`).

### `HipDpsOpInterface` (in-dialect marker + default reify)

The two interfaces above describe **what** the op promises the rest
of MLIR. To avoid 53× per-op restatements of the "reify-from-outs"
body, the dialect adds an in-dialect marker interface,
`HipDpsOpInterface` (declared in
[include/hip/Dialect/IR/HipDpsOpInterface.td](../../include/hip/Dialect/IR/HipDpsOpInterface.td),
body in [lib/Dialect/IR/HipDpsOpInterface.cpp](../../lib/Dialect/IR/HipDpsOpInterface.cpp)).
It carries one shared default `reifyResultShapes` that walks
`getDpsInits()` and lifts each init operand's shape via
`tensor::getMixedSizes` / `memref::getMixedSizes` — the
identity-on-DPS contract.

`Hip_DpsOp` (the TableGen base class for every DPS op) auto-emits the
per-op `ReifyRankedShapedTypeOpInterface::reifyResultShapes` dispatcher
via `extraClassDefinition`, forwarding to that interface default. A
3-bit parameter set on the base controls per-op opt-out:

| Parameter | Default | Effect |
|---|---|---|
| `autoReify` | `1` | Auto-emit a per-op `reifyResultShapes` that delegates to `HipDpsOp::reifyResultShapes`. Set `0` for ops with a tighter contract (matmul) that provide a hand-written body in `HipReifyResultShapesImpl.cpp`. |
| `autoInfer` | `0` | Auto-emit a per-op `inferReturnTypes` that reads the result type from the outs operand. Set `1` per-op when the op wants the InferType-aware `Op::create` builder. Opted in: `hip.matmul` on #260; `hip.rope`, `hip.rms_norm`, `hip.qmoe`, `hip.matmul_nbits`, `hip.gemm` on #262. Multi-result Hip_DpsOps (`hip.gqa`, `hip.layer_norm`, `hip.skip_rms_norm`) keep `0` because the auto-emit body pushes a single result type. |
| `declareInfer` | `0` | Declare `InferTypeOpInterface` on the op (so converters can use the inferred-type `Op::create` overload). Set `1` in lockstep with `autoInfer`. |
| `outsAccessor` | `"Output"` (37 ops) | ODS accessor name read by the auto-emitted `inferReturnTypes` body. Pass `"Y"` for ops with `$y` outs (12 ops), `"C"` for `hip.miopen.add` (`$C`), etc. |

The shape: the interface owns the default body in a sibling `.cpp`
file; per-op dispatchers are auto-emitted by TableGen from the
dialect's DPS base class. The opt-out lever (`autoReify=0`) gives
ops with a tighter shape contract — `[..., M, K] @ [..., K, N] ->
[..., M, N]` for matmul, or value-dependent contracts like
`hip.range` — a clean escape hatch with no boilerplate. Note that
`autoReify` and `autoInfer` are orthogonal: `hip.matmul` keeps
`autoReify=0` (runtime dim recovery from operand shapes) while taking
`autoInfer=1, declareInfer=1` (construction-time result type comes
verbatim from the typed outs operand). The same split applies to other
ops with bespoke reify (`hip.gemm`, `hip.qmoe`, `hip.matmul_nbits`)
plus the shape-preserving `hip.rope` / `hip.rms_norm` on #262;
`hip.range` joins later.

`ReifyRankedShapedTypeOpInterface` covers the orthogonal *dynamic-shape*
job: per-dim `OpFoldResult`s for `--hip-infer-shapes` and the test
pass `--resolve-shaped-type-result-dims`. Static dims fall out as
`IntegerAttr`, dynamic dims fall out as `tensor.dim` (or `memref.dim`)
of the operand they depend on. The interface and the test pass are
both shipped by MLIR — the dialect plugs into them rather than
forking equivalent infrastructure.

The two interfaces compose cleanly with no overlap: InferType makes
converters terse at op-construction time, Reify makes refinement work
at module-walk time (`--hip-infer-shapes` calls reify, narrows result
types in place, rebuilds outs producers). Verifier checks slot in on
top — for ops with a non-trivial shape contract — sharing the same
static shape helper (`inferMatmulShape`) that Reify lifts into
`OpFoldResult`s.

An earlier draft added a custom `HipShapeInferenceOpInterface` with an
`inferOutputShape(...) -> SmallVector<int64_t>` static method. Dropped:
the two MLIR interfaces above already cover both jobs (static type
inference, dynamic dim reify), doubling up adds maintenance with no
payoff, and the test-pass machinery is the test surface for free.

### Why a dedicated `--hip-infer-shapes` pass?

Two related MLIR passes already exist:

* **`--reify-result-shapes`** (memref dialect, MLIR 22): only `tensor::Pad`
  / `tensor::Concat`; skips DPS ops by design.
* **`--infer-static-shapes`** (newer): handles all
  `ReifyRankedShapedTypeOpInterface` ops, but is **not in the pinned
  MLIR** and is not DPS-aware (it does not refine the `outs` operand's
  producer to keep the type chain well-formed).

`--hip-infer-shapes` extends the same module-walk pattern with the
DPS-aware producer refinement: when the outs operand is a single-use
`tensor.empty`, the pass rebuilds it with the new static shape and
drops any dyn-dim operands that became static. Shared empties (more
than one use) and producers we don't know how to refine today
(function args, other DPS ops higher in the chain) are skipped — the
pass then leaves that result index at `?`.

The pass restricts itself to **HIP-dialect ops**. Non-HIP-dialect ops
that also implement the reify interface (`tensor::EmptyOp`,
`tensor::ExtractSliceOp`, `tensor::PadOp`, …) carry per-op invariants
between operand SSA values and the result shape (e.g.
`tensor.empty(%dyn)` requires the dynamic-size operand count to equal
the number of `?` dims in the result). An in-place `result.setType()`
narrow on those would desync those invariants, and we deliberately do
not duplicate those ops' folder/canonicalizer surface here. The
canonicalizer is the right tool for non-HIP-op refinement; this pass
is for HIP DPS op result types only.

### Why DPS-aware?

In a non-DPS world the pass could simply rewrite the op's result type
and call it a day. DPS ops require `result_type == outs_operand_type`,
so refining the result without matching the outs producer triggers a
verifier error.  The simplest DPS-aware refinement we can do safely is
to refine `tensor.empty` producers: that op is a pure constructor with
trivial type semantics and its dyn-dim operand list maps 1:1 to the
result's dynamic dims.

For other producers (function args, results of an op we don't refine),
the pass bails out at that result index and leaves the type as written.

### Per-op refinement, not whole-chain

When refining op `A`'s result, the pass inserts a `tensor.cast` back to
the original type for every non-DPS-init use of the new value. That
cast is a propagation BARRIER: a downstream consumer `B` that reads `A`
through a non-DPS edge sees the OLD type at its `ins`, so `B`'s own
`reifyResultShapes` runs against the OLD operand type. `B` can still
refine its result, but only through the static dims its own helper can
deduce locally — it does **not** transitively inherit `A`'s narrowing.

This is intentional: a transitive narrowing scheme would require either
retyping the cast (cascading into every downstream signature) or
trusting that consumers tolerate a more-static `ins` (false in
general). Per-op refinement is the local, terminating, verifier-safe
choice. See `refine_chained_matmul` in
`test/lit/Dialect/hip-infer-shapes.mlir` for the canonical example.

That said, the "barrier" is leaky in a *good* way: when the consumer's
`reifyResultShapes` materialises `tensor.dim %cast, i` via
`tensor::getMixedSize` (which uses `createOrFold`),
`tensor::DimOp::fold` looks through the cast and resolves to the
underlying static size. So a chained DPS op whose operand is a cast of
an already-refined value will still recover the static dim via the dim
op, **without** giving up the cast-as-barrier on the `ins` edge type.
The chained-matmul test exercises exactly this case.

### MLIR helpers we lean on

Where the implementation could be expressed via either a bespoke helper
or a library-provided one, it consistently picks the library helper:

| Need | Helper |
|---|---|
| Constant-int extraction from `OpFoldResult` | `mlir::getConstantIntValue(OpFoldResult)` (`mlir/Dialect/Utils/StaticValueUtils.h`) |
| Re-typing a `RankedTensorType` while preserving encoding | `RankedTensorType::clone(ArrayRef<int64_t>)` (`mlir/IR/BuiltinTypeInterfaces.td` shared decl) |
| `tensor.dim` of a value at a static-or-dynamic dim | `tensor::getMixedSize(OpBuilder &, Location, Value, int64_t)` |
| IR mutation through an observable rewriter | `mlir::IRRewriter` |

This keeps the HIP pass aligned with the rest of the dialect surface:
when a future MLIR release ships a more general successor pass (e.g.
`--infer-static-shapes`), retiring `--hip-infer-shapes` should be
mostly a matter of substituting the new pass into `Pipelines.cpp` and
re-running the LIT suite.

## Component layout

```text
include/hip/Dialect/IR/
  HipShapeUtils.h          -- public API: inferMatmulShape, verifyHipOpShape,
                              reifyDimOrConstant, reifyElementwiseSameShape
                              (added on #262 for ops whose result has the
                              same shape as one designated input).
                              Per-shape Tier-1 helpers reifyBroadcastShape /
                              reifyPadShape / reifyTileShape /
                              reifySliceShape / reifyExpandShape /
                              reifyRangeShape / reifyTransposeByPerm /
                              reifyGatherWithAxis / reifyGatherND /
                              reifyReductionWithKeepdims land with the
                              broadcast-op cleanup in #263 and the
                              bespoke-op cleanup in #264.
  HipDpsOpInterface.td     -- in-dialect `HipDpsOp` interface declaration;
                              carries the shared default `reifyResultShapes`
                              that all `Hip_DpsOp`s inherit unless they opt
                              out via `autoReify=0`
  HipOps.td                -- `Hip_DpsOp` base auto-emits per-op
                              `ReifyRankedShapedTypeOpInterface` dispatchers
                              via `extraClassDefinition`, and from #260
                              onward also auto-emits per-op
                              `InferTypeOpInterface::inferReturnTypes` for
                              ops that pass `autoInfer=1, declareInfer=1`
                              (matmul on #260; rope, rms_norm, qmoe,
                              matmul_nbits, gemm on #262). Per-op defs
                              only carry the `autoReify` / `autoInfer` /
                              `declareInfer` flags they need to override
                              and `outsAccessor` when the outs SSA name
                              is not `$output`. `let hasVerifier = 1`
                              only when the op has a non-trivial shape
                              contract (matmul today).

lib/Dialect/IR/
  HipShapeUtils.cpp                -- implementations + diagnostics
  HipDpsOpInterface.cpp            -- shared default `reifyResultShapes`
                                      body (walks `getDpsInits()`, lifts each
                                      via `tensor::getMixedSizes` /
                                      `memref::getMixedSizes`)
  HipDialect.cpp                   -- per-op `verify()` (matmul only as of
                                      #260, plus `LoopOp::verify()` added in
                                      #261), `getEffects()`,
                                      `getDpsInitsMutable()`, custom builders
                                      / printers / parsers; `LoopOp::inferReturnTypes`
                                      lives here next to its peers (control-flow op,
                                      not a DPS compute op)
  HipReifyResultShapesImpl.cpp     -- per-op `reifyResultShapes()` for ops
                                      that opt out (`autoReify=0`), one
                                      `// <OpName>` section per op. Matmul
                                      is the only op in this file on #260;
                                      `hip.rope` / `hip.rms_norm` / `hip.qmoe`
                                      / `hip.matmul_nbits` / `hip.gemm` join
                                      on #262; `hip.range` / `hip.nonzero` /
                                      `hip.pad` / `hip.tile` / `hip.expand` /
                                      `hip.slice` / `hip.gather*` /
                                      `hip.reduce_*` / `hip.transpose` /
                                      `hip.layer_norm` / `hip.skip_rms_norm` /
                                      `hip.gqa` / `hip.mha` /
                                      `hip.causal_conv_with_state` /
                                      `hip.linear_attention` /
                                      `hip.hipdnn_graph` join across
                                      #263-#264.

  (No HipResultTypeInferenceImpl.cpp today: every op that declares
   `InferTypeOpInterface` so far is single-result, and the auto-emitted
   body — push the outs operand's tensor type — is correct for them.
   When a future op needs a hand-written body — most likely a
   variadic-out op like `layer_norm` / `skip_rms_norm` / `hipdnn_graph`
   that wants to declare `InferTypeOpInterface` — add a new
   `HipResultTypeInferenceImpl.cpp` next to `HipReifyResultShapesImpl.cpp`,
   wire it into `lib/Dialect/IR/CMakeLists.txt`, and follow the same
   one-`// <OpName>`-section-per-op layout.)

include/hip/Dialect/Transforms/
  Passes.td                -- `def InferShapesPass` registration

lib/Dialect/Transforms/
  InferShapesPass.cpp      -- module-level walk; DPS-aware refinement

test/lit/Dialect/
  hip-matmul-shape-verifier.mlir   -- matmul shape-contract verifier
                                       positive + negative cases
  hip-matmul-reify-shapes.mlir     -- driven by the
                                       `--resolve-shaped-type-result-dims`
                                       test pass
  hip-loop-verifier.mlir           -- `hip.loop` v_init / result type
                                       contract (added by #261)
  hip-infer-shapes.mlir            -- consolidated `--hip-infer-shapes`
                                       LIT, one section per op rolled out
                                       across #260 - #264
```

### Why the per-interface split

`reifyResultShapes` lives in its own translation unit because it grows
on a different schedule from the rest of an op's machinery. Verifiers,
`getEffects`, `getDpsInitsMutable`, and op-local builders touch the op
once and rarely need maintenance after that. Reify implementations are
non-trivial shape arithmetic that benefits from being read alongside
each other (every dynamic-shape op solves a small variant of the same
"lift static dims to `IntegerAttr`, dynamic dims to `tensor.dim` of
the right operand" problem). The file follows a one-`// <OpName>`-
section-per-op convention.

`inferReturnTypes` does NOT have its own translation unit today: the
auto-emitted body in `Hip_DpsOp::extraClassDefinition` covers every op
that opts in (single-result, result type == outs operand type). When a
future op (most likely a variadic-out one) needs a hand-written body,
add `HipResultTypeInferenceImpl.cpp` next to `HipReifyResultShapesImpl.cpp`
following the same per-op-section layout.

The interface impls are **member functions, not external models** —
`attachInterface` / `ExternalModel<>` is only needed when implementing
an interface from *outside* the op's owning dialect. Inside the HIP
dialect we own both the op and the interface impl, so a normal
class-member definition in `HipReifyResultShapesImpl.cpp` is the right
shape (and the same shape applies to a future
`HipResultTypeInferenceImpl.cpp` when it lands).

When a future interface accumulates enough impls to deserve the same
treatment (e.g. a dialect-defined op interface, a tiling interface, or
a bufferization interface), follow the same pattern: a new
`Hip<InterfaceName>Impl.cpp` next to the existing files, sectioned by
op name, listed in `lib/Dialect/IR/CMakeLists.txt`.

## Pipeline placement

`--hip-infer-shapes` runs on `mlir::ModuleOp`, immediately after
`convert-onnx-to-hip` and before `one-shot-bufferize`:

```text
simplify-onnx -> hip-add-context-arg -> convert-onnx-to-hip
              -> hip-infer-shapes
              -> one-shot-bufferize -> ...
```

Wired in `lib/Dialect/Transforms/Pipelines.cpp` at the top of
`buildOnnxToHipPipelineTail` (the shared post-conversion segment used
by both `buildOnnxToHipPipeline` overloads), so every production
compile path gets it.

Refining tensor result types BEFORE bufferize is what makes the pass
pay rent: static dims then propagate through the alloc / pool sizing
computations, and bufferize stops emitting `memref.dim` ops at the top
of the function. The pass is idempotent (a second invocation that
finds no further refinements is a no-op) and a no-op on functions
whose ops either don't carry the interface or expose no further
refinable dims — safe to run unconditionally regardless of how much
HIP-op coverage the interface currently has. As more
HIP DPS ops (GQA, RmsNorm, SkipRmsNorm, Attention, RotaryEmbedding,
…) gain reify impls, the wiring lets each new op participate without
any change to `Pipelines.cpp`.

## How to add a new op

For most new HIP DPS ops the recipe collapses to **two** edits:
(1) declare the op via `Hip_DpsOp` in TableGen and (2) add a LIT case.
The `Hip_DpsOp` base auto-emits a working `reifyResultShapes` so the op
participates in dynamic-shape refinement without per-op `.cpp`
boilerplate. Construction-time `inferReturnTypes` is also auto-emitted
when the op opts in via `autoInfer=1, declareInfer=1` (matmul on #260;
five transformer-block ops on #262: `hip.rope`, `hip.rms_norm`,
`hip.qmoe`, `hip.matmul_nbits`, `hip.gemm`). Verifier and per-op
overrides remain optional escape hatches when the op has a shape
contract tighter than "result_type == outs_operand_type".

**Stack rollout context.** Infrastructure lands in #260: the in-dialect
`HipDpsOpInterface`, the parameterized `Hip_DpsOp` base, and matmul as
the dual worked example — opt out of the default reify
(`autoReify=0`, with a hand-written `MatmulOp::reifyResultShapes` that
recovers M/K/N from operand shapes) AND opt in to auto-emitted
inferReturnTypes (`autoInfer=1, declareInfer=1`, with the converter
callsite dropped to the inferred-type `Op::create` overload). #261
adds `InferTypeOpInterface` to `hip.loop` (region op, not DPS). #262
opts five transformer-block ops in to auto-emitted `inferReturnTypes`
(`hip.rope`, `hip.rms_norm`, `hip.qmoe`, `hip.matmul_nbits`,
`hip.gemm`) and gives them hand-written `reifyResultShapes`, so
`--hip-infer-shapes` refines through a typical decoder block end-to-end.
#263 introduces `Hip_DpsOp_Broadcast` and migrates 23 ops. #264 cleans
up the remaining bespoke ops.

### 1. Declare the op in TableGen

In [HipOps.td](../../include/hip/Dialect/IR/HipOps.td):

```tablegen
def Hip_MyOp : Hip_DpsOp<"my_op"> {
  // ... arguments, assemblyFormat ...
}
```

`Hip_DpsOp` declares `MemoryEffectsOpInterface`,
`DestinationStyleOpInterface`, `HipDpsOpInterface`, and
`ReifyRankedShapedTypeOpInterface` (with an auto-emitted body that
forwards to the shared default), and emits the
`Variadic<AnyRankedTensor>:$result_tensors` result for tensor mode.
Pass `outsAccessor` when the DPS init operand SSA name is not
`$output` (common for unary elementwise ops that use `$y`):

```tablegen
def Hip_SigmoidOp : Hip_DpsOp<"sigmoid", /*traits=*/[],
                              /*outsAccessor=*/"Y"> { /* ... */ }
```

Add `let hasVerifier = 1;` only if the op has a non-trivial static
shape contract worth verifying (matmul-style; default for new ops is
no verifier — see step 4).

### 2. (Optional) Per-op `reifyResultShapes` override

Skip this step unless the op has a shape contract tighter than
"result_type == outs_operand_type". The `Hip_DpsOp` base's
auto-emitted body forwards to `HipDpsOp::reifyResultShapes` (in
[lib/Dialect/IR/HipDpsOpInterface.cpp](../../lib/Dialect/IR/HipDpsOpInterface.cpp)),
which walks `getDpsInits()` and lifts each via
`tensor::getMixedSizes` / `memref::getMixedSizes`. For most ops this
is exactly the right behavior: result shape == outs shape by DPS
contract, and `--hip-infer-shapes` will narrow `?` dims in the result
type from any static dims the outs operand carries.

If the op has a tighter contract — matmul-style
`[..., M, K] @ [..., K, N] -> [..., M, N]`, or value-dependent
contracts like `hip.range` — opt out via `autoReify=0` and add a
`// MyOp` section banner at the bottom of
[HipReifyResultShapesImpl.cpp](../../lib/Dialect/IR/HipReifyResultShapesImpl.cpp).
Pick the reify helper that matches the op's shape contract:

| Op shape contract | Helper | Lands in |
|---|---|---|
| Result shape == outs operand shape (default) | (no helper — interface default in `HipDpsOpInterface.cpp`) | #260 |
| Result shape == named INPUT operand shape (e.g. rope, rms_norm, qmoe) | hand-written body calling `reifyElementwiseSameShape(b, loc, getInput())` | #262 |
| NumPy broadcast (add, mul, where, ...) | `Hip_DpsOp_Broadcast<[...]>` sub-base + `reifyBroadcastShape` | #263 |
| Permutation (`hip.transpose`) | `reifyTransposeByPerm` | #264 |
| Reduction with `keepdims` (reduce_sum, reduce_max) | `reifyReductionWithKeepdims` | #264 |
| Gather along an axis (`hip.gather`, `hip.gather_nd`) | `reifyGatherWithAxis` / `reifyGatherND` | #264 |
| Multi-init outs-lifting (gqa, mha, layer_norm, ...) | hand-written body in `HipReifyResultShapesImpl.cpp` walking `getDpsInits()` | #263-#264 |
| Fold-or-bail shape arithmetic (pad, tile, slice, expand, range) | one of `reifyPadShape` / `reifyTileShape` / `reifySliceShape` / `reifyExpandShape` / `reifyRangeShape` (return `failure()` on non-foldable; falls through to outs-lifting) | #264 |
| Matmul-shaped (`[..., M, K] @ [..., K, N] -> [..., M, N]`) | call `inferMatmulShape` to compute output extents, then `reifyDimOrConstant` per dim | #260 (matmul) |

If your op's contract is not in the table, write a new helper in
`HipShapeUtils.{h,cpp}` and add a row.

`reifyDimOrConstant` (used by the matmul-shaped row) returns
`IntegerAttr` when the dim is static and a `tensor.dim` / `memref.dim`
otherwise.

The interface declarations are already in TableGen (step 1), so no
header / `attachInterface` changes are needed — the linker resolves
each member function to whichever `.cpp` file defines it.

### 3. Data-dependent shape contracts (`hip.range`, `hip.nonzero`, ...)

Some ops have output dims that depend on operand **values** (not
operand types). `hip.range(start, end, step)` produces a 1D result
whose extent is `(end - start) / step`; `hip.nonzero(input)` produces
a 2D result whose first dim is the runtime count of nonzero entries
in the input.

The "fold-or-bail" Tier-1 helpers
(`reifyRangeShape` etc.) handle this in two cases:

1. **Constant value operands** — the helper folds `(end - start)` /
   `step` into an `IntegerAttr` and refines the result dim
   statically.
2. **Non-foldable** — the helper returns `failure()`. The dispatcher
   thunk then falls through to outs-lifting (the default), which
   reads the dim from the outs operand. If the outs operand was
   constructed via `tensor.empty(%dyn)` with the runtime extent as
   a dynamic-size operand, downstream consumers see a `tensor.dim`
   on the result that folds back to that operand.

Ops where the runtime extent has no SSA representation at all (e.g.
`hip.nonzero` whose count cannot be expressed before running the
kernel) leave the dim as `?` — the only honest answer.

### 4. Converter callsites (inferred-type `Op::create`)

For ops with `declareInfer=1, autoInfer=1` (matmul on #260; five
transformer-block ops on #262: `hip.rope`, `hip.rms_norm`, `hip.qmoe`,
`hip.matmul_nbits`, `hip.gemm`), drop the explicit `resultType`
argument from the `Op::create` callsite in
`lib/Conversion/OnnxToHip/<MyOp>Conversion.cpp`:

```cpp
// Before:
hip::MyOp::create(b, loc, /*resultType=*/outsType, ctx, lhs, rhs, outs);
// After (auto-emitted body reads outs.getType()):
hip::MyOp::create(b, loc, ctx, lhs, rhs, outs);
```

The InferType-aware `Op::create` overload is auto-generated by ODS
once `InferTypeOpInterface` is declared. Existing callers that still
pass an explicit type list keep working — the migration is mechanical
and per-op. See `lib/Conversion/OnnxToHip/MatMulConversion.cpp` for
the worked example on #260.

### 5. (Optional) Verifier in `HipDialect.cpp`

Only for ops with a non-trivial static shape contract — `hip.matmul`
is the only such op as of #264. Implement `verify()` under the
`// MyOp` section banner alongside the existing
`getDpsInitsMutable()` / `getEffects()`:

```cpp
LogicalResult MyOp::verify() {
  // Cross-cutting DPS contract check first.
  if (failed(verifyDpsComputeOp(*this, {getA(), getB(), getOutput()},
                                /*numInits=*/1)))
    return failure();

  // Static shape check via the shared helper. Empty result vector means
  // the helper already issued a diagnostic.
  return mlir::hip::verifyHipOpShape(
      *this, [&]() -> SmallVector<SmallVector<int64_t>> {
        SmallVector<int64_t> outShape = mlir::hip::inferMatmulShape(
            getShapeOf(getA()), getShapeOf(getB()),
            [&]() { return this->emitOpError(); });
        if (outShape.empty())
          return {};
        return {std::move(outShape)};
      });
}
```

Most new ops do **not** need this. The InferType-driven outs typing
already closes the DPS contract by construction, and a Reify impl
that returns the wrong shape is caught by LIT cases under
`--hip-infer-shapes` (step 6).

### 6. LIT coverage

Append cases to
[test/lit/Dialect/hip-infer-shapes.mlir](../../test/lit/Dialect/hip-infer-shapes.mlir),
one per shape pattern your op exercises (typically one static, one
dynamic; for fold-or-bail Tier-1 ops also one non-foldable case with
`CHECK-NOT: arith.subi` / `CHECK-NOT: arith.divsi` to guard the
no-IR-bloat fallback contract). Per-op verifier / reify files
(`hip-<op>-shape-verifier.mlir`, `hip-<op>-reify-shapes.mlir`) are
reserved for ops complex enough to deserve their own file (matmul has
both today; `hip.loop` has its own `hip-loop-verifier.mlir` for the
v_init / result-type contract added by #261).

## Open extensions

The recipe in [How to add a new op](#how-to-add-a-new-op) is the canonical
guide for adding shape inference to additional HIP DPS ops. Per-op rollout,
dialect generalisation, and backward dim-origin tracing across `func.return`
are tracked outside this document.

### `hip.matmul` rejects rank-1 operands

The verifier requires both `A` and `B` to have rank >= 2. ONNX `MatMul`
permits rank-1 operands (vector-matrix / matrix-vector) via implicit
unit-dim promotion and result unit-dim stripping. None of the currently
supported transformer models exercise this case. For future models that
do, the fix belongs in `lib/Conversion/OnnxToHip/MatMulConversion.cpp`:
insert a `tensor.expand_shape` to rank-2 before constructing `hip.matmul`,
then `tensor.collapse_shape` the result. Lowering (`MatmulLowering.cpp`)
and runtime assume rank >= 2 today (indexing `aShape[rank-2]` and
`bShape[rank-1]`); the verifier turns what would have been a
silently-wrong hipBLASLt call into an explicit diagnostic.
