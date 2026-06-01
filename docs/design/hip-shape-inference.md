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

1. **Result types tracked through the converter.** Every HIP DPS op
   declares `InferTypeOpInterface` so the converter can construct ops
   without restating the result type at every callsite — the
   auto-generated `Op::create` overload reads the result type(s) from
   the `outs` operand types via `appendDpsResultIfTensor`. This keeps
   the DPS contract `result_type == outs_operand_type` closed by
   construction.
2. **Reify dynamic dims.** When a result dim is `?` (`kDynamic`) at the
   type level but is computable in terms of operand dims, the op exposes
   that knowledge through upstream
   `ReifyRankedShapedTypeOpInterface::reifyResultShapes`. Downstream
   `--resolve-shaped-type-result-dims` then folds `tensor.dim` of the
   result into either an `arith.constant` (static dims) or a
   `tensor.dim` of the relevant input (dynamic dims).
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
(`reifyPadShape` etc.), and the generic `reifyResultsFromDpsInits` for
multi-result outs-lifting. See [How to add a new op](#how-to-add-a-new-op)
for the table.

## Design rationale

### Why `InferTypeOpInterface` + `ReifyRankedShapedTypeOpInterface`?

Upstream MLIR has three relevant interfaces:

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
produces the result types via `appendDpsResultIfTensor`. Every
`OnnxToHip*Conversion.cpp` callsite then drops its explicit type
list and relies on the inference. Memref-mode ops produce zero result
types — the destination operand carries the writes via its memref
descriptor and the SSA result list is empty.

`ReifyRankedShapedTypeOpInterface` covers the orthogonal *dynamic-shape*
job: per-dim `OpFoldResult`s for `--hip-infer-shapes` and the upstream
test pass `--resolve-shaped-type-result-dims`. Static dims fall out as
`IntegerAttr`, dynamic dims fall out as `tensor.dim` (or `memref.dim`)
of the operand they depend on. Linalg's named ops use exactly this
pattern; we reuse both the interface and the LLVM-project test pass so
our infrastructure does not fork.

The two interfaces compose cleanly with no overlap: InferType makes
converters terse at op-construction time, Reify makes refinement work
at module-walk time (`--hip-infer-shapes` calls reify, narrows result
types in place, rebuilds outs producers). Verifier checks slot in on
top — for ops with a non-trivial shape contract — sharing the same
static shape helper (`inferMatmulShape`) that Reify lifts into
`OpFoldResult`s.

An earlier draft added a custom `HipShapeInferenceOpInterface` with an
`inferOutputShape(...) -> SmallVector<int64_t>` static method. Dropped:
the two upstream interfaces above already cover both jobs (static type
inference, dynamic dim reify), doubling up adds maintenance with no
payoff, and the upstream test-pass machinery is the test surface for
free.

### Why a dedicated `--hip-infer-shapes` pass?

Upstream MLIR ships two related passes:

* **`--reify-result-shapes`** (memref dialect, MLIR 22): only `tensor::Pad`
  / `tensor::Concat`; skips DPS ops by design.
* **`--infer-static-shapes`** (newer): handles all
  `ReifyRankedShapedTypeOpInterface` ops, but is **not in the pinned
  MLIR** and is not DPS-aware (it does not refine the `outs` operand's
  producer to keep the type chain well-formed).

`--hip-infer-shapes` adapts the same pattern as upstream but adds the
DPS-aware producer refinement: when the outs operand is a `tensor.empty`,
the pass rebuilds it with the new static shape and drops any dyn-dim
operands that became static. Producers we don't know how to refine
today (function args, other DPS ops higher in the chain) are skipped —
the pass then leaves that result index at `?`.

The pass restricts itself to **HIP-dialect ops**. Upstream ops that
also implement the reify interface (`tensor::EmptyOp`,
`tensor::ExtractSliceOp`, `tensor::PadOp`, …) carry per-op invariants
between operand SSA values and the result shape (e.g.
`tensor.empty(%dyn)` requires the dynamic-size operand count to equal
the number of `?` dims in the result). An in-place `result.setType()`
narrow on those would desync those invariants, and we deliberately do
not duplicate every upstream op's folder/canonicalizer surface here.
The canonicalizer is the right tool for upstream-op refinement; this
pass is for HIP DPS op result types only.

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
`tensor::getMixedSize` (which uses `createOrFold`), upstream
`tensor::DimOp::fold` looks through the cast and resolves to the
underlying static size. So a chained DPS op whose operand is a cast of
an already-refined value will still recover the static dim via the dim
op, **without** giving up the cast-as-barrier on the `ins` edge type.
The chained-matmul test exercises exactly this case.

### Upstream helpers we lean on

Where the implementation could be expressed via either a bespoke helper
or an upstream one, it consistently picks upstream:

| Need | Upstream helper |
|---|---|
| Constant-int extraction from `OpFoldResult` | `mlir::getConstantIntValue(OpFoldResult)` (`mlir/Dialect/Utils/StaticValueUtils.h`) |
| Re-typing a `RankedTensorType` while preserving encoding | `RankedTensorType::clone(ArrayRef<int64_t>)` (`mlir/IR/BuiltinTypeInterfaces.td` shared decl) |
| `tensor.dim` of a value at a static-or-dynamic dim | `tensor::getMixedSize(OpBuilder &, Location, Value, int64_t)` |
| IR mutation through an observable rewriter | `mlir::IRRewriter` (matches upstream `--reify-result-shapes`) |

This keeps drift between the HIP pass and upstream `--reify-result-shapes`
to a minimum: when MLIR upgrades drop a more general successor (e.g.
`--infer-static-shapes`), retiring our pass should be mostly a matter of
substituting it into `Pipelines.cpp` and re-running the LIT suite.

## Component layout

```text
include/hip/Dialect/IR/
  HipShapeUtils.h          -- public API: inferMatmulShape,
                              verifyHipOpShape, appendDpsResultIfTensor,
                              reifyDimOrConstant, reifyElementwiseSameShape,
                              reifyBroadcastShape, reifyResultsFromDpsInits,
                              and per-shape Tier-1 helpers (reifyPadShape,
                              reifyTileShape, reifySliceShape,
                              reifyExpandShape, reifyRangeShape,
                              reifyTransposeByPerm, reifyGatherWithAxis,
                              reifyGatherND, reifyReductionWithKeepdims)
  HipOps.td                -- per-op declares
                              `DeclareOpInterfaceMethods<InferTypeOpInterface, ["inferReturnTypes"]>`
                              and `DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface>`;
                              `let hasVerifier = 1` only when the op has a
                              non-trivial shape contract (matmul today)

lib/Dialect/IR/
  HipShapeUtils.cpp                -- implementations + diagnostics
  HipDialect.cpp                   -- per-op `verify()` (matmul only as of
                                      #264, plus `LoopOp::verify()`),
                                      `getEffects()`, `getDpsInitsMutable()`,
                                      custom builders / printers / parsers;
                                      `LoopOp::inferReturnTypes` lives here
                                      next to its peers (control-flow op,
                                      not a DPS compute op)
  HipResultTypeInferenceImpl.cpp   -- per-op `inferReturnTypes()`,
                                      one `// <OpName>` section per op
  HipReifyResultShapesImpl.cpp     -- per-op `reifyResultShapes()`,
                                      one `// <OpName>` section per op

include/hip/Dialect/Transforms/
  Passes.td                -- `def InferShapesPass` registration

lib/Dialect/Transforms/
  InferShapesPass.cpp      -- module-level walk; DPS-aware refinement

test/lit/Dialect/
  hip-matmul-shape-verifier.mlir   -- matmul shape-contract verifier
                                       positive + negative cases
  hip-matmul-reify-shapes.mlir     -- driven by upstream
                                       `--resolve-shaped-type-result-dims`
  hip-loop-verifier.mlir           -- `hip.loop` v_init / result type
                                       contract (added by #261)
  hip-infer-shapes.mlir            -- consolidated `--hip-infer-shapes`
                                       LIT, one section per op rolled out
                                       across #260 - #264
```

### Why the per-interface split

`reifyResultShapes` and `inferReturnTypes` each live in their own
translation unit because they grow on a different schedule from the
rest of an op's machinery. Verifiers, `getEffects`,
`getDpsInitsMutable`, and op-local builders touch the op once and
rarely need maintenance after that. Reify implementations are
non-trivial shape arithmetic that benefits from being read alongside
each other (every dynamic-shape op solves a small variant of the same
"lift static dims to `IntegerAttr`, dynamic dims to `tensor.dim` of
the right operand" problem). InferType implementations are typically
mechanical (`appendDpsResultIfTensor` chained over outs operands) but
multiply across every DPS op — keeping them together makes the "every
op uniformly delegates to the same helper" pattern obvious. Both files
follow the same one-`// <OpName>`-section-per-op convention.

This mirrors IREE's LinalgExt convention (`LinalgExtOps.cpp` keeps
verifiers / builders; `TilingInterfaceImpl.cpp` and
`AggregatedOpInterfaceImpl.cpp` keep per-interface impls), which in
turn descends from the same upstream split between
`mlir/lib/Dialect/Linalg/IR/LinalgOps.cpp` (op identity) and the
various interface-impl `.cpp` files under
`mlir/lib/Dialect/Linalg/Transforms/`. The interface impls are
**member functions, not external models** — `attachInterface` /
`ExternalModel<>` is only needed when implementing an interface from
*outside* the op's owning dialect.

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

A new HIP DPS op participates in shape inference in **four** small
edits (TableGen + InferType + Reify + LIT), plus an **optional**
verifier when the op has a non-trivial static shape contract. The
matmul wiring is the canonical full-stack reference; for the rest of
the dialect the InferType + Reify path (no verifier) is the realized
pattern across #262 - #264.

**Stack rollout context.** Infrastructure lands in #260 with `hip.matmul`
as the worked example. #261 adds `InferTypeOpInterface` to `hip.loop`.
#262 - #264 roll out the InferType + Reify pattern across 51 more DPS
ops, picking from a small menu of reify helpers chosen by op shape
contract — see step 3.

### 1. Declare the two interfaces in TableGen

In [HipOps.td](../../include/hip/Dialect/IR/HipOps.td):

```tablegen
def Hip_MyOp : Hip_DpsOp<"my_op", [
    DeclareOpInterfaceMethods<InferTypeOpInterface, ["inferReturnTypes"]>,
    DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface>
  ]> {
  // ... arguments, assemblyFormat ...
}
```

`Hip_DpsOp` already declares `DestinationStyleOpInterface` and emits the
`Variadic<AnyRankedTensor>:$result_tensors` result for tensor mode. Add
`let hasVerifier = 1;` only if the op has a non-trivial static shape
contract worth verifying (matmul-style; default for new ops is no
verifier — see step 5).

### 2. Implement `inferReturnTypes()` in `HipResultTypeInferenceImpl.cpp`

Add a `// MyOp` section banner at the bottom of
[HipResultTypeInferenceImpl.cpp](../../lib/Dialect/IR/HipResultTypeInferenceImpl.cpp).
Single-result ops are typically a one-liner over the outs operand:

```cpp
LogicalResult MyOp::inferReturnTypes(
    MLIRContext *ctx, std::optional<Location> loc, MyOp::Adaptor adaptor,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  appendDpsResultIfTensor(adaptor.getOutput(), inferredReturnTypes);
  return success();
}
```

For multi-result ops, chain `appendDpsResultIfTensor` per outs operand;
when the op carries `AttrSizedOperandSegments` and you would otherwise
duplicate the chain, delegate to the generic helper used by
`hip.gqa` / `hip.layer_norm` / `hip.multi_head_attention`:

```cpp
return reifyResultsFromDpsInits(*this, b, reifiedReturnShapes);
```

Memref-mode ops produce zero result types — `appendDpsResultIfTensor`
is a no-op when the operand is a memref, so the same impl works for
both modes.

### 3. Implement `reifyResultShapes()` in `HipReifyResultShapesImpl.cpp`

Add a `// MyOp` section banner at the bottom of
[HipReifyResultShapesImpl.cpp](../../lib/Dialect/IR/HipReifyResultShapesImpl.cpp).
Pick the reify helper that matches the op's shape contract:

| Op shape contract | Helper |
|---|---|
| Result shape == operand shape (silu, sigmoid, cast, ...) | `reifyElementwiseSameShape` |
| NumPy broadcast (add, mul, where, ...) | `reifyBroadcastShape` |
| Permutation (`hip.transpose`) | `reifyTransposeByPerm` |
| Reduction with `keepdims` (reduce_sum, reduce_max, ...) | `reifyReductionWithKeepdims` |
| Gather along an axis (`hip.gather`, `hip.gather_nd`) | `reifyGatherWithAxis` / `reifyGatherND` |
| Multi-result outs-lifting (gqa, attention, layer_norm, ...) | `reifyResultsFromDpsInits` (mirrors IREE's `LinalgExtOp::reifyResultShapes` default) |
| Fold-or-bail shape arithmetic (pad, tile, slice, expand, range) | one of `reifyPadShape` / `reifyTileShape` / `reifySliceShape` / `reifyExpandShape` / `reifyRangeShape` (return `failure()` on non-foldable; the thunk falls through to outs-lifting) |
| Matmul-shaped (`[..., M, K] @ [..., K, N] -> [..., M, N]`) | call `inferMatmulShape` to compute output extents, then `reifyDimOrConstant` per dim |

If your op's contract is not in the table, write a new helper in
`HipShapeUtils.{h,cpp}` and add a row.

`reifyDimOrConstant` (used by the matmul-shaped row) returns
`IntegerAttr` when the dim is static and a `tensor.dim` / `memref.dim`
otherwise, mirroring upstream `linalg::createFoldedDimOp`.

The interface declarations are already in TableGen (step 1), so no
header / `attachInterface` changes are needed — the linker resolves
each member function to whichever `.cpp` file defines it.

### 4. Migrate converter callsites

Drop the explicit `resultType` argument from the `Op::create` callsite
in `lib/Conversion/OnnxToHip/<MyOp>Conversion.cpp`:

```cpp
// Before:
hip::MyOp::create(b, loc, /*resultType=*/outsType, ctx, lhs, rhs, outs);
// After:
hip::MyOp::create(b, loc, ctx, lhs, rhs, outs);
```

The InferType-aware `Op::create` overload is auto-generated by ODS
once `InferTypeOpInterface` is declared in step 1. Existing callers
that still pass an explicit type list keep working — the migration is
mechanical and per-op.

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
