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

1. **Static verification.** Every HIP DPS op verifies that its `outs`
   operand types are compatible with the shapes that fall out of the op
   contract (matmul: `[..., M, K] @ [..., K, N] -> [..., M, N]`;
   elementwise: NumPy broadcast over the inputs; etc).
2. **Reify dynamic dims.** When a result dim is `?` (`kDynamic`) at the
   type level but is computable in terms of operand dims, the op exposes
   that knowledge through upstream
   `ReifyRankedShapedTypeOpInterface::reifyResultShapes`.  Downstream
   `--resolve-shaped-type-result-dims` then folds `tensor.dim` of the
   result into either an `arith.constant` (static dims) or a
   `tensor.dim` of the relevant input (dynamic dims).
3. **Module-level refinement.** The `--hip-infer-shapes` pass walks the
   module, calls `reifyResultShapes` on every HIP-dialect op carrying
   the interface, and refines `?` dims in result types in place —
   including the DPS `outs` operand's `tensor.empty` producer, which is
   necessary to keep the IR well-formed (DPS contract:
   `result_type == outs_operand_type`). Wired into the production
   pipeline immediately after `convert-onnx-to-hip` (see
   [Pipeline placement](#pipeline-placement)).

These three layers share **one** static shape function per op
(`inferMatmulShape`, future `inferBroadcastShape`, ...). The verifier
checks the result against the helper; reify lifts the helper's output
into `OpFoldResult`s; the pass aggregates reify across the module.

## Design rationale

### Why `ReifyRankedShapedTypeOpInterface` and not a custom interface?

Upstream MLIR has three relevant interfaces:

| Interface | Static shape | Dynamic shape | Used by |
|---|---|---|---|
| `InferTypeOpInterface` | yes (`inferReturnTypes`) | no | Type inference at op build time |
| `InferShapedTypeOpInterface` | yes (`inferReturnTypeComponents`) | partial | Some ops (e.g. mhlo) |
| `ReifyRankedShapedTypeOpInterface` | implicit (IntegerAttr) | yes (Value) | linalg, tensor::Pad, tensor::Concat |

For DPS ops the result type is **tautologically tied** to the `outs`
operand type — you can't "infer" the result type from the inputs because
the result type IS the outs type. `InferTypeOpInterface` is therefore
not a fit: there's nothing for it to infer.

`ReifyRankedShapedTypeOpInterface` neatly steps around this: it lets the
op describe its result *shape* (per-dim `OpFoldResult`) without trying to
synthesise a result *type*. Static dims fall out as `IntegerAttr`,
dynamic dims fall out as `tensor.dim` (or `memref.dim`) of the operand
they depend on. Linalg's named ops use exactly this pattern; we reuse
both the interface and the LLVM-project test pass that drives it
(`--resolve-shaped-type-result-dims`) so our infrastructure does not
fork.

An earlier draft added a custom `HipShapeInferenceOpInterface` with an
`inferOutputShape(...) -> SmallVector<int64_t>` static method. Dropped:
the upstream interface and the static helper already cover both jobs
(static check, dynamic reify), doubling up adds maintenance with no
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
                              verifyHipOpShape, reifyDimOrConstant
  HipOps.td                -- per-op declares
                              `DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface>`
                              and `let hasVerifier = 1`

lib/Dialect/IR/
  HipShapeUtils.cpp                -- implementations + diagnostics
  HipDialect.cpp                   -- per-op `verify()`, `getEffects()`,
                                      `getDpsInitsMutable()`, custom
                                      builders / printers / parsers
  HipReifyResultShapesImpl.cpp     -- per-op `reifyResultShapes()`,
                                      one `// <OpName>` section per op

include/hip/Dialect/Transforms/
  Passes.td                -- `def InferShapesPass` registration

lib/Dialect/Transforms/
  InferShapesPass.cpp      -- module-level walk; DPS-aware refinement

test/lit/Dialect/
  hip-matmul-shape-verifier.mlir   -- verifier positive + negative
  hip-matmul-reify-shapes.mlir     -- driven by upstream
                                       `--resolve-shaped-type-result-dims`
  hip-infer-shapes.mlir            -- propagation pass behaviour
```

### Why the per-interface split

`reifyResultShapes` lives in its own translation unit because it grows
on a different schedule from the rest of an op's machinery. Verifiers,
`getEffects`, `getDpsInitsMutable`, and op-local builders touch the op
once and rarely need maintenance after that. Reify implementations are
non-trivial shape arithmetic that benefits from being read alongside
each other (every dynamic-shape op solves a small variant of the same
"lift static dims to `IntegerAttr`, dynamic dims to `tensor.dim` of the
right operand" problem). Keeping them together — and out of the
already-large `HipDialect.cpp` — makes both files easier to navigate as
the dialect picks up shape-inference support for more ops.

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

A new HIP op participates in shape inference in **five** small edits,
spread across the three files identified in the layout above. The
matmul wiring is the canonical reference; the same pattern applies to
every DPS compute op.

### 1. Add a shape helper (or reuse one)

In [HipShapeUtils.h](../../include/hip/Dialect/IR/HipShapeUtils.h), add a free
function that takes the operand shapes (and any op-specific attrs) and
returns the expected output shape vector. Use `ShapedType::kDynamic` for
dims that the helper cannot compute, and emit a diagnostic via the
supplied `emitError` callable on shape mismatch.

```cpp
// HipShapeUtils.h
SmallVector<int64_t> inferBroadcastShape(
    ArrayRef<int64_t> aShape, ArrayRef<int64_t> bShape,
    function_ref<InFlightDiagnostic()> emitError);
```

If your op fits an existing helper (e.g. another matmul-shaped op, or
another op with the same broadcast rules), just reuse it.

### 2. Declare the interface in TableGen

In [HipOps.td](../../include/hip/Dialect/IR/HipOps.td):

```tablegen
def Hip_MyOp : Hip_DpsOp<"my_op", [
    DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface>
  ]> {
  // ... arguments, assemblyFormat ...
  let hasVerifier = 1;
}
```

`Hip_DpsOp` already declares `DestinationStyleOpInterface` and emits the
`Variadic<AnyRankedTensor>:$result_tensors` result for tensor mode.

### 3. Implement `verify()` in `HipDialect.cpp`

In [HipDialect.cpp](../../lib/Dialect/IR/HipDialect.cpp), under the
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
        SmallVector<int64_t> outShape = mlir::hip::broadcastShapes(
            getShapeOf(getA()), getShapeOf(getB()),
            [&]() { return this->emitOpError(); });
        if (outShape.empty())
          return {};
        return {std::move(outShape)};
      });
}
```

### 4. Implement `reifyResultShapes()` in `HipReifyResultShapesImpl.cpp`

In [HipReifyResultShapesImpl.cpp](../../lib/Dialect/IR/HipReifyResultShapesImpl.cpp),
add a new `// MyOp` section banner at the bottom of the file and place
the impl underneath it:

```cpp
//===----------------------------------------------------------------------===//
// MyOp
//===----------------------------------------------------------------------===//

LogicalResult MyOp::reifyResultShapes(
    OpBuilder &b, ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  if (getNumResults() == 0)
    return failure(); // memref mode -- no tensor result
  // ... compute outShape via the same helper used in verify(), then for
  //     each dim: reifyDimOrConstant(b, loc, outShape[d], operand,
  //                                  operandDimIdx)
  // ... reifiedReturnShapes.assign({std::move(dims)});
  return success();
}
```

`reifyDimOrConstant` returns `IntegerAttr` when the dim is static and a
`tensor.dim` / `memref.dim` otherwise, mirroring upstream
`linalg::createFoldedDimOp`.

The interface declaration is already in TableGen (step 2), so no
header / `attachInterface` changes are needed — the linker resolves the
member function to whichever `.cpp` file defines it.

### 5. (Optional) LIT coverage

Add three small LIT tests under `test/lit/Dialect/`, modelled on the
matmul ones:

* `hip-<op>-shape-verifier.mlir`: positive + negative shape checks via
  `--verify-diagnostics`.
* `hip-<op>-reify-shapes.mlir`: drive `--resolve-shaped-type-result-dims`
  and FileCheck that `tensor.dim` of the op result folds correctly.
* (Optional) extend `test/lit/Dialect/hip-infer-shapes.mlir` with a case
  that exercises `--hip-infer-shapes` on a chain involving the new op.

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
