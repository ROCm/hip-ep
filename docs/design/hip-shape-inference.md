<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# HIP dialect shape inference

This document explains how the HIP dialect represents and refines the
shapes of its DPS (destination-passing-style) ops, and how to wire a new
op into the same machinery.

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
3. **Module-level refinement.** The optional `--hip-infer-shapes` pass
   walks the module, calls `reifyResultShapes` on every op, and refines
   `?` dims in result types in place — including the DPS `outs` operand's
   `tensor.empty` producer, which is necessary to keep the IR
   well-formed (DPS contract: `result_type == outs_operand_type`).

These three layers share **one** static shape function per op
(`inferContractionShape`, future `broadcastShapes`, ...). The verifier
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

We considered a custom `HipShapeInferenceOpInterface` early on. The
draft would have added a `inferOutputShape(...)` static method returning
`SmallVector<int64_t>`. That would have given us a clean separator
between "static helper used by the verifier" and "dynamic helper used by
the canonicaliser". We dropped it because:

* Both jobs are already covered by upstream interfaces — the static
  helper is a free function in `HipShapeUtils`, and the dynamic helper
  is the upstream interface method.
* Doubling up the surface adds a maintenance gradient with no payoff.
* The upstream test-pass machinery is the test surface for free.

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
  HipShapeUtils.h          -- public API: inferContractionShape,
                              verifyHipOpShape, reifyDimOrConstant
  HipOps.td                -- per-op declares
                              `DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface>`
                              and `let hasVerifier = 1`

lib/Dialect/IR/
  HipShapeUtils.cpp        -- implementations + diagnostics
  HipDialect.cpp           -- per-op `verify()` + `reifyResultShapes()`

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

## Pipeline placement

`--hip-infer-shapes` is **optional** and runs on `mlir::ModuleOp`. The
recommended placement is between `convert-onnx-to-hip` and
`one-shot-bufferize`:

```text
simplify-onnx -> hip-add-context-arg -> convert-onnx-to-hip
              -> hip-infer-shapes (NEW, optional)
              -> one-shot-bufferize -> ...
```

Refining tensor result types BEFORE bufferize propagates static dims
through the alloc / pool sizing computations and removes spurious
`memref.dim` ops at the top of the function.

The pass is idempotent: a second invocation that finds no further
refinements is a no-op.

## How to add a new op

A new HIP op participates in shape inference in **three** small edits.
The matmul wiring is the canonical reference; the same pattern applies
to every DPS compute op.

### 1. Add a shape helper (or reuse one)

In [HipShapeUtils.h](../../include/hip/Dialect/IR/HipShapeUtils.h), add a free
function that takes the operand shapes (and any op-specific attrs) and
returns the expected output shape vector. Use `ShapedType::kDynamic` for
dims that the helper cannot compute, and emit a diagnostic via the
supplied `emitError` callable on shape mismatch.

```cpp
// HipShapeUtils.h
SmallVector<int64_t> broadcastShapes(
    ArrayRef<int64_t> aShape, ArrayRef<int64_t> bShape,
    function_ref<InFlightDiagnostic()> emitError);
```

If your op fits an existing helper (e.g. another contraction shape),
just reuse it.

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

### 3. Implement `verify()` and `reifyResultShapes()`

In [HipDialect.cpp](../../lib/Dialect/IR/HipDialect.cpp), add the two methods
next to your existing `getDpsInitsMutable()` / `getEffects()` impls:

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

LogicalResult MyOp::reifyResultShapes(
    OpBuilder &b, ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  if (getNumResults() == 0)
    return failure(); // memref mode -- no tensor result
  // ... compute outShape via the same helper, then for each dim:
  //       reifyDimOrConstant(b, loc, outShape[d], operand, operandDimIdx)
  // ... reifiedReturnShapes.assign({std::move(dims)});
  return success();
}
```

`reifyDimOrConstant` returns `IntegerAttr` when the dim is static and a
`tensor.dim` / `memref.dim` otherwise, mirroring upstream
`linalg::createFoldedDimOp`.

### 4. (Optional) LIT coverage

Add three small LIT tests under `test/lit/Dialect/`, modelled on the
matmul ones:

* `hip-<op>-shape-verifier.mlir`: positive + negative shape checks via
  `--verify-diagnostics`.
* `hip-<op>-reify-shapes.mlir`: drive `--resolve-shaped-type-result-dims`
  and FileCheck that `tensor.dim` of the op result folds correctly.
* (Optional) extend `test/lit/Dialect/hip-infer-shapes.mlir` with a case
  that exercises `--hip-infer-shapes` on a chain involving the new op.

## Open extensions

The current first-pass coverage is intentionally narrow:

* **Producer refinement is `tensor.empty` only.** Adding refinement for
  other "shape-malleable" producers (`bufferization.alloc_tensor`, other
  HIP DPS ops) is a follow-up. The cleanest extension is to teach the
  pass a small dispatch table keyed on the producer's op name.
* **Element-type refinement is opt-in.** `verifyHipOpShape(... ,
  checkElementType=true)` exists but is off by default; dtype-changing
  ops (cast, equal, less, not, and) keep their op-local element-type
  checks.
* **Control-flow boundaries.** The pass does not propagate refined
  types into `scf.if` / `scf.while` block argument types. That requires
  a separate fixed-point pass keyed on `BranchOpInterface`.
* **Auto-derived builders.** Once a few more ops carry the interface,
  it becomes attractive to add a "shape-aware" op builder
  (`MyOp::create(b, loc, ctx, A, B)` which derives the output type
  automatically). Today every caller passes the explicit outs type.
