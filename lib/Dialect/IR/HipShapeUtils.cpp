/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- HipShapeUtils.cpp - Shape arithmetic + verifier helpers ------------===//
//
// Single source of truth for the static shape of every HIP DPS op.  Each
// shape category (contraction, elementwise broadcast, axis-driven reduction,
// ...) lives here as one helper that returns a `SmallVector<int64_t>` where
// each element is either a concrete dim or `ShapedType::kDynamic`. The same
// helper is consumed by:
//
//   1. The op's `verify()` -- via `verifyHipOpShape`, which compares the
//      computed shape against the actual `outs` operand types.
//   2. The op's `reifyResultShapes()` -- which lifts the same dims into
//      `OpFoldResult`s (IntegerAttr for static, tensor.dim/memref.dim
//      for kDynamic) so downstream `--resolve-shaped-type-result-dims`
//      / `--canonicalize` can fold `tensor.dim` of the op's result into
//      a constant or a dim-of-input.
//
// Per-op `reifyResultShapes` impls live next to the op's other methods in
// `HipDialect.cpp`. They call `inferContractionShape(...)` (or the
// equivalent for their op family) and pair each entry with
// `reifyDimOrConstant` to materialise the right `OpFoldResult`.
//
// Adding a new shape category
// ---------------------------
// Add a free function next to `inferContractionShape` (e.g.
// `broadcastShapes` for elementwise NumPy broadcast, `inferReductionShape`
// for axis-driven reduce ops). The new helper takes the operand shapes and
// any op-specific attributes (axes, perm, ...), and emits a diagnostic
// through the supplied `emitError` callable on shape mismatch. The op then
// wires up its `verify()` and `reifyResultShapes()` to call it through
// `verifyHipOpShape`.
//
//===----------------------------------------------------------------------===//

#include "hip/Dialect/IR/HipShapeUtils.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::hip;

namespace {

/// Pretty-print a shape vector with `?` for kDynamic. Used in diagnostics.
std::string formatShape(ArrayRef<int64_t> shape) {
  std::string out;
  llvm::raw_string_ostream os(out);
  os << "[";
  llvm::interleaveComma(shape, os, [&](int64_t d) {
    if (ShapedType::isDynamic(d))
      os << "?";
    else
      os << d;
  });
  os << "]";
  return os.str();
}

} // namespace

SmallVector<int64_t> mlir::hip::inferContractionShape(
    ArrayRef<int64_t> aShape, ArrayRef<int64_t> bShape,
    function_ref<InFlightDiagnostic()> emitError) {
  if (aShape.size() < 2) {
    emitError() << "matmul A must have rank >= 2, got rank " << aShape.size();
    return {};
  }
  if (bShape.size() < 2) {
    emitError() << "matmul B must have rank >= 2, got rank " << bShape.size();
    return {};
  }

  int64_t M = aShape[aShape.size() - 2];
  int64_t Ka = aShape.back();
  int64_t Kb = bShape[bShape.size() - 2];
  int64_t N = bShape.back();

  // Contraction K must agree (kDynamic on either side is a wildcard).
  if (!ShapedType::isDynamic(Ka) && !ShapedType::isDynamic(Kb) && Ka != Kb) {
    emitError() << "matmul contraction dim mismatch: A.shape[-1]=" << Ka
                << " vs B.shape[-2]=" << Kb;
    return {};
  }

  // NumPy-style batch broadcast: right-align, pad missing dims with 1.
  ArrayRef<int64_t> aBatch = aShape.drop_back(2);
  ArrayRef<int64_t> bBatch = bShape.drop_back(2);
  size_t batchRank = std::max(aBatch.size(), bBatch.size());
  size_t aPad = batchRank - aBatch.size();
  size_t bPad = batchRank - bBatch.size();

  SmallVector<int64_t> result;
  result.reserve(batchRank + 2);
  for (size_t i : llvm::seq<size_t>(0, batchRank)) {
    int64_t aDim = i < aPad ? 1 : aBatch[i - aPad];
    int64_t bDim = i < bPad ? 1 : bBatch[i - bPad];
    int64_t resDim;
    if (ShapedType::isDynamic(aDim) || ShapedType::isDynamic(bDim))
      resDim = ShapedType::kDynamic;
    else if (aDim == 1)
      resDim = bDim;
    else if (bDim == 1)
      resDim = aDim;
    else if (aDim == bDim)
      resDim = aDim;
    else {
      emitError() << "matmul batch dim broadcast failure at index " << i
                  << ": A=" << aDim << " B=" << bDim;
      return {};
    }
    result.push_back(resDim);
  }
  result.push_back(M);
  result.push_back(N);
  return result;
}

LogicalResult mlir::hip::verifyHipOpShape(
    Operation *op,
    function_ref<SmallVector<SmallVector<int64_t>>()> computeExpected,
    bool checkElementType) {
  auto dpsOp = dyn_cast<DestinationStyleOpInterface>(op);
  if (!dpsOp)
    return op->emitOpError(
        "verifyHipOpShape requires DestinationStyleOpInterface");

  SmallVector<SmallVector<int64_t>> expected = computeExpected();
  // Empty outer vector: the shape helper failed and already issued a
  // diagnostic. Don't double-emit.
  if (expected.empty())
    return failure();

  auto inits = dpsOp.getDpsInits();
  if (expected.size() != inits.size())
    return op->emitOpError("internal: shape helper produced ")
           << expected.size() << " expected shape(s) but op has "
           << inits.size() << " init operand(s)";

  for (auto [i, init] : llvm::enumerate(inits)) {
    auto initType = dyn_cast<ShapedType>(init.getType());
    if (!initType)
      return op->emitOpError("init #") << i << " is not a shaped type";
    ArrayRef<int64_t> actualShape = initType.getShape();
    ArrayRef<int64_t> expShape = expected[i];
    if (actualShape.size() != expShape.size())
      return op->emitOpError("rank mismatch on result #")
             << i << ": expected rank " << expShape.size() << " "
             << formatShape(expShape) << " but outs has rank "
             << actualShape.size() << " " << formatShape(actualShape);
    for (size_t d : llvm::seq<size_t>(0, actualShape.size())) {
      // kDynamic on either side is a wildcard.
      if (ShapedType::isDynamic(actualShape[d]) ||
          ShapedType::isDynamic(expShape[d]))
        continue;
      if (actualShape[d] != expShape[d])
        return op->emitOpError("dim ")
               << d << " of result #" << i << " mismatch: expected "
               << expShape[d] << " " << formatShape(expShape)
               << " but outs has " << actualShape[d] << " "
               << formatShape(actualShape);
    }
  }

  if (checkElementType) {
    auto inputs = dpsOp.getDpsInputs();
    if (!inputs.empty()) {
      if (auto inputType = dyn_cast<ShapedType>(inputs[0].getType())) {
        Type expectedET = inputType.getElementType();
        for (auto [i, init] : llvm::enumerate(inits)) {
          auto initType = dyn_cast<ShapedType>(init.getType());
          if (initType && initType.getElementType() != expectedET)
            return op->emitOpError("element type mismatch on result #")
                   << i << ": expected " << expectedET << " but got "
                   << initType.getElementType();
        }
      }
    }
  }
  return success();
}

OpFoldResult mlir::hip::reifyDimOrConstant(OpBuilder &b, Location loc,
                                           int64_t staticDim, Value source,
                                           int64_t sourceDim) {
  if (!ShapedType::isDynamic(staticDim))
    return b.getIndexAttr(staticDim);
  if (isa<RankedTensorType>(source.getType()))
    return tensor::getMixedSize(b, loc, source, sourceDim);
  // memref-mode operand: emit memref.dim. The op canonicalises to a
  // constant when the source memref type has a static size at sourceDim.
  return memref::DimOp::create(b, loc, source, sourceDim).getResult();
}
