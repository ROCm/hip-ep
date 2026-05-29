/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- HipShapeUtils.cpp - Shape arithmetic + verifier helpers ------------===//
//
// Implementation of the helpers declared in `HipShapeUtils.h`. See the
// per-symbol Doxygen comments there for the API contract, and
// `docs/design/hip-shape-inference.md` for the rationale, component
// layout, and the recipe for wiring a new op (or a new shape category)
// into the verify / reify / propagate pipeline.
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

SmallVector<int64_t>
mlir::hip::inferContractionShape(ArrayRef<int64_t> aShape,
                                 ArrayRef<int64_t> bShape,
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
    function_ref<SmallVector<SmallVector<int64_t>>()> computeExpected) {
  // Required-by-construction: every op that wires up `verifyHipOpShape`
  // also implements `DestinationStyleOpInterface` via TableGen. Use
  // asserting `cast<>` to express that contract — a missing interface is
  // a programmer error in the op's TableGen def, not a user-facing
  // diagnostic.
  auto dpsOp = cast<DestinationStyleOpInterface>(op);

  SmallVector<SmallVector<int64_t>> expected = computeExpected();
  // Empty outer vector: the shape helper failed and already issued a
  // diagnostic. Don't double-emit.
  if (expected.empty())
    return failure();

  // Programmer-error invariant: each shape helper returns one expected shape
  // per DPS init operand by construction. Assert in debug builds; the
  // `return failure()` keeps release builds safe by avoiding the
  // out-of-bounds `expected[i]` in the loop below.
  auto inits = dpsOp.getDpsInits();
  assert(expected.size() == inits.size() &&
         "shape helper must produce one expected shape per DPS init operand");
  if (expected.size() != inits.size())
    return failure();

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
  //
  // Note: `reifyResultShapes` is only invoked on tensor-result ops
  // (`ReifyRankedShapedTypeOpInterface` contract), and the
  // `--hip-infer-shapes` pass walks only `RankedTensorType` results.
  // No current caller reaches this branch — it's kept as defensive
  // code in case a future op exposes reify on a memref result type.
  return memref::DimOp::create(b, loc, source, sourceDim).getResult();
}
