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

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Traits.h"
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
mlir::hip::inferMatmulShape(ArrayRef<int64_t> aShape, ArrayRef<int64_t> bShape,
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

  // Batch broadcast (NumPy / ONNX MatMul) on the leading dims; see header
  // for the full case table.
  ArrayRef<int64_t> aBatch = aShape.drop_back(2);
  ArrayRef<int64_t> bBatch = bShape.drop_back(2);
  SmallVector<int64_t> result;
  if (!OpTrait::util::getBroadcastedShape(aBatch, bBatch, result)) {
    emitError() << "matmul batch broadcast failure: A.batch="
                << formatShape(aBatch) << " B.batch=" << formatShape(bBatch);
    return {};
  }
  result.reserve(result.size() + 2);
  result.push_back(M);
  result.push_back(N);
  return result;
}

LogicalResult mlir::hip::verifyHipOpShape(
    Operation *op,
    function_ref<SmallVector<SmallVector<int64_t>>()> computeExpected) {
  // Asserting cast: every op wired to verifyHipOpShape also implements DPS
  // via TableGen; a missing interface is a programmer error in the op def.
  auto dpsOp = cast<DestinationStyleOpInterface>(op);

  // Empty outer vector means the shape helper already emitted a diagnostic.
  SmallVector<SmallVector<int64_t>> expected = computeExpected();
  if (expected.empty())
    return failure();

  // Each helper returns one shape per DPS init by construction; assert in
  // debug, fail-safe in release to avoid OOB on `expected[i]` below.
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
  // Reify interface restricts callers to tensor results; if a memref
  // reify path is added later, also add a memref.dim branch + LIT case.
  return tensor::getMixedSize(b, loc, source, sourceDim);
}

SmallVector<OpFoldResult>
mlir::hip::reifyElementwiseSameShape(OpBuilder &b, Location loc, Value source) {
  // Caller must hand a ranked tensor; reify is only invoked in tensor mode
  // per the ReifyRankedShapedTypeOpInterface contract.
  auto sourceType = cast<RankedTensorType>(source.getType());
  ArrayRef<int64_t> shape = sourceType.getShape();
  SmallVector<OpFoldResult> dims;
  dims.reserve(shape.size());
  for (size_t i : llvm::seq<size_t>(0, shape.size()))
    dims.push_back(reifyDimOrConstant(b, loc, shape[i], source, i));
  return dims;
}

SmallVector<OpFoldResult>
mlir::hip::reifyBroadcastShape(OpBuilder &b, Location loc,
                               ValueRange operands) {
  if (operands.empty())
    return {};

  // Collect operand shapes; bail if any is non-ranked (verifier should
  // already have caught this on the op).
  SmallVector<ArrayRef<int64_t>> shapes;
  shapes.reserve(operands.size());
  for (Value v : operands) {
    auto t = dyn_cast<RankedTensorType>(v.getType());
    if (!t)
      return {};
    shapes.push_back(t.getShape());
  }

  // Compute the broadcast result shape via sequential pairwise reduction.
  // `getBroadcastedShape` follows NumPy/ONNX semantics:
  //   - 1 broadcasts against any other dim
  //   - dynamic + static>1 -> static (the strictly-correct tightening,
  //     since the dynamic side must equal the static side at runtime)
  //   - dynamic + dynamic -> dynamic
  //   - equal static -> static; unequal non-1 static -> failure
  SmallVector<int64_t> outShape(shapes[0].begin(), shapes[0].end());
  for (size_t k : llvm::seq<size_t>(1, shapes.size())) {
    SmallVector<int64_t> tmp;
    if (!OpTrait::util::getBroadcastedShape(outShape, shapes[k], tmp))
      return {};
    outShape = std::move(tmp);
  }

  size_t outRank = outShape.size();
  // Right-alignment padding per operand (operand `k` doesn't reach output
  // dims in `[0, pads[k])`; those positions are an implicit 1 contribution).
  SmallVector<size_t> pads(operands.size());
  for (size_t k : llvm::seq<size_t>(0, operands.size()))
    pads[k] = outRank - shapes[k].size();

  SmallVector<OpFoldResult> dims;
  dims.reserve(outRank);
  for (size_t i : llvm::seq<size_t>(0, outRank)) {
    // Pick the operand to reify this dim against:
    //   1. earliest operand that is in-range AND has a non-1 dim
    //      (that operand actually determines the runtime extent;
    //      `tensor.dim %that, i` folds to the constant when that
    //      operand's dim is static)
    //   2. else earliest operand that is in-range (all in-range
    //      operands have a 1 here, so reifying against any of them
    //      is correct; first wins for stability)
    //   3. else operand 0 dim 0 — defensive fallback that should be
    //      unreachable when the broadcast result rank == max input rank.
    Value bestSrc;
    size_t bestSrcDim = 0;
    bool foundCanonical = false;
    for (size_t k : llvm::seq<size_t>(0, operands.size())) {
      if (i < pads[k])
        continue;
      size_t kDim = i - pads[k];
      if (!bestSrc) {
        bestSrc = operands[k];
        bestSrcDim = kDim;
      }
      if (shapes[k][kDim] != 1) {
        bestSrc = operands[k];
        bestSrcDim = kDim;
        foundCanonical = true;
        break;
      }
    }
    (void)foundCanonical;
    if (!bestSrc) {
      bestSrc = operands[0];
      bestSrcDim = 0;
    }
    dims.push_back(reifyDimOrConstant(b, loc, outShape[i], bestSrc, bestSrcDim));
  }
  return dims;
}
