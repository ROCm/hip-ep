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
  size_t aRank = aShape.size();
  size_t bRank = bShape.size();
  if (aRank < 2) {
    emitError() << "matmul A must have rank >= 2, got rank " << aRank;
    return {};
  }
  if (bRank < 2) {
    emitError() << "matmul B must have rank >= 2, got rank " << bRank;
    return {};
  }

  // Tightened contract: matches what `MatMulConversion.cpp` +
  // `MatmulLowering.cpp` actually compute correctly today. Codegen
  // derives result rank and batchCount from `A`, with optional `B`
  // rank-2 broadcast (one matrix re-used across all batches). Anything
  // wider — `B`'s rank > `A`'s rank, mixed ranks where `B` is not
  // exactly rank-2, or per-dim batch broadcasting (`1` vs `>1`) —
  // would be miscompiled by codegen + runtime, so the verifier rejects
  // it here. Widening this contract requires matching widening in
  // codegen and the runtime's strided-batch layout (zero stride for
  // the broadcast side). Tracked under "Codegen contract" in
  // `docs/design/hip-shape-inference.md`.
  if (bRank > aRank) {
    emitError() << "matmul B's rank (" << bRank << ") exceeds A's rank ("
                << aRank
                << "); B-side batch broadcasting is not supported by codegen";
    return {};
  }
  if (aRank > bRank && bRank != 2) {
    emitError() << "matmul mixed-rank operands require B to be rank-2 (got A "
                   "rank "
                << aRank << ", B rank " << bRank << ")";
    return {};
  }

  int64_t M = aShape[aRank - 2];
  int64_t Ka = aShape.back();
  int64_t Kb = bShape[bRank - 2];
  int64_t N = bShape.back();

  // Contraction K must agree (kDynamic on either side is a wildcard).
  if (!ShapedType::isDynamic(Ka) && !ShapedType::isDynamic(Kb) && Ka != Kb) {
    emitError() << "matmul contraction dim mismatch: A.shape[-1]=" << Ka
                << " vs B.shape[-2]=" << Kb;
    return {};
  }

  ArrayRef<int64_t> aBatch = aShape.drop_back(2);
  ArrayRef<int64_t> bBatch = bShape.drop_back(2);
  SmallVector<int64_t> result;
  if (bBatch.empty()) {
    // ND x 2D: result batch = A's batch (one matrix B re-used across all
    // batches). Codegen path: MatmulLowering computes batchCount from A's
    // leading dims and runtime treats B as zero-stride for the batch loop.
    result.assign(aBatch.begin(), aBatch.end());
  } else {
    // Same-rank batched matmul: per-position batch dim equality.
    // kDynamic on either side is a wildcard; if one side is static and
    // the other dynamic, the static side is the strictly-correct
    // tightening (the dynamic side must equal it at runtime).
    assert(aBatch.size() == bBatch.size());
    for (size_t i : llvm::seq<size_t>(0, aBatch.size())) {
      int64_t ad = aBatch[i];
      int64_t bd = bBatch[i];
      bool aDyn = ShapedType::isDynamic(ad);
      bool bDyn = ShapedType::isDynamic(bd);
      if (!aDyn && !bDyn && ad != bd) {
        emitError() << "matmul batch dim mismatch at position " << i
                    << ": A=" << ad << " B=" << bd
                    << "; per-dim batch broadcasting (1 vs >1) is not "
                       "supported by codegen";
        return {};
      }
      result.push_back(aDyn ? bd : ad);
    }
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
