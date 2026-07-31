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

#include "hip/Dialect/IR/HipDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Traits.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <optional>

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
  return out;
}

FailureOr<OpFoldResult>
broadcastDim(OpBuilder &b, Location loc, OpFoldResult lhs, OpFoldResult rhs,
             function_ref<InFlightDiagnostic()> emitError) {
  if (lhs == rhs)
    return lhs;

  std::optional<int64_t> lhsStatic = getConstantIntValue(lhs);
  std::optional<int64_t> rhsStatic = getConstantIntValue(rhs);

  if (lhsStatic && rhsStatic) {
    if (*lhsStatic == 1)
      return rhs;
    if (*rhsStatic == 1 || *lhsStatic == *rhsStatic)
      return lhs;
    emitError() << "incompatible broadcast dimensions " << *lhsStatic << " and "
                << *rhsStatic;
    return failure();
  }

  // Under the ONNX broadcastability precondition, a dynamic extent paired
  // with a known non-unit extent must be either 1 or that known extent.
  if (lhsStatic)
    return *lhsStatic == 1 ? rhs : lhs;
  if (rhsStatic)
    return *rhsStatic == 1 ? lhs : rhs;

  Value lhsValue = getValueOrCreateConstantIndexOp(b, loc, lhs);
  Value rhsValue = getValueOrCreateConstantIndexOp(b, loc, rhs);
  Value one = arith::ConstantIndexOp::create(b, loc, 1);
  Value lhsIsOne =
      arith::CmpIOp::create(b, loc, arith::CmpIPredicate::eq, lhsValue, one);
  return OpFoldResult(
      arith::SelectOp::create(b, loc, lhsIsOne, rhsValue, lhsValue));
}

} // namespace

FailureOr<SmallVector<int64_t>>
mlir::hip::inferMatmulShape(ArrayRef<int64_t> aShape, ArrayRef<int64_t> bShape,
                            function_ref<InFlightDiagnostic()> emitError) {
  if (aShape.size() < 2) {
    emitError() << "matmul A must have rank >= 2, got rank " << aShape.size();
    return failure();
  }
  if (bShape.size() < 2) {
    emitError() << "matmul B must have rank >= 2, got rank " << bShape.size();
    return failure();
  }

  int64_t M = aShape[aShape.size() - 2];
  int64_t Ka = aShape.back();
  int64_t Kb = bShape[bShape.size() - 2];
  int64_t N = bShape.back();

  // Contraction K must agree (kDynamic on either side is a wildcard).
  if (!ShapedType::isDynamic(Ka) && !ShapedType::isDynamic(Kb) && Ka != Kb) {
    emitError() << "matmul contraction dim mismatch: A.shape[-1]=" << Ka
                << " vs B.shape[-2]=" << Kb;
    return failure();
  }

  ArrayRef<int64_t> aBatch = aShape.drop_back(2);
  ArrayRef<int64_t> bBatch = bShape.drop_back(2);
  SmallVector<int64_t> result;
  if (!OpTrait::util::getBroadcastedShape(aBatch, bBatch, result)) {
    emitError() << "matmul batch broadcast failure: A.batch="
                << formatShape(aBatch) << " B.batch=" << formatShape(bBatch);
    return failure();
  }
  result.reserve(result.size() + 2);
  result.push_back(M);
  result.push_back(N);
  return result;
}

namespace {

/// Whether `batch` -- one operand's leading extents, right-aligned against
/// `outputBatch` -- holds a matrix count that one constant stride can express.
///
/// A stride of 0 reuses a single matrix for every output batch and a stride of
/// the matrix size walks one matrix per output batch, so the operand's matrix
/// count must be either 1 or the output's. A partial broadcast lands strictly
/// between the two: `[2, 1]` against an output batch of `[2, 3]` holds 2
/// matrices where the output needs 6, so neither stride is correct.
///
/// Extents that are not statically 1 count as carrying batches and output
/// extents that are not statically 1 count as broadcast targets, so an unknown
/// extent is never assumed away.
bool isSingleStrideBatchLayout(ArrayRef<int64_t> batch,
                               ArrayRef<int64_t> outputBatch) {
  // Guaranteed by getBroadcastedShape, and load-bearing: the unsigned `pad`
  // below would wrap and silently report "representable" if it were violated.
  assert(batch.size() <= outputBatch.size() &&
         "operand batch cannot outrank the broadcasted output batch");
  size_t pad = outputBatch.size() - batch.size();
  bool broadcastsUp = false;
  bool carriesBatches = false;
  for (size_t i : llvm::seq<size_t>(0, outputBatch.size())) {
    // Axes below `pad` are the implicit leading ones of right-alignment.
    int64_t extent = i < pad ? 1 : batch[i - pad];
    if (extent == 1)
      broadcastsUp |= outputBatch[i] != 1;
    else
      carriesBatches = true;
  }
  return !(broadcastsUp && carriesBatches);
}

} // namespace

LogicalResult mlir::hip::verifyStridedBatchMatmul(
    ArrayRef<int64_t> aShape, ArrayRef<int64_t> bShape,
    function_ref<InFlightDiagnostic()> emitError) {
  if (aShape.size() < 2 || bShape.size() < 2) {
    emitError() << "strided-batch matmul requires rank >= 2 operands";
    return failure();
  }

  ArrayRef<int64_t> aBatch = aShape.drop_back(2);
  ArrayRef<int64_t> bBatch = bShape.drop_back(2);
  SmallVector<int64_t> outputBatch;
  if (!OpTrait::util::getBroadcastedShape(aBatch, bBatch, outputBatch)) {
    emitError() << "matmul batch broadcast failure: A.batch="
                << formatShape(aBatch) << " B.batch=" << formatShape(bBatch);
    return failure();
  }

  if (isSingleStrideBatchLayout(aBatch, outputBatch) &&
      isSingleStrideBatchLayout(bBatch, outputBatch))
    return success();

  emitError() << "matmul partial per-axis batch broadcast is not supported by "
                 "the strided-batch runtime: A.batch="
              << formatShape(aBatch) << " B.batch=" << formatShape(bBatch);
  return failure();
}

namespace {

/// NumPy-broadcast result shape of `shapes` (right-aligned) from static extents
/// only. Folds `OpTrait::util::getBroadcastedShape` pairwise so static
/// broadcast validation is identical to the matmul batch path.
FailureOr<SmallVector<int64_t>>
inferBroadcastShape(ArrayRef<ArrayRef<int64_t>> shapes,
                    function_ref<InFlightDiagnostic()> emitError) {
  if (shapes.empty()) {
    emitError() << "broadcast requires at least one input shape";
    return failure();
  }

  SmallVector<int64_t> result(shapes.front());
  for (ArrayRef<int64_t> shape : shapes.drop_front()) {
    SmallVector<int64_t> merged;
    if (!OpTrait::util::getBroadcastedShape(result, shape, merged)) {
      emitError() << "incompatible broadcast shapes " << formatShape(result)
                  << " and " << formatShape(shape);
      return failure();
    }
    result = std::move(merged);
  }
  return result;
}

} // namespace

FailureOr<SmallVector<int64_t>>
mlir::hip::inferGemmShape(ArrayRef<int64_t> aShape, ArrayRef<int64_t> bShape,
                          std::optional<ArrayRef<int64_t>> cShape,
                          int64_t transA, int64_t transB,
                          function_ref<InFlightDiagnostic()> emitError) {
  if (aShape.size() != 2 || bShape.size() != 2) {
    emitError() << "gemm A and B must be rank-2 tensors";
    return failure();
  }
  if ((transA != 0 && transA != 1) || (transB != 0 && transB != 1)) {
    emitError() << "gemm transA and transB must be 0 or 1";
    return failure();
  }

  // Contraction K must agree (kDynamic on either side is a wildcard).
  int64_t aK = aShape[transA ? 0 : 1];
  int64_t bK = bShape[transB ? 1 : 0];
  if (!ShapedType::isDynamic(aK) && !ShapedType::isDynamic(bK) && aK != bK) {
    emitError() << "gemm contraction dim mismatch: A has " << aK
                << " but B has " << bK;
    return failure();
  }

  SmallVector<int64_t> result = {aShape[transA ? 1 : 0],
                                 bShape[transB ? 0 : 1]};
  if (!cShape)
    return result;

  if (cShape->size() > 2) {
    emitError() << "gemm C must be a ranked tensor of rank at most 2";
    return failure();
  }
  // ONNX Gemm broadcasts C onto `{M, N}` unidirectionally: C never widens the
  // result, so every C extent must be 1 or equal to the output extent.
  size_t pad = result.size() - cShape->size();
  for (size_t i : llvm::seq<size_t>(pad, result.size())) {
    int64_t cDim = (*cShape)[i - pad];
    if (ShapedType::isDynamic(cDim) || ShapedType::isDynamic(result[i]) ||
        cDim == 1 || cDim == result[i])
      continue;
    emitError() << "gemm C dimension " << cDim
                << " is not broadcastable to output dimension " << result[i];
    return failure();
  }
  return result;
}

LogicalResult mlir::hip::verifyHipOpShape(
    Operation *op, function_ref<FailureOr<SmallVector<int64_t>>()> inferShape) {
  auto dpsOp = dyn_cast<DestinationStyleOpInterface>(op);
  if (!dpsOp)
    return op->emitOpError(
        "shape verification requires DestinationStyleOpInterface");
  auto inits = dpsOp.getDpsInits();
  if (inits.size() != 1)
    return op->emitOpError("expected a single DPS init operand, got ")
           << inits.size();

  // The shape helper has already emitted a diagnostic on failure.
  FailureOr<SmallVector<int64_t>> expected = inferShape();
  if (failed(expected))
    return failure();

  auto initType = dyn_cast<ShapedType>(inits.front().getType());
  if (!initType)
    return op->emitOpError("init operand is not a shaped type");
  ArrayRef<int64_t> actual = initType.getShape();
  if (actual.size() != expected->size())
    return op->emitOpError("rank mismatch on result: expected rank ")
           << expected->size() << " " << formatShape(*expected)
           << " but outs has rank " << actual.size() << " "
           << formatShape(actual);

  for (size_t d : llvm::seq<size_t>(0, actual.size())) {
    // kDynamic on either side is a wildcard.
    if (ShapedType::isDynamic(actual[d]) ||
        ShapedType::isDynamic((*expected)[d]))
      continue;
    if (actual[d] != (*expected)[d])
      return op->emitOpError("dim ")
             << d << " of result mismatch: expected " << (*expected)[d] << " "
             << formatShape(*expected) << " but outs has " << actual[d] << " "
             << formatShape(actual);
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

FailureOr<SmallVector<OpFoldResult>>
mlir::hip::reifyElementwiseSameShape(OpBuilder &b, Location loc, Value source) {
  auto sourceType = dyn_cast<RankedTensorType>(source.getType());
  if (!sourceType)
    return failure();
  ArrayRef<int64_t> shape = sourceType.getShape();
  SmallVector<OpFoldResult> dims;
  dims.reserve(shape.size());
  for (size_t i : llvm::seq<size_t>(0, shape.size()))
    dims.push_back(reifyDimOrConstant(b, loc, shape[i], source, i));
  return dims;
}

namespace {

/// NumPy-broadcast result shape from already-reified operand shapes. Callers
/// must have validated broadcastability against the static shapes first, since
/// this materializes index SSA as it folds.
FailureOr<SmallVector<OpFoldResult>>
reifyBroadcastShape(OpBuilder &b, Location loc,
                    ArrayRef<SmallVector<OpFoldResult>> inputShapes,
                    function_ref<InFlightDiagnostic()> emitError) {
  if (inputShapes.empty()) {
    emitError() << "broadcast requires at least one input shape";
    return failure();
  }

  size_t resultRank = 0;
  for (const SmallVector<OpFoldResult> &shape : inputShapes)
    resultRank = std::max(resultRank, shape.size());

  SmallVector<OpFoldResult> result(resultRank, b.getIndexAttr(1));
  for (const SmallVector<OpFoldResult> &shape : inputShapes) {
    size_t pad = resultRank - shape.size();
    for (size_t i : llvm::seq<size_t>(0, resultRank)) {
      OpFoldResult inputDim =
          i < pad ? OpFoldResult(b.getIndexAttr(1)) : shape[i - pad];
      FailureOr<OpFoldResult> merged =
          broadcastDim(b, loc, result[i], inputDim, emitError);
      if (failed(merged))
        return failure();
      result[i] = *merged;
    }
  }
  return result;
}

} // namespace

FailureOr<SmallVector<OpFoldResult>> mlir::hip::reifyBroadcastResultShape(
    OpBuilder &b, Location loc, ValueRange operands,
    function_ref<InFlightDiagnostic()> emitError) {
  if (operands.empty()) {
    emitError() << "broadcast requires at least one operand";
    return failure();
  }

  SmallVector<ArrayRef<int64_t>> staticShapes;
  staticShapes.reserve(operands.size());
  for (Value operand : operands) {
    auto operandType = dyn_cast<RankedTensorType>(operand.getType());
    if (!operandType) {
      emitError() << "broadcast operand must be a ranked tensor";
      return failure();
    }
    staticShapes.push_back(operandType.getShape());
  }
  // Validate broadcastability before emitting any `tensor.dim`, so a failure
  // leaves the IR unchanged (see the contract in HipShapeUtils.h).
  if (failed(inferBroadcastShape(staticShapes, emitError)))
    return failure();

  SmallVector<SmallVector<OpFoldResult>> shapes;
  shapes.reserve(operands.size());
  for (size_t i : llvm::seq<size_t>(0, operands.size())) {
    Value operand = operands[i];
    bool reused = false;
    // Reuse the first mixed shape for repeated SSA operands (e.g. x*x) so
    // broadcastDim sees identical OpFoldResults and emits no redundant
    // tensor.dim/cmpi/select chain. Broadcast arity is normally 2-3, so a
    // linear scan is simpler than maintaining a side map.
    for (size_t j : llvm::seq<size_t>(0, i)) {
      if (operand == operands[j]) {
        shapes.push_back(shapes[j]);
        reused = true;
        break;
      }
    }
    if (reused)
      continue;
    shapes.push_back(tensor::getMixedSizes(b, loc, operand));
  }
  return reifyBroadcastShape(b, loc, shapes, emitError);
}

FailureOr<SmallVector<OpFoldResult>> mlir::hip::reifyMatmulResultShape(
    OpBuilder &b, Location loc, Value A, Value B,
    function_ref<InFlightDiagnostic()> emitError) {
  auto aType = dyn_cast<RankedTensorType>(A.getType());
  auto bType = dyn_cast<RankedTensorType>(B.getType());
  if (!aType || !bType) {
    emitError() << "matmul operands must be ranked tensors";
    return failure();
  }
  // Validate before emitting any `tensor.dim`, so a failure leaves the IR
  // unchanged (see the contract in HipShapeUtils.h). Strided-batch
  // representability is a backend capability rather than a shape rule, so it
  // stays with `MatmulOp::verify` and the converter.
  if (failed(inferMatmulShape(aType.getShape(), bType.getShape(), emitError)))
    return failure();

  SmallVector<OpFoldResult> aSizes = tensor::getMixedSizes(b, loc, A);
  SmallVector<OpFoldResult> bSizes = tensor::getMixedSizes(b, loc, B);
  ArrayRef<OpFoldResult> aBatch = ArrayRef<OpFoldResult>(aSizes).drop_back(2);
  ArrayRef<OpFoldResult> bBatch = ArrayRef<OpFoldResult>(bSizes).drop_back(2);
  SmallVector<SmallVector<OpFoldResult>> batchShapes = {
      SmallVector<OpFoldResult>(aBatch.begin(), aBatch.end()),
      SmallVector<OpFoldResult>(bBatch.begin(), bBatch.end())};
  FailureOr<SmallVector<OpFoldResult>> result =
      reifyBroadcastShape(b, loc, batchShapes, emitError);
  if (failed(result))
    return failure();
  result->push_back(aSizes[aSizes.size() - 2]);
  result->push_back(bSizes.back());
  return result;
}

FailureOr<SmallVector<OpFoldResult>>
mlir::hip::reifyMatMulNBitsResultShape(OpBuilder &b, Location loc, Value A,
                                       int64_t N) {
  auto aType = dyn_cast<RankedTensorType>(A.getType());
  if (!aType || aType.getRank() < 1)
    return failure();

  ArrayRef<int64_t> aShape = aType.getShape();
  SmallVector<OpFoldResult> dims;
  dims.reserve(aShape.size());
  for (size_t i : llvm::seq<size_t>(0, aShape.size() - 1))
    dims.push_back(reifyDimOrConstant(b, loc, aShape[i], A, i));
  dims.push_back(b.getIndexAttr(N));
  return dims;
}

FailureOr<SmallVector<OpFoldResult>>
mlir::hip::reifyGemmResultShape(OpBuilder &b, Location loc, Value A, Value B,
                                Value optionalC, int64_t transA, int64_t transB,
                                function_ref<InFlightDiagnostic()> emitError) {
  auto aType = dyn_cast<RankedTensorType>(A.getType());
  auto bType = dyn_cast<RankedTensorType>(B.getType());
  if (!aType || !bType) {
    emitError() << "gemm A and B must be ranked tensors";
    return failure();
  }
  std::optional<ArrayRef<int64_t>> cShape;
  if (optionalC) {
    auto cType = dyn_cast<RankedTensorType>(optionalC.getType());
    if (!cType) {
      emitError() << "gemm C must be a ranked tensor";
      return failure();
    }
    cShape = cType.getShape();
  }
  // Validate before emitting any `tensor.dim`, so a failure leaves the IR
  // unchanged (see the contract in HipShapeUtils.h).
  if (failed(inferGemmShape(aType.getShape(), bType.getShape(), cShape, transA,
                            transB, emitError)))
    return failure();

  // C is validated above but never contributes an extent: Gemm's result is
  // exactly `{M, N}` from the transpose-aware A and B dimensions.
  SmallVector<OpFoldResult> aSizes = tensor::getMixedSizes(b, loc, A);
  SmallVector<OpFoldResult> bSizes = tensor::getMixedSizes(b, loc, B);
  return SmallVector<OpFoldResult>{aSizes[transA ? 1 : 0],
                                   bSizes[transB ? 0 : 1]};
}

FailureOr<SmallVector<OpFoldResult>>
mlir::hip::reifyTransposeByPerm(OpBuilder &b, Location loc, Value input,
                                ArrayRef<int64_t> perm) {
  auto inputType = dyn_cast<RankedTensorType>(input.getType());
  if (!inputType)
    return failure();
  ArrayRef<int64_t> inputShape = inputType.getShape();
  int64_t rank = inputType.getRank();
  if (static_cast<int64_t>(perm.size()) != rank)
    return failure();

  // Validate the entire permutation before materializing any tensor.dim. A
  // failure must leave the IR unchanged.
  llvm::SmallBitVector seen(rank);
  for (int64_t dim : perm) {
    if (dim < 0 || dim >= rank || seen.test(dim))
      return failure();
    seen.set(dim);
  }

  SmallVector<OpFoldResult> dims;
  dims.reserve(perm.size());
  for (int64_t dim : perm)
    dims.push_back(reifyDimOrConstant(b, loc, inputShape[dim], input, dim));
  return dims;
}

FailureOr<SmallVector<OpFoldResult>>
mlir::hip::reifyGatherWithAxis(OpBuilder &b, Location loc, Value data,
                               Value indices, int64_t axis) {
  auto dataType = dyn_cast<RankedTensorType>(data.getType());
  auto indicesType = dyn_cast<RankedTensorType>(indices.getType());
  if (!dataType || !indicesType)
    return failure();
  int64_t dataRank = dataType.getRank();
  int64_t indicesRank = indicesType.getRank();
  // Negative-axis normalization (ONNX convention).
  if (axis < 0)
    axis += dataRank;
  if (axis < 0 || axis >= dataRank)
    return failure();

  ArrayRef<int64_t> dataShape = dataType.getShape();
  ArrayRef<int64_t> indicesShape = indicesType.getShape();

  // Output = data.shape[:axis] ++ indices.shape ++ data.shape[axis+1:].
  SmallVector<OpFoldResult> dims;
  dims.reserve(dataRank - 1 + indicesRank);
  for (int64_t i : llvm::seq<int64_t>(0, axis))
    dims.push_back(reifyDimOrConstant(b, loc, dataShape[i], data, i));
  for (int64_t i : llvm::seq<int64_t>(0, indicesRank))
    dims.push_back(reifyDimOrConstant(b, loc, indicesShape[i], indices, i));
  for (int64_t i : llvm::seq<int64_t>(axis + 1, dataRank))
    dims.push_back(reifyDimOrConstant(b, loc, dataShape[i], data, i));
  return dims;
}

FailureOr<SmallVector<OpFoldResult>>
mlir::hip::reifyGatherND(OpBuilder &b, Location loc, Value data, Value indices,
                         int64_t batchDims) {
  auto dataType = dyn_cast<RankedTensorType>(data.getType());
  auto indicesType = dyn_cast<RankedTensorType>(indices.getType());
  if (!dataType || !indicesType)
    return failure();
  int64_t dataRank = dataType.getRank();
  int64_t indicesRank = indicesType.getRank();
  if (indicesRank < 1)
    return failure();
  ArrayRef<int64_t> dataShape = dataType.getShape();
  ArrayRef<int64_t> indicesShape = indicesType.getShape();

  // Trailing-tuple width must be statically known — it determines the
  // output rank (`q + r - tupleWidth - 1 - batch_dims`) and we can't
  // synthesise a rank with a dynamic count of dim entries.
  int64_t tupleWidth = indicesShape[indicesRank - 1];
  if (ShapedType::isDynamic(tupleWidth))
    return failure();
  if (batchDims < 0 || batchDims > indicesRank - 1 ||
      batchDims + tupleWidth > dataRank)
    return failure();

  // Output = data.shape[:batch_dims] (the shared batch prefix) ++
  //          indices.shape[batch_dims:-1] (the gathered tuple count) ++
  //          data.shape[batch_dims + tupleWidth:] (the per-element slice).
  SmallVector<OpFoldResult> dims;
  dims.reserve(batchDims + (indicesRank - 1 - batchDims) +
               (dataRank - batchDims - tupleWidth));
  for (int64_t i : llvm::seq<int64_t>(0, batchDims))
    dims.push_back(reifyDimOrConstant(b, loc, dataShape[i], data, i));
  for (int64_t i : llvm::seq<int64_t>(batchDims, indicesRank - 1))
    dims.push_back(reifyDimOrConstant(b, loc, indicesShape[i], indices, i));
  for (int64_t i : llvm::seq<int64_t>(batchDims + tupleWidth, dataRank))
    dims.push_back(reifyDimOrConstant(b, loc, dataShape[i], data, i));
  return dims;
}

namespace {

/// Try to extract the integer values from a rank-0 / rank-1 i64 / i32
/// tensor that came in via `arith.constant` with a `DenseIntElementsAttr`.
/// Returns true on success and writes the values into `out`. Returns false
/// for any pattern we don't recognise — caller falls back gracefully.
///
/// The reduce / pad / tile / expand converters in `lib/Conversion/OnnxToHip/`
/// materialize axes / pads / repeats / shape operands as `arith.constant`
/// with `DenseIntElementsAttr` when the ONNX node carries a constant
/// attribute or a folded constant initializer — that is the case this
/// helper recognises. Anything else (a runtime input, an unfolded chain
/// of arith ops) is intentionally left alone so reify falls through to
/// the no-op fallback.
bool extractConstantInts(Value v, SmallVectorImpl<int64_t> &out) {
  out.clear();
  IntegerAttr intAttr;
  DenseIntElementsAttr denseAttr;
  if (matchPattern(v, m_Constant(&intAttr))) {
    out.push_back(intAttr.getInt());
    return true;
  }
  if (matchPattern(v, m_Constant(&denseAttr))) {
    for (APInt e : denseAttr.getValues<APInt>())
      out.push_back(e.getSExtValue());
    return true;
  }
  return false;
}

} // namespace

namespace {

/// Map each output dimension of an ONNX reduction to the input dimension it
/// takes its extent from; `std::nullopt` marks a reduced axis retained by
/// `keepdims`, whose extent is the literal 1 rather than an input extent.
///
/// `axes` holds reduced axis indices in the ONNX negative-axis convention; an
/// empty list means no reduction. Returns failure when an axis is out of range.
///
/// This is the single source of truth behind `inferReductionShape` and
/// `reifyReductionResultShape`, so the static and mixed forms cannot disagree.
FailureOr<SmallVector<std::optional<int64_t>>>
computeReductionDimMap(int64_t dataRank, ArrayRef<int64_t> axes,
                       int64_t keepdims) {
  // Axis membership over the closed domain [0, dataRank).
  llvm::SmallBitVector reduced(dataRank);
  for (int64_t axis : axes) {
    int64_t normalized = axis < 0 ? axis + dataRank : axis;
    if (normalized < 0 || normalized >= dataRank)
      return failure();
    reduced.set(normalized);
  }

  SmallVector<std::optional<int64_t>> dimMap;
  dimMap.reserve(dataRank);
  for (int64_t i : llvm::seq<int64_t>(0, dataRank)) {
    if (!reduced.test(i))
      dimMap.push_back(i);
    else if (keepdims)
      dimMap.push_back(std::nullopt);
    // keepdims=0: the reduced axis leaves the output rank entirely.
  }
  return dimMap;
}

} // namespace

FailureOr<SmallVector<int64_t>>
mlir::hip::inferReductionShape(ArrayRef<int64_t> dataShape,
                               ArrayRef<int64_t> axes, int64_t keepdims) {
  FailureOr<SmallVector<std::optional<int64_t>>> dimMap =
      computeReductionDimMap(dataShape.size(), axes, keepdims);
  if (failed(dimMap))
    return failure();

  SmallVector<int64_t> shape;
  shape.reserve(dimMap->size());
  for (std::optional<int64_t> sourceDim : *dimMap)
    shape.push_back(sourceDim ? dataShape[*sourceDim] : 1);
  return shape;
}

FailureOr<SmallVector<OpFoldResult>>
mlir::hip::reifyReductionResultShape(OpBuilder &b, Location loc, Value data,
                                     ArrayRef<int64_t> axes, int64_t keepdims) {
  auto dataType = dyn_cast<RankedTensorType>(data.getType());
  if (!dataType)
    return failure();
  // Validate before emitting any `tensor.dim`, so a failure leaves the IR
  // unchanged (see the contract in HipShapeUtils.h).
  FailureOr<SmallVector<std::optional<int64_t>>> dimMap =
      computeReductionDimMap(dataType.getRank(), axes, keepdims);
  if (failed(dimMap))
    return failure();

  ArrayRef<int64_t> dataShape = dataType.getShape();
  SmallVector<OpFoldResult> dims;
  dims.reserve(dimMap->size());
  for (std::optional<int64_t> sourceDim : *dimMap)
    dims.push_back(sourceDim ? reifyDimOrConstant(b, loc, dataShape[*sourceDim],
                                                  data, *sourceDim)
                             : OpFoldResult(b.getIndexAttr(1)));
  return dims;
}

namespace {

/// Reduction result shape recovered from a constant `axes` operand.
///
/// Introspects `axes` as an `arith.constant` (the typical case after the
/// OnnxToHip converter materializes it from the ONNX attribute), resolves
/// ONNX's empty-axes semantics against `noop_with_empty_axes` — reduce every
/// axis when 0, reduce nothing when 1 — and delegates the shape rule to
/// `reifyReductionResultShape`.
///
/// Returns failure when `axes` is not a recognised constant, which is why this
/// returns `LogicalResult` and writes through `out`: a valid rank-0 reduction
/// result is a successful *empty* dim list and would otherwise be
/// indistinguishable from the bail path.
LogicalResult reifyReductionWithKeepdims(OpBuilder &b, Location loc, Value data,
                                         Value axes, int64_t keepdims,
                                         int64_t noopWithEmptyAxes,
                                         SmallVectorImpl<OpFoldResult> &out) {
  out.clear();
  auto dataType = dyn_cast<RankedTensorType>(data.getType());
  if (!dataType)
    return failure();

  SmallVector<int64_t> axesList;
  if (!extractConstantInts(axes, axesList))
    return failure();
  if (axesList.empty() && noopWithEmptyAxes == 0) {
    // Reduce every axis. `noop_with_empty_axes = 1` instead means "reduce
    // nothing", which the empty list already expresses.
    axesList = llvm::to_vector(llvm::seq<int64_t>(0, dataType.getRank()));
  }

  FailureOr<SmallVector<OpFoldResult>> dims =
      mlir::hip::reifyReductionResultShape(b, loc, data, axesList, keepdims);
  if (failed(dims))
    return failure();
  out.assign(dims->begin(), dims->end());
  return success();
}

} // namespace

LogicalResult
mlir::hip::reifyBroadcastShapeFor(OpBuilder &b, Location loc,
                                  ValueRange operands, Operation *op,
                                  ReifiedRankedShapedTypeDims &reified) {
  if (op->getNumResults() == 0)
    return failure();
  for (Value v : operands)
    if (!isa<RankedTensorType>(v.getType()))
      return failure();
  FailureOr<SmallVector<OpFoldResult>> dims = reifyBroadcastResultShape(
      b, loc, operands, [&]() { return op->emitOpError(); });
  if (failed(dims))
    return failure();
  reified.assign({std::move(*dims)});
  return success();
}

LogicalResult
mlir::hip::reifyReductionShape(OpBuilder &b, Location loc, Value data,
                               Value axes, int64_t keepdims,
                               int64_t noopWithEmptyAxes, Operation *op,
                               ReifiedRankedShapedTypeDims &reified) {
  if (op->getNumResults() == 0)
    return failure();
  if (!isa<RankedTensorType>(data.getType()))
    return failure();

  // Tier-1: introspect `axes` as an `arith.constant` and reify per-dim
  // from the input shape with keepdims awareness.
  SmallVector<OpFoldResult> dims;
  if (succeeded(reifyReductionWithKeepdims(b, loc, data, axes, keepdims,
                                           noopWithEmptyAxes, dims))) {
    reified.assign({std::move(dims)});
    return success();
  }

  // Fallback: lift the DPS `outs` operand's own shape via the shared
  // `HipDpsOp` default. `cast<HipDpsOp>(op).reifyResultShapes` dispatches
  // through the `HipDpsOp` interface concept and lands on the
  // interface-default body in `HipDpsOpInterface.cpp` — it walks
  // `getDpsInits()` and lifts each via `tensor::getMixedSizes` /
  // `memref::getMixedSizes`. Reduction ops override the SEPARATE
  // `ReifyRankedShapedTypeOpInterface::reifyResultShapes` (auto-emitted
  // by `Hip_DpsOp_Reduction` and the body of THIS function), but do not
  // override the `HipDpsOp` interface method, so the call below resolves
  // to the default body and does not recurse.
  return cast<HipDpsOp>(op).reifyResultShapes(b, reified);
}

LogicalResult mlir::hip::reifyPadShape(OpBuilder &b, Location loc, Value data,
                                       Value pads, Value axes,
                                       SmallVectorImpl<OpFoldResult> &out) {
  out.clear();
  auto dataType = dyn_cast<RankedTensorType>(data.getType());
  if (!dataType)
    return failure();
  ArrayRef<int64_t> dataShape = dataType.getShape();
  int64_t dataRank = dataType.getRank();

  // pads is the primary semantic info; without it nothing can be tightened
  // beyond outs (and the caller's Tier-2 fallback already handles that).
  SmallVector<int64_t> padsList;
  if (!extractConstantInts(pads, padsList))
    return failure();

  // axes default: full range. ONNX requires len(pads) == 2 * len(axes).
  SmallVector<int64_t> axesList;
  if (axes) {
    if (!extractConstantInts(axes, axesList))
      return failure();
    for (int64_t &a : axesList) {
      if (a < 0)
        a += dataRank;
      if (a < 0 || a >= dataRank)
        return failure();
    }
  } else {
    axesList.reserve(dataRank);
    for (int64_t i : llvm::seq<int64_t>(0, dataRank))
      axesList.push_back(i);
  }
  if (static_cast<int64_t>(padsList.size()) !=
      2 * static_cast<int64_t>(axesList.size()))
    return failure();

  // Index axes -> [pre, post]. Default 0 for non-listed axes.
  SmallVector<std::pair<int64_t, int64_t>> perAxis(dataRank, {0, 0});
  int64_t numAxes = axesList.size();
  for (int64_t i : llvm::seq<int64_t>(0, numAxes)) {
    int64_t a = axesList[i];
    perAxis[a] = {padsList[i], padsList[i + numAxes]};
  }

  // Fold-or-bail: each output dim must be statically computable. A
  // dynamic data dim under a non-zero pad gives a dynamic output dim
  // (would need arith.addi(tensor.dim, const)) -- we'd rather fall back
  // to Tier-2 outs-lifting than emit a non-foldable arith chain.
  SmallVector<int64_t> outShape;
  outShape.reserve(dataRank);
  for (int64_t d : llvm::seq<int64_t>(0, dataRank)) {
    if (ShapedType::isDynamic(dataShape[d]))
      return failure();
    outShape.push_back(dataShape[d] + perAxis[d].first + perAxis[d].second);
  }

  out.reserve(dataRank);
  for (int64_t v : outShape)
    out.push_back(b.getIndexAttr(v));
  return success();
}

LogicalResult mlir::hip::reifyTileShape(OpBuilder &b, Location loc, Value input,
                                        Value repeats,
                                        SmallVectorImpl<OpFoldResult> &out) {
  out.clear();
  auto inputType = dyn_cast<RankedTensorType>(input.getType());
  if (!inputType)
    return failure();
  ArrayRef<int64_t> inputShape = inputType.getShape();
  int64_t inputRank = inputType.getRank();

  SmallVector<int64_t> repeatsList;
  if (!extractConstantInts(repeats, repeatsList))
    return failure();
  if (static_cast<int64_t>(repeatsList.size()) != inputRank)
    return failure();

  // Same fold-or-bail logic as pad: skip dynamic input dims.
  out.reserve(inputRank);
  for (int64_t d : llvm::seq<int64_t>(0, inputRank)) {
    if (ShapedType::isDynamic(inputShape[d]))
      return failure();
    int64_t r = repeatsList[d];
    if (r < 0)
      return failure();
    out.push_back(b.getIndexAttr(inputShape[d] * r));
  }
  return success();
}

LogicalResult mlir::hip::reifyExpandShape(OpBuilder &b, Location loc,
                                          Value input, Value shape,
                                          SmallVectorImpl<OpFoldResult> &out) {
  out.clear();
  auto inputType = dyn_cast<RankedTensorType>(input.getType());
  if (!inputType)
    return failure();

  SmallVector<int64_t> shapeVals;
  if (!extractConstantInts(shape, shapeVals))
    return failure();

  // ONNX broadcast: right-aligned, leading-1 padded. Defer the actual
  // broadcast math to MLIR's `OpTrait::util::getBroadcastedShape` for
  // consistency with matmul / reifyBroadcastShape.
  SmallVector<int64_t> outShape;
  if (!OpTrait::util::getBroadcastedShape(inputType.getShape(), shapeVals,
                                          outShape))
    return failure();

  // Fold-or-bail: any dynamic in the broadcast result means we can't
  // produce a tight shape; let the Tier-2 fallback lift from outs.
  for (int64_t d : outShape)
    if (ShapedType::isDynamic(d))
      return failure();

  out.reserve(outShape.size());
  for (int64_t v : outShape)
    out.push_back(b.getIndexAttr(v));
  return success();
}

LogicalResult mlir::hip::reifySliceShape(OpBuilder &b, Location loc, Value data,
                                         Value starts, Value ends, Value axes,
                                         Value steps,
                                         SmallVectorImpl<OpFoldResult> &out) {
  out.clear();
  auto dataType = dyn_cast<RankedTensorType>(data.getType());
  if (!dataType)
    return failure();
  ArrayRef<int64_t> dataShape = dataType.getShape();
  int64_t dataRank = dataType.getRank();

  // ALL operands must be foldable -- partial constants on slice would
  // emit `arith.divsi(arith.subi(end, start), step)` chains per axis
  // that don't fold, persisting as dead IR. Tier-2 outs-lifting is the
  // safer fallback.
  SmallVector<int64_t> startsList, endsList, axesList, stepsList;
  if (!extractConstantInts(starts, startsList) ||
      !extractConstantInts(ends, endsList))
    return failure();
  if (axes) {
    if (!extractConstantInts(axes, axesList))
      return failure();
  } else {
    axesList.reserve(dataRank);
    for (int64_t i : llvm::seq<int64_t>(0, dataRank))
      axesList.push_back(i);
  }
  if (steps) {
    if (!extractConstantInts(steps, stepsList))
      return failure();
  } else {
    stepsList.assign(axesList.size(), 1);
  }

  if (startsList.size() != axesList.size() ||
      endsList.size() != axesList.size() || stepsList.size() != axesList.size())
    return failure();

  // Per-axis sliced extent; non-axis dims pass through.
  SmallVector<int64_t> outShape(dataShape.begin(), dataShape.end());
  for (size_t i : llvm::seq<size_t>(0, axesList.size())) {
    int64_t a = axesList[i];
    if (a < 0)
      a += dataRank;
    if (a < 0 || a >= dataRank)
      return failure();
    int64_t dim = dataShape[a];
    if (ShapedType::isDynamic(dim))
      return failure();
    int64_t step = stepsList[i];
    if (step == 0)
      return failure();
    int64_t s = startsList[i];
    int64_t e = endsList[i];
    // ONNX clamping: negative values offset by `dim`; out-of-range clamps.
    if (s < 0)
      s += dim;
    if (e < 0)
      e += dim;
    if (step > 0) {
      s = std::clamp<int64_t>(s, 0, dim);
      e = std::clamp<int64_t>(e, 0, dim);
      outShape[a] = (e > s) ? ((e - s + step - 1) / step) : 0;
    } else { // step < 0
      s = std::clamp<int64_t>(s, 0, dim - 1);
      // Negative-step end clamps to [-1, dim-1] (treat <-1 as -1).
      e = std::clamp<int64_t>(e, -1, dim - 1);
      int64_t span = s - e;
      int64_t k = -step;
      outShape[a] = (s > e) ? ((span + k - 1) / k) : 0;
    }
  }

  // All dims must be static (fold-or-bail).
  for (int64_t v : outShape)
    if (ShapedType::isDynamic(v))
      return failure();

  out.reserve(dataRank);
  for (int64_t v : outShape)
    out.push_back(b.getIndexAttr(v));
  return success();
}

LogicalResult mlir::hip::reifyRangeShape(OpBuilder &b, Location loc,
                                         Value start, Value limit, Value delta,
                                         SmallVectorImpl<OpFoldResult> &out) {
  out.clear();

  // Each operand is a rank-0 (scalar) integer tensor. extractConstantInts
  // returns a 1-element vector for the rank-0 / IntegerAttr case.
  SmallVector<int64_t> sList, lList, dList;
  if (!extractConstantInts(start, sList) ||
      !extractConstantInts(limit, lList) || !extractConstantInts(delta, dList))
    return failure();
  if (sList.size() != 1 || lList.size() != 1 || dList.size() != 1)
    return failure();
  int64_t s = sList[0], l = lList[0], d = dList[0];
  if (d == 0)
    return failure();

  // ONNX Range: count = max(0, ceil((limit - start) / delta)) for the
  // direction implied by sign(delta). Negative direction (delta < 0)
  // counts down from start to limit.
  int64_t count = 0;
  if ((d > 0 && l > s) || (d < 0 && l < s)) {
    int64_t diff = l - s;
    int64_t step = d;
    if (step < 0) {
      diff = -diff;
      step = -step;
    }
    count = (diff + step - 1) / step;
  }

  out.push_back(b.getIndexAttr(count));
  return success();
}
