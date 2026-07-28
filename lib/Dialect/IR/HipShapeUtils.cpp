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
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <functional>
#include <numeric>

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

LogicalResult mlir::hip::verifyStridedBatchMatmul(
    ArrayRef<int64_t> aShape, ArrayRef<int64_t> bShape,
    function_ref<InFlightDiagnostic()> emitError) {
  if (aShape.size() < 2 || bShape.size() < 2) {
    emitError() << "strided-batch matmul requires rank >= 2 operands";
    return failure();
  }

  ArrayRef<int64_t> aBatch = aShape.drop_back(2);
  ArrayRef<int64_t> bBatch = bShape.drop_back(2);
  auto isSingleMatrix = [](ArrayRef<int64_t> batch) {
    return llvm::all_of(batch, [](int64_t dim) { return dim == 1; });
  };
  if (isSingleMatrix(aBatch) || isSingleMatrix(bBatch))
    return success();
  // With at most one batch axis, every valid broadcast has operand matrix
  // counts in {1, output_count}, which one constant stride can represent.
  if (std::max(aBatch.size(), bBatch.size()) <= 1)
    return success();

  auto isDynamic = [](int64_t dim) { return ShapedType::isDynamic(dim); };
  if (llvm::any_of(aBatch, isDynamic) || llvm::any_of(bBatch, isDynamic)) {
    emitError() << "matmul with two nontrivial dynamic batch shapes is not "
                   "supported by the strided-batch runtime";
    return failure();
  }

  SmallVector<int64_t> outputBatch;
  if (!OpTrait::util::getBroadcastedShape(aBatch, bBatch, outputBatch)) {
    emitError() << "matmul batch broadcast failure: A.batch="
                << formatShape(aBatch) << " B.batch=" << formatShape(bBatch);
    return failure();
  }
  auto product = [](ArrayRef<int64_t> shape) {
    return std::accumulate(shape.begin(), shape.end(), int64_t{1},
                           std::multiplies<int64_t>());
  };
  int64_t outputCount = product(outputBatch);
  if (product(aBatch) == outputCount && product(bBatch) == outputCount)
    return success();

  emitError() << "matmul partial per-axis batch broadcast is not supported by "
                 "the strided-batch runtime";
  return failure();
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

FailureOr<SmallVector<OpFoldResult>>
mlir::hip::reifyBroadcastShape(OpBuilder &b, Location loc,
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

FailureOr<SmallVector<OpFoldResult>> mlir::hip::reifyBroadcastResultShape(
    OpBuilder &b, Location loc, ValueRange operands,
    function_ref<InFlightDiagnostic()> emitError) {
  if (operands.empty()) {
    emitError() << "broadcast requires at least one operand";
    return failure();
  }

  SmallVector<SmallVector<OpFoldResult>> shapes;
  shapes.reserve(operands.size());
  for (size_t i : llvm::seq<size_t>(0, operands.size())) {
    Value operand = operands[i];
    if (!isa<RankedTensorType>(operand.getType())) {
      emitError() << "broadcast operand must be a ranked tensor";
      return failure();
    }
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
  if (inferMatmulShape(aType.getShape(), bType.getShape(), emitError).empty())
    return failure();
  if (failed(verifyStridedBatchMatmul(aType.getShape(), bType.getShape(),
                                      emitError)))
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
mlir::hip::reifyGemmResultShape(OpBuilder &b, Location loc, Value A, Value B,
                                Value optionalC, int64_t transA, int64_t transB,
                                function_ref<InFlightDiagnostic()> emitError) {
  auto aType = dyn_cast<RankedTensorType>(A.getType());
  auto bType = dyn_cast<RankedTensorType>(B.getType());
  if (!aType || !bType || aType.getRank() != 2 || bType.getRank() != 2) {
    emitError() << "gemm A and B must be rank-2 tensors";
    return failure();
  }
  if ((transA != 0 && transA != 1) || (transB != 0 && transB != 1)) {
    emitError() << "gemm transA and transB must be 0 or 1";
    return failure();
  }

  int64_t aKDim = transA ? 0 : 1;
  int64_t bKDim = transB ? 1 : 0;
  int64_t aK = aType.getDimSize(aKDim);
  int64_t bK = bType.getDimSize(bKDim);
  if (!ShapedType::isDynamic(aK) && !ShapedType::isDynamic(bK) && aK != bK) {
    emitError() << "gemm contraction dim mismatch: A has " << aK
                << " but B has " << bK;
    return failure();
  }

  SmallVector<OpFoldResult> aSizes = tensor::getMixedSizes(b, loc, A);
  SmallVector<OpFoldResult> bSizes = tensor::getMixedSizes(b, loc, B);
  SmallVector<OpFoldResult> result = {aSizes[transA ? 1 : 0],
                                      bSizes[transB ? 0 : 1]};

  if (optionalC) {
    auto cType = dyn_cast<RankedTensorType>(optionalC.getType());
    if (!cType || cType.getRank() > 2) {
      emitError() << "gemm C must be a ranked tensor of rank at most 2";
      return failure();
    }
    SmallVector<OpFoldResult> cSizes = tensor::getMixedSizes(b, loc, optionalC);
    size_t pad = result.size() - cSizes.size();
    for (size_t i : llvm::seq<size_t>(0, result.size())) {
      if (i < pad)
        continue;
      std::optional<int64_t> cStatic = getConstantIntValue(cSizes[i - pad]);
      std::optional<int64_t> resultStatic = getConstantIntValue(result[i]);
      if (cStatic && resultStatic && *cStatic != 1 &&
          *cStatic != *resultStatic) {
        emitError() << "gemm C dimension " << *cStatic
                    << " is not broadcastable to output dimension "
                    << *resultStatic;
        return failure();
      }
    }
  }
  return result;
}

SmallVector<OpFoldResult>
mlir::hip::reifyTransposeByPerm(OpBuilder &b, Location loc, Value input,
                                ArrayRef<int64_t> perm) {
  auto inputType = dyn_cast<RankedTensorType>(input.getType());
  if (!inputType)
    return {};
  ArrayRef<int64_t> inputShape = inputType.getShape();
  int64_t rank = inputType.getRank();
  if (static_cast<int64_t>(perm.size()) != rank)
    return {};

  SmallVector<OpFoldResult> dims;
  dims.reserve(perm.size());
  for (int64_t pi : perm) {
    if (pi < 0 || pi >= rank)
      return {};
    dims.push_back(reifyDimOrConstant(b, loc, inputShape[pi], input, pi));
  }
  return dims;
}

SmallVector<OpFoldResult>
mlir::hip::reifyGatherWithAxis(OpBuilder &b, Location loc, Value data,
                               Value indices, int64_t axis) {
  auto dataType = dyn_cast<RankedTensorType>(data.getType());
  auto indicesType = dyn_cast<RankedTensorType>(indices.getType());
  if (!dataType || !indicesType)
    return {};
  int64_t dataRank = dataType.getRank();
  int64_t indicesRank = indicesType.getRank();
  // Negative-axis normalization (ONNX convention).
  if (axis < 0)
    axis += dataRank;
  if (axis < 0 || axis >= dataRank)
    return {};

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

SmallVector<OpFoldResult> mlir::hip::reifyGatherND(OpBuilder &b, Location loc,
                                                   Value data, Value indices,
                                                   int64_t batchDims) {
  auto dataType = dyn_cast<RankedTensorType>(data.getType());
  auto indicesType = dyn_cast<RankedTensorType>(indices.getType());
  if (!dataType || !indicesType)
    return {};
  int64_t dataRank = dataType.getRank();
  int64_t indicesRank = indicesType.getRank();
  if (indicesRank < 1)
    return {};
  ArrayRef<int64_t> dataShape = dataType.getShape();
  ArrayRef<int64_t> indicesShape = indicesType.getShape();

  // Trailing-tuple width must be statically known — it determines the
  // output rank (`q + r - tupleWidth - 1 - batch_dims`) and we can't
  // synthesise a rank with a dynamic count of dim entries.
  int64_t tupleWidth = indicesShape[indicesRank - 1];
  if (ShapedType::isDynamic(tupleWidth))
    return {};
  if (batchDims < 0 || batchDims > indicesRank - 1 ||
      batchDims + tupleWidth > dataRank)
    return {};

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

LogicalResult mlir::hip::reifyReductionWithKeepdims(
    OpBuilder &b, Location loc, Value data, Value axes, int64_t keepdims,
    int64_t noopWithEmptyAxes, SmallVectorImpl<OpFoldResult> &out) {
  out.clear();
  auto dataType = dyn_cast<RankedTensorType>(data.getType());
  if (!dataType)
    return failure();
  ArrayRef<int64_t> dataShape = dataType.getShape();
  int64_t dataRank = dataType.getRank();

  // Axes operand: try to fold to a constant int vector. ONNX semantics
  // allow a size-0 vector to mean "no axes specified" — combined with the
  // `noop_with_empty_axes` attribute that selects between "no-op" and
  // "reduce all axes".
  SmallVector<int64_t> axesList;
  if (!extractConstantInts(axes, axesList))
    return failure();

  // Empty axes branch: "no axes specified" semantics.
  if (axesList.empty()) {
    if (noopWithEmptyAxes != 0) {
      // No-op: output == data shape.
      out.reserve(dataRank);
      for (int64_t i : llvm::seq<int64_t>(0, dataRank))
        out.push_back(reifyDimOrConstant(b, loc, dataShape[i], data, i));
      return success();
    }
    // Reduce all axes: every dim is reduced.
    if (keepdims) {
      out.append(dataRank, b.getIndexAttr(1));
      return success();
    }
    // keepdims=0: output is rank-0 — `out` stays empty (a valid result).
    return success();
  }

  // Normalize negative axes (ONNX convention).
  llvm::SmallSet<int64_t, 8> reducedSet;
  for (int64_t a : axesList) {
    if (a < 0)
      a += dataRank;
    if (a < 0 || a >= dataRank)
      return failure();
    reducedSet.insert(a);
  }

  out.reserve(dataRank);
  for (int64_t i : llvm::seq<int64_t>(0, dataRank)) {
    if (reducedSet.contains(i)) {
      if (keepdims)
        out.push_back(b.getIndexAttr(1));
      // else: drop the dim from the output.
    } else {
      out.push_back(reifyDimOrConstant(b, loc, dataShape[i], data, i));
    }
  }
  return success();
}

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
