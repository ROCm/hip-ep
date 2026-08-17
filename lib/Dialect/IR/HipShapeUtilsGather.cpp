/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- HipShapeUtilsGather.cpp - Transpose and gather shape helpers -------===//
//
// Category implementation for the public shape helpers declared in
// `hip/Dialect/IR/HipShapeUtils.h`.
//
//===----------------------------------------------------------------------===//

#include "HipShapeUtilsInternal.h"
#include "hip/Dialect/IR/HipShapeUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Traits.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <limits>
#include <optional>

using namespace mlir;
using namespace mlir::hip;

FailureOr<SmallVector<int64_t>>
mlir::hip::inferTransposeShape(ArrayRef<int64_t> inputShape,
                               ArrayRef<int64_t> perm) {
  if (failed(detail::validatePermutation(perm, inputShape.size())))
    return failure();
  SmallVector<int64_t> result;
  result.reserve(perm.size());
  for (int64_t dim : perm)
    result.push_back(inputShape[dim]);
  return result;
}

FailureOr<SmallVector<OpFoldResult>>
mlir::hip::reifyTransposeByPerm(OpBuilder &b, Location loc, Value input,
                                ArrayRef<int64_t> perm) {
  auto inputType = dyn_cast<RankedTensorType>(input.getType());
  if (!inputType)
    return failure();
  ArrayRef<int64_t> inputShape = inputType.getShape();
  // Validate the entire permutation before materializing any tensor.dim. A
  // failure must leave the IR unchanged.
  if (failed(inferTransposeShape(inputShape, perm)))
    return failure();

  SmallVector<OpFoldResult> dims;
  dims.reserve(perm.size());
  for (int64_t dim : perm)
    dims.push_back(reifyDimOrConstant(b, loc, inputShape[dim], input, dim));
  return dims;
}

FailureOr<SmallVector<int64_t>>
mlir::hip::inferGatherShape(ArrayRef<int64_t> dataShape,
                            ArrayRef<int64_t> indicesShape, int64_t axis) {
  int64_t dataRank = dataShape.size();
  if (axis < 0)
    axis += dataRank;
  if (axis < 0 || axis >= dataRank)
    return failure();

  SmallVector<int64_t> result;
  result.reserve(dataRank - 1 + indicesShape.size());
  llvm::append_range(result, dataShape.take_front(axis));
  llvm::append_range(result, indicesShape);
  llvm::append_range(result, dataShape.drop_front(axis + 1));
  return result;
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
  if (failed(
          inferGatherShape(dataType.getShape(), indicesType.getShape(), axis)))
    return failure();
  // Negative-axis normalization (ONNX convention).
  if (axis < 0)
    axis += dataRank;

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

FailureOr<SmallVector<int64_t>> mlir::hip::inferGatherBlockQuantizedShape(
    ArrayRef<int64_t> dataShape, ArrayRef<int64_t> indicesShape,
    ArrayRef<int64_t> scalesShape,
    std::optional<ArrayRef<int64_t>> zeroPointsShape, int64_t bits,
    int64_t blockSize, int64_t gatherAxis, int64_t quantizeAxis,
    bool bytePackedInt4, bool uint8Storage,
    function_ref<InFlightDiagnostic()> emitError) {
  constexpr int64_t maxRank = 8;
  int64_t dataRank = dataShape.size();
  int64_t indicesRank = indicesShape.size();
  if (dataRank <= 1 || dataRank > maxRank) {
    emitError() << "gather_block_quantized data rank must be in [2, " << maxRank
                << "], got " << dataRank;
    return failure();
  }
  if (indicesRank > maxRank || indicesRank + dataRank - 1 > maxRank) {
    emitError() << "gather_block_quantized output rank "
                << indicesRank + dataRank - 1 << " exceeds the runtime maximum "
                << maxRank;
    return failure();
  }
  if (scalesShape.size() != dataShape.size()) {
    emitError() << "gather_block_quantized scales rank must match data rank";
    return failure();
  }
  if (zeroPointsShape && zeroPointsShape->size() != dataShape.size()) {
    emitError()
        << "gather_block_quantized zero_points rank must match data rank";
    return failure();
  }
  if (bits != 4 && bits != 8) {
    emitError() << "gather_block_quantized bits must be 4 or 8";
    return failure();
  }
  if (blockSize < 16 || (blockSize & (blockSize - 1)) != 0) {
    emitError() << "gather_block_quantized block_size must be a power of two "
                   "and at least 16";
    return failure();
  }
  if (bytePackedInt4 && bits != 4) {
    emitError() << "gather_block_quantized byte-packed int4 storage requires "
                   "bits = 4";
    return failure();
  }

  auto normalizeAxis = [dataRank](int64_t axis) {
    return axis < 0 ? axis + dataRank : axis;
  };
  int64_t normalizedGatherAxis = normalizeAxis(gatherAxis);
  int64_t normalizedQuantizeAxis = normalizeAxis(quantizeAxis);
  if (normalizedGatherAxis < 0 || normalizedGatherAxis >= dataRank) {
    emitError() << "gather_block_quantized gather_axis must be in [-"
                << dataRank << ", " << dataRank - 1 << "]";
    return failure();
  }
  if (normalizedQuantizeAxis < 0 || normalizedQuantizeAxis >= dataRank) {
    emitError() << "gather_block_quantized quantize_axis must be in [-"
                << dataRank << ", " << dataRank - 1 << "]";
    return failure();
  }
  if (uint8Storage && normalizedGatherAxis != 0) {
    emitError()
        << "gather_block_quantized gather_axis must be 0 for uint8 storage";
    return failure();
  }
  if (uint8Storage && normalizedQuantizeAxis != dataRank - 1) {
    emitError() << "gather_block_quantized quantize_axis must be the last "
                   "dimension for uint8 storage";
    return failure();
  }
  if (uint8Storage && normalizedGatherAxis == normalizedQuantizeAxis) {
    emitError() << "gather_block_quantized gather_axis and quantize_axis must "
                   "differ for uint8 storage";
    return failure();
  }

  SmallVector<int64_t> logicalDataShape(dataShape);
  int64_t packedDim = dataShape[normalizedQuantizeAxis];
  if (bytePackedInt4 && !ShapedType::isDynamic(packedDim)) {
    APInt logicalExtent = APInt(128, packedDim, /*isSigned=*/true) * 2;
    if (!logicalExtent.isSignedIntN(64) || logicalExtent.isNegative()) {
      emitError()
          << "gather_block_quantized logical quantized extent overflows i64";
      return failure();
    }
    logicalDataShape[normalizedQuantizeAxis] = logicalExtent.getSExtValue();
  }

  for (int64_t i : llvm::seq<int64_t>(0, dataRank)) {
    int64_t dataDim = logicalDataShape[i];
    int64_t scalesDim = scalesShape[i];
    if (ShapedType::isDynamic(dataDim) || ShapedType::isDynamic(scalesDim))
      continue;
    int64_t expectedScalesDim =
        i == normalizedQuantizeAxis
            ? dataDim / blockSize + (dataDim % blockSize != 0)
            : dataDim;
    if (scalesDim != expectedScalesDim) {
      emitError() << "gather_block_quantized data/scales mismatch at axis " << i
                  << ": expected scales extent " << expectedScalesDim
                  << " but got " << scalesDim;
      return failure();
    }
  }

  if (zeroPointsShape) {
    for (int64_t i : llvm::seq<int64_t>(0, dataRank)) {
      int64_t scalesDim = scalesShape[i];
      int64_t zeroPointsDim = (*zeroPointsShape)[i];
      if (ShapedType::isDynamic(scalesDim) ||
          ShapedType::isDynamic(zeroPointsDim))
        continue;
      bool compatible = zeroPointsDim == scalesDim;
      if (bytePackedInt4 && i == normalizedQuantizeAxis)
        compatible |= zeroPointsDim == scalesDim / 2 + (scalesDim % 2 != 0);
      if (!compatible) {
        emitError() << "gather_block_quantized zero_points extent at axis " << i
                    << " must match scales"
                    << (bytePackedInt4 && i == normalizedQuantizeAxis
                            ? " or its packed-byte extent"
                            : "");
        return failure();
      }
    }
  }

  SmallVector<int64_t> result;
  result.reserve(indicesRank + dataRank - 1);
  llvm::append_range(
      result,
      ArrayRef<int64_t>(logicalDataShape).take_front(normalizedGatherAxis));
  llvm::append_range(result, indicesShape);
  llvm::append_range(
      result,
      ArrayRef<int64_t>(logicalDataShape).drop_front(normalizedGatherAxis + 1));
  return result;
}

FailureOr<SmallVector<OpFoldResult>> mlir::hip::reifyGatherBlockQuantizedShape(
    OpBuilder &b, Location loc, Value data, Value indices, Value scales,
    Value zeroPoints, int64_t bits, int64_t blockSize, int64_t gatherAxis,
    int64_t quantizeAxis, bool bytePackedInt4, bool uint8Storage,
    function_ref<InFlightDiagnostic()> emitError) {
  auto dataType = dyn_cast<RankedTensorType>(data.getType());
  auto indicesType = dyn_cast<RankedTensorType>(indices.getType());
  auto scalesType = dyn_cast<RankedTensorType>(scales.getType());
  auto zeroPointsType = zeroPoints
                            ? dyn_cast<RankedTensorType>(zeroPoints.getType())
                            : RankedTensorType();
  if (!dataType || !indicesType || !scalesType ||
      (zeroPoints && !zeroPointsType)) {
    emitError() << "gather_block_quantized reification requires ranked tensor "
                   "operands";
    return failure();
  }

  std::optional<ArrayRef<int64_t>> zeroPointsShape;
  if (zeroPointsType)
    zeroPointsShape = zeroPointsType.getShape();
  FailureOr<SmallVector<int64_t>> inferred = inferGatherBlockQuantizedShape(
      dataType.getShape(), indicesType.getShape(), scalesType.getShape(),
      zeroPointsShape, bits, blockSize, gatherAxis, quantizeAxis,
      bytePackedInt4, uint8Storage, emitError);
  if (failed(inferred))
    return failure();

  int64_t dataRank = dataType.getRank();
  int64_t normalizedGatherAxis =
      gatherAxis < 0 ? gatherAxis + dataRank : gatherAxis;
  int64_t normalizedQuantizeAxis =
      quantizeAxis < 0 ? quantizeAxis + dataRank : quantizeAxis;
  auto reifyDataDim = [&](int64_t dim) -> FailureOr<OpFoldResult> {
    int64_t staticDim = dataType.getDimSize(dim);
    if (!bytePackedInt4 || dim != normalizedQuantizeAxis)
      return reifyDimOrConstant(b, loc, staticDim, data, dim);
    return detail::scaleAndOffsetDim(
        b, loc, tensor::getMixedSize(b, loc, data, dim), /*scale=*/2,
        /*offset=*/0);
  };

  SmallVector<OpFoldResult> result;
  result.reserve(indicesType.getRank() + dataRank - 1);
  for (int64_t i : llvm::seq<int64_t>(0, normalizedGatherAxis)) {
    FailureOr<OpFoldResult> extent = reifyDataDim(i);
    if (failed(extent))
      return failure();
    result.push_back(*extent);
  }
  for (int64_t i : llvm::seq<int64_t>(0, indicesType.getRank()))
    result.push_back(
        reifyDimOrConstant(b, loc, indicesType.getDimSize(i), indices, i));
  for (int64_t i : llvm::seq<int64_t>(normalizedGatherAxis + 1, dataRank)) {
    FailureOr<OpFoldResult> extent = reifyDataDim(i);
    if (failed(extent))
      return failure();
    result.push_back(*extent);
  }
  return result;
}

FailureOr<SmallVector<int64_t>>
mlir::hip::inferGatherNDShape(ArrayRef<int64_t> dataShape,
                              ArrayRef<int64_t> indicesShape,
                              int64_t batchDims) {
  int64_t dataRank = dataShape.size();
  int64_t indicesRank = indicesShape.size();
  if (indicesRank < 1 || batchDims < 0 || batchDims > indicesRank - 1)
    return failure();

  int64_t tupleWidth = indicesShape.back();
  if (ShapedType::isDynamic(tupleWidth) || tupleWidth < 0 ||
      tupleWidth > dataRank - batchDims)
    return failure();
  for (int64_t dim : llvm::seq<int64_t>(0, batchDims)) {
    int64_t dataDim = dataShape[dim];
    int64_t indicesDim = indicesShape[dim];
    if (!ShapedType::isDynamic(dataDim) && !ShapedType::isDynamic(indicesDim) &&
        dataDim != indicesDim)
      return failure();
  }

  SmallVector<int64_t> result;
  result.reserve(indicesRank - 1 + dataRank - tupleWidth - batchDims);
  llvm::append_range(result, dataShape.take_front(batchDims));
  llvm::append_range(
      result, indicesShape.slice(batchDims, indicesRank - 1 - batchDims));
  llvm::append_range(result, dataShape.drop_front(batchDims + tupleWidth));
  return result;
}

FailureOr<SmallVector<OpFoldResult>>
mlir::hip::reifyGatherND(OpBuilder &b, Location loc, Value data, Value indices,
                         int64_t batchDims) {
  auto dataType = dyn_cast<RankedTensorType>(data.getType());
  auto indicesType = dyn_cast<RankedTensorType>(indices.getType());
  if (!dataType || !indicesType || !indicesType.getElementType().isInteger(64))
    return failure();
  int64_t dataRank = dataType.getRank();
  int64_t indicesRank = indicesType.getRank();
  if (indicesRank < 1)
    return failure();
  ArrayRef<int64_t> dataShape = dataType.getShape();
  ArrayRef<int64_t> indicesShape = indicesType.getShape();

  // Validate every static precondition before materializing dimensions.
  if (failed(inferGatherNDShape(dataShape, indicesShape, batchDims)))
    return failure();
  int64_t tupleWidth = indicesShape[indicesRank - 1];

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
