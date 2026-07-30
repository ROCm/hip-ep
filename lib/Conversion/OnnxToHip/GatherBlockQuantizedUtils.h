/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- GatherBlockQuantizedUtils.h - GBQ shape / storage helpers ----------===//
//
// Shared helpers for com.microsoft.GatherBlockQuantized lowering. Maps ONNX
// (T1 element type, bits, data/scales shapes) to quantize_axis and unsigned
// storage semantics. See hip_custom_kernels.h for the runtime contract.
//
//===----------------------------------------------------------------------===//

#ifndef HIP_CONVERSION_GATHERBLOCKQUANTIZED_UTILS_H
#define HIP_CONVERSION_GATHERBLOCKQUANTIZED_UTILS_H

#include "hip/Dialect/IR/HipDialect.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "llvm/ADT/SmallVector.h"
#include <optional>

namespace mlir {
namespace hip {
namespace gbq {

/// MLIR ui8 element type => unsigned quant storage (uint4/uint8 semantics).
inline bool isUnsignedMlirElementType(Type elemType) {
  if (auto intTy = dyn_cast<IntegerType>(elemType))
    return intTy.isUnsigned();
  return false;
}

/// Walk backward through layout-preserving ops to the underlying value.
inline Value traceGbqDataToSource(Value value) {
  Value cur = value;
  for (int depth = 0; depth < 12; ++depth) {
    Operation *op = cur.getDefiningOp();
    if (!op)
      break;
    if (isa<bufferization::ToTensorOp>(op)) {
      cur = op->getOperand(0);
      continue;
    }
    if (op->getName().getStringRef() == "onnx.Constant")
      return cur;
    if (op->getNumOperands() == 0)
      break;
    StringRef name = op->getName().getStringRef();
    if (name == "onnx.Reshape" || name == "onnx.Transpose" ||
        name == "onnx.Squeeze" || name == "onnx.Unsqueeze" ||
        name == "onnx.Flatten" || name == "onnx.Gather" ||
        name == "onnx.Identity") {
      cur = op->getOperand(0);
      continue;
    }
    break;
  }
  return cur;
}

/// Resolve unsigned storage for GBQ from ONNX rules + MLIR types.
/// bits==8 always uses uint8 semantics. bits==4 uses ui8 source type or attr.
inline bool resolveUnsignedQuantStorage(int64_t bits, bool hasUnsignedAttr,
                                        Type dataElemType, Value dataValue) {
  if (hasUnsignedAttr)
    return true;
  if (bits == 8)
    return true;
  if (isUnsignedMlirElementType(dataElemType))
    return true;

  Value source = traceGbqDataToSource(dataValue);
  if (auto global = source.getDefiningOp<memref::GetGlobalOp>()) {
    if (isUnsignedMlirElementType(global.getType().getElementType()))
      return true;
  }
  if (auto *constOp = source.getDefiningOp()) {
    if (constOp->getName().getStringRef() == "onnx.Constant") {
      if (auto ty = dyn_cast<RankedTensorType>(constOp->getResult(0).getType()))
        if (isUnsignedMlirElementType(ty.getElementType()))
          return true;
    }
  }
  return false;
}

/// Returns true when `axis` satisfies GBQ block-quant shape invariants.
inline bool quantizeAxisMatches(ArrayRef<int64_t> dataShape,
                                ArrayRef<int64_t> scalesShape, int axis,
                                int64_t blockSize, int64_t bits) {
  if (blockSize <= 0 || axis < 0 ||
      axis >= static_cast<int>(dataShape.size()) ||
      axis >= static_cast<int>(scalesShape.size()))
    return false;

  for (int i = 0; i < static_cast<int>(dataShape.size()); ++i) {
    if (i == axis)
      continue;
    if (dataShape[i] != scalesShape[i])
      return false;
  }

  const int64_t dataDim = dataShape[axis];
  const int64_t scaleDim = scalesShape[axis];
  if (bits == 4) {
    if (scaleDim * blockSize == dataDim)
      return true;
    if (scaleDim * blockSize == dataDim * 2)
      return true;
    return false;
  }
  return scaleDim * blockSize == dataDim;
}

/// Infer quantize_axis from static shapes. Fails when zero or ambiguous
/// matches.
inline FailureOr<int64_t> inferQuantizeAxis(RankedTensorType dataType,
                                            RankedTensorType scalesType,
                                            int64_t blockSize, int64_t bits) {
  if (!dataType || !scalesType || blockSize <= 0)
    return failure();

  auto dataShape = dataType.getShape();
  auto scalesShape = scalesType.getShape();
  if (dataShape.size() != scalesShape.size())
    return failure();

  llvm::SmallVector<int, 4> matches;
  for (int axis = 0; axis < static_cast<int>(dataShape.size()); ++axis) {
    if (quantizeAxisMatches(dataShape, scalesShape, axis, blockSize, bits))
      matches.push_back(axis);
  }
  if (matches.size() != 1)
    return failure();
  return matches.front();
}

/// Resolve quantize_axis: use explicit attr when present, else infer.
inline FailureOr<int64_t>
resolveQuantizeAxis(RankedTensorType dataType, RankedTensorType scalesType,
                    int64_t blockSize, int64_t bits,
                    std::optional<int64_t> explicitAxis) {
  if (explicitAxis.has_value()) {
    int64_t axis = *explicitAxis;
    int rank = static_cast<int>(dataType.getRank());
    if (axis < 0)
      axis += rank;
    if (axis < 0 || axis >= rank)
      return failure();
    if (!quantizeAxisMatches(dataType.getShape(), scalesType.getShape(),
                             static_cast<int>(axis), blockSize, bits))
      return failure();
    return axis;
  }
  return inferQuantizeAxis(dataType, scalesType, blockSize, bits);
}

} // namespace gbq
} // namespace hip
} // namespace mlir

#endif // HIP_CONVERSION_GATHERBLOCKQUANTIZED_UTILS_H
