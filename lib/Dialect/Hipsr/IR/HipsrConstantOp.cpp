/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/Hipsr/IR/HipsrConstantOp.h"

using namespace mlir;
using namespace mlir::hipsr;

LogicalResult ConstantOp::verify() {
  bool hasValue = getValueAttr() != nullptr;
  bool hasSource = getSourceAttr() != nullptr;
  bool hasOffset = getOffsetAttr() != nullptr;
  bool hasSize = getSizeAttr() != nullptr;

  // A data source is always present: exactly one of value or source (never
  // both, never neither). Externalization only *adds* offset/size; it never
  // removes the data source, so getDataValues() keeps working
  // post-externalization.
  //   {value} | {source} | {value, offset, size} | {source, offset, size}
  if (hasValue == hasSource)
    return emitOpError("expected exactly one of {value} or {source}");

  // offset/size are the externalized-location marker; set together or not.
  if (hasOffset != hasSize)
    return emitOpError("`offset` and `size` must be set together");

  // Constants reside in VRAM: the result memref must be device memory.
  MemRefType memrefType = getResult().getType();
  auto deviceSpace =
      llvm::dyn_cast_or_null<MemorySpaceAttr>(memrefType.getMemorySpace());
  if (!deviceSpace || deviceSpace.getKind() != MemorySpaceKind::Device)
    return emitOpError("result must have #hipsr.mem<device> memory space");

  return success();
}

#define GET_OP_CLASSES
#include "hip/Dialect/Hipsr/IR/HipsrConstantOp.cpp.inc"
