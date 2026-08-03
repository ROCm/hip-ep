/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- HipShapeUtilsInternal.cpp - Private shape utility details ----------===//

#include "HipShapeUtilsInternal.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;

ArrayRef<int64_t> mlir::hip::detail::getShapeOf(Value value) {
  if (auto tensorType = dyn_cast<RankedTensorType>(value.getType()))
    return tensorType.getShape();
  if (auto memrefType = dyn_cast<MemRefType>(value.getType()))
    return memrefType.getShape();
  return {};
}

SmallVector<int64_t> mlir::hip::detail::getI64Array(ArrayAttr attr) {
  SmallVector<int64_t> values;
  values.reserve(attr.size());
  for (Attribute value : attr)
    values.push_back(cast<IntegerAttr>(value).getInt());
  return values;
}
