/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/Hipsr/IR/HipsrOps.h"

#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::hipsr;

MutableOperandRange ScatterNDOp::getDpsInitsMutable() {
  return getInitMutable();
}

LogicalResult ScatterNDOp::verify() {
  auto dataType = cast<ShapedType>(getData().getType());
  auto indicesType = cast<ShapedType>(getIndices().getType());
  auto updatesType = cast<ShapedType>(getUpdates().getType());
  auto outputType = cast<ShapedType>(getInit().getType());

  if (!indicesType.getElementType().isIntOrIndex()) {
    return emitOpError("indices element type must be an integer");
  }
  if (dataType.getElementType() != updatesType.getElementType() ||
      dataType.getElementType() != outputType.getElementType()) {
    return emitOpError("data, updates and output element types must match");
  }
  if (failed(
          verifyCompatibleShape(dataType.getShape(), outputType.getShape()))) {
    return emitOpError("output shape must match the data shape");
  }

  int64_t indicesRank = indicesType.getRank();
  if (indicesRank < 1) {
    return emitOpError("indices must have rank at least one, to hold a "
                       "trailing extent");
  }
  if (indicesType.isDynamicDim(indicesRank - 1)) {
    return emitOpError("the trailing indices extent selects which data axes a "
                       "row addresses, so it must be static");
  }
  int64_t indexDepth = indicesType.getDimSize(indicesRank - 1);
  if (indexDepth < 1 || indexDepth > dataType.getRank()) {
    return emitOpError("the trailing indices extent must be in [1, data rank]; "
                       "data rank is ")
           << dataType.getRank() << ", got " << indexDepth;
  }

  SmallVector<int64_t> expectedUpdates(indicesType.getShape().drop_back());
  llvm::append_range(expectedUpdates,
                     dataType.getShape().drop_front(indexDepth));
  if (failed(verifyCompatibleShape(expectedUpdates, updatesType.getShape()))) {
    return emitOpError("updates shape must be the leading indices extents "
                       "followed by the data extents no row addresses");
  }
  return success();
}
