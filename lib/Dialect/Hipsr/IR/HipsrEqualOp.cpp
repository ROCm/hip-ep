/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/Hipsr/IR/HipsrOps.h"

#include "mlir/Dialect/Traits.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::hipsr;

MutableOperandRange EqualOp::getDpsInitsMutable() { return getInitMutable(); }

LogicalResult EqualOp::verify() {
  auto lhsType = cast<ShapedType>(getLhs().getType());
  auto rhsType = cast<ShapedType>(getRhs().getType());
  auto outputType = cast<ShapedType>(getInit().getType());

  if (lhsType.getElementType() != rhsType.getElementType()) {
    return emitOpError("lhs and rhs element types must match");
  }
  // ONNX types the mask as bool, which importers spell as i1 or ui8.
  Type outputElementType = outputType.getElementType();
  if (!outputElementType.isInteger(1) && !outputElementType.isInteger(8)) {
    return emitOpError("output element type must be i1 or an 8-bit integer");
  }

  SmallVector<int64_t> broadcastShape;
  if (!OpTrait::util::getBroadcastedShape(lhsType.getShape(),
                                          rhsType.getShape(), broadcastShape)) {
    return emitOpError("lhs and rhs shapes are not broadcast compatible");
  }
  if (failed(verifyCompatibleShape(broadcastShape, outputType.getShape()))) {
    return emitOpError("output shape must be the broadcast of lhs and rhs");
  }
  return success();
}
