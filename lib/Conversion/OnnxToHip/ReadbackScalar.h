/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- ReadbackScalar.h - synchronized host readback of GPU scalars ------===//
//
// Several converters need a 0-D tensor value (a Range bound, a scalar Reshape
// target, a dynamic-Reshape `-1` shape entry, an Expand extent, a Loop trip
// count) on the HOST. That value is frequently GPU-computed -- canonical case:
// shape arithmetic packed via `tensor.from_elements` from values read out of an
// input like `image_grid_thw`. A bare `tensor.extract` lowers to an
// UNSYNCHRONIZED host `memref.load` of a device buffer and reads stale bytes on
// targets where the pool is true device memory (it only accidentally works on
// UMA-mapped host-accessible pools).
//
// These helpers fold compile-time constants (no device traffic) and otherwise
// emit `hip.readback_scalar` (D2H + stream sync) so the host observes what the
// producing kernel actually wrote.
//
//   Before:  %v = tensor.extract %t[]              // host load of device mem
//   After:   %v = hip.readback_scalar(%ctx, %t : tensor<i64>) -> i64
//
//===----------------------------------------------------------------------===//

#ifndef HIP_CONVERSION_ONNXTOHIP_READBACKSCALAR_H
#define HIP_CONVERSION_ONNXTOHIP_READBACKSCALAR_H

#include "hip/Dialect/IR/HipDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace hip {

/// `dense` attribute (arith.constant or inline onnx.Constant `value`) backing
/// `v`, or null when `v` is not such a constant.
inline mlir::DenseElementsAttr getConstantDense(mlir::Value v) {
  if (auto cst = v.getDefiningOp<mlir::arith::ConstantOp>())
    return mlir::dyn_cast<mlir::DenseElementsAttr>(cst.getValue());
  if (mlir::Operation *def = v.getDefiningOp())
    if (def->getName().getStringRef() == "onnx.Constant")
      return def->getAttrOfType<mlir::DenseElementsAttr>("value");
  return nullptr;
}

/// Materialize element `idx` of `dense` (int or float) as a host scalar const.
inline mlir::Value materializeConstScalar(mlir::OpBuilder &b,
                                          mlir::Location loc,
                                          mlir::DenseElementsAttr dense,
                                          mlir::Type elemTy, int64_t idx) {
  if (auto ity = mlir::dyn_cast<mlir::IntegerType>(elemTy))
    return mlir::arith::ConstantIntOp::create(
        b, loc, ity,
        (*(dense.getValues<llvm::APInt>().begin() + idx)).getSExtValue());
  auto fty = mlir::cast<mlir::FloatType>(elemTy);
  return mlir::arith::ConstantFloatOp::create(
      b, loc, fty, *(dense.getValues<llvm::APFloat>().begin() + idx));
}

/// Read a 0-D scalar tensor to a host SSA value. `ctx` must be a valid
/// !hip.context. Folds an arith.constant / onnx.Constant; otherwise emits a
/// synchronized hip.readback_scalar. See the file header for the race
/// rationale.
inline mlir::Value readbackScalarToHost(mlir::OpBuilder &b, mlir::Location loc,
                                        mlir::Value ctx,
                                        mlir::Value rank0Tensor) {
  mlir::Type elemTy = mlir::cast<mlir::RankedTensorType>(rank0Tensor.getType())
                          .getElementType();
  if (mlir::DenseElementsAttr dense = getConstantDense(rank0Tensor))
    if (dense.getNumElements() == 1)
      return materializeConstScalar(b, loc, dense, elemTy, 0);
  return mlir::hip::ReadbackScalarOp::create(b, loc, elemTy, ctx, rank0Tensor)
      .getResult();
}

/// Read element `idx` of a 1-D shape tensor to a host SSA value. `ctx` must be
/// a valid !hip.context. Folds a constant shape; otherwise slices the entry to
/// rank-0 and emits a synchronized hip.readback_scalar.
inline mlir::Value readbackShapeEntryToHost(mlir::OpBuilder &b,
                                            mlir::Location loc, mlir::Value ctx,
                                            mlir::Value shape, int64_t idx) {
  mlir::Type elemTy =
      mlir::cast<mlir::RankedTensorType>(shape.getType()).getElementType();
  if (mlir::DenseElementsAttr dense = getConstantDense(shape))
    if (idx < dense.getNumElements())
      return materializeConstScalar(b, loc, dense, elemTy, idx);
  llvm::SmallVector<mlir::OpFoldResult> offsets{b.getIndexAttr(idx)};
  llvm::SmallVector<mlir::OpFoldResult> sizes{b.getIndexAttr(1)};
  llvm::SmallVector<mlir::OpFoldResult> strides{b.getIndexAttr(1)};
  mlir::Value entry1d = mlir::tensor::ExtractSliceOp::create(
      b, loc, shape, offsets, sizes, strides);
  mlir::Value entry0d = mlir::tensor::CollapseShapeOp::create(
      b, loc, mlir::RankedTensorType::get({}, elemTy), entry1d,
      llvm::ArrayRef<mlir::ReassociationIndices>{});
  return mlir::hip::ReadbackScalarOp::create(b, loc, elemTy, ctx, entry0d)
      .getResult();
}

} // namespace hip
} // namespace mlir

#endif // HIP_CONVERSION_ONNXTOHIP_READBACKSCALAR_H
