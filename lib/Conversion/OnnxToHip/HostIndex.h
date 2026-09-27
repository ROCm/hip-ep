/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- HostIndex.h - host-side resolution of integer control tensors -----===//
//
// Slice bounds, Pad amounts and the CumSum axis are integer tensors that the
// runtime wrappers would otherwise copy back from the device and synchronize
// the stream on, every call. In exported graphs they are almost always either
// inline constants or shape arithmetic (`Shape`, `Sub`, `Concat`, ...) whose
// value the host can compute from `tensor.dim` without touching the device.
// These helpers recognize those producer chains during ONNX->HIP conversion
// and materialize each element as a host `index` SSA value.
//
//===----------------------------------------------------------------------===//

#ifndef HIP_CONVERSION_ONNXTOHIP_HOSTINDEX_H
#define HIP_CONVERSION_ONNXTOHIP_HOSTINDEX_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Value.h"

#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace hip {

/// Return the dense-elements attribute backing \p value if it can be
/// determined at compile time. Recognizes arith constants, inspectable
/// value carriers (`onnx.Constant` / `hip.constant` with a `value` attribute)
/// and a legacy initialized-global bridge.
mlir::DenseElementsAttr getCompileTimeConstantTensor(mlir::Value value);

/// Resolve element \p idx (row-major linear index) of an integer tensor \p v to
/// a host `index` SSA value, materializing `arith` ops as needed. Returns a
/// null Value when the element is not host-computable. Ops materialized by a
/// resolution that then fails are pure and get DCE'd.
mlir::Value resolveHostIndex(mlir::OpBuilder &b, mlir::Location loc,
                             mlir::Value v, int64_t idx, int depth = 0);

/// Resolve every element of the statically-shaped integer tensor \p v.
/// Returns false if \p v is not statically shaped or any element fails, in
/// which case \p out is left empty.
bool resolveHostIndices(mlir::OpBuilder &b, mlir::Location loc, mlir::Value v,
                        llvm::SmallVectorImpl<mlir::Value> &out);

} // namespace hip
} // namespace mlir

#endif // HIP_CONVERSION_ONNXTOHIP_HOSTINDEX_H
