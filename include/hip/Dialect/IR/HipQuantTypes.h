/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// Quantized element types (`!quant.uniform`) on hip tensors.
//
// A hip op whose operands/result are quantized dequantizes its inputs and
// quantizes its output; the same op on float tensors is the float op. The
// types therefore select the kernel, and fusion patterns stay identical for
// QDQ and float models. See docs/design/quant-types.md.
#ifndef HIP_DIALECT_IR_HIPQUANTTYPES_H
#define HIP_DIALECT_IR_HIPQUANTTYPES_H

#include "mlir/Dialect/Quant/IR/QuantTypes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/TypeUtilities.h"

namespace mlir::hip {

/// hip storage integer for `type`: `uiN` if unsigned, signless `iN` if signed.
IntegerType getHipStorageType(quant::QuantizedType type);

/// `type` (scalar or shaped) with any quantized element type replaced by its
/// hip storage type; other types are returned unchanged.
Type toHipStorageType(Type type);

/// True if the shaped or scalar `type` has a quantized element type.
bool isQuantized(Type type);

// Interface queries for any Operation *; an op without the interface is
// answered from its own element types.

/// The interface's isQuantized(), else any quantized operand or result.
bool isQuantizedOp(Operation *op);

/// The interface's getOperandQuantTypes(), else the element types of the
/// shaped operands, then the results.
SmallVector<Type> getOperandQuantTypes(Operation *op);

/// Per-tensor quant types of a quant-capable op (getQuantOperands order);
/// empty unless every one is per-tensor quantized.
SmallVector<quant::UniformQuantizedType, 3> getQuantElementTypes(Operation *op);

/// HipQuantCapableOpInterface's verifier: `operands` are all quantized (no
/// `quant_types`) or none are; then `quant_types` has one entry per operand,
/// all quantized with matching storage types or all equal to the operand types.
LogicalResult verifyQuantTypes(Operation *op, ValueRange operands);

} // namespace mlir::hip

#endif // HIP_DIALECT_IR_HIPQUANTTYPES_H
