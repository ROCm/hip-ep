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

/// True if a kernel exists for `op` with these quantized element types. This is
/// the single capability check used by fusion, legalization and lowering.
bool supportsQuantTypes(Operation *op, TypeRange inputElementTypes,
                        Type resultElementType);

/// True if `op` can carry quant types, i.e. it is the generic op that is
/// quantized or not depending on them. Such ops get `quant_types` on every
/// instance in hip-legalize-quant-types, float or quantized.
bool isQuantCapable(Operation *op);

/// True if `op` is quantized by its quant types: its `quant_types` entries
/// after hip-legalize-quant-types, its tensor element types before.
bool isQuantizedOp(Operation *op);

/// Per-tensor quantized [lhs, rhs, output] element types of an elementwise
/// op: from its tensor types before hip-legalize-quant-types, from its
/// `quant_types` attribute after. Empty for a float op.
SmallVector<quant::UniformQuantizedType, 3> getQuantElementTypes(Operation *op);

/// Verifies `op`'s quant types: either all of `operands` are quantized tensors
/// (and no `quant_types`), or none are and `quant_types` lists one element
/// type per operand: all quantized with matching storage types, or all
/// non-quantized and equal to the operand element types (a float op).
LogicalResult verifyQuantTypes(Operation *op, ValueRange operands);

} // namespace mlir::hip

#endif // HIP_DIALECT_IR_HIPQUANTTYPES_H
