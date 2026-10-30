/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef HIP_DIALECT_TRANSFORMS_REBIASUNSIGNEDQUANTIZE_H
#define HIP_DIALECT_TRANSFORMS_REBIASUNSIGNEDQUANTIZE_H

#include <mlir/Dialect/Func/IR/FuncOps.h>

namespace mlir::hip {

/// Retypes each connected tree of ui8 quantized values that reaches an integer
/// anchor to i8 and lowers every zero point in it by 128, so the anchor reads
/// signed bytes with no shift. The tree runs from quantizes and QLinear
/// results through ElementMoveOpInterface ops to dequantizes and QLinear
/// inputs. `isIntegerAnchor` tells which QLinear ops and dequantized convs
/// will become integer anchors; only those may join a tree.
void rebiasUnsignedQuantizedTrees(
    func::FuncOp funcOp, llvm::function_ref<bool(Operation *)> isIntegerAnchor);

} // namespace mlir::hip

#endif // HIP_DIALECT_TRANSFORMS_REBIASUNSIGNEDQUANTIZE_H
