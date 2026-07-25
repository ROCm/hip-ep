/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/Hipsr/IR/HipsrEmptyYieldOp.h"

#include "hip/Dialect/Hipsr/IR/HipsrEmptyOp.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::hipsr;

LogicalResult EmptyYieldOp::verify() {
  for (auto [idx, tensor] : llvm::enumerate(getTensors())) {
    if (!tensor.getDefiningOp<tensor::EmptyOp>()) {
      return emitOpError("operand #")
             << idx << " must be a tensor.empty result";
    }
  }
  return success();
}

#define GET_OP_CLASSES
#include "hip/Dialect/Hipsr/IR/HipsrEmptyYieldOp.cpp.inc"
