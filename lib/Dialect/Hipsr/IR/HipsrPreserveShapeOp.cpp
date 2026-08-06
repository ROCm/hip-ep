/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/Hipsr/IR/HipsrOps.h"

#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;
using namespace mlir::hipsr;

// PreserveShapeOp does not modify memory, but it must report a side effect
// to avoid being eliminated as a trivially dead operation.
void PreserveShapeOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(),
                       SideEffects::DefaultResource::get());
}
