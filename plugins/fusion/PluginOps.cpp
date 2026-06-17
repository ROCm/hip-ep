/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- PluginOps.cpp - Out-of-tree hip.fused_mul_add op definitions -------===//
//
// Op-class definitions for the plugin-contributed `hip.fused_mul_add` op, plus
// the DPS / memory-effect interface methods declared in FusedMulAddOps.td.
//
//===----------------------------------------------------------------------===//

#include "plugins/fusion/PluginOps.h"

#define GET_OP_CLASSES
#include "plugins/fusion/FusedMulAddOps.cpp.inc"

namespace mlir {
namespace hip {

// DPS init operand: the single `output` destination buffer/tensor.
MutableOperandRange FusedMulAddOp::getDpsInitsMutable() {
  return getOutputMutable();
}

// Memref inputs are reads, memref inits are writes; tensor / !hip.context
// operands carry no memory effect. Mirrors the in-tree emitDpsMemoryEffects.
void FusedMulAddOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  for (OpOperand *operand : getDpsInputOperands()) {
    if (!isa<MemRefType>(operand->get().getType()))
      continue;
    effects.emplace_back(MemoryEffects::Read::get(), operand,
                         SideEffects::DefaultResource::get());
  }
  for (OpOperand &operand : getDpsInitsMutable()) {
    if (!isa<MemRefType>(operand.get().getType()))
      continue;
    effects.emplace_back(MemoryEffects::Write::get(), &operand,
                         SideEffects::DefaultResource::get());
  }
}

} // namespace hip
} // namespace mlir
