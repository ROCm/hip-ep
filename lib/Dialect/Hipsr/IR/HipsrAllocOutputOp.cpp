/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/Hipsr/IR/HipsrOps.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir {
namespace hipsr {

void AllocOutputOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(),
                       SideEffects::DefaultResource::get());
}

LogicalResult AllocOutputOp::verify() {
  auto memrefTy = cast<MemRefType>(getMemref().getType());
  if (static_cast<int64_t>(getDynamicSizes().size()) !=
      memrefTy.getNumDynamicDims())
    return emitOpError("expected ")
           << memrefTy.getNumDynamicDims() << " dynamic size operand(s), got "
           << getDynamicSizes().size();
  return success();
}

} // namespace hipsr
} // namespace mlir
