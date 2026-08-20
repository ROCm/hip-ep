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
  // Two reasons for not using Alloc:
  // 1. The memory allocated by alloc_output is owned by the EP. If
  // MemoryEffects::Alloc
  //    is used, hip-pool-allocs treat it as a transient and pool it and
  //    may incorrectly free it.
  // 2. Marking it as Write prevents the operation from being removed by DCE.
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
