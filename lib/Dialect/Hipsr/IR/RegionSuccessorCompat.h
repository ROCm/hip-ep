/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef HIPSR_REGION_SUCCESSOR_COMPAT_H
#define HIPSR_REGION_SUCCESSOR_COMPAT_H

#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include <type_traits>

namespace mlir::hipsr::detail {
// LLVM's parent-edge representation changed from parent() to an operation
// constructor. Preserve the exit edge with either supported header version.
template <typename Successor = RegionSuccessor>
Successor parentSuccessor(Operation *operation) {
  if constexpr (std::is_constructible_v<Successor, Operation *>)
    return Successor(operation);
  else
    return Successor::parent();
}
} // namespace mlir::hipsr::detail

#endif
