/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/Hipsr/IR/HipsrTypes.h"

// Provides MemRefType plus the generated MemorySpaceAttr / MemorySpaceKind.
#include "hip/Dialect/Hipsr/IR/HipsrDialect.h"

#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;
using namespace mlir::hipsr;

namespace {

// True only when the type is a memref whose space is a #hipsr.mem attribute of
// this kind. A memref with no space, or a space set by some other attribute,
// returns false -- hipsr requires every memref to name its space.
bool memRefInSpace(Type type, MemorySpaceKind kind) {
  auto memref = dyn_cast<MemRefType>(type);
  if (!memref)
    return false;
  auto space = dyn_cast_or_null<MemorySpaceAttr>(memref.getMemorySpace());
  return space && space.getKind() == kind;
}

} // namespace

namespace mlir {
namespace hipsr {

bool isHostMemRef(Type type) {
  return memRefInSpace(type, MemorySpaceKind::Host);
}
bool isDeviceMemRef(Type type) {
  return memRefInSpace(type, MemorySpaceKind::Device);
}
bool isPinnedMemRef(Type type) {
  return memRefInSpace(type, MemorySpaceKind::Pinned);
}
bool isManagedMemRef(Type type) {
  return memRefInSpace(type, MemorySpaceKind::Managed);
}

} // namespace hipsr
} // namespace mlir
