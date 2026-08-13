/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/Hipsr/IR/HipsrTypes.h"

// Provides MemRefType plus the generated MemorySpaceAttr / MemorySpace.
#include "hip/Dialect/Hipsr/IR/HipsrDialect.h"

#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;
using namespace mlir::hipsr;

namespace {

// True only when the type is a memref whose space is a #hipsr.mem attribute
// naming this space. A memref with no space, or a space set by some other
// attribute, returns false -- hipsr requires every memref to name its space.
bool memRefInSpace(Type type, MemorySpace space) {
  auto memref = dyn_cast<MemRefType>(type);
  if (!memref) {
    return false;
  }
  auto spaceAttr = dyn_cast_or_null<MemorySpaceAttr>(memref.getMemorySpace());
  return spaceAttr && spaceAttr.getValue() == space;
}

} // namespace

namespace mlir {
namespace hipsr {

bool isHostMemRef(Type type) { return memRefInSpace(type, MemorySpace::Host); }
bool isDeviceMemRef(Type type) {
  return memRefInSpace(type, MemorySpace::Device);
}
bool isPinnedMemRef(Type type) {
  return memRefInSpace(type, MemorySpace::Pinned);
}
bool isManagedMemRef(Type type) {
  return memRefInSpace(type, MemorySpace::Managed);
}

} // namespace hipsr
} // namespace mlir
