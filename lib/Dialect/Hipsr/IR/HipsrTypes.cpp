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

// The #hipsr.mem space a memref names, or null for any other type, for a memref
// with no space, or for a space set by some other attribute -- hipsr requires
// every memref to name its space.
MemorySpaceAttr hipsrSpace(Type type) {
  auto memref = dyn_cast<MemRefType>(type);
  if (!memref) {
    return {};
  }
  return dyn_cast_or_null<MemorySpaceAttr>(memref.getMemorySpace());
}

bool memRefInSpace(Type type, MemorySpace space) {
  MemorySpaceAttr spaceAttr = hipsrSpace(type);
  return spaceAttr && spaceAttr.getValue() == space;
}

bool tensorInSpace(Type type, MemorySpace space) {
  auto tensor = dyn_cast<RankedTensorType>(type);
  if (!tensor) {
    return false;
  }
  auto spaceAttr = dyn_cast_or_null<MemorySpaceAttr>(tensor.getEncoding());
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

bool isAnySpaceMemRef(Type type) { return hipsrSpace(type) != nullptr; }

bool isHostRankedTensor(Type type) {
  return tensorInSpace(type, MemorySpace::Host);
}
bool isDeviceRankedTensor(Type type) {
  return tensorInSpace(type, MemorySpace::Device);
}

} // namespace hipsr
} // namespace mlir
