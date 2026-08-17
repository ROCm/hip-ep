/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef HIPSR_TYPES_H
#define HIPSR_TYPES_H

#include "mlir/IR/Types.h"

namespace mlir {
namespace hipsr {

// Predicates for the Hipsr_*MemRef type constraints in HipsrTypes.td. Each
// returns true only when the type is a memref in that memory space, or in any
// of them for isAnySpaceMemRef; a memref with no space (or a space set by some
// other attribute) returns false.
bool isHostMemRef(::mlir::Type type);
bool isDeviceMemRef(::mlir::Type type);
bool isPinnedMemRef(::mlir::Type type);
bool isManagedMemRef(::mlir::Type type);
bool isAnySpaceMemRef(::mlir::Type type);
bool isHostRankedTensor(::mlir::Type type);
bool isDeviceRankedTensor(::mlir::Type type);

} // namespace hipsr
} // namespace mlir

#endif // HIPSR_TYPES_H
