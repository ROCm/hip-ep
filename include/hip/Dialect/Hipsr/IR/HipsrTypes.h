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
// returns true only when the type is a memref in that memory space; a memref
// with no space (or a space set by some other attribute) returns false.
bool isHostMemRef(::mlir::Type type);
bool isDeviceMemRef(::mlir::Type type);
bool isPinnedMemRef(::mlir::Type type);
bool isManagedMemRef(::mlir::Type type);

// Predicates for the Hipsr_AnyHostRankedTensor / Hipsr_AnyDeviceRankedTensor
// type constraints in HipsrTypes.td. Each returns true only when the type is a
// ranked tensor whose encoding is a #hipsr.mem attribute naming that memory
// space; a tensor with no encoding (or an encoding set by some other attribute)
// returns false.
bool isHostRankedTensor(::mlir::Type type);
bool isDeviceRankedTensor(::mlir::Type type);

} // namespace hipsr
} // namespace mlir

#endif // HIPSR_TYPES_H
