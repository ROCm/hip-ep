/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef HIPSR_EMPTY_YIELD_OP_H
#define HIPSR_EMPTY_YIELD_OP_H

#include "hip/Dialect/Hipsr/IR/HipsrDialect.h"

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

// The .inc references EmptyOp (from the HasParent trait) only by name, so a
// forward declaration is enough. The full type is needed only in the .cpp,
// which includes its header.
namespace mlir {
namespace hipsr {
class EmptyOp;
} // namespace hipsr
} // namespace mlir

#define GET_OP_CLASSES
#include "hip/Dialect/Hipsr/IR/HipsrEmptyYieldOp.h.inc"

#endif // HIPSR_EMPTY_YIELD_OP_H
