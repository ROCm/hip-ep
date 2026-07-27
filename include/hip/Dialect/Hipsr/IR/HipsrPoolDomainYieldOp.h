/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef HIPSR_POOL_DOMAIN_YIELD_OP_H
#define HIPSR_POOL_DOMAIN_YIELD_OP_H

#include "hip/Dialect/Hipsr/IR/HipsrDialect.h"

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

// The .inc references PoolDomainOp (from the HasParent trait) only by name, so
// a forward declaration is enough. The full type is needed only in the .cpp,
// which includes its header.
namespace mlir {
namespace hipsr {
class PoolDomainOp;
} // namespace hipsr
} // namespace mlir

#include "hip/Dialect/Hipsr/IR/HipsrOps.h"

#endif // HIPSR_POOL_DOMAIN_YIELD_OP_H
