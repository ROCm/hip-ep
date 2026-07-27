/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef HIPSR_OPS_H
#define HIPSR_OPS_H

#include "hip/Dialect/Hipsr/IR/HipsrDialect.h"
#include "hip/Dialect/Hipsr/IR/HipsrShapeRegionInterface.h"

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir {
namespace hipsr {
class EmptyOp;
class EmptyYieldOp;
class PoolDomainOp;
class PoolDomainYieldOp;
} // namespace hipsr
} // namespace mlir

#define GET_OP_CLASSES
#include "hip/Dialect/Hipsr/IR/HipsrOps.h.inc"

#endif // HIPSR_OPS_H
