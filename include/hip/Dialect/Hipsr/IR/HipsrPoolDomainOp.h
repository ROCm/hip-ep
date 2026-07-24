/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef HIPSR_POOL_DOMAIN_OP_H
#define HIPSR_POOL_DOMAIN_OP_H

#include "hip/Dialect/Hipsr/IR/HipsrDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir {
namespace hipsr {
class PoolDomainYieldOp;
} // namespace hipsr
} // namespace mlir

#define GET_OP_CLASSES
#include "hip/Dialect/Hipsr/IR/HipsrPoolDomainOp.h.inc"

#endif // HIPSR_POOL_DOMAIN_OP_H
