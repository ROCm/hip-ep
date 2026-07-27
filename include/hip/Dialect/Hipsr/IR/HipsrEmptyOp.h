/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef HIPSR_EMPTY_OP_H
#define HIPSR_EMPTY_OP_H

#include "hip/Dialect/Hipsr/IR/HipsrDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir {
namespace hipsr {
class EmptyYieldOp;
} // namespace hipsr
} // namespace mlir

#include "hip/Dialect/Hipsr/IR/HipsrOps.h"

#endif // HIPSR_EMPTY_OP_H
