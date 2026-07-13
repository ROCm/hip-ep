/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef HIPSR_INFRASTRUCTURE_OPS_H
#define HIPSR_INFRASTRUCTURE_OPS_H

#include "hip/Dialect/Hipsr/IR/HipsrDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "hip/Dialect/Hipsr/IR/HipsrInfrastructureOps.h.inc"

#endif // HIPSR_INFRASTRUCTURE_OPS_H
