/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef HIP_DIALECT_H
#define HIP_DIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "HipDialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "HipTypes.h.inc"

#define GET_OP_CLASSES
#include "HipOps.h.inc"

#endif // HIP_DIALECT_H
