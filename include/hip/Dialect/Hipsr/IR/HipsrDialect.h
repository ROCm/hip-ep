/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef HIPSR_DIALECT_H
#define HIPSR_DIALECT_H

#include "mlir/IR/Dialect.h"
// Needed by the generated attribute parser/printer.
#include "mlir/IR/OpImplementation.h"

#include "hip/Dialect/Hipsr/IR/HipsrDialect.h.inc"

// Enum header first: MemorySpaceAttr uses MemorySpaceKind.
#include "hip/Dialect/Hipsr/IR/HipsrEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "hip/Dialect/Hipsr/IR/HipsrAttrs.h.inc"

// Predicates for the Hipsr_*MemRef type constraints (used by op verifiers once
// ops adopt them).
#include "hip/Dialect/Hipsr/IR/HipsrTypes.h"

#endif // HIPSR_DIALECT_H
