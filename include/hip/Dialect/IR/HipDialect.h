//===- HipDialect.h - HIP dialect class declarations ---------- *- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.  All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
#ifndef HIP_DIALECT_IR_HIPDIALECT_H
#define HIP_DIALECT_IR_HIPDIALECT_H

#include "hip/Dialect/IR/HipDialect.h.inc"

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_TYPEDEF_CLASSES
#include "hip/Dialect/IR/HipTypes.h.inc"

#define GET_OP_CLASSES
#include "hip/Dialect/IR/HipOps.h.inc"

#endif // HIP_DIALECT_IR_HIPDIALECT_H
