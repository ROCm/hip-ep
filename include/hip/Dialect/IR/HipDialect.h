/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef UDNA_COMPILER_DIALECT_HIP_IR_HIPDIALECT_H
#define UDNA_COMPILER_DIALECT_HIP_IR_HIPDIALECT_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/AllocationOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "udna-compiler/Dialect/Hip/IR/HipDialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "udna-compiler/Dialect/Hip/IR/HipTypes.h.inc"

#define GET_OP_CLASSES
#include "udna-compiler/Dialect/Hip/IR/HipOps.h.inc"

#endif // UDNA_COMPILER_DIALECT_HIP_IR_HIPDIALECT_H
