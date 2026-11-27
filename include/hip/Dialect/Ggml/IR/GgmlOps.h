/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef HIP_DIALECT_GGML_IR_GGMLOPS_H
#define HIP_DIALECT_GGML_IR_GGMLOPS_H

#include "hip/Dialect/Ggml/IR/GgmlDialect.h"
#include "hip/Dialect/Ggml/IR/GgmlShapeRule.h"

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

#define GET_OP_CLASSES
#include "hip/Dialect/Ggml/IR/GgmlOps.h.inc"

#endif // HIP_DIALECT_GGML_IR_GGMLOPS_H
