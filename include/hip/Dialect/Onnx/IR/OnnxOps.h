/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef ONNX_OPS_H
#define ONNX_OPS_H

#include "hip/Dialect/Onnx/IR/OnnxDialect.h"

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include <cstdint>

#define GET_OP_CLASSES
#include "hip/Dialect/Onnx/IR/OnnxOps.h.inc"

#endif // ONNX_OPS_H
