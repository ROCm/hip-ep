//===- MiniONNX.h - MiniONNX Dialect ----------------------------*- C++ -*-===//
//
// Minimal ONNX dialect for hip-ep conversion.
//
//===----------------------------------------------------------------------===//

#ifndef HIP_DIALECT_MINIONNX_IR_MINIONNX_H
#define HIP_DIALECT_MINIONNX_IR_MINIONNX_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "hip/Dialect/MiniONNX/IR/MiniONNXDialect.h.inc"

#define GET_OP_CLASSES
#include "hip/Dialect/MiniONNX/IR/MiniONNXOps.h.inc"

#endif // HIP_DIALECT_MINIONNX_IR_MINIONNX_H
