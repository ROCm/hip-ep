/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/Onnx/IR/OnnxDialect.h"

#include "hip/Dialect/Onnx/IR/OnnxOps.h"

using namespace mlir;
using namespace mlir::onnx;

#include "hip/Dialect/Onnx/IR/OnnxDialect.cpp.inc"

void OnnxDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "hip/Dialect/Onnx/IR/OnnxOps.cpp.inc"
      >();
  // A contrib-domain operation, or a standard one whose signature needs an ONNX
  // container type the dialect cannot spell, still has to parse and print.
  allowUnknownOperations();
}
