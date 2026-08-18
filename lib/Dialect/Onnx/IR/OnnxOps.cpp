/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/Onnx/IR/OnnxOps.h"

using namespace mlir;
using namespace mlir::onnx;

#define GET_OP_CLASSES
#include "hip/Dialect/Onnx/IR/OnnxOps.cpp.inc"
