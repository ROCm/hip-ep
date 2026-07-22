/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/Hipsr/IR/HipsrEmptyYieldOp.h"

// Needs the full EmptyOp type for the HasParent trait's generated check.
#include "hip/Dialect/Hipsr/IR/HipsrEmptyOp.h"

using namespace mlir;
using namespace mlir::hipsr;

#define GET_OP_CLASSES
#include "hip/Dialect/Hipsr/IR/HipsrEmptyYieldOp.cpp.inc"
