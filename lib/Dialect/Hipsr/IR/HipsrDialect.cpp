/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/Hipsr/IR/HipsrDialect.h"

#include "hip/Dialect/Hipsr/IR/HipsrInfrastructureOps.h"
#include "hip/Dialect/Hipsr/IR/HipsrOps.h"

using namespace mlir;
using namespace mlir::hipsr;

#include "hip/Dialect/Hipsr/IR/HipsrDialect.cpp.inc"

void HipsrDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "hip/Dialect/Hipsr/IR/HipsrInfrastructureOps.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "hip/Dialect/Hipsr/IR/HipsrOps.cpp.inc"
      >();
}
