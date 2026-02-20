/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::hip;

#include "HipDialect.cpp.inc"

void HipDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "HipTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "HipOps.cpp.inc"
      >();
}

// Type and op class implementations (parse/print/verify, TypeIDs)
#define GET_TYPEDEF_CLASSES
#include "HipTypes.cpp.inc"

#define GET_OP_CLASSES
#include "HipOps.cpp.inc"
