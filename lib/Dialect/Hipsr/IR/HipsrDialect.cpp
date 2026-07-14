/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/Hipsr/IR/HipsrDialect.h"

#include "hip/Dialect/Hipsr/IR/HipsrOps.h"
#include "hip/Dialect/Hipsr/IR/HipsrShapeYieldOp.h"

#include "llvm/ADT/TypeSwitch.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace mlir::hipsr;

#include "hip/Dialect/Hipsr/IR/HipsrDialect.cpp.inc"

// Enum code first: the attribute's parser/printer below calls these
// enum name<->value helpers.
#include "hip/Dialect/Hipsr/IR/HipsrEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "hip/Dialect/Hipsr/IR/HipsrAttrs.cpp.inc"

void HipsrDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "hip/Dialect/Hipsr/IR/HipsrShapeYieldOp.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "hip/Dialect/Hipsr/IR/HipsrOps.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "hip/Dialect/Hipsr/IR/HipsrAttrs.cpp.inc"
      >();
}
