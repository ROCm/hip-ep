//===- HipDialect.cpp - HIP dialect implementation ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "HipDialect.h"
#include "mlir/IR/DialectImplementation.h"

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
