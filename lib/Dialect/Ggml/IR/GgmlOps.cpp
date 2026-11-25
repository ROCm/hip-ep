/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===- GgmlOps.cpp - The operation classes themselves -------------------===//
//
// What TableGen generates for every operation, which has to be in exactly one
// translation unit. What each operation means -- its verifier and the rule for
// its result's extents -- is beside the operation, one file each, under `ops/`.
//
//===----------------------------------------------------------------------===//

#include "hip/Dialect/Ggml/IR/GgmlOps.h"

#include "hip/Dialect/Ggml/IR/GgmlLayout.h"
#include "hip/Dialect/Ggml/IR/GgmlShapeExpr.h"
#include "hip/Dialect/Ggml/IR/GgmlShapeRule.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::ggml;

#define GET_OP_CLASSES
#include "hip/Dialect/Ggml/IR/GgmlOps.cpp.inc"
