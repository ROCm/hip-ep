/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef HIPSR_SHAPE_REGION_INTERFACE_H
#define HIPSR_SHAPE_REGION_INTERFACE_H

#include "mlir/IR/OpDefinition.h"
#include "mlir/Support/LogicalResult.h"

// The generated verifier below uses both traits, and names
// SingleBlockExplicitTerminator with ShapeYieldOp, so the traits header and the
// full ShapeYieldOp type must be included first.
#include "hip/Dialect/Hipsr/IR/HipsrShapeYieldOp.h"
#include "hip/Dialect/Hipsr/IR/HipsrTraits.h"

#include "hip/Dialect/Hipsr/IR/HipsrShapeRegionInterface.h.inc"

#endif // HIPSR_SHAPE_REGION_INTERFACE_H
