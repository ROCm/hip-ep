/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef HIP_DIALECT_IR_HIP_SHAPE_UTILS_H
#define HIP_DIALECT_IR_HIP_SHAPE_UTILS_H

// Compatibility umbrella for consumers that implement or verify operations
// across several shape families. Prefer the category header in focused code.
#include "hip/Dialect/IR/HipShapeUtilsAttention.h"
#include "hip/Dialect/IR/HipShapeUtilsBroadcast.h"
#include "hip/Dialect/IR/HipShapeUtilsCommon.h"
#include "hip/Dialect/IR/HipShapeUtilsConvPool.h"
#include "hip/Dialect/IR/HipShapeUtilsGather.h"
#include "hip/Dialect/IR/HipShapeUtilsMatmulGemm.h"
#include "hip/Dialect/IR/HipShapeUtilsReduction.h"
#include "hip/Dialect/IR/HipShapeUtilsShapeOps.h"

#endif // HIP_DIALECT_IR_HIP_SHAPE_UTILS_H
