/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef HIP_DIALECT_GGML_IR_GGMLDIALECT_H
#define HIP_DIALECT_GGML_IR_GGMLDIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpImplementation.h"
// `#ggml.layout` is a tensor encoding, and the interface that lets
// `RankedTensorType` check it must be declared before the attribute carrying
// it.
#include "mlir/IR/TensorEncoding.h"

#include "hip/Dialect/Ggml/IR/GgmlDialect.h.inc"

#include "hip/Dialect/Ggml/IR/GgmlEnums.h.inc"

// Shape expression attributes implement this interface, so it must be declared
// before the attributes that carry it.
#include "hip/Dialect/Ggml/IR/GgmlShapeExprInterface.h.inc"

#define GET_ATTRDEF_CLASSES
#include "hip/Dialect/Ggml/IR/GgmlAttrs.h.inc"

#endif // HIP_DIALECT_GGML_IR_GGMLDIALECT_H
