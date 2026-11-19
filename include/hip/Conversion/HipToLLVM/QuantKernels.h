/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// Which quantized element types the native HIP runtime kernels take. Backend
// knowledge: the front end folds Q/DQ into quant types without asking this;
// hip-legalize-quant-types asks it before lowering.
#ifndef HIP_CONVERSION_HIPTOLLVM_QUANTKERNELS_H
#define HIP_CONVERSION_HIPTOLLVM_QUANTKERNELS_H

#include "mlir/IR/Operation.h"
#include "mlir/IR/TypeRange.h"

namespace mlir::hip {

/// True if a runtime kernel implements `op` with these quantized element types
/// (inputs in getQuantOperands order, then the result). False for an op kind
/// without a quantized kernel.
bool hasQuantKernel(Operation *op, TypeRange inputElementTypes,
                    Type resultElementType);

} // namespace mlir::hip

#endif // HIP_CONVERSION_HIPTOLLVM_QUANTKERNELS_H
