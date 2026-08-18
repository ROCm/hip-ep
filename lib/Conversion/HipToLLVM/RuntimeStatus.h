/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef HIP_CONVERSION_HIPTOLLVM_RUNTIME_STATUS_H
#define HIP_CONVERSION_HIPTOLLVM_RUNTIME_STATUS_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::hip {

/// Record every operation-runtime status and reject unused i32 call results.
LogicalResult recordAndVerifyRuntimeStatuses(ModuleOp module);

} // namespace mlir::hip

#endif // HIP_CONVERSION_HIPTOLLVM_RUNTIME_STATUS_H
