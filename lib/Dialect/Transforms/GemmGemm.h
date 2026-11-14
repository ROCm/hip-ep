/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef HIP_DIALECT_TRANSFORMS_GEMMGEMM_H
#define HIP_DIALECT_TRANSFORMS_GEMMGEMM_H

#include <mlir/Dialect/Func/IR/FuncOps.h>

#include <string>

namespace mlir::hip {

/// Merges each matmul or gemm rocMLIR kernel into the kernel that reads its
/// only result as the A operand of its own matmul or gemm, while the
/// gemm-elementwise-gemm gates pass. `counter` numbers the merged kernels.
/// Both gemms must read an element type listed in `accelTypes`.
void fuseGemmGemmKernels(func::FuncOp funcOp, int &counter,
                         ArrayRef<std::string> accelTypes);

} // namespace mlir::hip

#endif // HIP_DIALECT_TRANSFORMS_GEMMGEMM_H
