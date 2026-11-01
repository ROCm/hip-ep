/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef HIP_COMPILER_CONVERSION_HIPTOLLVM_PASSES_H
#define HIP_COMPILER_CONVERSION_HIPTOLLVM_PASSES_H

#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseSet.h"
#include <memory>

namespace mlir {
namespace hip {

/// Creates a pass that converts HIP dialect operations to LLVM dialect.
/// Lowers HIP GPU operations to LLVM calls to MIOpen/HIP runtime.
std::unique_ptr<Pass> createConvertHipToLLVMPass();

/// Return operations whose registered lowerings accept host-mapped inputs.
/// Include all loaded dialect alternatives. Unknown contracts are excluded.
DenseSet<OperationName> getHostMappedInputOps(MLIRContext *context);

} // namespace hip
} // namespace mlir

#endif // HIP_COMPILER_CONVERSION_HIPTOLLVM_PASSES_H
