/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef MORPHIZEN_MLIR_COMPILER_CONVERSION_HIPTOLLVM_PASSES_H
#define MORPHIZEN_MLIR_COMPILER_CONVERSION_HIPTOLLVM_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace hip {

/// Creates a pass that converts HIP dialect operations to LLVM dialect.
/// Lowers HIP GPU operations to LLVM calls to MIOpen/HIP runtime.
std::unique_ptr<Pass> createConvertHipToLLVMPass();

} // namespace hip
} // namespace mlir

#endif // MORPHIZEN_MLIR_COMPILER_CONVERSION_HIPTOLLVM_PASSES_H
