/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef HIP_PASSES_H
#define HIP_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace hip {

/// Create a pass to convert HIP operations to LLVM dialect.
std::unique_ptr<Pass> createConvertHipToLLVMPass();

/// Register all HIP passes.
void registerHipPasses();

} // namespace hip
} // namespace mlir

#endif // HIP_PASSES_H
