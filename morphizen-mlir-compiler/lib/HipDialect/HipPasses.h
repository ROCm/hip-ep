/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef HIP_PASSES_H
#define HIP_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
class OpPassManager;

namespace hip {

/// Create a pass to convert ONNX operations to HIP dialect.
std::unique_ptr<Pass> createConvertOnnxToHipPass();

/// Create a pass to convert HIP operations to LLVM dialect.
std::unique_ptr<Pass> createConvertHipToLLVMPass();

/// Create a pass to generate C interface wrapper functions.
std::unique_ptr<Pass> createGenerateInterfacePass();

/// Create a pass to optimize memory allocation with pooling.
std::unique_ptr<Pass> createMemoryPoolingPass();

/// Create a pass to insert hip.free operations for buffer deallocation.
std::unique_ptr<Pass> createHipBufferDeallocationPass();

/// Populate the complete ONNX→HIP→LLVM→Interface pass pipeline.
/// This is equivalent to the pipeline used in
/// CompilerPipeline::runMLIRPasses(). Registered via PassPipelineRegistration
/// as "--all-passes".
void populateCompleteOnnxToLLVMPipeline(OpPassManager& pm);

/// Register all HIP passes and pipelines.
void registerHipPasses();

} // namespace hip
} // namespace mlir

#endif // HIP_PASSES_H
