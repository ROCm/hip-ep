/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Compiler/Pipeline.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "hip/Compiler/Passes/Passes.h"
#include "hip/Conversion/Passes.h"
#include "hip/Dialect/Transforms/Passes.h"

namespace hip::compiler {
namespace compiler {

void populateMorphizenPipeline(mlir::OpPassManager& pm,
                               const hip::compiler::CompilationOptionsT& options,
                               morphizen::FileSystem* fs) {
  // Stage 0: Insert !hip.context argument into all functions
  // Must run before convert-onnx-to-hip so patterns can read ctx from arg 0
  pm.addPass(mlir::hip::createHipAddContextArgPass());

  // Stage 1: ONNX → HIP conversion (tensor-first, no allocation)
  // Lowers ONNX ops to tensor-mode HIP ops using tensor::EmptyOp as DPS init
  pm.addPass(mlir::hip::createConvertOnnxToHipPass(fs, options));

  // Stage 2: One-shot bufferization
  // Converts tensor-mode HIP ops to memref-mode via BufferizableOpInterface
  mlir::bufferization::OneShotBufferizePassOptions bufOpts;
  bufOpts.bufferizeFunctionBoundaries = true;
  pm.addPass(mlir::bufferization::createOneShotBufferizePass(bufOpts));

  // Stage 3: Convert function results to out-params (destination-passing style)
  // Transforms func return values into pointer output arguments for C ABI.
  // modifyPublicFunctions=true is required because @main_graph is public.
  mlir::bufferization::BufferResultsToOutParamsPassOptions outParamOpts;
  outParamOpts.modifyPublicFunctions = true;
  pm.addPass(
      mlir::bufferization::createBufferResultsToOutParamsPass(outParamOpts));

  // Stage 4: Canonicalization
  // Simplifies IR and applies standard optimizations after bufferization
  pm.addPass(mlir::createCanonicalizerPass());

  // Stage 5: Memory pooling optimization
  // Merges multiple memref.alloc ops into a single GPU memory pool
  pm.addPass(mlir::hip::createMemoryPoolingPass());

  // Stage 6: HIP → LLVM conversion
  // Lowers HIP dialect to LLVM dialect (calls to MIOpen/HIP runtime)
  pm.addPass(mlir::hip::createConvertHipToLLVMPass());

  // Stage 7: Interface generation
  // Generates C-ABI compatible wrapper functions for ONNX Runtime
  pm.addPass(createGenerateInterfacePass(options));
}

} // namespace compiler
} // namespace hip::compiler
