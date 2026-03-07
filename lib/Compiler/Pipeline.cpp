/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Compiler/Pipeline.h"
#include "hip/Compiler/Passes/Passes.h"
#include "hip/Conversion/Passes.h"
#include "hip/Dialect/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

namespace udna::compiler {
namespace compiler {

void populateMorphizenPipeline(mlir::OpPassManager &pm,
                               const udna::compiler::CompilationOptionsT &options,
                               morphizen::FileSystem *fs) {
  // Stage 0: Insert !hip.context argument into all functions
  pm.addPass(mlir::hip::createHipAddContextArgPass());

  // Stage 1: ONNX → HIP conversion (tensor-first, no allocation)
  pm.addPass(mlir::hip::createConvertOnnxToHipPass(fs, options));

  // Stage 2: One-shot bufferization
  // Converts tensor-mode HIP ops to memref-mode via BufferizableOpInterface.
  // IdentityLayoutMap lets OneShotBufferize see that function return buffers
  // are the same allocation as the DPS init (tensor.empty), so it assigns the
  // output buffer directly to tensor.empty instead of allocating a fresh one.
  // This eliminates the memref.alloc + memref.copy that would otherwise appear
  // after buffer-results-to-out-params.
  mlir::bufferization::OneShotBufferizePassOptions bufOpts;
  bufOpts.bufferizeFunctionBoundaries = true;
  bufOpts.functionBoundaryTypeConversion =
      mlir::bufferization::LayoutMapOption::IdentityLayoutMap;
  pm.addPass(mlir::bufferization::createOneShotBufferizePass(bufOpts));

  // Stage 3: Convert function results to out-params (destination-passing style)
  // Transforms func return values into pointer output arguments for C ABI.
  // modifyPublicFunctions=true is required because @main_graph is public.
  // hoistStaticAllocs=true: when the returned memref comes from a fresh
  // memref.alloc, replace all uses of the alloc with the new out-param arg
  // and erase the alloc — no memref.copy is inserted. The HIP DPS op writes
  // directly into the caller-provided output buffer.
  mlir::bufferization::BufferResultsToOutParamsPassOptions outParamOpts;
  outParamOpts.modifyPublicFunctions = true;
  outParamOpts.hoistStaticAllocs = true;
  pm.addPass(
      mlir::bufferization::createBufferResultsToOutParamsPass(outParamOpts));

  // Stage 4: Canonicalization
  pm.addPass(mlir::createCanonicalizerPass());

  // Stage 5: Memory pooling optimization
  // Packs multiple memref.alloc ops into a single byte pool
  pm.addPass(mlir::hip::createPoolAllocsPass());

  // Stage 6: HIP → LLVM conversion
  pm.addPass(mlir::hip::createConvertHipToLLVMPass());

  // Stage 7: Interface generation
  pm.addPass(createGenerateInterfacePass(options));
}

} // namespace compiler
} // namespace udna::compiler
