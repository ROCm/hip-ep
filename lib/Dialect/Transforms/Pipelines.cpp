/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/Transforms/Pipelines.h"
#include "hip/Conversion/OnnxToHip/Passes.h"
#include "hip/Conversion/OnnxToHipDNN/Passes.h"
#include "hip/Dialect/Transforms/Passes.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/BufferizationToMemRef/BufferizationToMemRef.h"
#include "mlir/Dialect/Bufferization/Pipelines/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

#include "compilation_options_generated.h"

using namespace mlir;

/// Common tail of the ONNX-to-HIP pipeline after the OnnxToHip pass.
static void buildOnnxToHipPipelineTail(OpPassManager &pm) {
  // 2. Bufferize tensor IR to memref IR
  //
  // Use IdentityLayoutMap for function boundaries: all EP inputs/outputs come
  // from the ORT runtime as contiguous (row-major) tensors, and
  // GenerateInterface::buildMemrefDescriptor always constructs memref
  // descriptors with contiguous strides (stride[i] = product of sizes[i+1..]).
  // The default InferLayoutMap falls back to FullyDynamicLayoutMap for ops like
  // collapse_shape/expand_shape, producing strided<[?, ?, ?], offset: ?>
  // memrefs that cannot be lowered to LLVM.
  bufferization::OneShotBufferizePassOptions bufferizeOpts;
  bufferizeOpts.bufferizeFunctionBoundaries = true;
  bufferizeOpts.functionBoundaryTypeConversion =
      bufferization::LayoutMapOption::IdentityLayoutMap;
  pm.addPass(bufferization::createOneShotBufferizePass(bufferizeOpts));

  // 3. Convert tensor function results to out-params (memref)
  bufferization::BufferResultsToOutParamsPassOptions outParamsOpts;
  outParamsOpts.hoistStaticAllocs = true;
  outParamsOpts.hoistDynamicAllocs = true;
  outParamsOpts.addResultAttribute = true;
  outParamsOpts.modifyPublicFunctions = true;
  pm.addPass(bufferization::createBufferResultsToOutParamsPass(outParamsOpts));

  // 4. Insert ownership-based buffer deallocation
  bufferization::BufferDeallocationPipelineOptions deallocOpts;
  bufferization::buildBufferDeallocationPipeline(pm, deallocOpts);

  // 5. Clean up after bufferization
  pm.addPass(createCSEPass());
  pm.addPass(createCanonicalizerPass());

  // 6. HIP-specific buffer optimizations
  pm.addNestedPass<func::FuncOp>(hip::createOptimizeMemRefsPass());

  // 6a. Promote strided memref operands of hip.* ops to contiguous
  //     temporaries.  Required because the HIP runtime call ABI (used by
  //     --convert-hip-to-llvm) only forwards a bare alignedPtr per memref
  //     operand and has no channel for offset / per-dim strides; without
  //     this pass, hip.* ops that consume memref.subview results read the
  //     base of the parent buffer rather than the slice.
  //
  //     Placement: after OptimizeMemRefs (so we don't fight its
  //     subview-folding) and before PoolAllocs (so the new transient
  //     memref.alloc / memref.dealloc pairs flow through pool views and do
  //     not trigger extra hipMalloc calls per inference).
  pm.addNestedPass<func::FuncOp>(hip::createPromoteStridedHipOperandsPass());

  // 6b. Redirect tiny host-fed memref.alloc ops (bufferized
  //     `tensor.from_elements` for shape arithmetic — rank-0 / 1xi64) away
  //     from the GPU pool to a runtime-owned host-mapped scratch buffer.
  //
  //     Placement is load-bearing: MUST run AFTER PromoteStridedHipOperands
  //     (so any contiguous-temporary memref.alloc that pass introduces is
  //     also visible to the candidate scan) and BEFORE PoolAllocs (so
  //     candidates are removed from its input set).  If PoolAllocs runs
  //     first, it absorbs the alloc into a memref.view over GPU pool memory
  //     and the subsequent host store SEGVs on targets where the GPU pool is
  //     real device memory (other targets silently worked because hipMalloc
  //     returned UMA-mapped host memory there, masking the bug).  See
  //     MaterializeHostScalars.cpp file header for the full pinned-mapped
  //     story; the static-shape lockdown test under
  //     test/lit/Pipelines/ asserts this ordering does not regress.
  pm.addNestedPass<func::FuncOp>(hip::createMaterializeHostScalarsPass());

  pm.addNestedPass<func::FuncOp>(hip::createPoolAllocsPass());

  // 7. Lower remaining bufferization ops to memref
  pm.addPass(createConvertBufferizationToMemRefPass());

  pm.addPass(createCSEPass());
  pm.addPass(createCanonicalizerPass());

  // 8. Replace memref.alloc with hip.alloc/hip.free
  pm.addNestedPass<func::FuncOp>(hip::createLowerAllocsPass());

  // 9. Resolve extern constants → memref.view into constants blob argument
  pm.addPass(hip::createResolveExternConstantsPass());

  // 10. Final cleanup: LowerAllocs and ResolveExternConstants both introduce
  //     new constants and ops that benefit from deduplication and hoisting.
  pm.addPass(createCSEPass());
  pm.addPass(createCanonicalizerPass());
}

void mlir::hip::buildOnnxToHipPipeline(OpPassManager &pm,
                                       const OnnxToHipPipelineOptions &options,
                                       morphizen::FileSystem *fs) {
  // Pre-lowering ONNX-dialect simplifications (currently: CastLike -> Cast
  // + drop dead type-donor function arguments). Pure ONNX dialect; runs
  // BEFORE hip-add-context-arg so it operates in the original ONNX
  // function index space and has no HIP-dialect dependency.
  pm.addPass(createSimplifyOnnxPass());
  pm.addPass(createHipAddContextArgPass());

  // Outline onnx.Loop bodies into separate func.func ops before the main
  // onnx-to-hip conversion runs. That way each outlined body's onnx.* ops
  // get the same treatment as ops in main_graph (constant lowering, op
  // mapping, etc.) -- the conversion pass already iterates all func.func
  // ops in the module.
  pm.addPass(createOnnxLoopOutlinePass());

  // Re-infer onnx.* result types inside each outlined body func from the
  // refined operand types (loop-outline already corrected the function
  // signature; the cloned ops still carry stale result types from the
  // original onnx.Loop body annotation). Propagates the refined return
  // type into the body func signature and the enclosing hip.loop op's
  // result types. Runs BEFORE convert-onnx-to-hip so the body converts
  // with correct types.
  pm.addPass(createRefineLoopBodyTypesPass());

  if (fs) {
    pm.addPass(mlir::hip::createConvertOnnxToHipPass(
        fs, options.externalizeMinNumElements, options.skipConstantData));
  } else {
    ConvertOnnxToHipPassOptions onnxToHipOpts;
    onnxToHipOpts.externalizeOutputDir = options.externalizeOutputDir;
    onnxToHipOpts.externalizeMinNumElements = options.externalizeMinNumElements;
    pm.addPass(createConvertOnnxToHipPass(std::move(onnxToHipOpts)));
  }

  buildOnnxToHipPipelineTail(pm);
}

void mlir::hip::buildOnnxToHipPipeline(OpPassManager &pm,
                                       const OnnxToHipPipelineOptions &options,
                                       morphizen::FileSystem *fs,
                                       hipdnnHandle_t handle,
                                       CompiledGraphMap output_graphs) {
  // See sibling overload for rationale.
  pm.addPass(createSimplifyOnnxPass());
  pm.addPass(createHipAddContextArgPass());
  pm.addPass(createOnnxLoopOutlinePass());
  pm.addPass(createRefineLoopBodyTypesPass());

  if (handle) {
    pm.addPass(createOutlineOnnxToHipDNNPass());
    pm.addPass(createCompileHipDNNGraphsPass(handle, std::move(output_graphs)));
  }

  if (fs) {
    pm.addPass(mlir::hip::createConvertOnnxToHipPass(
        fs, options.externalizeMinNumElements, options.skipConstantData));
  } else {
    ConvertOnnxToHipPassOptions onnxToHipOpts;
    onnxToHipOpts.externalizeOutputDir = options.externalizeOutputDir;
    onnxToHipOpts.externalizeMinNumElements = options.externalizeMinNumElements;
    pm.addPass(createConvertOnnxToHipPass(std::move(onnxToHipOpts)));
  }

  buildOnnxToHipPipelineTail(pm);
}

void mlir::hip::buildHipToLLVMPipeline(
    OpPassManager &pm, const HipToLLVMPipelineOptions &options) {
  // Decompose memref.collapse_shape / memref.expand_shape into
  // memref.reinterpret_cast + arithmetic.
  // populateFinalizeMemRefToLLVMConversionPatterns (used by ConvertHipToLLVM)
  // does not include patterns for these ops; expand-strided-metadata rewrites
  // them into ops that it can lower.
  pm.addPass(memref::createExpandStridedMetadataPass());

  // ExpandStridedMetadata can emit affine.apply for collapse-shape stride
  // products (e.g. flattening an MoE expert-major 3D memref to 2D). The
  // ConvertHipToLLVM lowering does not include affine→arith patterns, so any
  // surviving affine.apply leaves builtin.unrealized_conversion_cast in the
  // final LLVM IR and "Failed to translate MLIR to LLVM IR" aborts compile.
  pm.addPass(createLowerAffinePass());

  pm.addPass(createConvertHipToLLVMPass());

  mlir::hip::CompilationOptionsT compOpts;
  compOpts.constants_file = options.constantsFile;
  pm.addPass(createGenerateInterfacePass(compOpts));
}

void mlir::hip::buildHipdnnPipeline(OpPassManager &pm,
                                    const HipdnnPipelineOptions &options) {
  OnnxToHipPipelineOptions onnxOpts;
  onnxOpts.externalizeOutputDir = options.constantsDir;
  onnxOpts.externalizeMinNumElements = options.externalizeMinNumElements;
  buildOnnxToHipPipeline(pm, onnxOpts);

  HipToLLVMPipelineOptions llvmOpts;
  llvmOpts.constantsFile = options.constantsFile;
  buildHipToLLVMPipeline(pm, llvmOpts);
}

void mlir::hip::registerHipPipelines() {
  PassPipelineRegistration<OnnxToHipPipelineOptions>(
      "onnx-to-hip-pipeline",
      "Lower ONNX IR to bufferized HIP memref IR with optional constant "
      "externalization",
      [](OpPassManager &pm, const OnnxToHipPipelineOptions &opts) {
        buildOnnxToHipPipeline(pm, opts);
      });

  PassPipelineRegistration<HipToLLVMPipelineOptions>(
      "hip-to-llvm-pipeline",
      "Lower HIP memref IR to LLVM dialect and generate C interface",
      buildHipToLLVMPipeline);

  PassPipelineRegistration<HipdnnPipelineOptions>(
      "hipdnn-pipeline", "Complete HIPDNN ONNX→HIP→LLVM→Interface pipeline",
      buildHipdnnPipeline);
}
