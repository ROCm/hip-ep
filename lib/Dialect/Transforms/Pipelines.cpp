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
#include "mlir/Dialect/Arith/Transforms/Passes.h"
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
  // 1b. Phase 2 of slot-buffer-coalesce: reserve fresh runtime slot ids
  // for translucent propagators whose result dims transitively depend on
  // a Cat-C `RuntimeSlot` leaf. Runs while the IR is still in tensor
  // form so `getResultDimSpec` can walk the SSA graph from the value
  // side. See docs/design/slot-buffer-coalesce.md (Phase 2.3).
  pm.addPass(mlir::hip::createReservePropagatorSlotsPass());

  // 1c. Phase 4 of slot-buffer-coalesce: erase identity-shaped
  // propagator ops (transpose with perm=[0..rank), cast same-dtype,
  // full-range slice, etc.) before bufferize so the downstream
  // bufferize + dealloc + pool-allocs / Phase 3 coalesce passes see
  // a simpler IR. Runs in tensor form because the pass relies on
  // `op->getResult(0)` being a real SSA value (post-bufferize HIP
  // DPS ops have zero results — the DPS-init memref takes the place
  // of the result).
  pm.addPass(mlir::hip::createIdentityPropagatorRebindPass());

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

  // 5b. Phase 1 of slot-buffer-coalesce: shrink the DPS-init allocs of
  // Cat-C slot publishers (NonZero, Range Cat-C, ConstantOfShape Cat-C)
  // to 0-byte placeholders so PoolAllocs reserves no bytes for them.
  // See docs/design/slot-buffer-coalesce.md.
  pm.addNestedPass<func::FuncOp>(hip::createElideSlotPublisherAllocsPass());

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
  // Compose per-op output DimSpecs into model-output DimSpecs and attach
  // as a module attribute. Runs while the IR is still in func.func / hip
  // / memref form so the walker can traverse SSA values; the result is
  // pure metadata, no IR rewriting.
  pm.addPass(mlir::hip::createComposeDimSpecsPass());

  // Annotate every consumer op with the (operand_idx, dim_idx, slot_id)
  // tuples of any dynamic dim that resolves (per the per-op DimSpec
  // system) to a runtime-published slot. The HipToLLVM lowering of
  // affected consumers reads this attribute and substitutes a
  // `hipdnn_ep_state_read_dim` call for the descriptor `sizes[d]`
  // load on the relevant dim, so the kernel sees the actual published
  // count rather than the upper-bound pool allocation size.
  pm.addPass(mlir::hip::createAnnotateInputDimSlotsPass());

  // Phase 3: coalesce slot ids whose canonical DimSpec + lifetime
  // overlap allow them to share one dyn-pool buffer at runtime.
  // Must run AFTER annotation (so per-operand input_dim_slots /
  // input_slot_buffers attrs exist and get rewritten consistently)
  // and BEFORE the HipToLLVM lowering (so the emitted runtime calls
  // see the post-coalesce slot ids in op attributes).
  pm.addPass(mlir::hip::createSlotLifetimeCoalescePass());

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

  // ConvertHipToLLVM uses populateArithToLLVMConversionPatterns, which does
  // NOT include lowering for ceildivsi / floordivsi / ceildivui. The dynamic
  // wrap_range fallback path in OnnxToHip emits arith.ceildivsi for the
  // Range output-length computation. Running arith-expand here decomposes
  // those ops into basic divisions + compares + selects that the LLVM
  // converter can lower; without it, ceildivsi survives the convert pass
  // and translation to LLVM IR aborts with
  // "LLVM Translation failed for operation: arith.ceildivsi".
  // Module-scope pass so it walks both main_graph and any helpers.
  pm.addPass(arith::createArithExpandOpsPass());

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
