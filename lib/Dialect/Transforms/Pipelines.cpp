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
///
/// \p useOutputAllocator selects the allocator pipeline. The classic pipeline
/// runs `buffer-results-to-out-params` at slot 3 (each returned memref becomes
/// a trailing out-param). The allocator pipeline SKIPS that pass and instead
/// runs `hip-use-output-allocator` at slot 4.5, whose load-bearing placement
/// (after buffer-deallocation, before pool-allocs) is explained at that slot.
/// See docs/design/output-allocator-design.md.
static void buildOnnxToHipPipelineTail(OpPassManager &pm,
                                       bool useOutputAllocator) {
  // 1b. Refine `?` (kDynamic) dims on HIP DPS op result types using each
  //     op's `ReifyRankedShapedTypeOpInterface` impl. Placed here so the
  //     refinements propagate through bufferize and into pool / alloc
  //     sizing computations downstream, and run BEFORE the unrefined
  //     `?` dims would otherwise materialise as `memref.dim` ops on the
  //     pool buffer at the top of the function.
  //
  //     Idempotent and a no-op on functions whose ops either don't carry
  //     the interface or expose no further refinable dims — safe to run
  //     unconditionally regardless of how many HIP ops currently
  //     implement the interface. See `docs/design/hip-shape-inference.md`
  //     for the design and `test/lit/Dialect/hip-infer-shapes.mlir` for
  //     the reference cases.
  pm.addPass(hip::createInferShapesPass());

  // 1c. Fold `tensor.dim` queries on `tensor.expand_shape` /
  //     `tensor.collapse_shape` chains into arithmetic on the chain
  //     root's dims, in the tensor domain, before one-shot-bufferize.
  //     The reshape ops' shape SSA (`output_shape` and reassociation
  //     maps) is opaque to the post-bufferize `memref.dim` patterns,
  //     so this is the last useful position.  Without it, downstream
  //     `--hip-pool-allocs` sees one scattered dim query per reshape
  //     site and fragments them into many single-alloc dominance
  //     domains -- a pooling-efficiency cost (one tiny pool each), not
  //     a failure -- on graphs with per-layer same-rank dynamic
  //     `onnx.Reshape` (typical: norm / projection chains).  See
  //     `ResolveTensorDims.cpp`.
  pm.addNestedPass<func::FuncOp>(hip::createResolveTensorDimsPass());

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

  // 3. Classic pipeline only: convert tensor function results to out-params
  //    (memref). The allocator pipeline keeps results as returned memrefs here
  //    and defers output handling to slot 4.5 (after buffer-deallocation) --
  //    see that comment for the ownership/clone reason.
  if (!useOutputAllocator) {
    bufferization::BufferResultsToOutParamsPassOptions outParamsOpts;
    outParamsOpts.hoistStaticAllocs = true;
    outParamsOpts.hoistDynamicAllocs = true;
    outParamsOpts.addResultAttribute = true;
    outParamsOpts.modifyPublicFunctions = true;
    pm.addPass(
        bufferization::createBufferResultsToOutParamsPass(outParamsOpts));
  }

  // 4. Insert ownership-based buffer deallocation
  bufferization::BufferDeallocationPipelineOptions deallocOpts;
  bufferization::buildBufferDeallocationPipeline(pm, deallocOpts);

  // 4.5. Allocator pipeline only: hip-use-output-allocator (FuncOp) rewrites
  //      each returned `memref.alloc` into `hip.alloc_output` (EP-owned,
  //      allocated in-graph at runtime via the output-allocator callback) and
  //      stamps the `hipdnn.use_output_allocator` module BoolAttr that
  //      convert-hip-to-llvm + generate-interface read to select the allocator
  //      ABI. Leaves the function signature + `return` intact.
  //
  //      Ordering is load-bearing -- the rewrite MUST run AFTER buffer-
  //      deallocation: `hip.alloc_output` has a Write effect but NO Allocate
  //      effect, so the ownership-based deallocation pass would treat it as an
  //      unowned value at the `func.return` and clone it (`%c =
  //      bufferization.clone %out; return %c`), adding a per-inference alloc +
  //      full-output copy. By running here the deallocation pass sees the
  //      output as a plain `memref.alloc` (Allocate effect => owned), so it
  //      returns it directly with no clone and no dealloc. Still runs BEFORE
  //      pool-allocs (slot 6) so the EP-owned output never enters the GPU pool,
  //      which only absorbs `memref.alloc`. Verified by test/lit/Pipeline/
  //      output-allocator-dealloc.mlir (both orderings).
  if (useOutputAllocator)
    pm.addNestedPass<func::FuncOp>(hip::createUseOutputAllocatorPass());

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

  // 6c. Hoist speculatable size arithmetic feeding `memref.alloc` dynamic
  //     operands above the earliest dynamic alloc in the entry block.
  //     PoolAllocs's single-block dominator-emit phase requires every
  //     dyn-operand SSA def to dominate the earliest pooled alloc; this
  //     pass establishes that precondition for IR where canonicalize left
  //     speculatable arith interleaved with allocs.  No-op for
  //     already-feasible IR; PoolAllocs.cpp itself is unchanged.  Uses
  //     `mlir::isSpeculatable` (the same predicate upstream LICM uses) so
  //     traps (e.g. `arith.divsi` with a runtime-zero divisor) are not
  //     speculated across the move.  See HoistAllocSizeArith.cpp for the
  //     algorithm and rationale.
  pm.addNestedPass<func::FuncOp>(hip::createHoistAllocSizeArithPass());

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
  //
  // ONNX-side shape refinement is intentionally NOT done here. The
  // importer is responsible for emitting `tensor<*xT>` (unranked) for
  // values whose shape it does not know, and `tensor<>` (rank-0) only
  // for genuine scalars. Any unranked tensors that survive into the
  // HIP dialect are refined post-conversion by `--hip-infer-shapes`
  // via `ReifyRankedShapedTypeOpInterface`. See
  // `docs/design/unranked-tensor-handling.md` for the full contract.
  //
  // TODO(unranked-import-contract): the unranked-import contract on
  // the importer side ships in MorphiZen PR #228
  // (https://github.com/ROCm/MorphiZen/pull/228). Until that PR is
  // merged AND the `3rd-party/morphizen` submodule here is bumped
  // past the merge, the importer still emits `tensor<>` (rank-0) for
  // values it has no shape for, which `--convert-onnx-to-hip` will
  // misinterpret as a genuine scalar on Loop-heavy models (any
  // `onnx.Concat` / `onnx.Add` etc. inside an outlined body whose
  // operand was unranked at import will fail rank-aware conversion).
  // When this submodule is bumped: re-run the LIT suite here and
  // every Python perf test under `test/python/` on a Loop-heavy
  // model end-to-end, then delete this TODO.
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

  buildOnnxToHipPipelineTail(pm, options.useOutputAllocator);
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

  buildOnnxToHipPipelineTail(pm, options.useOutputAllocator);
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
  // Both convert-hip-to-llvm (above) and generate-interface read the
  // `hipdnn.use_output_allocator` module attribute (set by
  // hip-use-output-allocator in the ONNX-to-HIP half) to choose the 2-arg
  // allocator vs 3-arg classic ABI. No pipeline option is needed here -- one
  // generator handles both modes.
  pm.addPass(createGenerateInterfacePass(compOpts));
}

void mlir::hip::buildHipdnnPipeline(OpPassManager &pm,
                                    const HipdnnPipelineOptions &options) {
  OnnxToHipPipelineOptions onnxOpts;
  onnxOpts.externalizeOutputDir = options.constantsDir;
  onnxOpts.externalizeMinNumElements = options.externalizeMinNumElements;
  // Only the ONNX-to-HIP half consumes the flag: it picks slot-3
  // buffer-results-to-out-params (classic) vs the slot-4.5
  // hip-use-output-allocator pass, which stamps the
  // `hipdnn.use_output_allocator` module attr. The HIP-to-LLVM half reads that
  // attr, so it needs no option of its own.
  onnxOpts.useOutputAllocator = options.useOutputAllocator;
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
