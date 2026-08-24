/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/Transforms/Pipelines.h"
#include "hip/Compiler/PluginRegistry.h"
#include "hip/Conversion/OnnxToHip/Passes.h"
#include "hip/Conversion/OnnxToHipDNN/Passes.h"
#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/Transforms/Passes.h"
#include "hip/debug_log.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/BufferizationToMemRef/BufferizationToMemRef.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/Support/raw_ostream.h"

#include "compilation_options_generated.h"

using namespace mlir;

namespace {

/// Append every plugin pass requested for `slot` to `pm`, resolving each by
/// name through MLIR's pass registry (the same path `--pass=foo` takes), so a
/// typo in a plugin's requestPipelineSlot surfaces as a warning, not a silent
/// miss. No-op (one vector lookup) when no plugin requested this slot.
//
// Name-lookup caveat: this TU has `using namespace mlir;` AND includes
// PluginRegistry.h, so both `mlir::hip` (the dialect) and `::hip` (plugin
// infra) are visible -- an unqualified `hip::` is AMBIGUOUS here. Spell dialect
// passes `mlir::hip::create...Pass()` and registry calls
// `::hip::compiler::...`. (A bare `hip::` compiles on main, where
// PluginRegistry.h is absent, but not here.)
void addPluginPassesForSlot(OpPassManager &pm,
                            ::hip::compiler::PipelineSlot slot) {
  auto passNames = ::hip::compiler::pluginPassesForSlot(slot);
  if (passNames.empty())
    return;
  for (llvm::StringRef passName : passNames) {
    // Resolve into this (module-level) pass manager, exactly like
    // --pass-pipeline; warn rather than silently skip on failure (causes
    // below).
    if (failed(parsePassPipeline(passName, pm))) {
      llvm::errs()
          << "[plugin] WARNING: could not add pass pipeline '" << passName
          << "' requested for pipeline slot " << static_cast<int>(slot)
          << ". Check that the pass was registered (registerPass<>()) and the "
             "plugin is statically linked into this host, and that the string "
             "carries the right anchor nesting (e.g. func.func(<arg>) for a "
             "FuncOp pass).\n";
    }
  }
}

/// Add shape normalizations used only to improve hip-pool-allocs:
///   - resolve-memref-dims folds `memref.dim` through view chains to the root
///     buffer.
///   - CSE removes repeated size queries exposed by late allocation rewrites.
///   - hoist-alloc-size-arith moves pure size computations above the first used
///     allocation.
/// Omitting these steps preserves correctness but may split allocations into
/// more dominance domains and increase peak memory. Run these after view
/// creation and immediately before pooling.
void addPoolAllocsShapePreconditionPasses(OpPassManager &pm) {
  pm.addNestedPass<func::FuncOp>(mlir::hip::createResolveMemRefDimsPass());
  // Allocation and view rewrites run after the pipeline's earlier CSE and may
  // leave repeated memref.dim queries on the same source. Deduplicate them
  // before hoisting so equivalent sizes do not open separate pool domains.
  pm.addNestedPass<func::FuncOp>(mlir::createCSEPass());
  pm.addNestedPass<func::FuncOp>(mlir::hip::createHoistAllocSizeArithPass());
}

struct VerifyNoConstantCarriersPass
    : PassWrapper<VerifyNoConstantCarriersPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VerifyNoConstantCarriersPass)

  void runOnOperation() override {
    bool found = false;
    getOperation().walk([&](mlir::hip::ConstantOp op) {
      op.emitError("hip.constant survived past hip-externalize-constants; "
                   "plugins must emit carriers at AfterConvertOnnxToHip, not "
                   "at BeforeBufferization");
      found = true;
    });
    if (found)
      signalPassFailure();
  }
};

std::unique_ptr<Pass> createVerifyNoConstantCarriersPass() {
  return std::make_unique<VerifyNoConstantCarriersPass>();
}

} // namespace

/// Common tail of the ONNX-to-HIP pipeline after the OnnxToHip pass.
///
/// Graph outputs use the output-allocator ABI: `hip-use-output-allocator` runs
/// at slot 4.5, rewriting each returned `memref.alloc` into `hip.alloc_output`
/// (allocated in-graph at runtime via the EP's output-allocator callback).
/// It must run before pool-allocs (slot 6); the reason is at that slot. See
/// docs/design/output-allocator-design.md.
static void
buildOnnxToHipPipelineTail(OpPassManager &pm,
                           const mlir::hip::OnnxToHipPipelineOptions &options,
                           morphizen::FileSystem *fs) {
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
  pm.addPass(mlir::hip::createInferShapesPass());

  // Apply constant storage policy only after HIP shape inference has consumed
  // every inspectable dense carrier payload. This keeps compile-time shape
  // values visible through conversion and reification while still replacing
  // every carrier before canonicalization, CSE, and bufferization.
  if (fs) {
    pm.addPass(mlir::hip::createExternalizeConstantsPass(
        fs, options.externalizeMinNumElements, options.skipConstantData));
  } else {
    mlir::hip::ExternalizeConstantsPassOptions externalizeOptions;
    externalizeOptions.externalizeOutputDir = options.externalizeOutputDir;
    externalizeOptions.externalizeMinNumElements =
        options.externalizeMinNumElements;
    externalizeOptions.skipConstantData = options.skipConstantData;
    pm.addPass(mlir::hip::createExternalizeConstantsPass(
        std::move(externalizeOptions)));
  }

  // 1b'. Canonicalize + CSE immediately after shape inference and constant
  //      externalization. The dynamic-shape op conversions (e.g. pool /
  //      reduce) size each dynamic result dim by emitting `tensor.dim` of a
  //      (statically-typed) producer plus a little index arithmetic, then build
  //      the `tensor.empty` init from those values. InferShapes above has just
  //      tightened many of those producers to static dims, so canonicalization
  //      now folds `tensor.dim` of a static dim to a constant, collapses the
  //      dependent arithmetic, and DCEs the dead shape computations; CSE
  //      dedups the identical per-dim recomputations the per-op conversions
  //      emit independently. Both run before bufferize so
  //      `--hip-pool-allocs` sees folded constant dims instead of fragmented
  //      `memref.dim` chains (un-folded chains split the pool and pessimize
  //      buffer reuse).
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  // 1b''. Repair the one unsafe consequence of the CSE above: when it merges
  //       the seed `tensor.empty`s of two init operands of the SAME op, the
  //       tied results bufferize into one buffer and an op that reads back its
  //       own outputs (e.g. `hip.gqa` present_key/present_value) miscompiles.
  //       This re-points each such duplicate at a fresh, non-aliasing
  //       `bufferization.alloc_tensor`. It MUST run after the CSE that creates
  //       the duplication and before `one-shot-bufferize` locks in the shared
  //       buffer; it is the surgical alternative to a blanket
  //       `empty-tensor-to-alloc-tensor` (which would also undo the benign
  //       merges the CSE above is here to make). See
  //       SplitDuplicateDpsInits.cpp.
  pm.addNestedPass<func::FuncOp>(mlir::hip::createSplitDuplicateDpsInitsPass());

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
  pm.addNestedPass<func::FuncOp>(mlir::hip::createResolveTensorDimsPass());

  // Plugin slot: BeforeBufferization. Vendors often want to lower or
  // canonicalize hip.* ops before bufferization fixes the type system
  // to memrefs.
  addPluginPassesForSlot(pm,
                         ::hip::compiler::PipelineSlot::BeforeBufferization);

  // A BeforeBufferization plugin runs after the supported constant-producer
  // boundary. Diagnose any carrier it introduced before One-Shot Bufferize
  // reports the much less actionable "op was not bufferized".
  pm.addPass(createVerifyNoConstantCarriersPass());

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
  // Opt-in: skip One-Shot's RaW-conflict analysis (super-linear in op count;
  // the sole hotspot on very large single-function graphs) by copying before
  // every write instead of proving in-place safety. Gated because those extra
  // copies cost runtime performance, so default OFF keeps the optimal in-place
  // path; set HIPDNN_EP_BUFFERIZE_COPY_BEFORE_WRITE=1 to trade that for compile
  // time on huge models.
  // TODO: revisit the known upstream super-linear scaling in One-Shot's
  // RaW/alias analysis for a real fix -- e.g. memoize the aliasing
  // read/write-set construction -- so large models keep the in-place path
  // without this fallback.
  if (!hip_get_env("HIPDNN_EP_BUFFERIZE_COPY_BEFORE_WRITE").empty())
    bufferizeOpts.copyBeforeWrite = true;
  pm.addPass(bufferization::createOneShotBufferizePass(bufferizeOpts));

  // 3. Promote outlined `*_loop_body_*` helpers to the out-param ABI
  //    LoopLowering expects. @main_graph keeps its returned memrefs here and
  //    defers output handling to slot 4.5 (hip-use-output-allocator). This pass
  //    is scoped to the private loop bodies (modifyPublicFunctions = false) and
  //    must run before the pool/lowering passes that consume the out-param ABI.
  pm.addPass(mlir::hip::createLoopBodyToOutParamsPass());

  // 4. Ownership-based buffer deallocation is intentionally NOT run. One-Shot
  //    Bufferize emits a `memref.alloc` per transient, but this pipeline never
  //    frees them individually: `hip-pool-allocs` (slot 6) rewrites each into a
  //    `memref.view` over one of the session-owned pools (`hip.get_pool`),
  //    reusing slots across disjoint lifetimes and freeing the pools once at
  //    session cleanup; graph outputs become runtime-owned `hip.alloc_output`
  //    (slot 4.5). So no individually allocated buffer survives -- there is
  //    nothing to deallocate. Running the pass is also a compile-time hazard:
  //    buffer-deallocation-simplification's O(n^2) pairwise `isSameAllocation`
  //    scan over the one giant `bufferization.dealloc` it emits per block hangs
  //    large single-block functions. (Dropping the bundle also omits
  //    expand-realloc, fine here: the pipeline emits no `memref.realloc`.)

  // 4.5. hip-use-output-allocator (FuncOp) rewrites each returned
  //      `memref.alloc` into `hip.alloc_output` (EP-owned, allocated in-graph
  //      at runtime via the output-allocator callback). Leaves the function
  //      signature + `return` intact. Must run BEFORE pool-allocs (slot 6) so
  //      the EP-owned output never enters the GPU pool, which only absorbs
  //      `memref.alloc` (`hip.alloc_output` carries a Write but no Allocate
  //      effect).
  pm.addNestedPass<func::FuncOp>(mlir::hip::createUseOutputAllocatorPass());

  // 4.6. Rewrite frozen Concat-accumulator offsets in outlined hip.loop bodies
  //      to iter-driven offsets (the loop trampoline aliases v_in/v_out onto
  //      one descriptor, freezing `memref.dim %v_in` so a growing accumulator
  //      keeps only its last chunk). Placement is load-bearing: AFTER
  //      out-param promotion (slot 3, which is what creates the in-place-writer
  //      alias) and BEFORE the pool/hoist passes (so the synthesized readback +
  //      index_cast flow through them). No-op on non-loop-body funcs. See
  //      FixLoopAccumulatorOffset.cpp.
  pm.addNestedPass<func::FuncOp>(
      mlir::hip::createFixLoopAccumulatorOffsetPass());

  // 5. Clean up after bufferization
  pm.addPass(createCSEPass());
  pm.addPass(createCanonicalizerPass());

  // 5a. Convert any memref-level `linalg.*` (today only the rank-0
  //     `linalg.fill` that `ConstantOfShapeAsScalar` emits for a
  //     `ConstantOfShape -> Where` fill-value buffer) to `scf` loops +
  //     `memref.store`.  ConvertHipToLLVM has no linalg patterns, so a
  //     surviving `linalg.fill` would leave an unrealized cast and abort
  //     translation.
  //
  //     Placement is load-bearing: it MUST run BEFORE MaterializeHostScalars
  //     (6b).  The rank-0 fill bufferizes to `memref.alloc + linalg.fill`;
  //     only AFTER this pass does it become `memref.alloc + memref.store`,
  //     which is the shape MaterializeHostScalars recognises as a tiny
  //     host-fed scalar and redirects to host-mapped scratch.  If this ran
  //     after MaterializeHostScalars, the alloc would still carry a
  //     `linalg.fill` user (not a `memref.store`), the candidate scan would
  //     skip it, and the rank-0 buffer would land in the GPU pool with a
  //     host store on it -- a device fault on targets whose pool is real
  //     device memory.  No-op on graphs that emit no linalg ops.
  pm.addNestedPass<func::FuncOp>(createConvertLinalgToLoopsPass());

  // 6. HIP-specific buffer optimizations
  pm.addNestedPass<func::FuncOp>(mlir::hip::createOptimizeMemRefsPass());

  // The remaining pre-pooling passes have two distinct roles:
  //   - Correctness/ABI passes that must introduce or redirect allocations
  //     before pooling (6a and 6b).
  //   - Shape preconditions used only to improve pool quality (grouped in 6c).

  // 6a. Correctness/ABI: materialize identity-layout temporaries for
  //     non-identity-layout DPS-input memrefs. The interface-based selection
  //     covers plugin ops whose bare-pointer runtime ABI carries no descriptor
  //     offset or strides. Runs after linalg cleanup and OptimizeMemRefs, and
  //     before pooling so the temporaries become pool views rather than
  //     per-inference allocations.
  pm.addNestedPass<func::FuncOp>(
      mlir::hip::createPromoteStridedHipOperandsPass());

  // 6b. Correctness/ABI: redirect tiny host-fed memref.alloc ops (bufferized
  //     `tensor.from_elements` shape arithmetic) from the GPU pool to
  //     runtime-owned host-mapped scratch. Must run after 6a so its candidate
  //     analysis sees the final allocation set, and before pooling so
  //     host-written buffers never become views into device memory. See
  //     MaterializeHostScalars.cpp; pipeline-static-shape-lockdown.mlir locks
  //     down this ordering.
  pm.addNestedPass<func::FuncOp>(mlir::hip::createMaterializeHostScalarsPass());

  // 6c. Pool quality: normalize dynamic allocation sizes after all view and
  //     allocation rewrites, immediately before hip-pool-allocs.
  addPoolAllocsShapePreconditionPasses(pm);

  pm.addNestedPass<func::FuncOp>(mlir::hip::createPoolAllocsPass());

  // Plugin slot: AfterPoolAllocs. Useful for vendor passes that
  // analyze or transform memref allocations after pooling
  // (e.g., custom allocator tagging).
  addPluginPassesForSlot(pm, ::hip::compiler::PipelineSlot::AfterPoolAllocs);

  // 7. Lower remaining bufferization ops to memref
  pm.addPass(createConvertBufferizationToMemRefPass());

  pm.addPass(createCSEPass());
  pm.addPass(createCanonicalizerPass());

  // 8. Replace memref.alloc with hip.alloc/hip.free
  pm.addNestedPass<func::FuncOp>(mlir::hip::createLowerAllocsPass());

  // 9. Resolve extern constants → memref.view into constants blob argument
  pm.addPass(mlir::hip::createResolveExternConstantsPass());

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

  // Plugin slot: AfterSimplifyOnnx. Vendor passes here see canonical
  // ONNX dialect IR with no HIP context arg yet.
  addPluginPassesForSlot(pm, ::hip::compiler::PipelineSlot::AfterSimplifyOnnx);

  pm.addPass(createHipAddContextArgPass());

  // Outline onnx.Loop bodies into separate func.func ops before the main
  // onnx-to-hip conversion runs. That way each outlined body's onnx.* ops
  // get the same treatment as ops in main_graph (constant lowering, op
  // mapping, etc.) -- the conversion pass already iterates all func.func
  // ops in the module.
  //
  // General ONNX-side shape refinement is intentionally NOT done here. The
  // importer is responsible for emitting `tensor<*xT>` (unranked) for values
  // whose shape it does not know, and `tensor<>` (rank-0) only for genuine
  // scalars. The narrow pre-conversion pass below establishes rank where
  // outlined-loop conversion requires it; post-conversion
  // `--hip-infer-shapes` only narrows `?` dims on ranked HIP results. See
  // `docs/design/unranked-tensor-handling.md` for the full contract.
  //
  // TODO(unranked-import-contract): the unranked-import contract on
  // the importer side ships in MorphiZen PR #228
  // (https://github.com/ROCm/MorphiZen/pull/228). Until that PR is
  // merged AND the `morphizen` subtree here is updated past the
  // merge, the importer still emits `tensor<>` (rank-0) for
  // values it has no shape for, which `--convert-onnx-to-hip` will
  // misinterpret as a genuine scalar on Loop-heavy models (any
  // `onnx.Concat` / `onnx.Add` etc. inside an outlined body whose
  // operand was unranked at import will fail rank-aware conversion).
  // When this submodule is bumped: re-run the LIT suite here and
  // every Python perf test under `test/python/` on a Loop-heavy
  // model end-to-end, then delete this TODO.
  pm.addPass(createOnnxLoopOutlinePass());
  pm.addPass(createOnnxIfOutlinePass());

  // Rank-establish unranked tensors inside outlined loop bodies (e.g. a
  // loop-carried `onnx.Concat` output the importer left as `tensor<*xT>`)
  // BEFORE conversion: `convert-onnx-to-hip` converters require ranked
  // results and otherwise leave the op unconverted, breaking the body
  // func signature and later bufferization.
  pm.addPass(createInferLoopBodyShapesPass());

  // Plugin slot: AfterOnnxLoopOutline. Operate on outlined ONNX loop
  // bodies before the main lowering runs.
  addPluginPassesForSlot(pm,
                         ::hip::compiler::PipelineSlot::AfterOnnxLoopOutline);

  {
    ConvertOnnxToHipPassOptions convertOptions;
    convertOptions.kvShareBuffer = options.kvShareBuffer;
    pm.addPass(createConvertOnnxToHipPass(std::move(convertOptions)));
  }

  // Plugin slot: AfterConvertOnnxToHip. The most common slot for
  // vendor lowerings of `onnx.Custom` ops or vendor-specific hip.*
  // canonicalisations. This is the supported plugin boundary for emitting
  // hip.constant carriers.
  addPluginPassesForSlot(pm,
                         ::hip::compiler::PipelineSlot::AfterConvertOnnxToHip);

  buildOnnxToHipPipelineTail(pm, options, fs);
}

void mlir::hip::buildOnnxToHipPipeline(OpPassManager &pm,
                                       const OnnxToHipPipelineOptions &options,
                                       morphizen::FileSystem *fs,
                                       hipdnnHandle_t handle,
                                       CompiledGraphMap output_graphs) {
  // See sibling overload for rationale.
  pm.addPass(createSimplifyOnnxPass());
  addPluginPassesForSlot(pm, ::hip::compiler::PipelineSlot::AfterSimplifyOnnx);

  pm.addPass(createHipAddContextArgPass());
  pm.addPass(createOnnxLoopOutlinePass());
  pm.addPass(createOnnxIfOutlinePass());
  pm.addPass(createInferLoopBodyShapesPass());
  addPluginPassesForSlot(pm,
                         ::hip::compiler::PipelineSlot::AfterOnnxLoopOutline);

  if (handle) {
    pm.addPass(createOutlineOnnxToHipDNNPass());
    pm.addPass(createCompileHipDNNGraphsPass(handle, std::move(output_graphs)));
  }

  {
    ConvertOnnxToHipPassOptions convertOptions;
    convertOptions.kvShareBuffer = options.kvShareBuffer;
    pm.addPass(createConvertOnnxToHipPass(std::move(convertOptions)));
  }

  addPluginPassesForSlot(pm,
                         ::hip::compiler::PipelineSlot::AfterConvertOnnxToHip);

  buildOnnxToHipPipelineTail(pm, options, fs);
}

void mlir::hip::buildHipToLLVMPipeline(
    OpPassManager &pm, const HipToLLVMPipelineOptions &options) {
  // Rewrite multi-dyn-per-group memref.expand_shape ops into
  // memref.reinterpret_cast BEFORE expand-strided-metadata.  Upstream
  // expand-strided-metadata asserts at most one dynamic size per
  // reassociation group, but our IR legitimately produces 2-dyn groups
  // (e.g. ONNX `Range -> Reshape([bs, ss])` for 2-D position_ids).  This
  // local pass handles only that case; everything else passes through
  // untouched and is handled by upstream.  See RelaxMultiDynExpandShape.cpp
  // header for the IR snippet and the retirement path.
  pm.addNestedPass<func::FuncOp>(
      mlir::hip::createRelaxMultiDynExpandShapePass());

  // Plugin slot: BeforeConvertHipToLLVM. Last chance to operate on
  // hip.* / memref IR before the lowering to LLVM dialect erases it.
  addPluginPassesForSlot(pm,
                         ::hip::compiler::PipelineSlot::BeforeConvertHipToLLVM);

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

  // Op-state slots (docs/design/op-state-slots-design.md). Assign one slot per
  // stateful op instance, then emit @hipdnn_ep_op_states_init_fn while the HIP
  // ops (and their OpStateOpInterface impls) still exist. Both run BEFORE the
  // lowering below: assign so the slot is available to the wrap_* lowering and
  // to generate-op-state-init; generate-op-state-init because generate-
  // interface (after lowering) only calls the function this builds. Both are
  // no-ops on modules with no stateful ops.
  pm.addPass(createAssignOpStateSlotsPass());
  pm.addPass(createGenerateOpStateInitPass());

  // Lower `scf.for` / `scf.if` introduced by convert-linalg-to-loops (5a) to
  // unstructured control flow, then reconcile any leftover unrealized casts.
  // ConvertHipToLLVM has no SCF patterns, so a surviving scf.for would fail
  // translation. No-op on graphs whose linalg ops all folded to rank-0
  // stores (no loop emitted); kept for any future linalg lowering that does
  // emit real loops.
  pm.addPass(createSCFToControlFlowPass());
  pm.addPass(createReconcileUnrealizedCastsPass());

  pm.addPass(createConvertHipToLLVMPass());

  mlir::hip::CompilationOptionsT compOpts;
  compOpts.constants_file = options.constantsFile;
  // convert-hip-to-llvm (above) rewrites @main_graph to the 2-arg
  // (ctx, inputs) form; generate-interface emits the matching 2-arg
  // inference_compute. Outputs are allocated inside the graph via
  // hip.alloc_output. This is the only ABI.
  pm.addPass(createGenerateInterfacePass(compOpts));

  // Plugin slot: AfterGenerateInterface. The C interface (inference_init,
  // inference_compute, ...) is in place; vendor passes can stamp metadata
  // attributes or add LLVM-dialect ops alongside it.
  addPluginPassesForSlot(pm,
                         ::hip::compiler::PipelineSlot::AfterGenerateInterface);
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
