/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- OnnxToHip.cpp - Convert ONNX dialect to HIP dialect (tensor DPS) ---===//
//
// Converts ONNX dialect IR into HIP dialect IR using destination-passing style
// (DPS) with tensor types.  ONNX ops are matched by name via the generic MLIR
// Operation API, so no onnx-mlir headers or libraries are required.
// Bufferization to memref is handled by a separate downstream pass.
//
//===----------------------------------------------------------------------===//

#include "OnnxDimParams.h"
#include "OnnxToHipUtils.h"
#include "ShapeProvenanceAnalysis.h"

#include "hip/debug_log.h"
#include "hip/timing.h"

#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

#include <limits>
#include <map>
#include <optional>

#define DEBUG_TYPE "convert-onnx-to-hip"

namespace mlir {
namespace hip {

#define GEN_PASS_DEF_CONVERTONNXTOHIPPASS
#include "hip/Dialect/Transforms/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Constant carrier lowering
//===----------------------------------------------------------------------===//

constexpr llvm::StringLiteral kOrtMemoryAddressLocation = "*/_ORT_MEM_ADDR_/*";

/// Lower every onnx.Constant to a policy-neutral hip.constant carrier.
///
/// Both conversion sweeps call this helper: the first handles imported
/// constants after all value-based pre-folds, and the second handles constants
/// synthesized by compute-op conversion. Externalization policy and artifact
/// I/O deliberately live in the later hip-externalize-constants module pass.
static mlir::LogicalResult lowerOnnxConstants(mlir::func::FuncOp funcOp,
                                              int64_t &nextOrder) {
  llvm::SmallVector<mlir::Operation *> constants;
  funcOp.walk([&](mlir::Operation *op) {
    if (op->getName().getStringRef() == "onnx.Constant")
      constants.push_back(op);
  });

  for (mlir::Operation *constOp : constants) {
    auto tensorType =
        mlir::dyn_cast<mlir::RankedTensorType>(constOp->getResult(0).getType());
    if (!tensorType)
      return constOp->emitError(
          "onnx.Constant requires a ranked tensor result");
    if (nextOrder == std::numeric_limits<int64_t>::max())
      return constOp->emitError("hip.constant order overflows int64");

    mlir::OpBuilder builder(constOp);
    auto orderAttr = builder.getI64IntegerAttr(nextOrder);
    mlir::hip::ConstantOp carrier;
    if (auto valueAttr = mlir::dyn_cast_or_null<mlir::DenseElementsAttr>(
            constOp->getAttrOfType<mlir::ElementsAttr>("value"))) {
      carrier = mlir::hip::ConstantOp::create(builder, constOp->getLoc(),
                                              tensorType, valueAttr);
    } else if (auto location =
                   constOp->getAttrOfType<mlir::StringAttr>("location")) {
      auto offset = constOp->getAttrOfType<mlir::IntegerAttr>("offset");
      auto size = constOp->getAttrOfType<mlir::IntegerAttr>("size");
      if (!offset || !size)
        return constOp->emitError(
            "onnx.Constant external source requires location/offset/size");
      carrier = location.getValue() == kOrtMemoryAddressLocation
                    ? mlir::hip::ConstantOp::create(builder, constOp->getLoc(),
                                                    tensorType, offset, size)
                    : mlir::hip::ConstantOp::create(builder, constOp->getLoc(),
                                                    tensorType, location,
                                                    offset, size);
    } else {
      return constOp->emitError(
          "unsupported onnx.Constant form (expected dense value attribute "
          "or location attribute)");
    }
    ++nextOrder;
    carrier.setSerializationOrderAttr(orderAttr);

    auto nodeName = constOp->getAttrOfType<mlir::StringAttr>("onnx_node_name");
    if (nodeName) {
      carrier.setSymbolNameHintAttr(nodeName);
      carrier.setSourceNameAttr(nodeName);
    } else if (auto outputs =
                   constOp->getAttrOfType<mlir::ArrayAttr>("node.outputs")) {
      if (!outputs.empty())
        if (auto outputName = mlir::dyn_cast<mlir::StringAttr>(outputs[0]))
          carrier.setSourceNameAttr(outputName);
    }
    constOp->getResult(0).replaceAllUsesWith(carrier.getResult());
    constOp->erase();
  }
  return mlir::success();
}

/// Replace onnx.Return terminators with func.return.
///
/// In onnx-mlir's own pipeline a dedicated StandardFuncReturnPass handles
/// this before lowering.  Since we bypass that pipeline we must do it
/// ourselves.
static void lowerOnnxReturns(mlir::func::FuncOp funcOp) {
  llvm::SmallVector<mlir::Operation *> returns;
  funcOp.walk([&](mlir::Operation *op) {
    if (op->getName().getStringRef() == "onnx.Return")
      returns.push_back(op);
  });

  for (mlir::Operation *returnOp : returns) {
    mlir::OpBuilder builder(returnOp);
    mlir::func::ReturnOp::create(builder, returnOp->getLoc(),
                                 returnOp->getOperands());
    returnOp->erase();
  }
}

//===----------------------------------------------------------------------===//
// convertComputeOps implementation
//===----------------------------------------------------------------------===//

static mlir::LogicalResult convertComputeOps(mlir::func::FuncOp funcOp,
                                             mlir::MLIRContext *ctx) {
  mlir::RewritePatternSet patterns(ctx);
  populateMatMulConversionPatterns(patterns, ctx);
  populateTransposeConversionPatterns(patterns, ctx);
  populateElementwiseConversionPatterns(patterns, ctx);
  populatePowerConversionPatterns(patterns, ctx);
  populateActivationConversionPatterns(patterns, ctx);
  populateBiasGeluConversionPatterns(patterns, ctx);
  populateFastGeluConversionPatterns(patterns, ctx);
  populateCastConversionPatterns(patterns, ctx);
  populateReductionConversionPatterns(patterns, ctx);
  populateGatherConversionPatterns(patterns, ctx);
  populateCompressConversionPatterns(patterns, ctx);
  populateOneHotConversionPatterns(patterns, ctx);
  populateGatherElementsConversionPatterns(patterns, ctx);
  populateTopKConversionPatterns(patterns, ctx);
  populateScatterElementsConversionPatterns(patterns, ctx);
  populateShapeConversionPatterns(patterns, ctx);
  populateConvConversionPatterns(patterns, ctx);
  populateConvTransposeConversionPatterns(patterns, ctx);
  populateNormConversionPatterns(patterns, ctx);
  populateRotaryEmbeddingConversionPatterns(patterns, ctx);
  populateOnnxRotaryEmbeddingConversionPatterns(patterns, ctx);
  populateGqaConversionPatterns(patterns, ctx);
  populateMultiHeadAttentionConversionPatterns(patterns, ctx);
  populateAttentionConversionPatterns(patterns, ctx);
  populateOnnxAttentionConversionPatterns(patterns, ctx);
  populateMatMulNBitsConversionPatterns(patterns, ctx);
  populateQMoEConversionPatterns(patterns, ctx);
  populateGatherBlockQuantizedConversionPatterns(patterns, ctx);
  populateReshapeConversionPatterns(patterns, ctx);
  populateCausalConvWithStateConversionPatterns(patterns, ctx);
  populateGemmConversionPatterns(patterns, ctx);
  populateWhereConversionPatterns(patterns, ctx);
  populateLinearAttentionConversionPatterns(patterns, ctx);
  populateRangeConversionPatterns(patterns, ctx);
  populateEqualConversionPatterns(patterns, ctx);
  populateDivConversionPatterns(patterns, ctx);
  populateMinConversionPatterns(patterns, ctx);
  populateMaxConversionPatterns(patterns, ctx);
  populateNotConversionPatterns(patterns, ctx);
  populateCosConversionPatterns(patterns, ctx);
  populateErfConversionPatterns(patterns, ctx);
  populateSinConversionPatterns(patterns, ctx);
  populateCeilConversionPatterns(patterns, ctx);
  populateRoundConversionPatterns(patterns, ctx);
  populateAtanConversionPatterns(patterns, ctx);
  populateFloorConversionPatterns(patterns, ctx);
  populateExpConversionPatterns(patterns, ctx);
  populateLogConversionPatterns(patterns, ctx);
  populateCumSumConversionPatterns(patterns, ctx);
  populatePadConversionPatterns(patterns, ctx);
  populateTileConversionPatterns(patterns, ctx);
  populateExpandConversionPatterns(patterns, ctx);
  populateLessConversionPatterns(patterns, ctx);
  populateGreaterConversionPatterns(patterns, ctx);
  populateGreaterOrEqualConversionPatterns(patterns, ctx);
  populateLessOrEqualConversionPatterns(patterns, ctx);
  populateGatherNDConversionPatterns(patterns, ctx);
  populateSignConversionPatterns(patterns, ctx);
  populateModConversionPatterns(patterns, ctx);
  populateConstantOfShapeConversionPatterns(patterns, ctx);
  populateSliceConversionPatterns(patterns, ctx);
  populateScatterNDConversionPatterns(patterns, ctx);
  populateIdentityConversionPatterns(patterns, ctx);
  populateOrConversionPatterns(patterns, ctx);
  populateAndConversionPatterns(patterns, ctx);
  populateAbsConversionPatterns(patterns, ctx);
  populateSizeConversionPatterns(patterns, ctx);
  populateNonZeroConversionPatterns(patterns, ctx);
  populateConcatConversionPatterns(patterns, ctx);
  populateReluConversionPatterns(patterns, ctx);
  populateLeakyReluConversionPatterns(patterns, ctx);
  populateClipConversionPatterns(patterns, ctx);
  populatePoolConversionPatterns(patterns, ctx);
  populateResizeConversionPatterns(patterns, ctx);
  populateGridSampleConversionPatterns(patterns, ctx);
  populateGlobalPoolConversionPatterns(patterns, ctx);
  populateFlattenConversionPatterns(patterns, ctx);

  mlir::GreedyRewriteConfig config;
  config.setStrictness(mlir::GreedyRewriteStrictness::ExistingOps);
  if (mlir::failed(
          mlir::applyPatternsGreedily(funcOp, std::move(patterns), config)))
    return mlir::failure();
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// Module metadata generation
//===----------------------------------------------------------------------===//

/// Generate module metadata attributes required by GenerateInterfacePass.
/// Must be called BEFORE patterns transform function signatures.
static mlir::LogicalResult generateModuleMetadata(mlir::ModuleOp module) {
  auto mainFunc = module.lookupSymbol<mlir::func::FuncOp>("main_graph");
  if (!mainFunc) {
    module.emitError("expected @main_graph function for metadata generation");
    return mlir::failure();
  }

  auto originalFuncType = mainFunc.getFunctionType();
  mlir::OpBuilder builder(module.getContext());

  int64_t inputCount = originalFuncType.getNumInputs();
  llvm::SmallVector<mlir::Attribute> inputShapes;
  llvm::SmallVector<int64_t> inputElementSizes;

  for (mlir::Type inputType : originalFuncType.getInputs()) {
    if (mlir::isa<mlir::hip::ContextType>(inputType)) {
      --inputCount;
      continue;
    }
    if (auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(inputType)) {
      auto elemType = tensorType.getElementType();
      if (!elemType.isIntOrFloat()) {
        mainFunc.emitError("unsupported element type in @main_graph input: ")
            << elemType;
        return mlir::failure();
      }
      llvm::SmallVector<int64_t> shape(tensorType.getShape().begin(),
                                       tensorType.getShape().end());
      inputShapes.push_back(builder.getDenseI64ArrayAttr(shape));
      inputElementSizes.push_back(elemType.getIntOrFloatBitWidth() / 8);
    } else {
      mainFunc.emitError("non-tensor input type in @main_graph: ") << inputType;
      return mlir::failure();
    }
  }

  int64_t outputCount = originalFuncType.getNumResults();
  llvm::SmallVector<mlir::Attribute> outputShapes;
  llvm::SmallVector<int64_t> outputElementSizes;

  for (mlir::Type resultType : originalFuncType.getResults()) {
    if (auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(resultType)) {
      auto elemType = tensorType.getElementType();
      if (!elemType.isIntOrFloat()) {
        mainFunc.emitError("unsupported element type in @main_graph output: ")
            << elemType;
        return mlir::failure();
      }
      llvm::SmallVector<int64_t> shape(tensorType.getShape().begin(),
                                       tensorType.getShape().end());
      outputShapes.push_back(builder.getDenseI64ArrayAttr(shape));
      outputElementSizes.push_back(elemType.getIntOrFloatBitWidth() / 8);
    } else {
      mainFunc.emitError("non-tensor output type in @main_graph: ")
          << resultType;
      return mlir::failure();
    }
  }

  module->setAttr("hipdnn.input_count", builder.getI64IntegerAttr(inputCount));
  module->setAttr("hipdnn.input_shapes", builder.getArrayAttr(inputShapes));
  module->setAttr("hipdnn.input_element_sizes",
                  builder.getDenseI64ArrayAttr(inputElementSizes));
  module->setAttr("hipdnn.output_count",
                  builder.getI64IntegerAttr(outputCount));
  module->setAttr("hipdnn.output_shapes", builder.getArrayAttr(outputShapes));
  module->setAttr("hipdnn.output_element_sizes",
                  builder.getDenseI64ArrayAttr(outputElementSizes));

  LLVM_DEBUG({
    llvm::dbgs() << "[convert-onnx-to-hip] module metadata:"
                 << " input_count=" << inputCount
                 << " input_shapes=" << builder.getArrayAttr(inputShapes)
                 << " input_element_sizes="
                 << builder.getDenseI64ArrayAttr(inputElementSizes)
                 << " output_count=" << outputCount
                 << " output_shapes=" << builder.getArrayAttr(outputShapes)
                 << " output_element_sizes="
                 << builder.getDenseI64ArrayAttr(outputElementSizes) << "\n";
  });

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// ConvertOnnxToHip Pass
//===----------------------------------------------------------------------===//

struct ConvertOnnxToHipPass
    : public impl::ConvertOnnxToHipPassBase<ConvertOnnxToHipPass> {
  using ConvertOnnxToHipPassBase::ConvertOnnxToHipPassBase;

  void runOnOperation() override;
};

void ConvertOnnxToHipPass::runOnOperation() {
  mlir::ModuleOp module = getOperation();
  mlir::MLIRContext *ctx = module.getContext();
  const bool timing = hipdnn_ep_timing_enabled();

  bool hasUntrustedPlan = false;
  module.walk([&](Operation *op) {
    if (op->hasAttr(kBroadcastDimSourcesAttr)) {
      op->emitError("pre-existing broadcast dimension-source plan is not "
                    "trusted compiler metadata");
      hasUntrustedPlan = true;
    }
  });
  if (hasUntrustedPlan)
    return signalPassFailure();
  llvm::scope_exit clearPlans([&]() {
    module.walk(
        [&](Operation *op) { op->removeAttr(kBroadcastDimSourcesAttr); });
  });

  std::optional<OnnxDimParams> dimParams;
  if (module->hasAttr(kOnnxDimParamsModuleAttr)) {
    auto parsed = OnnxDimParams::parse(module);
    if (failed(parsed))
      return signalPassFailure();
    dimParams.emplace(std::move(*parsed));
    module->removeAttr(kOnnxDimParamsModuleAttr);
  }

  auto passStart = timing_now();
  auto phaseStart = passStart;

  auto logSubpass = [&](const char *name, const char *extra = nullptr) {
    if (!timing)
      return;
    double sec = record_elapsed(phaseStart);
    if (extra)
      llvm::errs() << "[ConvertOnnxToHipPass] " << name << ": "
                   << llvm::format("%.3f", sec) << "s  " << extra << "\n";
    else
      llvm::errs() << "[ConvertOnnxToHipPass] " << name << ": "
                   << llvm::format("%.3f", sec) << "s\n";
  };

  // NOTE: onnx.CastLike -> onnx.Cast + dead-type-donor function-argument
  // drop is handled by the standalone simplify-onnx pass, which must run
  // upstream of this one (see lib/Dialect/Transforms/Pipelines.cpp). We
  // capture metadata directly from the (already-simplified) signatures.
  if (mlir::failed(generateModuleMetadata(module)))
    return signalPassFailure();
  logSubpass("metadata");

  int64_t constantOrder = 0;
  for (auto funcOp :
       llvm::make_early_inc_range(module.getOps<mlir::func::FuncOp>())) {
    if (funcOp.isDeclaration())
      continue;
    // Pre-lowering ONNX rewrites that must run BEFORE constants become
    // hip.constant carriers:
    //   * Gather(Shape(x), const_idx) -> tensor.from_elements(tensor.dim),
    //     collapsing the dynseqlen runtime-shape arithmetic chain to a
    //     single 0-D / 1-element result (narrows the host-store-into-pool
    //     footprint the late `--hip-materialize-host-scalars` pass must
    //     redirect out of the GPU pool).
    //   * Inlined FastGelu primitive chain (Pow/Mul/Sum/Tanh) ->
    //     onnx.Gelu(approximate="tanh"), restoring the MorphiZen-supported
    //     form for ORT paths that inline the Gelu function body.
    //   * Projector/vision decompositions (patch-embed Conv-ND -> Gemm,
    //     AveragePool(kernel==stride, no pad) -> Reshape/Transpose/ReduceMean
    //     (overlap / pad cases fall through to hip.pool in PoolConversion),
    //     Pow(x,c) -> Mul chain, broadcasting Div -> Mul(Reciprocal)).
    //     ProjectorOpsRewrites emits NEW `onnx.*` ops (Reshape, Gemm,
    //     ReduceMean, ...) that a subsequent round must visit (e.g. the
    //     AveragePool decomposition's emitted Reshape feeds the next
    //     round's ReduceMean handling), so the set is applied in a
    //     fixed-point loop until quiescence rather than a single pass.
    //   * GatherBlockQuantized INT4 legalize (packed-byte weight shapes,
    //     unsigned_quant_storage, quantize_axis inference) on
    //     com.microsoft.GatherBlockQuantized custom ops.
    // All patterns are value-based and require literal values to remain on
    // `onnx.Constant` until the first carrier sweep below.
    // ExistingOps strictness is sufficient: the patterns either rewrite to
    // tensor.* (Gather) or emit `onnx.*` ops. FastGelu (-> onnx.Gelu) is
    // convergent.
    // ProjectorOpsRewrites emits NEW `onnx.*` ops (Reshape, Gemm,
    // ReduceMean, ...) that a subsequent round must visit (e.g. the
    // AveragePool decomposition's emitted Reshape feeds the next round's
    // ReduceMean handling), so the set is applied in a fixed-point loop until
    // quiescence rather than a single pass. Newly-emitted ops are given their
    // result types in-place at emission (constructed explicitly from the dims
    // the rewriter already knows), so no separate ONNX-level shape-inference
    // pass is run between rounds — the HIP-dialect `--hip-infer-shapes` pass
    // (pipeline tail, post-conversion) resolves any residual dynamic dims. A
    // tiny RewriterBase::Listener flips a flag on any IR mutation; the loop
    // breaks the first round that mutates nothing (capped at kMaxRounds as a
    // safety net).
    {
      struct ChangeFlagListener final : public mlir::RewriterBase::Listener {
        bool changed = false;
        void notifyOperationInserted(mlir::Operation *,
                                     mlir::OpBuilder::InsertPoint) override {
          changed = true;
        }
        void notifyOperationModified(mlir::Operation *) override {
          changed = true;
        }
        void notifyOperationReplaced(mlir::Operation *,
                                     mlir::ValueRange) override {
          changed = true;
        }
        void notifyOperationErased(mlir::Operation *) override {
          changed = true;
        }
      };
      constexpr int kMaxRounds = 4;
      bool quiesced = false;
      for (int round = 0; round < kMaxRounds; ++round) {
        mlir::RewritePatternSet preLoweringPatterns(ctx);
        populateGatherShapeFoldPatterns(preLoweringPatterns, ctx);
        populateTransposeMatMulFoldPatterns(preLoweringPatterns, ctx);
        populateGatherBlockQuantizedPreparePatterns(preLoweringPatterns, ctx);
        populateSliceShapeFoldPatterns(preLoweringPatterns, ctx);
        populatePackBroadcastTo4DPatterns(preLoweringPatterns, ctx);
        populateConstantOfShapePreLoweringPatterns(preLoweringPatterns, ctx);
        populateFastGeluFusionPatterns(preLoweringPatterns, ctx);
        populateErfGeluFusionPatterns(preLoweringPatterns, ctx);
        populateProjectorOpsRewritePatterns(preLoweringPatterns, ctx);
        populateLpNormalizationConversionPatterns(preLoweringPatterns, ctx);
        populatePowDecompositionPatterns(preLoweringPatterns, ctx);
        ChangeFlagListener listener;
        mlir::GreedyRewriteConfig preLoweringConfig;
        preLoweringConfig.setStrictness(
            mlir::GreedyRewriteStrictness::ExistingOps);
        preLoweringConfig.setListener(&listener);
        if (mlir::failed(mlir::applyPatternsGreedily(
                funcOp, std::move(preLoweringPatterns), preLoweringConfig)))
          return signalPassFailure();
        if (!listener.changed) {
          quiesced = true;
          break;
        }
      }
      // If the loop never settles a future pattern set may rely on a
      // rewrite the safety cap silently dropped — surface it so the next
      // maintainer can raise the cap or find the bouncing pattern.
      if (!quiesced)
        funcOp.emitWarning()
            << "convert-onnx-to-hip: pre-lowering round loop hit kMaxRounds="
            << kMaxRounds << " without quiescence";
    }
    // Run one function-level solve over the quiesced generic ONNX IR while
    // `onnx.Constant` values are still inline, before the first carrier sweep.
    // Sparse dataflow shares producer facts across Reshapes and conservatively
    // joins block arguments. Materialization stamps the proof contract consumed
    // and revalidated if conversion reaches the dynamic runtime-shaped
    // fallback.
    if (hasEligibleReshapeShapeProvenanceCandidate(funcOp)) {
      ShapeProvenanceAnalysis analysis(funcOp);
      if (mlir::failed(analysis.run()) ||
          mlir::failed(materializeReshapeShapeOperands(funcOp, analysis)))
        return signalPassFailure();
    }
    if (dimParams && funcOp.getSymName() == "main_graph")
      if (failed(dimParams->annotateBroadcastDimSources(funcOp)))
        return signalPassFailure();
    if (mlir::failed(lowerOnnxConstants(funcOp, constantOrder)))
      return signalPassFailure();
    lowerOnnxReturns(funcOp);
    if (mlir::failed(convertComputeOps(funcOp, ctx)))
      return signalPassFailure();
    // Phase 2: lower any `onnx.Constant` ops SYNTHESIZED during
    // convertComputeOps. Some ONNX->HIP patterns introduce a fresh literal as
    // an `onnx.Constant` (e.g. Relu(x) = hip.max(x, 0) emits a 0-D zero), which
    // the Phase-1 lowerOnnxConstants above could not see because it ran before
    // these ops existed. Without this second call those constants survive the
    // pass and one-shot-bufferize aborts with "op was not bufferized" (the EP
    // then silently falls back to CPU). Re-running creates carriers for them
    // exactly like every imported constant.
    if (mlir::failed(lowerOnnxConstants(funcOp, constantOrder)))
      return signalPassFailure();
  }

  logSubpass("constant carriers + compute ops");

  // Clean up onnx.NoValue and onnx.EntryPoint, plus any other unregistered
  // onnx.* op that ended up with no uses after conversion. The latter case
  // is the dead-shape-arithmetic pattern shipped by some HF ONNX exports:
  // a Shape/Gather/Unsqueeze/Concat chain whose computed shape feeds a
  // Reshape that lowered to tensor.expand_shape via static type info, so
  // the computed-shape operand is never read. Without this DCE,
  // one-shot-bufferize trips on the unregistered op because it has
  // tensor-typed operands but no bufferization interface, and the whole
  // pipeline aborts with "op was not bufferized" — which is silent (CPU
  // fallback) at the EP level.
  //
  // The FastGelu fusion erases its primitive chain inline in
  // reverse-topological order via the rewriter, and the Gather/Shape
  // fold leaves its now-unused `onnx.Shape` and `onnx.Constant` operands
  // alive on purpose (they may be shared across many Gather sites). This
  // walk catches all those single-layer `use_empty` survivors — Shape
  // ops shared across Gather instances that all folded, index constants,
  // and dead-shape-arithmetic survivors from upstream exports.
  llvm::SmallVector<mlir::Operation *> toErase;
  module.walk([&](mlir::Operation *op) {
    llvm::StringRef name = op->getName().getStringRef();
    if (name == "onnx.EntryPoint")
      toErase.push_back(op);
    else if (name.starts_with("onnx.") && op->use_empty())
      toErase.push_back(op);
  });
  for (auto *op : toErase)
    op->erase();

  // Diagnostic: after conversion + dead-op DCE, any op still named `onnx.*`
  // and still USED is an operator with no ONNX->HIP converter (or one whose
  // matcher bailed out). These are the ops that will later make
  // one-shot-bufferize abort with `op was not bufferized` — which is silent at
  // the EP level (CPU fallback). Report them here, grouped by op name with a
  // per-name count, so the missing converters are visible before bufferize.
  // Gated on HIPDNN_EP_DEBUG=1 to avoid noise on normal compiles.
  // `onnx.NoValue` is excluded — it is a legal placeholder consumed by later
  // lowering, not an unconverted compute op.
  if (hipdnn_ep_debug_enabled()) {
    std::map<std::string, int64_t> unconverted;
    module.walk([&](mlir::Operation *op) {
      llvm::StringRef name = op->getName().getStringRef();
      if (!name.starts_with("onnx.") || name == "onnx.NoValue" ||
          op->use_empty())
        return;
      // onnx.Custom is a generic container: the actual op is selected by the
      // `function_name` attribute (MatMulNBits / GroupQueryAttention /
      // Attention / GatherBlockQuantized / ...). Reporting bare "onnx.Custom"
      // would collapse distinct unconverted ops, so append the function name.
      std::string key = name.str();
      if (name == "onnx.Custom") {
        if (auto fn = op->getAttrOfType<mlir::StringAttr>("function_name"))
          key += "[" + fn.getValue().str() + "]";
      }
      ++unconverted[key];
    });
    if (!unconverted.empty()) {
      llvm::errs() << "[convert-onnx-to-hip] " << unconverted.size()
                   << " unconverted onnx op type(s) still live (no ONNX->HIP "
                      "converter):\n";
      for (auto &kv : unconverted)
        llvm::errs() << "  " << kv.first << " x" << kv.second << "\n";
    }
  }

  // ONNX-MLIR tags func.func results with attributes (e.g. "onnx_node_name").
  // Downstream bufferization skips any result that still has attributes, which
  // leaves the signature unconverted and breaks later lowering. Clear them so
  // every result is handled.
  module.walk([&](mlir::func::FuncOp funcOp) {
    unsigned numResults = funcOp.getNumResults();
    if (numResults > 0) {
      llvm::SmallVector<mlir::DictionaryAttr> emptyResAttrs(
          numResults, mlir::DictionaryAttr::get(ctx));
      funcOp.setAllResultAttrs(emptyResAttrs);
    }
  });

  if (timing) {
    llvm::errs() << "[ConvertOnnxToHipPass] total: "
                 << llvm::format("%.3f", elapsed_since(passStart)) << "s\n";
  }
}

} // namespace

} // namespace hip
} // namespace mlir
