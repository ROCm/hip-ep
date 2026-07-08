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

#include "OnnxToHipUtils.h"

#include "hip/Conversion/OnnxToHip/ConstantsIO.h"
#include "hip/Support/DiskFileSystem.h"
#include "hip/timing.h"
#include "morphizen-foundation/file_io.hpp"

#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4244) // Conversion warnings in LLVM JSON.h
#endif
#include "llvm/Support/JSON.h"
#ifdef _MSC_VER
#pragma warning(pop)
#endif
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <chrono>
#include <cstring>
#include <fstream>
#include <vector>

#define DEBUG_TYPE "convert-onnx-to-hip"

namespace mlir {
namespace hip {

#define GEN_PASS_DEF_CONVERTONNXTOHIPPASS
#include "hip/Dialect/Transforms/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Constant carrier lowering
//===----------------------------------------------------------------------===//
//
// onnx.Constant is lowered to a neutral hip.constant carrier (below). The
// externalization core (state, source-kind helpers, the constants.bin finalize
// with its three emit modes) moved to the standalone
// --hip-externalize-constants pass
// (lib/Dialect/Transforms/ExternalizeConstants.cpp), which serializes both
// these carriers and plugin-emitted hip.constant ops uniformly.

/// Lower onnx.Constant ops to neutral hip.constant carriers. The
/// externalize-vs-inline decision and the constants.bin write are deferred to
/// the standalone --hip-externalize-constants pass (which also serializes
/// plugin-emitted hip.constant ops). A value constant carries its dense
/// `value`; an ORT external-data constant carries `location`/`offset`/`size`.
/// The onnx_node_name / node.outputs naming hints are carried through so the
/// externalize pass reproduces identical externalized-global symbol names.
///
/// Before: %c = "onnx.Constant"() {value = dense<...>} : () -> tensor<8xf32>
/// After:  %c = hip.constant {value = dense<...>} : tensor<8xf32>
static mlir::LogicalResult lowerOnnxConstants(mlir::func::FuncOp funcOp) {
  llvm::SmallVector<mlir::Operation *> constants;
  funcOp.walk([&](mlir::Operation *op) {
    if (op->getName().getStringRef() == "onnx.Constant")
      constants.push_back(op);
  });

  for (mlir::Operation *constOp : constants) {
    auto tensorType =
        mlir::dyn_cast<mlir::RankedTensorType>(constOp->getResult(0).getType());
    if (!tensorType)
      return constOp->emitError("onnx.Constant has non-ranked result type");

    bool hasValue = mlir::isa_and_present<mlir::DenseElementsAttr>(
        constOp->getAttrOfType<mlir::ElementsAttr>("value"));
    bool hasLocation = constOp->hasAttr("location");
    if (!hasValue && !hasLocation)
      return constOp->emitError(
          "unsupported onnx.Constant form (expected dense value attribute "
          "or location attribute)");

    mlir::OpBuilder builder(constOp);
    auto hipConst =
        mlir::hip::ConstantOp::create(builder, constOp->getLoc(), tensorType);
    if (hasValue) {
      hipConst->setAttr("value", constOp->getAttr("value"));
    } else {
      // Carry the ORT external-data ref verbatim; offset/size may be absent on
      // malformed input, in which case the externalize pass emits the
      // diagnostic (guard avoids setting a null attribute here).
      hipConst->setAttr("location", constOp->getAttr("location"));
      if (auto o = constOp->getAttr("offset"))
        hipConst->setAttr("offset", o);
      if (auto s = constOp->getAttr("size"))
        hipConst->setAttr("size", s);
    }
    if (auto n = constOp->getAttr("onnx_node_name"))
      hipConst->setAttr("onnx_node_name", n);
    if (auto n = constOp->getAttr("node.outputs"))
      hipConst->setAttr("node.outputs", n);

    constOp->getResult(0).replaceAllUsesWith(hipConst.getResult());
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
  populateCastConversionPatterns(patterns, ctx);
  populateReduceSumConversionPatterns(patterns, ctx);
  populateReduceMeanConversionPatterns(patterns, ctx);
  populateGatherConversionPatterns(patterns, ctx);
  populateShapeConversionPatterns(patterns, ctx);
  populateConvConversionPatterns(patterns, ctx);
  populateConvTransposeConversionPatterns(patterns, ctx);
  populateNormConversionPatterns(patterns, ctx);
  populateRotaryEmbeddingConversionPatterns(patterns, ctx);
  populateGqaConversionPatterns(patterns, ctx);
  populateMultiHeadAttentionConversionPatterns(patterns, ctx);
  populateAttentionConversionPatterns(patterns, ctx);
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
  populateReduceMaxConversionPatterns(patterns, ctx);
  populateMinConversionPatterns(patterns, ctx);
  populateMaxConversionPatterns(patterns, ctx);
  populateNotConversionPatterns(patterns, ctx);
  populateCosConversionPatterns(patterns, ctx);
  populateSinConversionPatterns(patterns, ctx);
  populateExpConversionPatterns(patterns, ctx);
  populateCumSumConversionPatterns(patterns, ctx);
  populatePadConversionPatterns(patterns, ctx);
  populateTileConversionPatterns(patterns, ctx);
  populateExpandConversionPatterns(patterns, ctx);
  populateReduceProdConversionPatterns(patterns, ctx);
  populateLessConversionPatterns(patterns, ctx);
  populateGreaterOrEqualConversionPatterns(patterns, ctx);
  populateLessOrEqualConversionPatterns(patterns, ctx);
  populateGatherNDConversionPatterns(patterns, ctx);
  populateSignConversionPatterns(patterns, ctx);
  populateModConversionPatterns(patterns, ctx);
  populateConstantOfShapeConversionPatterns(patterns, ctx);
  populateSliceConversionPatterns(patterns, ctx);
  populateScatterNDConversionPatterns(patterns, ctx);
  populateIdentityConversionPatterns(patterns, ctx);
  populateAndConversionPatterns(patterns, ctx);
  populateSizeConversionPatterns(patterns, ctx);
  populateNonZeroConversionPatterns(patterns, ctx);
  populateConcatConversionPatterns(patterns, ctx);
  populateReluConversionPatterns(patterns, ctx);
  populateLeakyReluConversionPatterns(patterns, ctx);
  populateClipConversionPatterns(patterns, ctx);
  populatePoolConversionPatterns(patterns, ctx);
  populateResizeConversionPatterns(patterns, ctx);
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

  for (auto funcOp :
       llvm::make_early_inc_range(module.getOps<mlir::func::FuncOp>())) {
    if (funcOp.isDeclaration())
      continue;
    // Pre-lowering ONNX rewrites that must run BEFORE constants are
    // externalized:
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
    // All patterns are value-based and require the literal constants to
    // still be inline in `onnx.Constant` `value` attributes — once the
    // constants are externalized to memref.get_global the matchers break.
    // ExistingOps strictness is sufficient: the patterns either rewrite to
    // tensor.* (Gather) or emit `onnx.*` ops. FastGelu (-> onnx.Gelu) and
    // ReshapeShapeFold (roots on onnx.Reshape, only swaps its shape operand
    // in place; the re-visit fails the "operand1 is onnx.Shape" guard) are
    // convergent. ProjectorOpsRewrites emits NEW `onnx.*` ops (Reshape, Gemm,
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
        populateReshapeShapeFoldPatterns(preLoweringPatterns, ctx);
        populatePadShapeFoldPatterns(preLoweringPatterns, ctx);
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
    // Run ConstantOfShape folding BEFORE `lowerOnnxConstants` so it can
    // still see the original `onnx.Constant` (or `onnx.Shape`) as the
    // shape input.  Once `lowerOnnxConstants` externalises the constant,
    // the IR becomes `memref.global` with a null `initial_value` (data
    // lives in `constants.bin`) and the fold can no longer reach it.
    // Roots on `onnx.ConstantOfShape`, disjoint from the pre-lowering
    // patterns above (which root on `onnx.Gather` and `onnx.Tanh`), so
    // ordering and pattern-set separation are both safe.
    {
      mlir::RewritePatternSet preFoldPatterns(ctx);
      populateConstantOfShapeConversionPatterns(preFoldPatterns, ctx);
      mlir::GreedyRewriteConfig cfg;
      cfg.setStrictness(mlir::GreedyRewriteStrictness::ExistingOps);
      if (mlir::failed(mlir::applyPatternsGreedily(
              funcOp, std::move(preFoldPatterns), cfg)))
        return signalPassFailure();
    }
    if (mlir::failed(lowerOnnxConstants(funcOp)))
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
    // then silently falls back to CPU). Re-running externalizes / inlines them
    // exactly like every other constant.
    if (mlir::failed(lowerOnnxConstants(funcOp)))
      return signalPassFailure();
  }

  logSubpass("constants + compute ops");

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
