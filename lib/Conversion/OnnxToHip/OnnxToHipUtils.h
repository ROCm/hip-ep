/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- OnnxToHipUtils.h - Shared helpers for ONNX-to-HIP patterns --------===//
//
// Shared utility functions and forward declarations used by per-operator
// conversion files.
//
//===----------------------------------------------------------------------===//

#ifndef HIP_CONVERSION_ONNXTOHIP_UTILS_H
#define HIP_CONVERSION_ONNXTOHIP_UTILS_H

#include "ReadbackScalar.h"

#include "hip/Conversion/OnnxToHip/Passes.h"
#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "convert-onnx-to-hip"

namespace mlir {
namespace hip {

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// Sanitize an arbitrary string (typically an ONNX node name) into a valid
/// MLIR bare identifier fragment.  Non-alphanumeric characters are replaced
/// with '_', consecutive underscores are collapsed, and leading/trailing
/// underscores are stripped.  Returns an empty string if the input yields
/// no usable characters.
inline std::string sanitizeForMlirIdentifier(llvm::StringRef raw) {
  std::string sanitized;
  sanitized.reserve(raw.size());
  for (char c : raw) {
    if (std::isalnum(static_cast<unsigned char>(c)) || c == '_')
      sanitized.push_back(c);
    else
      sanitized.push_back('_');
  }
  // Collapse runs of underscores and trim leading/trailing ones.
  std::string result;
  result.reserve(sanitized.size());
  bool lastWasUnderscore = true; // suppress leading '_'
  for (char c : sanitized) {
    if (c == '_') {
      if (!lastWasUnderscore)
        result.push_back(c);
      lastWasUnderscore = true;
    } else {
      result.push_back(c);
      lastWasUnderscore = false;
    }
  }
  while (!result.empty() && result.back() == '_')
    result.pop_back();
  return result;
}

/// Create a tensor.empty for a DPS init operand.  Dynamic dimension sizes
/// are extracted from \p source using tensor.dim at each dynamic index.
/// Suitable for ops where the output shape aligns positionally with one input
/// (e.g., softmax, element-wise).
inline mlir::Value createEmptyTensor(mlir::OpBuilder &builder,
                                     mlir::Location loc,
                                     mlir::RankedTensorType resultType,
                                     mlir::Value source) {
  llvm::SmallVector<mlir::Value> dynSizes;
  for (int64_t dimIdx : llvm::seq<int64_t>(resultType.getRank())) {
    if (resultType.isDynamicDim(dimIdx))
      dynSizes.push_back(
          mlir::tensor::DimOp::create(builder, loc, source, dimIdx));
  }
  return mlir::tensor::EmptyOp::create(builder, loc, resultType.getShape(),
                                       resultType.getElementType(), dynSizes);
}

/// Resolve the ranked result type of an ONNX reduction op (ReduceMax / Sum /
/// Mean / Prod / ...).
///
/// Usually this is just the op's own result type. The ONNX importer can,
/// however, leave a reduction result UNRANKED (`tensor<*xT>`) when the input
/// carries dynamic symbolic dims and the axes are not explicit -- e.g. Phi's
/// `ReduceMax(position_ids)` feeding `GreaterOrEqual`. A bare
/// `mlir::cast<RankedTensorType>` on an unranked type is unchecked in release
/// builds and dereferences garbage -> crash. In that case, infer the result
/// type from the (ranked) input + statically-known reduced axes + keepdims,
/// per ONNX reduction shape semantics:
///   keepdims=1: reduced axes become size 1, other dims preserved.
///   keepdims=0: reduced axes are dropped.
///
/// \p reducedAxes          reduced axis indices (may be negative; normalized
///                         here). For the all-axes default the caller passes
///                         every axis; for a noop (empty axes) it passes none.
/// \p axesStaticallyKnown  false when axes are only known at runtime, in which
///                         case an unranked result cannot be inferred.
/// Returns failure only when the result is unranked AND cannot be inferred
/// (unranked/absent input type, or runtime-only axes).
inline mlir::FailureOr<mlir::RankedTensorType>
inferReduceResultType(mlir::Operation *op, mlir::Value data,
                      llvm::ArrayRef<int64_t> reducedAxes,
                      bool axesStaticallyKnown, int64_t keepdims) {
  if (auto ranked =
          mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType()))
    return ranked;
  auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(data.getType());
  if (!inputType || !axesStaticallyKnown)
    return mlir::failure();
  int64_t rank = inputType.getRank();
  llvm::SmallVector<bool> reduced(rank, false);
  for (int64_t a : reducedAxes)
    reduced[a < 0 ? a + rank : a] = true;
  llvm::SmallVector<int64_t> outShape;
  for (int64_t i = 0; i < rank; ++i) {
    if (reduced[i]) {
      if (keepdims)
        outShape.push_back(1);
    } else {
      outShape.push_back(inputType.getDimSize(i));
    }
  }
  return mlir::RankedTensorType::get(outShape, inputType.getElementType());
}

/// Create a tensor.empty for a DPS init whose shape is the NumPy-style
/// broadcast of \p operands. Operand shapes are right-aligned with the
/// result. For each dynamic dimension of \p resultType, the size is taken
/// from the first operand that truly contributes at that axis -- i.e. whose
/// corresponding dim is not statically 1. Shorter-rank operands (left-padded
/// with 1) and statically-1 dims are skipped. If every spanning operand is
/// statically 1 at the axis, fall back to the first operand that spans it.
///
/// Use this for binary/multinary broadcast elementwise ops (Add, Mul, Where,
/// ...). Do NOT use `createEmptyTensor(resultType, source)` when operands can
/// disagree on which side supplies a dynamic extent (e.g. `[?x1] + [1x?] ->
/// [?x?]` -- dim 0 from lhs, dim 1 from rhs).
inline mlir::FailureOr<mlir::Value>
createBroadcastEmptyTensor(mlir::OpBuilder &builder, mlir::Location loc,
                           mlir::RankedTensorType resultType,
                           mlir::ValueRange operands) {
  int64_t resultRank = resultType.getRank();
  llvm::SmallVector<mlir::Value> dynSizes;
  for (int64_t dimIdx : llvm::seq<int64_t>(resultRank)) {
    if (!resultType.isDynamicDim(dimIdx))
      continue;

    mlir::Value chosen;
    int64_t chosenDim = -1;
    mlir::Value fallback;
    int64_t fallbackDim = -1;
    for (mlir::Value operand : operands) {
      auto t = mlir::dyn_cast<mlir::RankedTensorType>(operand.getType());
      if (!t)
        continue;
      int64_t offset = resultRank - t.getRank();
      if (dimIdx < offset)
        continue;
      int64_t operandDim = dimIdx - offset;
      if (!fallback) {
        fallback = operand;
        fallbackDim = operandDim;
      }
      if (!t.isDynamicDim(operandDim) && t.getDimSize(operandDim) == 1)
        continue;
      chosen = operand;
      chosenDim = operandDim;
      break;
    }
    if (!chosen) {
      chosen = fallback;
      chosenDim = fallbackDim;
    }
    if (!chosen)
      return mlir::failure();
    dynSizes.push_back(
        mlir::tensor::DimOp::create(builder, loc, chosen, chosenDim));
  }
  return mlir::Value(
      mlir::tensor::EmptyOp::create(builder, loc, resultType.getShape(),
                                    resultType.getElementType(), dynSizes));
}

/// Get !hip.context from function argument 0. Returns failure if the
/// function has no arguments or the first argument is not !hip.context.
inline mlir::FailureOr<mlir::Value>
getContextArg(mlir::Operation *op, mlir::PatternRewriter &rewriter) {
  auto funcOp = op->getParentOfType<mlir::func::FuncOp>();
  if (!funcOp)
    return rewriter.notifyMatchFailure(op, "not inside a function");
  auto &entry = funcOp.getBody().front();
  if (entry.getNumArguments() == 0)
    return rewriter.notifyMatchFailure(op, "function has no arguments");
  mlir::Value ctx = entry.getArgument(0);
  if (!mlir::isa<mlir::hip::ContextType>(ctx.getType()))
    return rewriter.notifyMatchFailure(op,
                                       "first argument is not !hip.context");
  return ctx;
}

/// Build a hip.gqa op for the Whisper-MHA / Whisper-encoder-Attention paths.
///
/// Emits one `hip.gqa` op with the 19-slot AttrSizedOperandSegments layout
/// and replaces \p op's results positionally with the hip.gqa's results
/// (callers with fewer than 3 results — e.g. cross-attn or fused-QKV
/// Attention — simply drop the trailing present_* outputs on the floor).
///
/// Defined in HipGqaBuilder.cpp.  Shared between
/// MultiHeadAttentionConversion.cpp and AttentionConversion.cpp.
///
/// \param numHeads        N (== kv_num_heads; HPG=1, multi-head, not GQA)
/// \param scale           ONNX `scale` attr, or 0.0 sentinel for "auto-compute
///                        1/sqrt(head_size) at runtime"
/// \param noCausal        true for bidirectional (encoder / cross-attn);
///                        false for causal (decoder self-attn)
/// \param pastKey/pastValue  Past KV-cache buffers; nullptr when absent
///                        (encoder / cross-attn).  When null the corresponding
///                        operand_segment_size is 0.
/// \param seqlensK        1-D [B] i32 (must already be in scope; either a
///                        runtime arg or an arith.constant)
/// \param totalSeqLen     scalar i32 (must already be in scope)
/// \param outputType      hip.gqa output[0] type (== query type)
/// \param presentKeyType  hip.gqa present_key type (BNSH); for cross-attn or
///                        encoder paths an unused-but-required DPS init buffer
/// \param presentValueType ditto for present_value
mlir::LogicalResult
buildHipGqaCall(mlir::Operation *op, mlir::PatternRewriter &rewriter,
                mlir::Value context, mlir::Value query, mlir::Value key,
                mlir::Value value, mlir::Value pastKey, mlir::Value pastValue,
                mlir::Value seqlensK, mlir::Value totalSeqLen, int64_t numHeads,
                float scale, bool noCausal, mlir::RankedTensorType outputType,
                mlir::RankedTensorType presentKeyType,
                mlir::RankedTensorType presentValueType);

/// `readbackScalarToHost` recovering !hip.context from `op`. Falls back to a
/// bare `tensor.extract` when the function has no context arg (utility funcs /
/// pre-context-arg conversions) -- the only case where the unsynchronized load
/// is acceptable because there is no runtime to sync against.
inline mlir::Value
readbackScalarToHostOrExtract(mlir::PatternRewriter &rewriter,
                              mlir::Location loc, mlir::Operation *op,
                              mlir::Value rank0Tensor) {
  auto ctx = getContextArg(op, rewriter);
  if (mlir::failed(ctx))
    return mlir::tensor::ExtractOp::create(rewriter, loc, rank0Tensor,
                                           mlir::ValueRange{})
        .getResult();
  return readbackScalarToHost(rewriter, loc, *ctx, rank0Tensor);
}

/// `readbackShapeEntryToHost` recovering !hip.context from `op`. Falls back to
/// a bare `tensor.extract %shape[idx]` when there is no context arg.
inline mlir::Value
readbackShapeEntryToHostOrExtract(mlir::PatternRewriter &rewriter,
                                  mlir::Location loc, mlir::Operation *op,
                                  mlir::Value shape, int64_t idx) {
  auto ctx = getContextArg(op, rewriter);
  if (mlir::failed(ctx)) {
    mlir::Value cidx = mlir::arith::ConstantIndexOp::create(rewriter, loc, idx);
    return mlir::tensor::ExtractOp::create(rewriter, loc, shape,
                                           mlir::ValueRange{cidx})
        .getResult();
  }
  return readbackShapeEntryToHost(rewriter, loc, *ctx, shape, idx);
}

// Pattern population functions (one per operator file)
void populateMatMulConversionPatterns(RewritePatternSet &patterns,
                                      MLIRContext *ctx);
void populateTransposeConversionPatterns(RewritePatternSet &patterns,
                                         MLIRContext *ctx);
void populateElementwiseConversionPatterns(RewritePatternSet &patterns,
                                           MLIRContext *ctx);
void populatePowerConversionPatterns(RewritePatternSet &patterns,
                                     MLIRContext *ctx);
void populateActivationConversionPatterns(RewritePatternSet &patterns,
                                          MLIRContext *ctx);
void populateBiasGeluConversionPatterns(RewritePatternSet &patterns,
                                        MLIRContext *ctx);
void populateFastGeluConversionPatterns(RewritePatternSet &patterns,
                                        MLIRContext *ctx);
void populateCastConversionPatterns(RewritePatternSet &patterns,
                                    MLIRContext *ctx);
void populateReduceSumConversionPatterns(RewritePatternSet &patterns,
                                         MLIRContext *ctx);
void populateReduceMeanConversionPatterns(RewritePatternSet &patterns,
                                          MLIRContext *ctx);
void populateMatMulNBitsConversionPatterns(RewritePatternSet &patterns,
                                           MLIRContext *ctx);
void populateQMoEConversionPatterns(RewritePatternSet &patterns,
                                    MLIRContext *ctx);
void populateGatherBlockQuantizedConversionPatterns(RewritePatternSet &patterns,
                                                    MLIRContext *ctx);
void populateConvConversionPatterns(RewritePatternSet &patterns,
                                    MLIRContext *ctx);
void populateConvTransposeConversionPatterns(RewritePatternSet &patterns,
                                             MLIRContext *ctx);
void populateNormConversionPatterns(RewritePatternSet &patterns,
                                    MLIRContext *ctx);
void populateRotaryEmbeddingConversionPatterns(RewritePatternSet &patterns,
                                               MLIRContext *ctx);
void populateGqaConversionPatterns(RewritePatternSet &patterns,
                                   MLIRContext *ctx);
void populateMultiHeadAttentionConversionPatterns(RewritePatternSet &patterns,
                                                  MLIRContext *ctx);
void populateAttentionConversionPatterns(RewritePatternSet &patterns,
                                         MLIRContext *ctx);
void populateGatherConversionPatterns(RewritePatternSet &patterns,
                                      MLIRContext *ctx);
void populateCompressConversionPatterns(RewritePatternSet &patterns,
                                        MLIRContext *ctx);
void populateOneHotConversionPatterns(RewritePatternSet &patterns,
                                      MLIRContext *ctx);
void populateGatherElementsConversionPatterns(RewritePatternSet &patterns,
                                              MLIRContext *ctx);
void populateScatterElementsConversionPatterns(RewritePatternSet &patterns,
                                               MLIRContext *ctx);
void populateShapeConversionPatterns(RewritePatternSet &patterns,
                                     MLIRContext *ctx);
void populateReshapeConversionPatterns(RewritePatternSet &patterns,
                                       MLIRContext *ctx);
void populateCausalConvWithStateConversionPatterns(RewritePatternSet &patterns,
                                                   MLIRContext *ctx);
void populateGemmConversionPatterns(RewritePatternSet &patterns,
                                    MLIRContext *ctx);
void populateWhereConversionPatterns(RewritePatternSet &patterns,
                                     MLIRContext *ctx);
void populateLinearAttentionConversionPatterns(RewritePatternSet &patterns,
                                               MLIRContext *ctx);
void populateRangeConversionPatterns(RewritePatternSet &patterns,
                                     MLIRContext *ctx);
void populateEqualConversionPatterns(RewritePatternSet &patterns,
                                     MLIRContext *ctx);
void populateDivConversionPatterns(RewritePatternSet &patterns,
                                   MLIRContext *ctx);
void populateMinConversionPatterns(RewritePatternSet &patterns,
                                   MLIRContext *ctx);
void populateMaxConversionPatterns(RewritePatternSet &patterns,
                                   MLIRContext *ctx);
void populateReduceMaxConversionPatterns(RewritePatternSet &patterns,
                                         MLIRContext *ctx);
void populateReduceMinConversionPatterns(RewritePatternSet &patterns,
                                         MLIRContext *ctx);
void populateNotConversionPatterns(RewritePatternSet &patterns,
                                   MLIRContext *ctx);
void populateCosConversionPatterns(RewritePatternSet &patterns,
                                   MLIRContext *ctx);
void populateSinConversionPatterns(RewritePatternSet &patterns,
                                   MLIRContext *ctx);
void populateCeilConversionPatterns(RewritePatternSet &patterns,
                                    MLIRContext *ctx);
void populateExpConversionPatterns(RewritePatternSet &patterns,
                                   MLIRContext *ctx);
void populateLogConversionPatterns(RewritePatternSet &patterns,
                                   MLIRContext *ctx);
void populateCumSumConversionPatterns(RewritePatternSet &patterns,
                                      MLIRContext *ctx);
void populatePadConversionPatterns(RewritePatternSet &patterns,
                                   MLIRContext *ctx);
void populateTileConversionPatterns(RewritePatternSet &patterns,
                                    MLIRContext *ctx);
void populateExpandConversionPatterns(RewritePatternSet &patterns,
                                      MLIRContext *ctx);
void populateReduceProdConversionPatterns(RewritePatternSet &patterns,
                                          MLIRContext *ctx);
void populateLessConversionPatterns(RewritePatternSet &patterns,
                                    MLIRContext *ctx);
void populateGreaterConversionPatterns(RewritePatternSet &patterns,
                                       MLIRContext *ctx);
void populateGreaterOrEqualConversionPatterns(RewritePatternSet &patterns,
                                              MLIRContext *ctx);
void populateLessOrEqualConversionPatterns(RewritePatternSet &patterns,
                                           MLIRContext *ctx);
void populateGatherNDConversionPatterns(RewritePatternSet &patterns,
                                        MLIRContext *ctx);
void populateSignConversionPatterns(RewritePatternSet &patterns,
                                    MLIRContext *ctx);
void populateModConversionPatterns(RewritePatternSet &patterns,
                                   MLIRContext *ctx);
void populateConstantOfShapeConversionPatterns(RewritePatternSet &patterns,
                                               MLIRContext *ctx);
void populateSliceConversionPatterns(RewritePatternSet &patterns,
                                     MLIRContext *ctx);
void populateScatterNDConversionPatterns(RewritePatternSet &patterns,
                                         MLIRContext *ctx);
void populateIdentityConversionPatterns(RewritePatternSet &patterns,
                                        MLIRContext *ctx);
void populateOrConversionPatterns(RewritePatternSet &patterns,
                                  MLIRContext *ctx);
void populateAndConversionPatterns(RewritePatternSet &patterns,
                                   MLIRContext *ctx);
void populateAbsConversionPatterns(RewritePatternSet &patterns,
                                   MLIRContext *ctx);
void populateSizeConversionPatterns(RewritePatternSet &patterns,
                                    MLIRContext *ctx);
void populateNonZeroConversionPatterns(RewritePatternSet &patterns,
                                       MLIRContext *ctx);
void populateConcatConversionPatterns(RewritePatternSet &patterns,
                                      MLIRContext *ctx);
void populateReluConversionPatterns(RewritePatternSet &patterns,
                                    MLIRContext *ctx);
void populateLeakyReluConversionPatterns(RewritePatternSet &patterns,
                                         MLIRContext *ctx);
void populateClipConversionPatterns(RewritePatternSet &patterns,
                                    MLIRContext *ctx);
void populatePoolConversionPatterns(RewritePatternSet &patterns,
                                    MLIRContext *ctx);
void populateResizeConversionPatterns(RewritePatternSet &patterns,
                                      MLIRContext *ctx);
void populateGlobalPoolConversionPatterns(RewritePatternSet &patterns,
                                          MLIRContext *ctx);
void populateFlattenConversionPatterns(RewritePatternSet &patterns,
                                       MLIRContext *ctx);

/// Pre-lowering pattern set: collapse the Gather(Shape(x), const_idx)
/// idiom into tensor.from_elements over a tensor.dim of x. Must run
/// BEFORE lowerOnnxConstants so the index value is still inline in the
/// onnx.Constant `value` attribute. See GatherShapeFold.cpp for the
/// dynseqlen-regression rationale.
void populateGatherShapeFoldPatterns(RewritePatternSet &patterns,
                                     MLIRContext *ctx);

/// Pre-lowering pattern set: rewrite the `Reshape(data, Shape(src))` idiom
/// so the shape operand becomes an explicit
/// `tensor.from_elements(tensor.dim(src, *))`. This lets ReshapeConversion's
/// `tensor.reshape` fallback recover per-output-dim sizes when the result has
/// >1 dynamic dim in one reassociation group (otherwise ReshapeConversion
/// ignores its second operand and emits the same SSA dim twice — the [N, N]
/// bug). Sibling of GatherShapeFold; must run BEFORE lowerOnnxConstants.
/// See ReshapeShapeFold.cpp.
void populateReshapeShapeFoldPatterns(RewritePatternSet &patterns,
                                      MLIRContext *ctx);

/// Pre-lowering pattern set: stamp `onnx.Pad`'s compile-time `pads` (and
/// optional `axes`) constant onto the op as `hipdnn.pad_amounts` /
/// `hipdnn.pad_axes` attributes so PadConversion can compute the dynamic
/// output shape from them without reading the (by-then externalized) operand.
/// Sibling of GatherShapeFold; must run BEFORE lowerOnnxConstants. See
/// PadShapeFold.cpp.
void populatePadShapeFoldPatterns(RewritePatternSet &patterns,
                                  MLIRContext *ctx);

/// Pre-lowering pattern set: collapse ORT's inlined `FastGelu` primitive
/// chain (Pow / Mul / Sum / Tanh) back into a single
/// `onnx.Gelu(approximate="tanh")`. ORT inlines the Gelu function body
/// for some loading paths (notably dynamic-shape models) and the inlined
/// primitives have no MorphiZen converters. Must run BEFORE
/// `lowerOnnxConstants` so the literal float values of the embedded
/// constants (3.0, 0.044715, sqrt(2/π), 1.0, 0.5) are still inline.
/// See FastGeluFusion.cpp.
void populateFastGeluFusionPatterns(RewritePatternSet &patterns,
                                    MLIRContext *ctx);

/// Pre-lowering pattern set: decompose `onnx.Pow(x, c)` (constant scalar
/// exponent c, optionally wrapped in `onnx.Cast`/`onnx.CastLike`) into ONNX
/// primitives (`onnx.Mul` / `onnx.Sqrt` / `onnx.Reciprocal`), which then flow
/// through their own ONNX→HIP converters in `convertComputeOps`. Must run
/// BEFORE `lowerOnnxConstants` because production builds externalize every
/// `onnx.Constant` (incl. 1-element scalars) into a memref.global with the
/// value moved to the constants sidecar — at that point the exponent is no
/// longer recoverable from IR. See PowerConversion.cpp.
void populatePowDecompositionPatterns(RewritePatternSet &patterns,
                                      MLIRContext *ctx);
/// Pre-lowering pattern set: collapse the inlined erf-form `Gelu` primitive
/// chain (Div / Erf / Sum / Mul, optionally wrapped in CastLike scalars and
/// `Sqrt(2.0)`) back into a single `onnx.Gelu(approximate="none")`. Some
/// exports (e.g. ConvNeXt) inline the exact erf-based Gelu definition
/// `0.5 * x * (1 + erf(x / sqrt(2)))` as primitives that have no MorphiZen
/// converters. Must run BEFORE `lowerOnnxConstants` so the literal float
/// values (1.0, 2.0, 0.5) of the wrapped constants are still inline.
/// See ErfGeluFusion.cpp.
void populateErfGeluFusionPatterns(RewritePatternSet &patterns,
                                   MLIRContext *ctx);

/// Pre-lowering pattern set: decompose vision/projector ops that have no
/// direct MorphiZen converter into supported primitives — patch-embed
/// Conv-ND → Reshape/Gemm/Reshape, AveragePool(kernel==stride) →
/// Reshape/Transpose/ReduceMean, Pow(x, c) → repeated Mul, ReduceMean →
/// ReduceSum·(1/N), and broadcasting Div → Mul(x, Reciprocal). Emits
/// `onnx.*` ops with result types built explicitly from the dims the
/// rewriter already knows (no separate shape pass needed at emission;
/// `--hip-infer-shapes` resolves any residual dynamic dims post-conversion).
/// Runs in the same ExistingOps pre-lowering set as FastGeluFusion. See
/// ProjectorOpsRewrites.cpp.
void populateProjectorOpsRewritePatterns(RewritePatternSet &patterns,
                                         MLIRContext *ctx);

/// Pre-lowering pattern set: decompose `onnx.LpNormalization` into a small
/// chain of already-supported ONNX primitives (Mul / Sqrt / ReduceSum /
/// Div). Lives in the pre-lowering loop next to FastGeluFusion /
/// ProjectorOpsRewrites so the freshly-emitted onnx.* primitives become
/// "existing" ops by the next round and reach `convertComputeOps`'s
/// converters as ordinary onnx.* ops. See LpNormalizationConversion.cpp.
void populateLpNormalizationConversionPatterns(RewritePatternSet &patterns,
                                               MLIRContext *ctx);

} // namespace hip
} // namespace mlir

#endif // HIP_CONVERSION_ONNXTOHIP_UTILS_H
