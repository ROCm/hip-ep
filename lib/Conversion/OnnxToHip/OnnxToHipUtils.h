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

#include "hip/Conversion/OnnxToHip/Passes.h"
#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
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
void populateCastConversionPatterns(RewritePatternSet &patterns,
                                    MLIRContext *ctx);
void populateReduceSumConversionPatterns(RewritePatternSet &patterns,
                                         MLIRContext *ctx);
void populateMatMulNBitsConversionPatterns(RewritePatternSet &patterns,
                                           MLIRContext *ctx);
void populateQMoEConversionPatterns(RewritePatternSet &patterns,
                                    MLIRContext *ctx);
void populateConvConversionPatterns(RewritePatternSet &patterns,
                                    MLIRContext *ctx);
void populateNormConversionPatterns(RewritePatternSet &patterns,
                                    MLIRContext *ctx);
void populateRotaryEmbeddingConversionPatterns(RewritePatternSet &patterns,
                                               MLIRContext *ctx);
void populateGqaConversionPatterns(RewritePatternSet &patterns,
                                   MLIRContext *ctx);
void populateMultiHeadAttentionConversionPatterns(RewritePatternSet &patterns,
                                                  MLIRContext *ctx);
void populateGatherConversionPatterns(RewritePatternSet &patterns,
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
void populateReduceMaxConversionPatterns(RewritePatternSet &patterns,
                                         MLIRContext *ctx);
void populateNotConversionPatterns(RewritePatternSet &patterns,
                                   MLIRContext *ctx);
void populateCosConversionPatterns(RewritePatternSet &patterns,
                                   MLIRContext *ctx);
void populateSinConversionPatterns(RewritePatternSet &patterns,
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
void populateAndConversionPatterns(RewritePatternSet &patterns,
                                   MLIRContext *ctx);
void populateSizeConversionPatterns(RewritePatternSet &patterns,
                                    MLIRContext *ctx);
void populateNonZeroConversionPatterns(RewritePatternSet &patterns,
                                       MLIRContext *ctx);
void populateConcatConversionPatterns(RewritePatternSet &patterns,
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

} // namespace hip
} // namespace mlir

#endif // HIP_CONVERSION_ONNXTOHIP_UTILS_H
