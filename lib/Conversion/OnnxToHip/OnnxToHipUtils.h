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

/// True iff `v` is an entry-block argument of an ancestor `func.func`. Used
/// by Range / ConstantOfShape conversions to decide whether their shape-
/// tensor operands trace to a host-readable EP input (Category B) or to an
/// intermediate value produced inside the graph (Category C — needs a
/// RuntimeSlot publisher). The same predicate is used by the EP-side
/// resolver to decide between InputValueI64 leaves and RuntimeSlot leaves.
inline bool operandIsFuncEntryBlockArg(mlir::Value v) {
  auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(v);
  if (!blockArg)
    return false;
  auto *owner = blockArg.getOwner();
  if (!owner || !owner->getParent())
    return false;
  auto funcOp = mlir::dyn_cast<mlir::func::FuncOp>(owner->getParentOp());
  if (!funcOp)
    return false;
  return &funcOp.getBody().front() == owner;
}

/// If `operand` already has the shape and rank of `targetType`, return it
/// unchanged.  Otherwise insert a `hip.expand` that broadcasts `operand`
/// to `targetType` (preserving its element type) and return the expanded
/// result.
///
/// This is the canonical solution to the binary-elementwise broadcasting
/// problem: the HipToLLVM elementwise lowerings (Equal / Less / And /
/// Add / Mul / Sub / ...) pass a single `num_elements` to the runtime
/// and require both operand buffers to already have identical layouts
/// (see the comment block at the top of
/// `3rd-party/custom_kernels/hip/elementwise_binary_kernel.hip`).
/// ONNX, however, allows NumPy-style broadcasting; the canonical
/// trigger is `Equal(input_ids, scalar_const)`, which would otherwise
/// have the kernel read past the 1-element constant buffer and produce
/// uninitialised junk.
///
/// Dynamic target dims are sourced from `shapeSource`, which MUST have
/// the same shape as `targetType`. The standard caller is a binary
/// elementwise op: `shapeSource` is the operand whose shape already
/// equals the result type (i.e. the "wide" side of the broadcast).
/// For each dynamic dim of `targetType`, the helper emits a
/// `tensor.dim shapeSource, i` and uses those values to (a) size the
/// `tensor.empty` for the DPS init operand and (b) build the
/// `hip.expand` shape operand via `tensor.from_elements`. Static dims
/// are baked into the shape operand as `arith.constant index` literals
/// so the IR remains valid even when targetType is fully static.
///
/// All `tensor.dim` SSA values are created at the current rewriter
/// insertion point — which is the user op's rewrite location — so they
/// dominate the consuming `tensor.empty` and `hip.expand`. This is
/// load-bearing for dynamic-shape graphs (SSA dominance violations are
/// the leading failure class when broadcastToShape is called from a
/// binary op converter; see Phase 2a in
/// docs/bisecting-ep-compile-failures.md).
inline mlir::Value broadcastToShape(mlir::PatternRewriter &rewriter,
                                    mlir::Location loc, mlir::Value context,
                                    mlir::Value operand,
                                    mlir::RankedTensorType targetType,
                                    mlir::Value shapeSource) {
  auto opType = mlir::dyn_cast<mlir::RankedTensorType>(operand.getType());
  if (opType && opType.getShape() == targetType.getShape())
    return operand;

  // Materialise one tensor.dim per dynamic target dim, pulled from
  // shapeSource. Static dims become arith.constant index literals so
  // we can build the rank-1 shape tensor uniformly via
  // tensor.from_elements (which takes any mix of dynamic + static
  // index values).
  int64_t targetRank = targetType.getRank();
  llvm::SmallVector<mlir::Value> dynSizes;
  llvm::SmallVector<mlir::Value> shapeElements;
  shapeElements.reserve(targetRank);
  for (int64_t i = 0; i < targetRank; ++i) {
    mlir::Value dim;
    if (targetType.isDynamicDim(i)) {
      assert(shapeSource && "broadcastToShape needs a shapeSource with the "
                            "result shape when target has dynamic dims");
      dim = mlir::tensor::DimOp::create(rewriter, loc, shapeSource, i);
      dynSizes.push_back(dim);
    } else {
      dim = mlir::arith::ConstantIndexOp::create(rewriter, loc,
                                                 targetType.getDimSize(i));
    }
    // hip.expand expects an i64 shape tensor; tensor.dim / arith.constant
    // index produce `index`, so cast through arith.index_cast.
    shapeElements.push_back(mlir::arith::IndexCastOp::create(
        rewriter, loc, rewriter.getI64Type(), dim));
  }

  mlir::Type elemType =
      opType ? opType.getElementType() : targetType.getElementType();
  auto resultType =
      mlir::RankedTensorType::get(targetType.getShape(), elemType);
  mlir::Value init =
      mlir::tensor::EmptyOp::create(rewriter, loc, resultType.getShape(),
                                    resultType.getElementType(), dynSizes);

  auto shapeTensorType =
      mlir::RankedTensorType::get({targetRank}, rewriter.getI64Type());
  mlir::Value shapeTensor = mlir::tensor::FromElementsOp::create(
      rewriter, loc, shapeTensorType, shapeElements);

  return mlir::hip::ExpandOp::create(rewriter, loc, resultType, context,
                                     operand, shapeTensor, init)
      ->getResult(0);
}

/// Static-only overload kept for callers that have always-static target
/// shapes (the historical signature; covered by the assertion in the
/// dynamic path). Forwards to the dynamic overload with no shapeSource
/// — safe because targetType.hasStaticShape() means no tensor.dim is
/// emitted.
inline mlir::Value broadcastToShape(mlir::PatternRewriter &rewriter,
                                    mlir::Location loc, mlir::Value context,
                                    mlir::Value operand,
                                    mlir::RankedTensorType targetType) {
  assert(targetType.hasStaticShape() &&
         "static-only overload of broadcastToShape called with a "
         "dynamic targetType; use the 5-arg overload and pass a "
         "shapeSource value with the result shape");
  return broadcastToShape(rewriter, loc, context, operand, targetType,
                          /*shapeSource=*/nullptr);
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
void populateShapeConversionPatterns(RewritePatternSet &patterns,
                                     MLIRContext *ctx);
void populateConcatConversionPatterns(RewritePatternSet &patterns,
                                      MLIRContext *ctx);

} // namespace hip
} // namespace mlir

#endif // HIP_CONVERSION_ONNXTOHIP_UTILS_H
