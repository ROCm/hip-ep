/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- LpNormalizationConversion.cpp - Decompose onnx.LpNormalization ----===//
//
// `onnx.LpNormalization` (opset 22) computes
//
//     output = input / Lp_norm(input, axis)
//
// where `Lp_norm` reduces along a single user-specified `axis` and `p` is
// either 1 or 2. There is no MIOpen / hipBLASLt primitive that maps 1:1 to
// this op; instead we rewrite it at the ONNX level as a small chain of ops
// that already have HIP converters:
//
//   p == 2:
//     %x_sq    = onnx.Mul(%x, %x)                        // same shape as %x
//     %sum_sq  = onnx.ReduceSum(%x_sq, [axis], keepdims=1)
//     %norm    = onnx.Sqrt(%sum_sq)
//     %result  = onnx.Div(%x, %norm)                     // broadcasts on axis
//
//   p == 1:
//     %x_sq    = onnx.Mul(%x, %x)
//     %abs_x   = onnx.Sqrt(%x_sq)                        // |x| via sqrt(x^2)
//     %norm    = onnx.ReduceSum(%abs_x, [axis], keepdims=1)
//     %result  = onnx.Div(%x, %norm)
//
// The pattern runs in the convert-onnx-to-hip pre-lowering loop next to
// FastGeluFusion / ProjectorOpsRewrites. Putting it there matters: the
// pre-lowering loop rewrites with `GreedyRewriteStrictness::ExistingOps`
// over MULTIPLE rounds, so the freshly-emitted onnx.Mul / onnx.Sqrt /
// onnx.ReduceSum / onnx.Div ops become "existing" by the next round and
// reach the main `convertComputeOps` driver below as ordinary onnx.* ops.
// A pattern living inside `convertComputeOps` itself would not get the
// same treatment — that driver is single-shot and would leave the
// freshly-created onnx.* primitives unconverted, tripping the survival
// check at the bottom of the pass.
//
// The broadcasting Div with a smaller-rank rhs is intentionally LEFT to
// `BroadcastDivToMulReciprocal` (also in the pre-lowering loop) which
// rewrites it into Mul(x, Reciprocal(norm)) — `hip.div`'s flat
// element-wise kernel does not broadcast.
//
// Spec note: ONNX 22's text says "When the Lp norm is zero ... the output
// is defined to be zero to avoid division by zero." The accompanying
// example, however, computes `y = x / norm` directly, which produces
// NaN/Inf when the norm is zero. Matching the canonical ORT CPU
// behaviour (and the example) we do the simple division here. Adding
// strict spec compliance would cost an Equal + Where pair around the
// final Div; revisit if a downstream model trips the all-zero edge case.
//
// Before:
//   %y = onnx.LpNormalization(%x) {axis = -1, p = 2}
//        : (tensor<2x3xf32>) -> tensor<2x3xf32>
//
// After (p=2, conceptual — the pre-lowering loop's Div rewriter then
// turns the broadcasting Div into Mul + Reciprocal before hip lowering):
//   %x_sq   = onnx.Mul(%x, %x) : tensor<2x3xf32>
//   %axes   = onnx.Constant {value = dense<[1]>}      : tensor<1xi64>
//   %sum_sq = onnx.ReduceSum(%x_sq, %axes) {keepdims = 1, ...}
//             : (tensor<2x3xf32>, tensor<1xi64>) -> tensor<2x1xf32>
//   %norm   = onnx.Sqrt(%sum_sq) : tensor<2x1xf32>
//   %y      = onnx.Div(%x, %norm)
//             : (tensor<2x3xf32>, tensor<2x1xf32>) -> tensor<2x3xf32>
//
//===----------------------------------------------------------------------===//

#include "OnnxToHipUtils.h"

#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Debug.h"

#include <cstdlib>
#include <cstring>

#define DEBUG_TYPE "lpnormalization-decompose"

STATISTIC(NumLpNormalizationRewrites,
          "onnx.LpNormalization rewrites into Mul / ReduceSum / Sqrt / Div");
STATISTIC(NumLpNormalizationFused,
          "onnx.LpNormalization fused into a single hip.l2_norm op");

namespace mlir {
namespace hip {

namespace {

// Opt-in toggle for the fused last-axis hip.l2_norm fast path. Off by default
// so the baseline (decomposition) is unchanged until the A/B measurement
// justifies flipping the default. Enabled when HIPDNN_EP_FUSE_LPNORM is set to
// a non-zero / non-"off" value.
inline bool lpNormFuseEnabled() {
  const char *v = std::getenv("HIPDNN_EP_FUSE_LPNORM");
  if (!v || !*v)
    return false;
  return std::strcmp(v, "0") != 0 && std::strcmp(v, "off") != 0 &&
         std::strcmp(v, "false") != 0;
}

/// Decomposes `onnx.LpNormalization` into a chain of already-supported ONNX
/// primitives. See file header for the rewrite shape and the rationale for
/// living in the pre-lowering loop instead of `convertComputeOps`.
struct LpNormalizationDecompose : public mlir::RewritePattern {
  LpNormalizationDecompose(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.LpNormalization", /*benefit=*/2, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 1 || op->getNumResults() != 1)
      return rewriter.notifyMatchFailure(op, "lpnorm.arity");

    mlir::Value x = op->getOperand(0);
    auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(x.getType());
    auto outputType =
        mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
    if (!inputType || !outputType)
      return rewriter.notifyMatchFailure(op, "lpnorm.not_ranked");
    if (inputType.getShape() != outputType.getShape() ||
        inputType.getElementType() != outputType.getElementType())
      return rewriter.notifyMatchFailure(op, "lpnorm.in_out_type_mismatch");

    int64_t rank = inputType.getRank();
    if (rank == 0)
      return rewriter.notifyMatchFailure(op, "lpnorm.scalar_input");

    // Read `axis` (default -1).
    int64_t axis = -1;
    if (auto axisAttr = op->getAttrOfType<mlir::IntegerAttr>("axis"))
      axis = axisAttr.getSInt();
    int64_t normAxis = axis < 0 ? axis + rank : axis;
    if (normAxis < 0 || normAxis >= rank)
      return rewriter.notifyMatchFailure(op, "lpnorm.axis_oob");

    // Read `p` (default 2). Spec: only 1 and 2 are supported.
    int64_t p = 2;
    if (auto pAttr = op->getAttrOfType<mlir::IntegerAttr>("p"))
      p = pAttr.getSInt();
    if (p != 1 && p != 2)
      return rewriter.notifyMatchFailure(op, "lpnorm.p_not_1_or_2");

    // ONNX type-constraint T: bfloat16 / float16 / float / double. The
    // decomposed primitives (Mul, ReduceSum, Sqrt, Div) must each see a
    // floating-point element type — refuse anything else loudly so a
    // future opset extension surfaces here instead of in some downstream
    // converter.
    auto elemType = inputType.getElementType();
    if (!elemType.isF16() && !elemType.isF32() && !elemType.isBF16() &&
        !elemType.isF64())
      return rewriter.notifyMatchFailure(op, "lpnorm.elem_type_not_float");

    mlir::Location loc = op->getLoc();
    mlir::MLIRContext *ctx = rewriter.getContext();

    // Reduced shape: same as input but with a `1` along the reduce axis
    // (keepdims=1). Dynamic dims on other axes pass through unchanged.
    llvm::SmallVector<int64_t> reducedShape(inputType.getShape().begin(),
                                            inputType.getShape().end());
    reducedShape[normAxis] = 1;
    auto reducedType = mlir::RankedTensorType::get(reducedShape, elemType);

    // Step 1: %x_sq = Mul(x, x). Element-wise, same shape/type as input.
    mlir::OperationState mulState(loc, "onnx.Mul");
    mulState.addOperands({x, x});
    mulState.addTypes(inputType);
    mlir::Value xSq = rewriter.create(mulState)->getResult(0);

    // Step 2 (p=1 only): %abs_x = Sqrt(%x_sq) = |x|.  We avoid emitting an
    // explicit `onnx.Abs` because there is no `onnx.Abs` converter in this
    // pipeline and we don't want to add one just for this one decomposition.
    mlir::Value reduceInput = xSq;
    if (p == 1) {
      mlir::OperationState sqrtState(loc, "onnx.Sqrt");
      sqrtState.addOperands(xSq);
      sqrtState.addTypes(inputType);
      reduceInput = rewriter.create(sqrtState)->getResult(0);
    }

    // Step 3: %reduced = ReduceSum(reduceInput, axes=[normAxis], keepdims=1).
    // Build the axes constant inline; the existing ReduceSum converter
    // accepts axes either as an attribute or as an operand. We use the
    // operand form so the reduction axis is visible to InferOnnxShapes.
    auto i64Type = rewriter.getI64Type();
    auto axesType = mlir::RankedTensorType::get({1}, i64Type);
    auto axesValueAttr = mlir::DenseElementsAttr::get(
        axesType, llvm::ArrayRef<int64_t>{normAxis});
    mlir::OperationState axesState(loc, "onnx.Constant");
    axesState.addTypes(axesType);
    axesState.addAttribute("value", axesValueAttr);
    mlir::Value axes = rewriter.create(axesState)->getResult(0);

    // ONNX importers / hand-written IR alike normally encode `keepdims` and
    // `noop_with_empty_axes` as `si64` IntegerAttr. Construct them
    // explicitly so the existing ReduceSum converter reads the right value
    // through `getSInt()`.
    auto sintType = mlir::IntegerType::get(ctx, 64, mlir::IntegerType::Signed);
    mlir::OperationState reduceState(loc, "onnx.ReduceSum");
    reduceState.addOperands({reduceInput, axes});
    reduceState.addTypes(reducedType);
    reduceState.addAttribute("keepdims", mlir::IntegerAttr::get(sintType, 1));
    reduceState.addAttribute("noop_with_empty_axes",
                             mlir::IntegerAttr::get(sintType, 0));
    mlir::Value reduced = rewriter.create(reduceState)->getResult(0);

    // Step 4 (p=2 only): %norm = Sqrt(%reduced).  For p=1 the reduce result
    // already IS the norm.
    mlir::Value norm = reduced;
    if (p == 2) {
      mlir::OperationState sqrtState(loc, "onnx.Sqrt");
      sqrtState.addOperands(reduced);
      sqrtState.addTypes(reducedType);
      norm = rewriter.create(sqrtState)->getResult(0);
    }

    // Step 5: %result = Div(%x, %norm).  `norm` has shape `[..., 1, ...]`
    // (the reduce axis is 1, all others match `x`), so this Div broadcasts
    // along the reduce axis. `BroadcastDivToMulReciprocal` (also in the
    // pre-lowering loop) will rewrite this into Mul(x, Reciprocal(norm))
    // before HIP lowering — `hip.div` itself is non-broadcasting.
    mlir::OperationState divState(loc, "onnx.Div");
    divState.addOperands({x, norm});
    divState.addTypes(outputType);
    mlir::Value result = rewriter.create(divState)->getResult(0);

    rewriter.replaceOp(op, result);
    ++NumLpNormalizationRewrites;
    LLVM_DEBUG(llvm::dbgs()
               << "[" DEBUG_TYPE "] decomposed LpNormalization " << inputType
               << " axis=" << normAxis << " p=" << p << "\n");
    return mlir::success();
  }
};

/// Fuses `onnx.LpNormalization` into a single `hip.l2_norm` op when the
/// normalization axis is the trailing (contiguous) axis with a STATIC extent
/// and `p` is 1 or 2. Higher benefit than `LpNormalizationDecompose`, so the
/// greedy driver tries this first and only falls back to the decomposition
/// when this pattern reports a match failure (toggle off, non-last axis,
/// dynamic last dim, or an unsupported dtype). Emits the hip.* op directly
/// (like the other Norm conversions) — legal in the pre-lowering loop because
/// the ctx handle is the enclosing function's first argument.
struct LpNormToHip : public mlir::RewritePattern {
  LpNormToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.LpNormalization", /*benefit=*/3, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    if (!lpNormFuseEnabled())
      return rewriter.notifyMatchFailure(op, "lpnorm.fuse_disabled");

    if (op->getNumOperands() != 1 || op->getNumResults() != 1)
      return rewriter.notifyMatchFailure(op, "lpnorm.arity");

    mlir::Value x = op->getOperand(0);
    auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(x.getType());
    auto outputType =
        mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
    if (!inputType || !outputType)
      return rewriter.notifyMatchFailure(op, "lpnorm.not_ranked");
    if (inputType.getShape() != outputType.getShape() ||
        inputType.getElementType() != outputType.getElementType())
      return rewriter.notifyMatchFailure(op, "lpnorm.in_out_type_mismatch");

    int64_t rank = inputType.getRank();
    if (rank == 0)
      return rewriter.notifyMatchFailure(op, "lpnorm.scalar_input");

    int64_t axis = -1;
    if (auto axisAttr = op->getAttrOfType<mlir::IntegerAttr>("axis"))
      axis = axisAttr.getSInt();
    int64_t normAxis = axis < 0 ? axis + rank : axis;
    if (normAxis < 0 || normAxis >= rank)
      return rewriter.notifyMatchFailure(op, "lpnorm.axis_oob");

    // Fast path requires the trailing (contiguous) axis with a static extent:
    // the kernel reshapes to [outer, norm_size] and norm_size is a compile-time
    // constant. Anything else falls back to the decomposition.
    if (normAxis != rank - 1)
      return rewriter.notifyMatchFailure(op, "lpnorm.not_last_axis");
    if (inputType.isDynamicDim(normAxis))
      return rewriter.notifyMatchFailure(op, "lpnorm.dynamic_norm_dim");

    int64_t p = 2;
    if (auto pAttr = op->getAttrOfType<mlir::IntegerAttr>("p"))
      p = pAttr.getSInt();
    if (p != 1 && p != 2)
      return rewriter.notifyMatchFailure(op, "lpnorm.p_not_1_or_2");

    // Kernel supports fp16 / fp32 only. Other float types (bf16, f64) stay on
    // the decomposition path.
    auto elemType = inputType.getElementType();
    if (!elemType.isF16() && !elemType.isF32())
      return rewriter.notifyMatchFailure(op, "lpnorm.dtype_not_f16_f32");

    auto ctxOrFailure = getContextArg(op, rewriter);
    if (mlir::failed(ctxOrFailure))
      return rewriter.notifyMatchFailure(op, "lpnorm.missing_context");
    mlir::Value context = *ctxOrFailure;

    mlir::Location loc = op->getLoc();
    mlir::Value init = createEmptyTensor(rewriter, loc, outputType, x);

    auto axisI64Attr = rewriter.getI64IntegerAttr(normAxis);
    auto pI64Attr = rewriter.getI64IntegerAttr(p);

    // DPS: result type inferred from `init`. Arg order matches HipOps.td:
    // (ctx, input, output, axis, p).
    auto hipOp = mlir::hip::L2NormOp::create(rewriter, loc, context, x, init,
                                             axisI64Attr, pI64Attr);

    rewriter.replaceOp(op, hipOp->getResult(0));
    ++NumLpNormalizationFused;
    LLVM_DEBUG(llvm::dbgs()
               << "[" DEBUG_TYPE "] fused LpNormalization " << inputType
               << " axis=" << normAxis << " p=" << p << " -> hip.l2_norm\n");
    return mlir::success();
  }
};

} // namespace

void populateLpNormalizationConversionPatterns(RewritePatternSet &patterns,
                                               MLIRContext *ctx) {
  patterns.add<LpNormToHip, LpNormalizationDecompose>(ctx);
}

} // namespace hip
} // namespace mlir
