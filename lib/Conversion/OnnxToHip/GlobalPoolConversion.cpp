/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

//===----------------------------------------------------------------------===//
// onnx.GlobalAveragePool / onnx.GlobalMaxPool / onnx.GlobalLpPool
//   -> hip.global_pool {mode, p}  (native direct lowering)
//===----------------------------------------------------------------------===//
//
// All three ONNX global pool ops reduce the input across every spatial
// dimension (dims 2..r-1), keeping those dims as size-1 in the output:
//
//   Y[n, c, 0, 0, ..., 0] = REDUCE(X[n, c, *, *, ..., *])
//
// Only the reduction kernel differs (mean vs. max vs. lp-norm). N and C are
// passthrough; the spatial extents fully determine the reduction. We map
// every ONNX op onto the same `hip.global_pool` DPS op, distinguishing them
// by the `mode` attribute. The actual reduction is performed by the runtime
// custom HIP kernel (one block per `(n, c)` slice, fp accumulator).
//
// LP-pool additionally carries a `p` integer attribute (default 2) that the
// runtime reads only when `mode == LP`; it is ignored for AVG / MAX.
//
// Before:
//   %y = "onnx.GlobalAveragePool"(%x)
//          : (tensor<NxCxD_1x...xD_kxT>) -> tensor<NxCx1x...x1xT>
//   %y = "onnx.GlobalMaxPool"(%x)        : (...) -> (...)
//   %y = "onnx.GlobalLpPool"(%x) {p = 3 : i64} : (...) -> (...)
//
// After:
//   %init = tensor.empty(...) : tensor<NxCx1x...x1xT>      // dyn dims via
//   tensor.dim %y    = hip.global_pool(%ctx)
//             ins(%x   : tensor<NxCxD_1x...xD_kxT>)
//             outs(%init : tensor<NxCx1x...x1xT>)
//             {mode = 0|1|2 : i64, p = 2|<p> : i64}

// Mode encoding shared with HipToLLVMUtils.h / hipdnn_ep_runtime.h.
//   0 = AVERAGE, 1 = MAX, 2 = LP.
constexpr int64_t kGlobalPoolModeAverage = 0;
constexpr int64_t kGlobalPoolModeMax = 1;
constexpr int64_t kGlobalPoolModeLp = 2;

/// Common rank / element-type checks plus DPS rewrite. Returns failure with
/// a notification on any rejection; otherwise replaces \p op with a fresh
/// `hip.global_pool` carrying the supplied mode and p.
static mlir::LogicalResult buildHipGlobalPool(mlir::Operation *op,
                                              mlir::PatternRewriter &rewriter,
                                              int64_t mode, int64_t p,
                                              llvm::StringRef opLabel) {
  if (op->getNumOperands() != 1 || op->getNumResults() != 1)
    return rewriter.notifyMatchFailure(op, llvm::Twine(opLabel) +
                                               " expects 1 input and 1 output");

  auto ctxOrFailure = getContextArg(op, rewriter);
  if (mlir::failed(ctxOrFailure))
    return mlir::failure();
  mlir::Value context = *ctxOrFailure;

  mlir::Value input = op->getOperand(0);
  auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(input.getType());
  auto resultType =
      mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
  if (!inputType || !resultType)
    return rewriter.notifyMatchFailure(op, "expected ranked tensor types");

  int64_t rank = inputType.getRank();
  if (rank < 3 || rank != resultType.getRank())
    return rewriter.notifyMatchFailure(
        op,
        llvm::Twine(opLabel) + " needs rank >= 3 and matching in/out ranks");
  if (!mlir::isa<mlir::FloatType>(inputType.getElementType()) ||
      inputType.getElementType() != resultType.getElementType())
    return rewriter.notifyMatchFailure(
        op, llvm::Twine(opLabel) + " expects matching float element types (T)");

  mlir::Location loc = op->getLoc();

  // The output's leading two dims (N, C) mirror the input; the trailing
  // (rank-2) spatial dims are always 1 in the output (verified statically
  // when present). `createEmptyTensor` resolves dynamic N / C from `input`
  // — the static "1" dims need no runtime size.
  mlir::Value init = createEmptyTensor(rewriter, loc, resultType, input);

  auto hipOp = mlir::hip::GlobalPoolOp::create(rewriter, loc, resultType,
                                               context, input, init,
                                               /*mode=*/mode, /*p=*/p);
  rewriter.replaceOp(op, hipOp->getResult(0));
  return mlir::success();
}

struct GlobalAveragePoolToHip : public mlir::RewritePattern {
  GlobalAveragePoolToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.GlobalAveragePool", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    return buildHipGlobalPool(op, rewriter, kGlobalPoolModeAverage,
                              /*p=*/2, "GlobalAveragePool");
  }
};

struct GlobalMaxPoolToHip : public mlir::RewritePattern {
  GlobalMaxPoolToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.GlobalMaxPool", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    return buildHipGlobalPool(op, rewriter, kGlobalPoolModeMax,
                              /*p=*/2, "GlobalMaxPool");
  }
};

struct GlobalLpPoolToHip : public mlir::RewritePattern {
  GlobalLpPoolToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.GlobalLpPool", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    // ONNX `p` is an optional i64 (default 2). Negative / zero p makes the
    // LP norm undefined, so reject early — the runtime kernel never sees
    // it. Spec: `p >= 1`.
    int64_t p = 2;
    if (auto pAttr = op->getAttrOfType<mlir::IntegerAttr>("p"))
      p = pAttr.getInt();
    if (p < 1)
      return rewriter.notifyMatchFailure(
          op, "GlobalLpPool requires p >= 1 (ONNX spec)");
    return buildHipGlobalPool(op, rewriter, kGlobalPoolModeLp, p,
                              "GlobalLpPool");
  }
};

} // namespace

void populateGlobalPoolConversionPatterns(RewritePatternSet &patterns,
                                          MLIRContext *ctx) {
  patterns.add<GlobalAveragePoolToHip, GlobalMaxPoolToHip, GlobalLpPoolToHip>(
      ctx);
}

} // namespace hip
} // namespace mlir
