/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/IR/HipDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace mlir::hip;

namespace {

// Pool modes; the values mirror HIPDNN_EP_POOL_* in hipdnn_ep_runtime.h.
constexpr uint64_t kPoolAverage = 0;

// An average pool whose one window covers the whole input is the mean over
// the spatial dims.
//
// Before:
//   %y = hip.pool(%ctx) ins(%x : tensor<?x96x56x56xf16>)
//          outs(%i : tensor<?x96x1x1xf16>)
//          {pool_mode = 0, kernel_shape = [56, 56], strides = [56, 56],
//           pads = [0, 0, 0, 0], dilations = [1, 1], ...}
// After:
//   %axes = arith.constant dense<[2, 3]> : tensor<2xi64>
//   %y = hip.reduce_mean(%ctx) ins(%x, %axes : ...)
//          outs(%i : tensor<?x96x1x1xf16>)
//          {keepdims = 1, reduced_axes = array<i64: 2, 3>}
//
// The window covers the input when, on every spatial axis, the input extent
// is static, the kernel is at least that extent, and there is no padding or
// dilation; the output extent is then 1 whatever the stride or ceil_mode.
// Without padding the divisor of either count_include_pad mode is the number
// of input elements, so the result is the mean, also for a kernel larger
// than the input. The reduction is a single kernel over contiguous spatial
// elements instead of a sliding-window pass.
struct WholeInputAvgPoolToReduceMean : OpRewritePattern<PoolOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(PoolOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getPoolMode() != kPoolAverage || op->getNumResults() != 1 ||
        op.getOutputs().size() != 1)
      return rewriter.notifyMatchFailure(op, "not a tensor average pool");
    auto inputType = dyn_cast<RankedTensorType>(op.getInput().getType());
    auto resultType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!inputType || !resultType || inputType.getRank() < 3 ||
        resultType.getRank() != inputType.getRank())
      return rewriter.notifyMatchFailure(op, "expected ranked NC... tensors");
    // hip.reduce_mean computes in f16 and f32 only.
    Type elementType = inputType.getElementType();
    if (!elementType.isF16() && !elementType.isF32())
      return rewriter.notifyMatchFailure(op, "reduce_mean element type");

    const int64_t rank = inputType.getRank();
    const int64_t spatialRank = rank - 2;
    ArrayAttr kernel = op.getKernelShape();
    ArrayAttr dilations = op.getDilations();
    ArrayAttr pads = op.getPads();
    if (static_cast<int64_t>(kernel.size()) != spatialRank ||
        static_cast<int64_t>(dilations.size()) != spatialRank ||
        static_cast<int64_t>(pads.size()) != 2 * spatialRank)
      return rewriter.notifyMatchFailure(op, "attribute arity");
    auto value = [](Attribute attr) {
      return cast<IntegerAttr>(attr).getInt();
    };
    if (llvm::any_of(pads, [&](Attribute pad) { return value(pad) != 0; }))
      return rewriter.notifyMatchFailure(op, "padded");
    SmallVector<int64_t> axes;
    for (int64_t i : llvm::seq<int64_t>(spatialRank)) {
      const int64_t axis = 2 + i;
      if (inputType.isDynamicDim(axis) ||
          value(kernel[i]) < inputType.getDimSize(axis) ||
          value(dilations[i]) != 1 || resultType.getDimSize(axis) != 1)
        return rewriter.notifyMatchFailure(op, "window is not the whole input");
      axes.push_back(axis);
    }

    auto axesType = RankedTensorType::get({spatialRank}, rewriter.getI64Type());
    Value axesValue = arith::ConstantOp::create(
        rewriter, op.getLoc(), axesType,
        DenseIntElementsAttr::get(axesType, ArrayRef<int64_t>(axes)));
    auto mean = ReduceMeanOp::create(
        rewriter, op.getLoc(), op.getCtx(), op.getInput(), axesValue,
        op.getOutputs().front(), rewriter.getI64IntegerAttr(1),
        rewriter.getI64IntegerAttr(0), rewriter.getDenseI64ArrayAttr(axes));
    rewriter.replaceOp(op, mean->getResults());
    return success();
  }
};

} // namespace

void PoolOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                         MLIRContext *context) {
  patterns.add<WholeInputAvgPoolToReduceMean>(context);
}
