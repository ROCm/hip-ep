/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//
// onnx.ScatterND -> hip.scatter_nd
//
// ONNX semantics (opset 13):
//   output = data
//   for i in [0, prod(indices.shape[:-1])):
//     idx   = indices[i, :indices.shape[-1]]
//     slice = updates[i, ...]
//     reduce(output[idx, ...], slice)        per `reduction` attr
//
// Supported reductions: "none" (overwrite), "add" (atomic add).  The Kokoro
// iSTFT decoder uses "none"; "add" is provided for other models that
// commonly need it.  "mul" / "max" / "min" return failure.
//
// We follow the standard DPS convention used by the rest of the dialect:
// the output buffer starts uninitialised and the runtime wrapper copies
// `data` into `output` before applying the per-index updates.

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

// Reduction values must match the runtime's switch in scatter_nd.cpp /
// scatter_nd_kernel.hip.
static constexpr int64_t kReduceNone = 0;
static constexpr int64_t kReduceAdd = 1;

struct ScatterNdToHip : public RewritePattern {
  ScatterNdToHip(MLIRContext *ctx)
      : RewritePattern("onnx.ScatterND", /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto ctxOrFailure = getContextArg(op, rewriter);
    if (failed(ctxOrFailure))
      return failure();
    Value context = *ctxOrFailure;

    if (op->getNumOperands() != 3)
      return rewriter.notifyMatchFailure(
          op, "onnx.ScatterND needs 3 operands (data, indices, updates)");
    Value data = op->getOperand(0);
    Value indices = op->getOperand(1);
    Value updates = op->getOperand(2);
    auto dataType = dyn_cast<RankedTensorType>(data.getType());
    auto indicesType = dyn_cast<RankedTensorType>(indices.getType());
    auto updatesType = dyn_cast<RankedTensorType>(updates.getType());
    auto resultType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!dataType || !indicesType || !updatesType || !resultType)
      return rewriter.notifyMatchFailure(
          op, "onnx.ScatterND requires ranked tensor operands");

    int64_t reduction = kReduceNone;
    if (auto reductionAttr = op->getAttrOfType<StringAttr>("reduction")) {
      StringRef s = reductionAttr.getValue();
      if (s == "none")
        reduction = kReduceNone;
      else if (s == "add")
        reduction = kReduceAdd;
      else
        return rewriter.notifyMatchFailure(
            op, "onnx.ScatterND reduction must be 'none' or 'add' "
                "(mul/min/max not implemented)");
    }

    Location loc = op->getLoc();
    // DPS init aligns with `data` (output has the same shape as data).
    Value init = createEmptyTensor(rewriter, loc, resultType, data);
    auto hipOp = ScatterNdOp::create(rewriter, loc, resultType, context, data,
                                      indices, updates, init,
                                      rewriter.getI64IntegerAttr(reduction));
    rewriter.replaceOp(op, hipOp->getResult(0));
    return success();
  }
};

} // namespace

void mlir::hip::populateScatterNdConversionPatterns(RewritePatternSet &patterns,
                                                    MLIRContext *ctx) {
  patterns.add<ScatterNdToHip>(ctx);
}

} // namespace hip
} // namespace mlir
