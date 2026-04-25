/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//
// Lowers onnx.Slice (opset 10/13) to hip.slice.  Negative indices, omitted
// axes, and explicit `step != 1` are all normalised at conversion time so
// the runtime kernel only sees a uniform per-axis (start, step) pair plus
// the input strides (carried by the lowering pass).

#include "OnnxToHipUtils.h"

#include "mlir/IR/BuiltinAttributes.h"

namespace mlir {
namespace hip {
namespace {

/// Pull a 1-D int64 constant into a SmallVector.  Accepts both onnx.Constant
/// (when running in the pre-lower pass before lowerOnnxConstants has folded
/// constants) and arith.constant (in case the pattern fires later).
static FailureOr<SmallVector<int64_t>> extractI64Constant(Value v) {
  if (!v)
    return failure();
  Operation *def = v.getDefiningOp();
  if (!def)
    return failure();
  ElementsAttr valueAttr;
  StringRef name = def->getName().getStringRef();
  if (name == "onnx.Constant") {
    valueAttr = def->getAttrOfType<ElementsAttr>("value");
  } else if (auto arithConst = dyn_cast<mlir::arith::ConstantOp>(def)) {
    valueAttr = dyn_cast<ElementsAttr>(arithConst.getValue());
  } else {
    return failure();
  }
  if (!valueAttr)
    return failure();
  auto dense = dyn_cast<DenseElementsAttr>(valueAttr);
  if (!dense)
    return failure();
  Type elem = dense.getElementType();
  SmallVector<int64_t> out;
  if (elem.isInteger(64)) {
    for (int64_t v : dense.getValues<int64_t>())
      out.push_back(v);
  } else if (elem.isInteger(32)) {
    for (int32_t v : dense.getValues<int32_t>())
      out.push_back(static_cast<int64_t>(v));
  } else {
    return failure();
  }
  return out;
}

struct SliceToHip : public RewritePattern {
  SliceToHip(MLIRContext *ctx)
      : RewritePattern("onnx.Slice", /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto ctxOrFailure = getContextArg(op, rewriter);
    if (failed(ctxOrFailure))
      return failure();
    Value context = *ctxOrFailure;

    Location loc = op->getLoc();
    if (op->getNumOperands() < 3 || op->getNumOperands() > 5)
      return rewriter.notifyMatchFailure(
          op, "onnx.Slice expects 3..5 operands (data, starts, ends [, axes [, steps]])");

    Value input = op->getOperand(0);
    auto inputType = dyn_cast<RankedTensorType>(input.getType());
    if (!inputType || !inputType.hasStaticShape())
      return rewriter.notifyMatchFailure(
          op, "onnx.Slice lowering requires a static-shape input");

    auto resultType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!resultType || !resultType.hasStaticShape())
      return rewriter.notifyMatchFailure(
          op, "onnx.Slice lowering requires a static-shape output");

    auto startsOr = extractI64Constant(op->getOperand(1));
    auto endsOr = extractI64Constant(op->getOperand(2));
    if (failed(startsOr) || failed(endsOr))
      return rewriter.notifyMatchFailure(
          op, "onnx.Slice starts/ends must be onnx.Constant int64 tensors");
    SmallVector<int64_t> starts = *startsOr;
    SmallVector<int64_t> ends = *endsOr;

    SmallVector<int64_t> axes;
    if (op->getNumOperands() >= 4) {
      auto axesOr = extractI64Constant(op->getOperand(3));
      if (failed(axesOr))
        return rewriter.notifyMatchFailure(
            op, "onnx.Slice axes must be an onnx.Constant int64 tensor");
      axes = *axesOr;
    } else {
      for (int64_t i = 0; i < (int64_t)starts.size(); ++i)
        axes.push_back(i);
    }

    SmallVector<int64_t> steps;
    if (op->getNumOperands() >= 5) {
      auto stepsOr = extractI64Constant(op->getOperand(4));
      if (failed(stepsOr))
        return rewriter.notifyMatchFailure(
            op, "onnx.Slice steps must be an onnx.Constant int64 tensor");
      steps = *stepsOr;
    } else {
      steps.assign(starts.size(), 1);
    }

    if (starts.size() != ends.size() || starts.size() != axes.size() ||
        starts.size() != steps.size())
      return rewriter.notifyMatchFailure(
          op, "onnx.Slice starts/ends/axes/steps lengths must match");

    int64_t inputRank = inputType.getRank();
    SmallVector<int64_t> normStarts(inputRank, 0);
    SmallVector<int64_t> normSteps(inputRank, 1);
    for (size_t i = 0; i < axes.size(); ++i) {
      int64_t ax = axes[i];
      if (ax < 0)
        ax += inputRank;
      if (ax < 0 || ax >= inputRank)
        return rewriter.notifyMatchFailure(op, "onnx.Slice axis out of range");
      int64_t dimSize = inputType.getDimSize(ax);
      int64_t s = starts[i];
      int64_t step = steps[i];
      if (step == 0)
        return rewriter.notifyMatchFailure(op, "onnx.Slice step must be != 0");
      // Per spec:  for step > 0, clamp start to [0, dim], end to [0, dim].
      // Negative indices count from the end.
      if (s < 0)
        s += dimSize;
      if (step > 0)
        s = std::clamp<int64_t>(s, 0, dimSize);
      else
        s = std::clamp<int64_t>(s, 0, dimSize - 1);
      normStarts[ax] = s;
      normSteps[ax] = step;
    }

    Value init = createEmptyTensor(rewriter, loc, resultType, input);
    auto hipOp = SliceOp::create(rewriter, loc, resultType, context, input,
                                  init, rewriter.getI64ArrayAttr(normStarts),
                                  rewriter.getI64ArrayAttr(normSteps));
    rewriter.replaceOp(op, hipOp->getResult(0));
    return success();
  }
};

} // namespace

void mlir::hip::populateSliceConversionPatterns(RewritePatternSet &patterns,
                                                 MLIRContext *ctx) {
  patterns.add<SliceToHip>(ctx);
}

} // namespace hip
} // namespace mlir
