/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

static LogicalResult extractConstI64Vector(Value splitOperand,
                                           SmallVectorImpl<int64_t> &values) {
  if (!splitOperand)
    return failure();

  Operation *defOp = splitOperand.getDefiningOp();
  if (!defOp)
    return failure();

  DenseIntElementsAttr denseAttr;
  if (auto arithConst = dyn_cast<arith::ConstantOp>(defOp)) {
    denseAttr = dyn_cast<DenseIntElementsAttr>(arithConst.getValue());
  } else {
    auto valueAttr = defOp->getAttrOfType<ElementsAttr>("value");
    denseAttr = dyn_cast_or_null<DenseIntElementsAttr>(valueAttr);
  }
  if (!denseAttr)
    return failure();

  for (const APInt &v : denseAttr.getValues<APInt>())
    values.push_back(v.getSExtValue());
  return success();
}

static Value createSplitInitTensor(PatternRewriter &rewriter, Location loc,
                                   Value data, RankedTensorType outputType,
                                   int64_t axis, int64_t splitSize) {
  SmallVector<Value> dynSizes;
  for (int64_t i : llvm::seq<int64_t>(outputType.getRank())) {
    if (!outputType.isDynamicDim(i))
      continue;
    if (i == axis) {
      dynSizes.push_back(
          arith::ConstantIndexOp::create(rewriter, loc, splitSize));
    } else {
      dynSizes.push_back(tensor::DimOp::create(rewriter, loc, data, i));
    }
  }
  return tensor::EmptyOp::create(rewriter, loc, outputType.getShape(),
                                 outputType.getElementType(), dynSizes);
}

/// onnx.Split -> one hip.split op per output chunk.
/// Current support requires compile-time split sizes (either from static output
/// types or a constant split-lengths operand).
struct SplitToHip : public RewritePattern {
  SplitToHip(MLIRContext *ctx) : RewritePattern("onnx.Split", 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto ctxOrFailure = getContextArg(op, rewriter);
    if (failed(ctxOrFailure))
      return rewriter.notifyMatchFailure(op, "missing context argument");
    Value context = *ctxOrFailure;

    Value data = op->getOperand(0);
    auto inputType = dyn_cast<RankedTensorType>(data.getType());
    if (!inputType)
      return rewriter.notifyMatchFailure(op, "input must be ranked tensor");

    int64_t axis = 0;
    if (auto axisAttr = op->getAttrOfType<IntegerAttr>("axis"))
      axis = axisAttr.getSInt();
    if (axis < 0)
      axis += inputType.getRank();
    if (axis < 0 || axis >= inputType.getRank())
      return rewriter.notifyMatchFailure(op, "axis out of range");

    const int64_t numOutputs = static_cast<int64_t>(op->getNumResults());
    SmallVector<int64_t> splitSizes;

    bool hasSplitOperand = false;
    if (op->getNumOperands() > 1) {
      Value splitOperand = op->getOperand(1);
      if (splitOperand.getDefiningOp() &&
          splitOperand.getDefiningOp()->getName().getStringRef() !=
              "onnx.NoValue") {
        hasSplitOperand = true;
        if (failed(extractConstI64Vector(splitOperand, splitSizes))) {
          return rewriter.notifyMatchFailure(
              op, "split lengths must be compile-time constant");
        }
      }
    }

    if (hasSplitOperand &&
        static_cast<int64_t>(splitSizes.size()) != numOutputs)
      return rewriter.notifyMatchFailure(
          op, "split lengths count must equal number of outputs");

    if (!hasSplitOperand) {
      splitSizes.reserve(numOutputs);
      int64_t staticAxisDim = inputType.getDimSize(axis);
      if (staticAxisDim == ShapedType::kDynamic)
        return rewriter.notifyMatchFailure(
            op, "equal split requires static input axis dimension");
      int64_t base = staticAxisDim / numOutputs;
      int64_t rem = staticAxisDim % numOutputs;
      for (int64_t i = 0; i < numOutputs; ++i)
        splitSizes.push_back(base + (i < rem ? 1 : 0));
    }

    if (inputType.getDimSize(axis) != ShapedType::kDynamic) {
      int64_t expected = inputType.getDimSize(axis);
      int64_t actual = 0;
      for (int64_t v : splitSizes)
        actual += v;
      if (actual != expected) {
        return rewriter.notifyMatchFailure(
            op, "split lengths sum does not match input axis dimension");
      }
    }

    SmallVector<Value> newResults;
    newResults.reserve(op->getNumResults());
    Location loc = op->getLoc();

    int64_t currentOffset = 0;
    for (int64_t i = 0; i < numOutputs; ++i) {
      int64_t splitSize = splitSizes[i];
      if (splitSize < 0)
        return rewriter.notifyMatchFailure(op,
                                           "split size must be non-negative");

      auto resultType = dyn_cast<RankedTensorType>(op->getResult(i).getType());
      if (!resultType)
        return rewriter.notifyMatchFailure(op, "result must be ranked tensor");

      int64_t resultAxisDim = resultType.getDimSize(axis);
      if (resultAxisDim != ShapedType::kDynamic && resultAxisDim != splitSize) {
        return rewriter.notifyMatchFailure(
            op, "result axis dimension mismatch with split size");
      }

      Value init = createSplitInitTensor(rewriter, loc, data, resultType, axis,
                                         splitSize);
      auto splitOp =
          hip::SplitOp::create(rewriter, loc, resultType, context, data, init,
                               rewriter.getI64IntegerAttr(axis),
                               rewriter.getI64IntegerAttr(currentOffset));
      newResults.push_back(splitOp->getResult(0));
      currentOffset += splitSize;
    }

    rewriter.replaceOp(op, newResults);
    return success();
  }
};

} // namespace

void mlir::hip::populateSplitConversionPatterns(RewritePatternSet &patterns,
                                                MLIRContext *ctx) {
  patterns.add<SplitToHip>(ctx);
}

} // namespace hip
} // namespace mlir
