/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//
// Conversion patterns for Tier 5 sequence/recurrent ops:
//
//   - onnx.CumSum -> hip.cumsum
//
// LSTM and STFT live in their own conversion files (TODO).

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

/// Pull a scalar i64 constant from an onnx.Constant or arith.constant.
static FailureOr<int64_t> extractScalarI64(Value v) {
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
  if (!dense || dense.getNumElements() != 1)
    return failure();
  Type elem = dense.getElementType();
  if (elem.isInteger(64))
    return *dense.value_begin<int64_t>();
  if (elem.isInteger(32))
    return static_cast<int64_t>(*dense.value_begin<int32_t>());
  return failure();
}

struct CumSumToHip : public RewritePattern {
  CumSumToHip(MLIRContext *ctx)
      : RewritePattern("onnx.CumSum", /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto ctxOrFailure = getContextArg(op, rewriter);
    if (failed(ctxOrFailure))
      return failure();
    Value context = *ctxOrFailure;

    if (op->getNumOperands() != 2)
      return rewriter.notifyMatchFailure(
          op, "onnx.CumSum needs 2 operands (input, axis)");
    Value input = op->getOperand(0);
    auto inType = dyn_cast<RankedTensorType>(input.getType());
    auto resultType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!inType || !resultType || !inType.hasStaticShape() ||
        !resultType.hasStaticShape())
      return rewriter.notifyMatchFailure(
          op, "onnx.CumSum lowering requires static shapes");

    auto axisOr = extractScalarI64(op->getOperand(1));
    if (failed(axisOr))
      return rewriter.notifyMatchFailure(
          op, "onnx.CumSum axis input must be a scalar int constant");
    int64_t axis = *axisOr;
    int64_t rank = inType.getRank();
    if (axis < 0)
      axis += rank;
    if (axis < 0 || axis >= rank)
      return rewriter.notifyMatchFailure(op, "onnx.CumSum axis out of range");

    int64_t exclusive = 0, reverse = 0;
    if (auto e = op->getAttrOfType<IntegerAttr>("exclusive"))
      exclusive = e.getValue().getSExtValue();
    if (auto r = op->getAttrOfType<IntegerAttr>("reverse"))
      reverse = r.getValue().getSExtValue();

    Location loc = op->getLoc();
    Value init = createEmptyTensor(rewriter, loc, resultType, input);
    auto hipOp = CumSumOp::create(
        rewriter, loc, resultType, context, input, init,
        rewriter.getI64IntegerAttr(axis),
        rewriter.getI64IntegerAttr(exclusive),
        rewriter.getI64IntegerAttr(reverse));
    rewriter.replaceOp(op, hipOp->getResult(0));
    return success();
  }
};

} // namespace

void mlir::hip::populateTier5SeqConversionPatterns(RewritePatternSet &patterns,
                                                   MLIRContext *ctx) {
  patterns.add<CumSumToHip>(ctx);
}

} // namespace hip
} // namespace mlir
