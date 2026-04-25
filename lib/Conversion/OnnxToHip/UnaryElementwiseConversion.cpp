/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//
// Lowers ONNX unary element-wise ops to a single hip.unary_elementwise op
// with a numeric `kind` discriminator.  alpha/beta carry op-specific scalar
// attributes (LeakyRelu negative slope; Clip min/max).
//
// Supported ONNX ops (Tier 1 in docs/kokoro_tts_plan.md):
//   onnx.Sin, onnx.Cos, onnx.Exp, onnx.Tanh, onnx.Floor, onnx.Round,
//   onnx.Atan, onnx.LeakyRelu, onnx.Clip
//
// Kind values must match hip_unary_kind_t in
// 3rd-party/custom_kernels/include/hip_custom_kernels.h.

#include "OnnxToHipUtils.h"

#include "mlir/IR/BuiltinAttributes.h"

#include <limits>

namespace mlir {
namespace hip {
namespace {

// kind values match hip_unary_kind_t
static constexpr int64_t kKindSin = 0;
static constexpr int64_t kKindCos = 1;
static constexpr int64_t kKindExp = 2;
static constexpr int64_t kKindTanh = 3;
static constexpr int64_t kKindFloor = 4;
static constexpr int64_t kKindRound = 5;
static constexpr int64_t kKindAtan = 6;
static constexpr int64_t kKindLeakyRelu = 7;
static constexpr int64_t kKindClip = 8;

/// Build a hip.unary_elementwise op from a 1-input ONNX op.
static LogicalResult buildUnary(Operation *op, PatternRewriter &rewriter,
                                int64_t kind, float alpha, float beta) {
  auto ctxOrFailure = getContextArg(op, rewriter);
  if (failed(ctxOrFailure))
    return failure();
  Value context = *ctxOrFailure;

  Location loc = op->getLoc();
  Value input = op->getOperand(0);
  if (!isa<RankedTensorType>(input.getType()))
    return rewriter.notifyMatchFailure(
        op, "hip.unary_elementwise lowering needs a ranked tensor input");
  if (!isa<RankedTensorType>(op->getResult(0).getType()))
    return rewriter.notifyMatchFailure(
        op, "hip.unary_elementwise lowering needs a ranked tensor result");

  auto resultType = cast<RankedTensorType>(op->getResult(0).getType());
  Value init = createEmptyTensor(rewriter, loc, resultType, input);

  auto hipOp = UnaryElementwiseOp::create(
      rewriter, loc, resultType, context, input, init,
      rewriter.getI64IntegerAttr(kind),
      rewriter.getF32FloatAttr(alpha), rewriter.getF32FloatAttr(beta));
  rewriter.replaceOp(op, hipOp->getResult(0));
  return success();
}

template <int64_t Kind>
struct SimpleUnaryToHip : public RewritePattern {
  SimpleUnaryToHip(MLIRContext *ctx, StringRef opName)
      : RewritePattern(opName, /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    return buildUnary(op, rewriter, Kind, 0.0f, 0.0f);
  }
};

struct LeakyReluToHip : public RewritePattern {
  LeakyReluToHip(MLIRContext *ctx)
      : RewritePattern("onnx.LeakyRelu", /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    float alpha = 0.01f; // ONNX LeakyRelu default
    if (auto attr = op->getAttrOfType<FloatAttr>("alpha"))
      alpha = static_cast<float>(attr.getValueAsDouble());
    return buildUnary(op, rewriter, kKindLeakyRelu, alpha, 0.0f);
  }
};

struct ClipToHip : public RewritePattern {
  ClipToHip(MLIRContext *ctx)
      : RewritePattern("onnx.Clip", /*benefit=*/1, ctx) {}

  /// Try to extract a scalar f32 from an ONNX Clip min/max operand.  Both
  /// inputs are optional; missing inputs (`onnx.NoValue`) keep the default.
  /// Accepts both onnx.Constant and the arith.constant emitted by
  /// lowerOnnxConstants in the same conversion pass.
  static FailureOr<float> extractScalar(Value v, float defaultVal) {
    if (!v)
      return defaultVal;
    Operation *def = v.getDefiningOp();
    if (!def)
      return failure();
    StringRef name = def->getName().getStringRef();
    if (name == "onnx.NoValue")
      return defaultVal;
    ElementsAttr valueAttr;
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
    if (elem.isF32())
      return *dense.value_begin<float>();
    if (elem.isF16() || elem.isBF16()) {
      // Take the first APFloat and convert to f32.
      auto first = *dense.value_begin<APFloat>();
      bool losesInfo = false;
      first.convert(APFloat::IEEEsingle(), APFloat::rmNearestTiesToEven,
                    &losesInfo);
      return first.convertToFloat();
    }
    return failure();
  }

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumOperands() < 1 || op->getNumOperands() > 3)
      return rewriter.notifyMatchFailure(op,
                                         "onnx.Clip expects 1 to 3 operands");

    float minVal = -std::numeric_limits<float>::infinity();
    float maxVal = std::numeric_limits<float>::infinity();
    if (op->getNumOperands() >= 2) {
      auto minOr = extractScalar(op->getOperand(1), minVal);
      if (failed(minOr))
        return rewriter.notifyMatchFailure(
            op, "onnx.Clip min input must be a scalar f32 constant");
      minVal = *minOr;
    }
    if (op->getNumOperands() >= 3) {
      auto maxOr = extractScalar(op->getOperand(2), maxVal);
      if (failed(maxOr))
        return rewriter.notifyMatchFailure(
            op, "onnx.Clip max input must be a scalar f32 constant");
      maxVal = *maxOr;
    }
    return buildUnary(op, rewriter, kKindClip, minVal, maxVal);
  }
};

} // namespace

void mlir::hip::populateUnaryElementwiseConversionPatterns(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.add<SimpleUnaryToHip<kKindSin>>(ctx, "onnx.Sin");
  patterns.add<SimpleUnaryToHip<kKindCos>>(ctx, "onnx.Cos");
  patterns.add<SimpleUnaryToHip<kKindExp>>(ctx, "onnx.Exp");
  patterns.add<SimpleUnaryToHip<kKindTanh>>(ctx, "onnx.Tanh");
  patterns.add<SimpleUnaryToHip<kKindFloor>>(ctx, "onnx.Floor");
  patterns.add<SimpleUnaryToHip<kKindRound>>(ctx, "onnx.Round");
  patterns.add<SimpleUnaryToHip<kKindAtan>>(ctx, "onnx.Atan");
  patterns.add<LeakyReluToHip>(ctx);
  patterns.add<ClipToHip>(ctx);
}

} // namespace hip
} // namespace mlir
