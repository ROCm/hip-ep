/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//
// Lowers onnx.STFT (opset 17) to hip.stft.
//
// ONNX STFT signature:
//   inputs : signal (batch, signal_length, [1])
//            frame_step (scalar i64)
//            window     (frame_length,) -- optional
//            frame_length (scalar i64) -- optional
//   attrs  : onesided (default 1)
//   output : (batch, n_frames, n_freqs, 2)
//
// Kokoro's iSTFTNet decoder uses STFT with onesided=1, frame_length=20,
// frame_step=4, Hann window over an fp32 signal.  We require the
// `frame_step` and `frame_length` operands to be compile-time int64
// constants (true for Kokoro -- they come straight from onnx.Constant
// nodes).  The 3-D `(batch, signal_length, 1)` form is collapsed back to
// 2-D before passing to the runtime; the trailing unit dim only exists so
// ORT can reuse a generic tensor layout.

#include "OnnxToHipUtils.h"

#include "mlir/IR/BuiltinAttributes.h"

namespace mlir {
namespace hip {

// Local fallback declaration.  Also expected in OnnxToHipUtils.h, but
// kept inline so the build stays green if a parallel coverage commit
// reformats that header before ours lands.
void populateStftConversionPatterns(RewritePatternSet &patterns,
                                    MLIRContext *ctx);

namespace {

/// Pull a 1-D int64 constant into a SmallVector.  Mirrors the helper used
/// by Tier2ShapeConversion / SliceConversion -- accepts onnx.Constant,
/// arith.constant, bufferization.to_tensor with `hip.inline_value`, and
/// any tensor.expand/collapse/cast wrappers ORT inserts around scalar
/// shape constants.
static FailureOr<SmallVector<int64_t>> extractI64Constant(Value v) {
  if (!v)
    return failure();
  Operation *def = v.getDefiningOp();
  if (!def)
    return failure();
  StringRef name = def->getName().getStringRef();
  ElementsAttr valueAttr;
  if (name == "onnx.Constant") {
    valueAttr = def->getAttrOfType<ElementsAttr>("value");
  } else if (auto arithConst = dyn_cast<mlir::arith::ConstantOp>(def)) {
    valueAttr = dyn_cast<ElementsAttr>(arithConst.getValue());
  } else if (auto toT = dyn_cast<mlir::bufferization::ToTensorOp>(def)) {
    valueAttr = toT->getAttrOfType<ElementsAttr>("hip.inline_value");
  } else if (auto expandShape =
                 dyn_cast<mlir::tensor::ExpandShapeOp>(def)) {
    return extractI64Constant(expandShape.getSrc());
  } else if (auto collapseShape =
                 dyn_cast<mlir::tensor::CollapseShapeOp>(def)) {
    return extractI64Constant(collapseShape.getSrc());
  } else if (auto castOp = dyn_cast<mlir::tensor::CastOp>(def)) {
    return extractI64Constant(castOp.getSource());
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

static FailureOr<int64_t> extractScalarI64(Value v) {
  auto vec = extractI64Constant(v);
  if (failed(vec) || vec->empty())
    return failure();
  return vec->front();
}

/// Returns the input value with a possible trailing-unit dim collapsed away.
/// `(batch, signal_length, 1)` -> `(batch, signal_length)`.  If the input is
/// already rank-2 it is returned as-is.
static Value collapseTrailingUnit(OpBuilder &builder, Location loc,
                                  Value signal) {
  auto t = dyn_cast<RankedTensorType>(signal.getType());
  if (!t || t.getRank() != 3)
    return signal;
  if (t.getDimSize(2) != 1)
    return signal; // not the (B, T, 1) layout we know how to fold
  auto newType = RankedTensorType::get(
      {t.getDimSize(0), t.getDimSize(1)}, t.getElementType());
  // tensor.collapse_shape with reassoc {{0}, {1, 2}}.
  SmallVector<ReassociationIndices> reassoc = {{0}, {1, 2}};
  return mlir::tensor::CollapseShapeOp::create(builder, loc, newType, signal,
                                                reassoc);
}

struct StftToHip : public RewritePattern {
  StftToHip(MLIRContext *ctx)
      : RewritePattern("onnx.STFT", /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto ctxOrFailure = getContextArg(op, rewriter);
    if (failed(ctxOrFailure))
      return failure();
    Value context = *ctxOrFailure;

    if (op->getNumOperands() < 2 || op->getNumOperands() > 4)
      return rewriter.notifyMatchFailure(
          op, "onnx.STFT expects 2..4 operands "
              "(signal, frame_step [, window [, frame_length]])");

    Value signal = op->getOperand(0);
    Value frameStepV = op->getOperand(1);
    Value window;
    Value frameLengthV;
    if (op->getNumOperands() >= 3)
      window = op->getOperand(2);
    if (op->getNumOperands() >= 4)
      frameLengthV = op->getOperand(3);

    // Drop NoValue placeholders.
    auto isNoValue = [](Value v) {
      if (!v)
        return false;
      Operation *def = v.getDefiningOp();
      return def && def->getName().getStringRef() == "onnx.NoValue";
    };
    if (isNoValue(window))
      window = nullptr;
    if (isNoValue(frameLengthV))
      frameLengthV = nullptr;

    auto signalType = dyn_cast<RankedTensorType>(signal.getType());
    auto resultType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!signalType || !resultType)
      return rewriter.notifyMatchFailure(
          op, "onnx.STFT lowering requires ranked tensors");
    if (resultType.getRank() != 4)
      return rewriter.notifyMatchFailure(
          op, "onnx.STFT lowering requires a rank-4 output "
              "(batch, n_frames, n_freqs, 2)");
    if (!resultType.getElementType().isF32())
      return rewriter.notifyMatchFailure(
          op, "onnx.STFT lowering currently supports f32 output only");

    int64_t onesided = 1;
    if (auto a = op->getAttrOfType<IntegerAttr>("onesided"))
      onesided = a.getValue().getSExtValue();

    auto frameStepOr = extractScalarI64(frameStepV);
    if (failed(frameStepOr))
      return rewriter.notifyMatchFailure(
          op, "onnx.STFT requires a constant frame_step");
    int64_t frameStep = *frameStepOr;

    int64_t frameLength = 0;
    if (frameLengthV) {
      auto fl = extractScalarI64(frameLengthV);
      if (failed(fl))
        return rewriter.notifyMatchFailure(
            op, "onnx.STFT requires a constant frame_length when provided");
      frameLength = *fl;
    } else if (window) {
      auto wt = dyn_cast<RankedTensorType>(window.getType());
      if (!wt || wt.getRank() != 1 || wt.isDynamicDim(0))
        return rewriter.notifyMatchFailure(
            op, "onnx.STFT cannot infer frame_length from a non-static "
                "1-D window");
      frameLength = wt.getDimSize(0);
    } else {
      return rewriter.notifyMatchFailure(
          op, "onnx.STFT requires either window or frame_length");
    }
    if (frameStep <= 0 || frameLength <= 0)
      return rewriter.notifyMatchFailure(
          op, "onnx.STFT frame_step/frame_length must be > 0");

    Location loc = op->getLoc();

    // Collapse a (batch, signal_length, 1) signal back to 2-D.
    signal = collapseTrailingUnit(rewriter, loc, signal);
    auto signalType2D = dyn_cast<RankedTensorType>(signal.getType());
    if (!signalType2D || signalType2D.getRank() != 2)
      return rewriter.notifyMatchFailure(
          op, "onnx.STFT lowering requires a 2-D (batch, signal_length) "
              "signal after collapsing the trailing unit dim");
    if (!signalType2D.getElementType().isF32())
      return rewriter.notifyMatchFailure(
          op, "onnx.STFT lowering currently supports f32 signals only");

    // Build the empty output tensor.  All four output dims are normally
    // static for Kokoro; only batch is allowed to remain dynamic.
    SmallVector<Value> dynSizes;
    for (int64_t d = 0; d < resultType.getRank(); ++d) {
      if (!resultType.isDynamicDim(d))
        continue;
      if (d == 0) {
        dynSizes.push_back(
            mlir::tensor::DimOp::create(rewriter, loc, signal, 0));
      } else {
        return rewriter.notifyMatchFailure(
            op, "onnx.STFT lowering only supports a dynamic batch dim "
                "(other dims must be static)");
      }
    }
    Value init = mlir::tensor::EmptyOp::create(
        rewriter, loc, resultType.getShape(), resultType.getElementType(),
        dynSizes);

    auto hipOp = StftOp::create(
        rewriter, loc, resultType, context, signal, window, init,
        rewriter.getI64IntegerAttr(frameStep),
        rewriter.getI64IntegerAttr(frameLength),
        rewriter.getI64IntegerAttr(onesided));
    rewriter.replaceOp(op, hipOp->getResult(0));
    return success();
  }
};

} // namespace

void mlir::hip::populateStftConversionPatterns(RewritePatternSet &patterns,
                                               MLIRContext *ctx) {
  patterns.add<StftToHip>(ctx);
}

} // namespace hip
} // namespace mlir
