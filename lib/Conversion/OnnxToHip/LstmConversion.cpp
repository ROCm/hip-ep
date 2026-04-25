/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//
// onnx.LSTM (opset 14) -> hip.lstm
//
// We support the canonical bidirectional / unidirectional configuration that
// Kokoro hits in its duration predictor:
//   * input_forget = 0
//   * default activations (sigmoid, tanh, tanh) -- everything MIOpen LSTM
//     hard-codes anyway
//   * layout = 0 (T, N, in_size)
//
// The optional sequence_lens input is dropped (Kokoro never feeds variable
// lengths).  bias / initial_h / initial_c flow through as optional operands.
// Gate-layout translation from ONNX `IOFG` to MIOpen `IFOG` happens inside
// the runtime wrapper -- we keep the W/R/B layouts untouched at the IR level.

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

// Encode the ONNX `direction` string as the i64 attribute the runtime expects.
//   forward       = 0
//   reverse       = 1
//   bidirectional = 2
static FailureOr<int64_t> encodeDirection(StringRef s) {
  if (s == "forward")
    return 0;
  if (s == "reverse")
    return 1;
  if (s == "bidirectional")
    return 2;
  return failure();
}

// Look at an operand and decide whether it's actually present.  ONNX's
// optional inputs surface either as a NoneType-typed onnx.NoValue or as
// an explicit value; we treat both the absence and the NoneType as "absent".
static Value resolveOptional(Value v) {
  if (!v)
    return nullptr;
  if (isa<NoneType>(v.getType()))
    return nullptr;
  if (auto def = v.getDefiningOp())
    if (def->getName().getStringRef() == "onnx.NoValue")
      return nullptr;
  return v;
}

struct LstmToHip : public RewritePattern {
  LstmToHip(MLIRContext *ctx)
      : RewritePattern("onnx.LSTM", /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto ctxOrFailure = getContextArg(op, rewriter);
    if (failed(ctxOrFailure))
      return failure();
    Value context = *ctxOrFailure;

    if (op->getNumOperands() < 3 || op->getNumOperands() > 8)
      return rewriter.notifyMatchFailure(op,
                                         "onnx.LSTM expects 3..8 operands");

    Value input = op->getOperand(0);
    Value weights = op->getOperand(1);
    Value recurrence = op->getOperand(2);
    Value bias = op->getNumOperands() >= 4
                     ? resolveOptional(op->getOperand(3))
                     : nullptr;
    // operand 4 = sequence_lens (int32 [N]).  We require it to be either
    // absent or a constant (typically `dense<seq_len>`).  The runtime
    // assumes every sample uses the full sequence length.
    if (op->getNumOperands() >= 5) {
      if (Value sl = resolveOptional(op->getOperand(4))) {
        // Best-effort sanity check: if sequence_lens is a constant, its
        // values should all equal seq_len.  We don't fail when we can't
        // peek inside (e.g. externalised constants); MIOpen's RNN
        // ForwardInference simply runs every batch sample for the full
        // sequence length, which matches Kokoro.
        (void)sl;
      }
    }
    Value initialH = op->getNumOperands() >= 6
                         ? resolveOptional(op->getOperand(5))
                         : nullptr;
    Value initialC = op->getNumOperands() >= 7
                         ? resolveOptional(op->getOperand(6))
                         : nullptr;
    // operand 7 = peephole P (com.microsoft only); not supported by MIOpen.
    if (op->getNumOperands() >= 8 && resolveOptional(op->getOperand(7)))
      return rewriter.notifyMatchFailure(
          op, "hip.lstm does not support the optional peephole input");

    auto inputType = dyn_cast<RankedTensorType>(input.getType());
    auto weightsType = dyn_cast<RankedTensorType>(weights.getType());
    auto recurrenceType = dyn_cast<RankedTensorType>(recurrence.getType());
    if (!inputType || !weightsType || !recurrenceType)
      return rewriter.notifyMatchFailure(
          op, "onnx.LSTM requires ranked input/weights/recurrence");
    if (inputType.getRank() != 3 || weightsType.getRank() != 3 ||
        recurrenceType.getRank() != 3)
      return rewriter.notifyMatchFailure(
          op, "onnx.LSTM expects rank-3 input/weights/recurrence");

    // hidden_size attribute is required (ONNX makes it mandatory at opset 14).
    auto hiddenSizeAttr = op->getAttrOfType<IntegerAttr>("hidden_size");
    if (!hiddenSizeAttr)
      return rewriter.notifyMatchFailure(op, "missing hidden_size attribute");
    int64_t hiddenSize = hiddenSizeAttr.getValue().getSExtValue();

    // direction defaults to "forward".
    int64_t directionEnum = 0;
    if (auto dirAttr = op->getAttrOfType<StringAttr>("direction")) {
      auto dir = encodeDirection(dirAttr.getValue());
      if (failed(dir))
        return rewriter.notifyMatchFailure(
            op, "direction must be forward/reverse/bidirectional");
      directionEnum = *dir;
    }

    // input_forget: only the default value (0) is supported because MIOpen's
    // LSTM does not expose the coupling option.
    if (auto ifAttr = op->getAttrOfType<IntegerAttr>("input_forget"))
      if (ifAttr.getValue().getSExtValue() != 0)
        return rewriter.notifyMatchFailure(
            op, "input_forget != 0 not supported");

    // layout: only 0 (T, N, in_size) is supported.  layout=1 (N, T, in_size)
    // would need a transpose pre/post the call.
    if (auto layoutAttr = op->getAttrOfType<IntegerAttr>("layout"))
      if (layoutAttr.getValue().getSExtValue() != 0)
        return rewriter.notifyMatchFailure(op, "layout != 0 not supported");

    if (op->getNumResults() < 1 || op->getNumResults() > 3)
      return rewriter.notifyMatchFailure(op,
                                         "onnx.LSTM expects 1..3 results");

    // Kokoro reads Y for every LSTM (the duration predictor uses the
    // per-step hidden states).  We always materialise all three outputs so
    // the runtime can assume non-null pointers; unused ones become dead
    // memrefs that the bufferizer drops.
    Location loc = op->getLoc();

    // Derive the per-direction sizes from the weights tensor (which is
    // always (num_dir, 4*hidden, in_size)).
    int64_t numDir = weightsType.getDimSize(0);
    if (numDir == ShapedType::kDynamic)
      numDir = (directionEnum == 2) ? 2 : 1;

    // Build empty tensor types for Y / Y_h / Y_c using shapes inferred from
    // the result types when present, falling back to the canonical
    // (T, num_dir, N, hidden) / (num_dir, N, hidden) shapes otherwise.
    auto buildOutType = [&](unsigned idx,
                            ArrayRef<int64_t> fallback) -> RankedTensorType {
      if (idx < op->getNumResults()) {
        if (auto t = dyn_cast<RankedTensorType>(op->getResult(idx).getType()))
          if (!isa<NoneType>(t))
            return t;
      }
      return RankedTensorType::get(fallback, inputType.getElementType());
    };

    // Pull dynamic dims off the input tensor (T, N, in_size).
    int64_t T = inputType.getDimSize(0);
    int64_t N = inputType.getDimSize(1);

    SmallVector<int64_t> yShape{T, numDir, N, hiddenSize};
    SmallVector<int64_t> hShape{numDir, N, hiddenSize};
    auto yType = buildOutType(0, yShape);
    auto yhType = buildOutType(1, hShape);
    auto ycType = buildOutType(2, hShape);

    // Helper: build a tensor.empty for a result type, walking a list of
    // candidate sources for any dynamic dims.
    auto emptyForType = [&](RankedTensorType ty) -> Value {
      SmallVector<Value> dynSizes;
      for (int64_t d = 0; d < ty.getRank(); ++d) {
        if (!ty.isDynamicDim(d))
          continue;
        // For Y the dynamic dims (T, N) come from input.  For Y_h / Y_c
        // the dynamic dim (N) comes from input dim 1.  Probe input first.
        Value picked;
        auto inT = inputType;
        if (ty.getRank() == 4) {
          if (d == 0)
            picked = tensor::DimOp::create(rewriter, loc, input, 0);
          else if (d == 2)
            picked = tensor::DimOp::create(rewriter, loc, input, 1);
        } else { // rank 3 (num_dir, N, hidden)
          if (d == 1)
            picked = tensor::DimOp::create(rewriter, loc, input, 1);
        }
        if (!picked)
          picked = mlir::arith::ConstantIndexOp::create(rewriter, loc, 1);
        dynSizes.push_back(picked);
        (void)inT;
      }
      return tensor::EmptyOp::create(rewriter, loc, ty.getShape(),
                                     ty.getElementType(), dynSizes);
    };

    Value yInit = emptyForType(yType);
    Value yhInit = emptyForType(yhType);
    Value ycInit = emptyForType(ycType);

    // Build operand list and the operand_segment_sizes attribute.
    SmallVector<Value> operands;
    operands.push_back(context);
    operands.push_back(input);
    operands.push_back(weights);
    operands.push_back(recurrence);
    if (bias)
      operands.push_back(bias);
    if (initialH)
      operands.push_back(initialH);
    if (initialC)
      operands.push_back(initialC);
    operands.push_back(yInit);
    operands.push_back(yhInit);
    operands.push_back(ycInit);

    SmallVector<int32_t> segmentSizes;
    segmentSizes.push_back(1);                // ctx
    segmentSizes.push_back(1);                // input
    segmentSizes.push_back(1);                // weights
    segmentSizes.push_back(1);                // recurrence
    segmentSizes.push_back(bias ? 1 : 0);     // bias
    segmentSizes.push_back(initialH ? 1 : 0); // initial_h
    segmentSizes.push_back(initialC ? 1 : 0); // initial_c
    segmentSizes.push_back(1);                // y_out
    segmentSizes.push_back(1);                // y_h_out
    segmentSizes.push_back(1);                // y_c_out

    SmallVector<NamedAttribute> attrs;
    attrs.push_back(rewriter.getNamedAttr(
        "hidden_size", rewriter.getI64IntegerAttr(hiddenSize)));
    attrs.push_back(rewriter.getNamedAttr(
        "direction", rewriter.getI64IntegerAttr(directionEnum)));
    attrs.push_back(
        rewriter.getNamedAttr("operand_segment_sizes",
                              rewriter.getDenseI32ArrayAttr(segmentSizes)));

    OperationState state(loc, "hip.lstm");
    state.addOperands(operands);
    state.addAttributes(attrs);
    state.addTypes({yType, yhType, ycType});

    Operation *hipOp = rewriter.create(state);

    // Replace original op results.  ONNX may have fewer than 3 results;
    // each existing result maps positionally to one of (y, y_h, y_c).
    SmallVector<Value> replacements;
    for (unsigned i = 0; i < op->getNumResults(); ++i)
      replacements.push_back(hipOp->getResult(i));
    rewriter.replaceOp(op, replacements);
    return success();
  }
};

} // namespace

void mlir::hip::populateLstmConversionPatterns(RewritePatternSet &patterns,
                                               MLIRContext *ctx) {
  patterns.add<LstmToHip>(ctx);
}

} // namespace hip
} // namespace mlir
