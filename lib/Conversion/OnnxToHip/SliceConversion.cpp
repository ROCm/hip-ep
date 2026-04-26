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
  } else if (auto toT = dyn_cast<mlir::bufferization::ToTensorOp>(def)) {
    valueAttr = toT->getAttrOfType<ElementsAttr>("hip.inline_value");
  } else if (auto expandShape =
                 dyn_cast<mlir::tensor::ExpandShapeOp>(def)) {
    // Look through expand_shape of a constant scalar/1-d tensor.  Kokoro
    // emits this for the istft slice ends (a scalar arith.constant
    // expanded to <1xi64>).  We just extract the underlying constant
    // values; the expand_shape preserves them.
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
    if (!inputType)
      return rewriter.notifyMatchFailure(
          op, "onnx.Slice lowering requires a ranked input");

    auto resultType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(
          op, "onnx.Slice lowering requires a ranked output");

    // Degenerate scalar slice: rank-0 input passes through unchanged.
    if (inputType.getRank() == 0 && resultType.getRank() == 0) {
      rewriter.replaceOp(op, input);
      return success();
    }

    auto startsOr = extractI64Constant(op->getOperand(1));
    auto endsOr = extractI64Constant(op->getOperand(2));
    if (failed(startsOr) || failed(endsOr)) {
      // Non-constant starts/ends: lower to tensor.extract_slice directly.
      // Extract runtime offsets and sizes from the starts/ends tensors.
      Value startsTensor = op->getOperand(1);
      Value endsTensor = op->getOperand(2);

      SmallVector<int64_t> axes;
      if (op->getNumOperands() >= 4) {
        auto axesOr = extractI64Constant(op->getOperand(3));
        if (failed(axesOr))
          return rewriter.notifyMatchFailure(
              op, "onnx.Slice axes must be an int64 constant for "
                  "dynamic starts/ends fallback");
        axes = *axesOr;
      } else {
        auto stType = dyn_cast<RankedTensorType>(startsTensor.getType());
        int64_t nSlice = stType ? stType.getDimSize(0) : 0;
        if (nSlice <= 0)
          return rewriter.notifyMatchFailure(
              op, "onnx.Slice cannot determine number of slice axes");
        for (int64_t i = 0; i < nSlice; ++i)
          axes.push_back(i);
      }

      SmallVector<int64_t> steps;
      if (op->getNumOperands() >= 5) {
        auto stepsOr = extractI64Constant(op->getOperand(4));
        if (failed(stepsOr))
          return rewriter.notifyMatchFailure(
              op, "onnx.Slice steps must be constant for dynamic fallback");
        steps = *stepsOr;
      } else {
        steps.assign(axes.size(), 1);
      }

      Location loc2 = op->getLoc();
      int64_t inputRank = inputType.getRank();
      Value zeroIdx = mlir::arith::ConstantIndexOp::create(rewriter, loc2, 0);
      Value oneIdx = mlir::arith::ConstantIndexOp::create(rewriter, loc2, 1);

      SmallVector<OpFoldResult> offsets(inputRank, OpFoldResult(zeroIdx));
      SmallVector<OpFoldResult> sizes(inputRank);
      SmallVector<OpFoldResult> strides(inputRank, OpFoldResult(oneIdx));

      // Default sizes = input dims
      for (int64_t d = 0; d < inputRank; ++d) {
        if (inputType.isDynamicDim(d))
          sizes[d] = OpFoldResult(
              mlir::tensor::DimOp::create(rewriter, loc2, input, d)
                  .getResult());
        else
          sizes[d] = rewriter.getIndexAttr(inputType.getDimSize(d));
      }

      for (size_t i = 0; i < axes.size(); ++i) {
        int64_t ax = axes[i] < 0 ? axes[i] + inputRank : axes[i];
        Value idxV = mlir::arith::ConstantIndexOp::create(rewriter, loc2, i);
        Value startI64 = mlir::tensor::ExtractOp::create(
            rewriter, loc2, startsTensor, ValueRange{idxV});
        Value endI64 = mlir::tensor::ExtractOp::create(
            rewriter, loc2, endsTensor, ValueRange{idxV});
        Value startIdx = mlir::arith::IndexCastOp::create(
            rewriter, loc2, rewriter.getIndexType(), startI64);
        Value endIdx = mlir::arith::IndexCastOp::create(
            rewriter, loc2, rewriter.getIndexType(), endI64);

        Value dimV = inputType.isDynamicDim(ax)
            ? mlir::tensor::DimOp::create(rewriter, loc2, input, ax)
                  .getResult()
            : mlir::arith::ConstantIndexOp::create(
                  rewriter, loc2, inputType.getDimSize(ax))
                  .getResult();

        // Handle negative starts/ends: s = s < 0 ? s + dim : s
        Value negCheck = mlir::arith::CmpIOp::create(
            rewriter, loc2, mlir::arith::CmpIPredicate::slt, startIdx, zeroIdx);
        Value sAdj = mlir::arith::AddIOp::create(rewriter, loc2, startIdx, dimV);
        startIdx = mlir::arith::SelectOp::create(
            rewriter, loc2, negCheck, sAdj, startIdx);
        // Clamp start to [0, dim]
        startIdx = mlir::arith::MaxSIOp::create(rewriter, loc2, startIdx, zeroIdx);
        startIdx = mlir::arith::MinSIOp::create(rewriter, loc2, startIdx, dimV);

        negCheck = mlir::arith::CmpIOp::create(
            rewriter, loc2, mlir::arith::CmpIPredicate::slt, endIdx, zeroIdx);
        Value eAdj = mlir::arith::AddIOp::create(rewriter, loc2, endIdx, dimV);
        endIdx = mlir::arith::SelectOp::create(
            rewriter, loc2, negCheck, eAdj, endIdx);
        endIdx = mlir::arith::MaxSIOp::create(rewriter, loc2, endIdx, zeroIdx);
        endIdx = mlir::arith::MinSIOp::create(rewriter, loc2, endIdx, dimV);

        // size = max(0, (end - start + step - 1) / step) for positive step
        Value diff = mlir::arith::SubIOp::create(rewriter, loc2, endIdx, startIdx);
        int64_t step = steps[i];
        Value stepV = mlir::arith::ConstantIndexOp::create(
            rewriter, loc2, std::abs(step));
        if (step != 1) {
          Value stepM1 = mlir::arith::SubIOp::create(rewriter, loc2, stepV, oneIdx);
          Value num = mlir::arith::AddIOp::create(rewriter, loc2, diff, stepM1);
          diff = mlir::arith::DivSIOp::create(rewriter, loc2, num, stepV);
        }
        diff = mlir::arith::MaxSIOp::create(rewriter, loc2, diff, zeroIdx);

        offsets[ax] = OpFoldResult(startIdx);
        sizes[ax] = OpFoldResult(diff);
        if (step != 1)
          strides[ax] = OpFoldResult(stepV);
      }

      auto sliceOp = mlir::tensor::ExtractSliceOp::create(
          rewriter, loc2, input, offsets, sizes, strides);
      Value result = sliceOp.getResult();
      if (result.getType() != resultType)
        result = mlir::tensor::CastOp::create(rewriter, loc2, resultType,
                                               result)
                     .getResult();
      rewriter.replaceOp(op, result);
      return success();
    }
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
    // For each dynamic OUTPUT dim, we may need to compute its runtime
    // size from the input dim and the constant start/end/step.  Track
    // the per-dim raw start/end/step so we can emit arith ops below.
    struct DynPlan {
      int64_t axis;
      int64_t start;  // ONNX-spec value (may be negative)
      int64_t end;    // ONNX-spec value (may be negative or INT64_MAX)
      int64_t step;
    };
    SmallVector<DynPlan> dynPlans;
    for (size_t i = 0; i < axes.size(); ++i) {
      int64_t ax = axes[i];
      if (ax < 0)
        ax += inputRank;
      if (ax < 0 || ax >= inputRank)
        return rewriter.notifyMatchFailure(op, "onnx.Slice axis out of range");
      int64_t step = steps[i];
      if (step == 0)
        return rewriter.notifyMatchFailure(op, "onnx.Slice step must be != 0");

      int64_t s = starts[i];
      int64_t e = ends[i];
      if (inputType.isDynamicDim(ax)) {
        // Defer normalization to runtime arith ops below.  We still
        // pass the *raw* start to the kernel as the slice offset;
        // negative values are folded against tensor.dim at runtime.
        normStarts[ax] = s;
        normSteps[ax] = step;
        dynPlans.push_back({ax, s, e, step});
        continue;
      }
      int64_t dimSize = inputType.getDimSize(ax);
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

    // Build init.  resultType may have dynamic dims; for those we
    // compute the runtime size from the input dim and the constant
    // start/end/step using arith ops (so the init memref is the right
    // size and downstream Reshape '-1' inference produces a positive
    // dim).  For axes not in axes[], or non-sliced dynamic dims, the
    // output dim equals the input dim.
    Location loc2 = loc;
    SmallVector<Value> dynSizes;
    for (int64_t d = 0; d < inputRank; ++d) {
      if (!resultType.isDynamicDim(d))
        continue;
      // Find a DynPlan for this axis.
      const DynPlan *plan = nullptr;
      for (const auto &p : dynPlans)
        if (p.axis == d) { plan = &p; break; }
      if (!plan) {
        // Dim is dynamic but not sliced -- forward the input dim.
        dynSizes.push_back(mlir::tensor::DimOp::create(rewriter, loc2, input, d));
        continue;
      }
      // Compute runtime: clamp(end, 0, dim) - clamp(start, 0, dim),
      // then ceil-divide by abs(step).
      Value dimV = mlir::tensor::DimOp::create(rewriter, loc2, input, d);
      Value zeroIdx = mlir::arith::ConstantIndexOp::create(rewriter, loc2, 0);
      auto addDim = [&](int64_t k) {
        if (k == 0) return zeroIdx;
        if (k == std::numeric_limits<int64_t>::max()) return dimV;
        Value c = mlir::arith::ConstantIndexOp::create(rewriter, loc2, k);
        if (k > 0) return c;
        // negative: dim + k
        return mlir::arith::AddIOp::create(rewriter, loc2, dimV, c).getResult();
      };
      Value startV = addDim(plan->start);
      Value endV = addDim(plan->end);
      // clamp to [0, dim]
      auto clampV = [&](Value v) {
        Value lo = mlir::arith::MaxSIOp::create(rewriter, loc2, v, zeroIdx);
        return mlir::arith::MinSIOp::create(rewriter, loc2, lo, dimV)
            .getResult();
      };
      Value sC = clampV(startV);
      Value eC = clampV(endV);
      Value diff = mlir::arith::SubIOp::create(rewriter, loc2, eC, sC);
      // For step != 1 we'd need ceil-div; for now Kokoro only uses step=1.
      if (plan->step != 1 && plan->step != -1) {
        Value stepV =
            mlir::arith::ConstantIndexOp::create(rewriter, loc2,
                                                  std::abs(plan->step));
        Value stepM1 = mlir::arith::SubIOp::create(
            rewriter, loc2, stepV,
            mlir::arith::ConstantIndexOp::create(rewriter, loc2, 1));
        Value num = mlir::arith::AddIOp::create(rewriter, loc2, diff, stepM1);
        diff = mlir::arith::DivSIOp::create(rewriter, loc2, num, stepV);
      }
      // diff can go negative if start > end -- clamp to 0.
      diff = mlir::arith::MaxSIOp::create(rewriter, loc2, diff, zeroIdx);
      dynSizes.push_back(diff);
    }
    Value init = mlir::tensor::EmptyOp::create(
        rewriter, loc2, resultType.getShape(), resultType.getElementType(),
        dynSizes);
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
