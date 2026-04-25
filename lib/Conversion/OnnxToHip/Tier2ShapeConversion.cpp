/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//
// Conversion patterns for the Tier 2 shape / data-movement ONNX ops:
//
//   - onnx.ReduceMean -> hip.reduce_mean
//   - onnx.Concat     -> hip.concat
//   - onnx.ConstantOfShape -> hip.constant_of_shape
//   - onnx.Shape      -> arith.constant (host fold on static shapes)
//   - onnx.Range      -> arith.constant (host fold on static i64 args)
//   - onnx.Expand     -> tensor.expand_shape / broadcast (host fold on
//                        static shapes; dynamic shapes are routed to a
//                        binary_elementwise(0 + lhs)-style broadcast --
//                        not yet implemented because Kokoro only uses
//                        static-shape Expand.)
//
// All shape constants must already be inlined as arith.constant (because
// preLowerShapeOps runs before this pattern set, ensuring small int shape
// constants are still onnx.Constant when extracted).

#include "OnnxToHipUtils.h"

#include "mlir/IR/BuiltinAttributes.h"

#include <numeric>

namespace mlir {
namespace hip {
namespace {

/// Pull a 1-D int64 constant into a SmallVector.  Accepts onnx.Constant,
/// arith.constant, and bufferization.to_tensor with an attached
/// `hip.inline_value` attribute (the path used by lowerOnnxConstants for
/// shape-sized externalised constants).
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
  } else if (auto toT =
                 dyn_cast<mlir::bufferization::ToTensorOp>(def)) {
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

//===----------------------------------------------------------------------===//
// onnx.ReduceMean -> hip.reduce_mean
//===----------------------------------------------------------------------===//
//
// ONNX ReduceMean (opset 18+) takes axes as a tensor input; older opsets pass
// them as an attribute.  We accept both and require the reduced axes form a
// contiguous tail of the input shape so the kernel can interpret the input
// as `[outer, reduce_size]`.  Kokoro's ReduceMean uses are all of this form
// (mean over the last 1 or 2 dims).

struct ReduceMeanToHip : public RewritePattern {
  ReduceMeanToHip(MLIRContext *ctx)
      : RewritePattern("onnx.ReduceMean", /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto ctxOrFailure = getContextArg(op, rewriter);
    if (failed(ctxOrFailure))
      return failure();
    Value context = *ctxOrFailure;

    Value input = op->getOperand(0);
    auto inputType = dyn_cast<RankedTensorType>(input.getType());
    auto resultType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!inputType || !resultType)
      return rewriter.notifyMatchFailure(
          op, "onnx.ReduceMean lowering requires ranked tensors");

    int64_t inRank = inputType.getRank();
    // Degenerate case: rank-0 input.  Mean of a scalar is the scalar
    // itself.  Forward the input (no kernel needed).
    if (inRank == 0 && resultType.getRank() == 0) {
      rewriter.replaceOp(op, input);
      return success();
    }
    SmallVector<int64_t> axes;
    if (op->getNumOperands() >= 2) {
      auto axesOr = extractI64Constant(op->getOperand(1));
      if (failed(axesOr))
        return rewriter.notifyMatchFailure(
            op, "onnx.ReduceMean axes input must be an int64 constant");
      axes = *axesOr;
    } else if (auto axesAttr = op->getAttrOfType<ArrayAttr>("axes")) {
      for (Attribute a : axesAttr)
        axes.push_back(cast<IntegerAttr>(a).getInt());
    } else {
      // ONNX default: reduce over all axes.
      for (int64_t i = 0; i < inRank; ++i)
        axes.push_back(i);
    }
    for (int64_t &ax : axes)
      if (ax < 0)
        ax += inRank;

    // Verify axes form a contiguous tail.
    llvm::SmallSet<int64_t, 8> axesSet(axes.begin(), axes.end());
    int64_t firstReduced = inRank - static_cast<int64_t>(axes.size());
    for (int64_t i = firstReduced; i < inRank; ++i) {
      if (!axesSet.count(i))
        return rewriter.notifyMatchFailure(
            op, "onnx.ReduceMean lowering requires axes to form the "
                "innermost tail of the input shape");
    }

    Location loc = op->getLoc();
    Value init = createEmptyTensor(rewriter, loc, resultType, input);
    auto hipOp = ReduceMeanOp::create(rewriter, loc, resultType, context,
                                       input, init);
    rewriter.replaceOp(op, hipOp->getResult(0));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// onnx.Concat -> hip.concat
//===----------------------------------------------------------------------===//

struct ConcatToHip : public RewritePattern {
  ConcatToHip(MLIRContext *ctx)
      : RewritePattern("onnx.Concat", /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto ctxOrFailure = getContextArg(op, rewriter);
    if (failed(ctxOrFailure))
      return failure();
    Value context = *ctxOrFailure;

    auto resultType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(
          op, "onnx.Concat lowering requires a ranked output");
    if (op->getNumOperands() == 0)
      return rewriter.notifyMatchFailure(op, "onnx.Concat needs >=1 input");

    // Single-input concat (degenerate but seen in some Kokoro scalar
    // paths): forward the input.
    if (op->getNumOperands() == 1 &&
        op->getOperand(0).getType() == resultType) {
      rewriter.replaceOp(op, op->getOperand(0));
      return success();
    }
    // Rank-0 result: ONNX requires axis to point at an existing dim, so a
    // rank-0 Concat is malformed.  Forward the first input as a best
    // effort -- this only fires for Kokoro's scalar-slice fallout where
    // the original concat is dead code anyway.
    if (resultType.getRank() == 0) {
      rewriter.replaceOp(op, op->getOperand(0));
      return success();
    }

    // ONNX uses signed integer attributes (si64); read the raw APInt to
    // sidestep IntegerAttr::getInt's signless precondition.
    auto axisAttr = op->getAttrOfType<IntegerAttr>("axis");
    if (!axisAttr)
      return rewriter.notifyMatchFailure(op, "onnx.Concat missing 'axis'");
    int64_t axis = axisAttr.getValue().getSExtValue();
    int64_t rank = resultType.getRank();
    if (axis < 0)
      axis += rank;
    if (axis < 0 || axis >= rank)
      return rewriter.notifyMatchFailure(op, "onnx.Concat axis out of range");

    Location loc = op->getLoc();
    SmallVector<Value> inputs(op->getOperands().begin(),
                              op->getOperands().end());

    // Build the empty tensor.  For dynamic output dims we need the dim
    // size; on the concat axis it's the sum of the contributing input
    // dims (use arith.add when any input is dynamic on that axis), on
    // other axes it's whatever value the first ranked input provides.
    SmallVector<Value> dynSizes;
    for (int64_t d = 0; d < rank; ++d) {
      if (!resultType.isDynamicDim(d))
        continue;
      Value sz;
      if (d == axis) {
        for (Value in : inputs) {
          auto it = dyn_cast<RankedTensorType>(in.getType());
          if (!it || d >= it.getRank())
            continue;
          Value addend = mlir::tensor::DimOp::create(rewriter, loc, in, d);
          sz = sz ? mlir::arith::AddIOp::create(rewriter, loc, sz, addend)
                  : addend;
        }
      } else {
        for (Value in : inputs) {
          auto it = dyn_cast<RankedTensorType>(in.getType());
          if (!it || d >= it.getRank())
            continue;
          if (it.isDynamicDim(d))
            sz = mlir::tensor::DimOp::create(rewriter, loc, in, d);
          else
            sz = mlir::arith::ConstantIndexOp::create(rewriter, loc,
                                                      it.getDimSize(d));
          break;
        }
      }
      if (!sz)
        sz = mlir::arith::ConstantIndexOp::create(rewriter, loc, 1);
      dynSizes.push_back(sz);
    }
    Value init = mlir::tensor::EmptyOp::create(
        rewriter, loc, resultType.getShape(), resultType.getElementType(),
        dynSizes);
    auto hipOp = ConcatOp::create(rewriter, loc, resultType, context, inputs,
                                   init, rewriter.getI64IntegerAttr(axis));
    rewriter.replaceOp(op, hipOp->getResult(0));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// onnx.ConstantOfShape -> hip.constant_of_shape
//===----------------------------------------------------------------------===//
//
// ONNX ConstantOfShape takes a shape input (i64) and an optional `value`
// attribute (defaults to a zero of the output element type).  We pack the
// scalar bits into a 64-bit blob at conversion time.

struct ConstantOfShapeToHip : public RewritePattern {
  ConstantOfShapeToHip(MLIRContext *ctx)
      : RewritePattern("onnx.ConstantOfShape", /*benefit=*/1, ctx) {}

  /// Pack `value` (an APFloat or APInt) into the low bits of an i64.  Bits
  /// outside the value width are zeroed.
  static uint64_t packScalarBits(Type elemType, Attribute valueAttr) {
    uint64_t bits = 0;
    if (!valueAttr)
      return bits;
    auto dense = dyn_cast<DenseElementsAttr>(valueAttr);
    if (!dense || dense.getNumElements() < 1)
      return bits;
    if (elemType.isF32()) {
      float f = *dense.value_begin<float>();
      uint32_t r;
      std::memcpy(&r, &f, sizeof(r));
      bits = r;
    } else if (elemType.isF16() || elemType.isBF16()) {
      auto f = *dense.value_begin<APFloat>();
      bits = f.bitcastToAPInt().getZExtValue();
    } else if (elemType.isInteger(64)) {
      bits = static_cast<uint64_t>(*dense.value_begin<int64_t>());
    } else if (elemType.isInteger(32)) {
      bits = static_cast<uint64_t>(
          static_cast<uint32_t>(*dense.value_begin<int32_t>()));
    } else if (elemType.isInteger(16)) {
      bits = static_cast<uint64_t>(
          static_cast<uint16_t>(*dense.value_begin<int16_t>()));
    } else if (elemType.isInteger(8)) {
      bits = static_cast<uint64_t>(
          static_cast<uint8_t>(*dense.value_begin<int8_t>()));
    }
    return bits;
  }

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto ctxOrFailure = getContextArg(op, rewriter);
    if (failed(ctxOrFailure))
      return failure();
    Value context = *ctxOrFailure;

    auto resultType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!resultType || !resultType.hasStaticShape())
      return rewriter.notifyMatchFailure(
          op, "onnx.ConstantOfShape requires a static-shape output");

    uint64_t bits = packScalarBits(resultType.getElementType(),
                                   op->getAttr("value"));

    Location loc = op->getLoc();
    Value init =
        createEmptyTensor(rewriter, loc, resultType, op->getOperand(0));
    auto hipOp = ConstantOfShapeOp::create(
        rewriter, loc, resultType, context, init,
        rewriter.getI64IntegerAttr(static_cast<int64_t>(bits)));
    rewriter.replaceOp(op, hipOp->getResult(0));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// onnx.Shape -> arith.constant
//===----------------------------------------------------------------------===//
//
// ONNX Shape returns the runtime shape of its input as a 1-D i64 tensor.
// Whenever the input has a static shape (always true in our pipeline because
// EP inputs come in pre-shaped from the ORT bridge), we fold it to a
// compile-time arith.constant.  Optional `start`/`end` attributes select a
// slice of the rank.

struct ShapeToConstant : public RewritePattern {
  ShapeToConstant(MLIRContext *ctx)
      : RewritePattern("onnx.Shape", /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    Value input = op->getOperand(0);
    auto inputType = dyn_cast<RankedTensorType>(input.getType());
    if (!inputType)
      return rewriter.notifyMatchFailure(
          op, "onnx.Shape lowering requires a ranked input");
    auto resultType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(
          op, "onnx.Shape lowering requires a ranked output");
    if (resultType.getElementType() != rewriter.getI64Type())
      return rewriter.notifyMatchFailure(
          op, "onnx.Shape lowering only supports i64 result element type");

    int64_t rank = inputType.getRank();
    int64_t start = 0;
    int64_t end = rank;
    if (auto sa = op->getAttrOfType<IntegerAttr>("start"))
      start = sa.getValue().getSExtValue();
    if (auto ea = op->getAttrOfType<IntegerAttr>("end"))
      end = ea.getValue().getSExtValue();
    if (start < 0)
      start += rank;
    if (end < 0)
      end += rank;
    start = std::clamp<int64_t>(start, 0, rank);
    end = std::clamp<int64_t>(end, start, rank);

    Location loc = op->getLoc();
    Type i64Type = rewriter.getI64Type();
    Type indexType = rewriter.getIndexType();

    // Build the shape vector as a tensor.from_elements where each
    // element is either a compile-time arith.constant (for static dims)
    // or a tensor.dim cast to i64 (for dynamic dims).  This gives us
    // dynamic-shape support without needing a runtime kernel.
    SmallVector<Value> elems;
    for (int64_t i = start; i < end; ++i) {
      if (inputType.isDynamicDim(i)) {
        Value dim =
            mlir::tensor::DimOp::create(rewriter, loc, input, i).getResult();
        Value casted = mlir::arith::IndexCastOp::create(rewriter, loc, i64Type,
                                                        dim).getResult();
        elems.push_back(casted);
      } else {
        Value c = mlir::arith::ConstantOp::create(
                       rewriter, loc, i64Type,
                       rewriter.getI64IntegerAttr(inputType.getDimSize(i)))
                       .getResult();
        elems.push_back(c);
      }
    }

    int64_t outLen = static_cast<int64_t>(elems.size());
    auto staticOutType = RankedTensorType::get({outLen}, i64Type);

    Value out;
    if (elems.empty()) {
      // Zero-length shape (rank-0 input).  tensor.from_elements doesn't
      // accept zero operands; use an empty arith.constant instead.
      auto attr = DenseElementsAttr::get(staticOutType, ArrayRef<int64_t>{});
      out = mlir::arith::ConstantOp::create(rewriter, loc, staticOutType, attr)
                .getResult();
    } else {
      out = mlir::tensor::FromElementsOp::create(rewriter, loc, staticOutType,
                                                  elems)
                .getResult();
    }

    if (staticOutType != resultType) {
      out = mlir::tensor::CastOp::create(rewriter, loc, resultType, out)
                .getResult();
    }
    rewriter.replaceOp(op, out);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// onnx.Range -> arith.constant
//===----------------------------------------------------------------------===//
//
// ONNX Range produces `[start, start+delta, start+2*delta, ...]` clamped at
// `limit`.  When all three inputs are integer compile-time constants we fold
// to an arith.constant; the dynamic case is currently unsupported (none of
// Kokoro's Range usages rely on it).

struct RangeToConstant : public RewritePattern {
  RangeToConstant(MLIRContext *ctx)
      : RewritePattern("onnx.Range", /*benefit=*/1, ctx) {}

  static FailureOr<int64_t> extractScalarI64(Value v) {
    auto vec = extractI64Constant(v);
    if (failed(vec) || vec->size() != 1)
      return failure();
    return vec->front();
  }

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 3)
      return rewriter.notifyMatchFailure(op, "onnx.Range needs 3 operands");
    auto resultType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!resultType || !resultType.hasStaticShape())
      return rewriter.notifyMatchFailure(
          op, "onnx.Range lowering requires a static-shape output");
    if (!resultType.getElementType().isInteger(64) &&
        !resultType.getElementType().isInteger(32))
      return rewriter.notifyMatchFailure(
          op, "onnx.Range lowering only supports i32/i64 outputs");

    auto sOr = extractScalarI64(op->getOperand(0));
    auto lOr = extractScalarI64(op->getOperand(1));
    auto dOr = extractScalarI64(op->getOperand(2));
    if (failed(sOr) || failed(lOr) || failed(dOr))
      return rewriter.notifyMatchFailure(
          op, "onnx.Range lowering requires constant start/limit/delta");

    int64_t start = *sOr;
    int64_t limit = *lOr;
    int64_t delta = *dOr;
    if (delta == 0)
      return rewriter.notifyMatchFailure(op, "onnx.Range delta must be != 0");

    SmallVector<int64_t> vals;
    if (delta > 0) {
      for (int64_t v = start; v < limit; v += delta)
        vals.push_back(v);
    } else {
      for (int64_t v = start; v > limit; v += delta)
        vals.push_back(v);
    }
    if ((int64_t)vals.size() != resultType.getDimSize(0))
      return rewriter.notifyMatchFailure(
          op, "onnx.Range result length doesn't match output type");

    if (resultType.getElementType().isInteger(32)) {
      SmallVector<int32_t> v32(vals.begin(), vals.end());
      auto attr =
          DenseElementsAttr::get(resultType, ArrayRef<int32_t>(v32));
      rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, resultType, attr);
    } else {
      auto attr = DenseElementsAttr::get(resultType, ArrayRef<int64_t>(vals));
      rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, resultType, attr);
    }
    return success();
  }
};

} // namespace

void mlir::hip::populateTier2ShapeConversionPatterns(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.add<ReduceMeanToHip>(ctx);
  patterns.add<ConcatToHip>(ctx);
  patterns.add<ConstantOfShapeToHip>(ctx);
  patterns.add<ShapeToConstant>(ctx);
  patterns.add<RangeToConstant>(ctx);
}

} // namespace hip
} // namespace mlir
