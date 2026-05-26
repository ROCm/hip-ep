/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

//===----------------------------------------------------------------------===//
// Concat -> tensor.from_elements (axis=0, rank-1 shape-vector case)
//===----------------------------------------------------------------------===//
//
// The only ONNX Concat shape we actually need today is the "shape vector
// concat" produced by dynamic-shape graphs immediately upstream of a
// Reshape — i.e.
//
//     B = Gather(Shape(K), 0)          ; rank-1[1] i64 (or rank-0)
//     S = Gather(Shape(K), 1)          ; rank-1[1] i64 (or rank-0)
//     shape = Concat([B], [S]) {axis=0} ; rank-1[2] i64
//     pos   = Reshape(range_out, shape) ; [batch_seq] -> [B, S]
//
// In Qwen3.5 text.onnx (and similar dyn-shape graphs) every Concat node
// is of this form: a small (2-4 input) axis=0 concat over rank-1[1] or
// rank-1[N] i64 tensors used purely to build a shape operand for a
// downstream Reshape / Expand / ConstantOfShape.
//
// The cleanest zero-cost lowering is `tensor.from_elements`, because:
//
//   * Each input element is `tensor.extract`ed once at MLIR level.
//   * Downstream `tensor.extract %from_elements[%c_k]` immediately folds
//     back to the original SSA value via tensor canonicalization, so the
//     dim values feed straight into the Reshape's `tensor.expand_shape`
//     output_shape operands without any runtime cost.
//   * No HIP dialect op, no LLVM lowering, no runtime kernel needed.
//
// For non-shape-vector Concat (e.g. concatenating large feature tensors
// along an arbitrary axis), this pattern deliberately fails to match.
// Those cases need a real hip.concat op + runtime kernel; they don't
// appear in any dyn-shape model we currently care about, and silently
// expanding `tensor.from_elements` to a large tensor would generate
// pathological IR (one tensor.extract per element).
struct ConcatShapeVectorToFromElements : public mlir::RewritePattern {
  ConcatShapeVectorToFromElements(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Concat", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    if (op->getNumResults() != 1)
      return rewriter.notifyMatchFailure(op, "expected exactly 1 result");

    auto resultType =
        mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(op, "expected ranked tensor result");

    // axis attribute is REQUIRED on onnx.Concat (no default in the spec).
    auto axisAttr = op->getAttrOfType<mlir::IntegerAttr>("axis");
    if (!axisAttr)
      return rewriter.notifyMatchFailure(op, "missing axis attribute");
    int64_t axis = axisAttr.getSInt();
    int64_t resultRank = resultType.getRank();
    if (axis < 0)
      axis += resultRank;

    // Shape-vector case constraints: rank-1 output, axis=0, integer
    // element type (i32/i64; index would also work but ONNX Concat never
    // emits it). These three together exclude the heavy feature-tensor
    // concat case.
    if (resultRank != 1)
      return rewriter.notifyMatchFailure(
          op, "shape-vector concat path requires a rank-1 result");
    if (axis != 0)
      return rewriter.notifyMatchFailure(
          op, "shape-vector concat path requires axis=0");
    mlir::Type elemTy = resultType.getElementType();
    if (!elemTy.isIntOrIndex())
      return rewriter.notifyMatchFailure(
          op, "shape-vector concat path requires an integer element type");

    // Walk each operand and emit per-element tensor.extract. We accept
    // two shapes per operand:
    //   * rank-0 scalar tensor                      -> 1 extract, no idx
    //   * rank-1 [K] tensor with static K >= 0      -> K extracts at i=0..K-1
    // ONNX Concat rejects rank-0 operands by spec (all inputs share the
    // result rank), but real graphs do contain rank-0 i64 scalars from
    // a preceding Gather; we accept both to be friendly.
    //
    // Dynamic-rank-1 inputs (K = ShapedType::kDynamic) cannot be expanded
    // at compile time without a runtime loop, so fall back to a no-match
    // (the only producer we've seen for shape-vector concat operands is
    // a Gather indexed by a compile-time constant, which always yields
    // statically-known K).
    mlir::Location loc = op->getLoc();
    llvm::SmallVector<mlir::Value> elements;
    elements.reserve(op->getNumOperands());

    // Cache the c0 index once; reused for every rank-1 extract and for
    // the rank-0 case (extract uses an empty ValueRange).
    mlir::Value c0Index;
    auto getC0 = [&]() -> mlir::Value {
      if (!c0Index)
        c0Index = mlir::arith::ConstantIndexOp::create(rewriter, loc, 0);
      return c0Index;
    };

    for (mlir::Value operand : op->getOperands()) {
      auto operandType =
          mlir::dyn_cast<mlir::RankedTensorType>(operand.getType());
      if (!operandType)
        return rewriter.notifyMatchFailure(op, "non-ranked Concat operand");
      if (operandType.getElementType() != elemTy)
        return rewriter.notifyMatchFailure(
            op, "Concat operand element type does not match result");

      int64_t opRank = operandType.getRank();
      if (opRank == 0) {
        mlir::Value v = mlir::tensor::ExtractOp::create(rewriter, loc, operand,
                                                        mlir::ValueRange{});
        elements.push_back(v);
        continue;
      }
      if (opRank != 1)
        return rewriter.notifyMatchFailure(
            op, "shape-vector concat operand must be rank-0 or rank-1");
      if (operandType.isDynamicDim(0))
        return rewriter.notifyMatchFailure(
            op, "shape-vector concat operand with dynamic length is not "
                "compile-time expandable");

      int64_t K = operandType.getDimSize(0);
      // K == 0 is legal (an empty concat operand contributes nothing); the
      // result rank-1 length still adds up to the sum of K's via the
      // remaining operands.
      for (int64_t i = 0; i < K; ++i) {
        mlir::Value idx;
        if (i == 0)
          idx = getC0();
        else
          idx = mlir::arith::ConstantIndexOp::create(rewriter, loc, i);
        mlir::Value v = mlir::tensor::ExtractOp::create(rewriter, loc, operand,
                                                        mlir::ValueRange{idx});
        elements.push_back(v);
      }
    }

    // The result shape must be statically known (rank-1[N] with N = sum
    // of operand K's). If the upstream shape inference lost track of the
    // length we cannot build tensor.from_elements; bail.
    if (resultType.isDynamicDim(0))
      return rewriter.notifyMatchFailure(
          op, "shape-vector concat result length must be static");
    int64_t expected = resultType.getDimSize(0);
    if (static_cast<int64_t>(elements.size()) != expected)
      return rewriter.notifyMatchFailure(
          op, "Concat operand element count does not match result length");

    mlir::Value fromElements = mlir::tensor::FromElementsOp::create(
        rewriter, loc, resultType, elements);
    rewriter.replaceOp(op, fromElements);
    return mlir::success();
  }
};

} // namespace

void populateConcatConversionPatterns(RewritePatternSet &patterns,
                                      MLIRContext *ctx) {
  patterns.add<ConcatShapeVectorToFromElements>(ctx);
}

} // namespace hip
} // namespace mlir
