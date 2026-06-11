/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- ReshapeShapeFold.cpp - Pre-lowering Reshape(_, Shape) folding ------===//
//
// Sibling pass to GatherShapeFold.  Recognises the
//
//     %shape = onnx.Shape(%src)            : tensor<Nxi64>
//     %out   = onnx.Reshape(%data, %shape) : tensor<...?...>
//
// idiom and rewrites the Reshape's shape operand into a
//
//     %d_i  = tensor.dim %src, i                            : index
//     %s_i  = arith.index_cast %d_i : index to i64          ;; for dyn dims
//     %s_j  = arith.constant <static_size> : i64            ;; for static dims
//     %new  = tensor.from_elements %s_0, ..., %s_{N-1}      : tensor<Nxi64>
//
//     %out  = onnx.Reshape(%data, %new)
//
// so ReshapeConversion's `buildExpandShapeOutputShape` multi-dyn-per-group
// branch (which already understands tensor.from_elements) can recover
// per-output-dim sizes when the result has >1 dynamic dim in the same
// reassociation group.
//
// Why this matters
// ----------------
// PR #212 GatherShapeFold rewrites Gather(Shape, idx) so the canonical
// transformer-shape-arithmetic chain
//
//     bs   = Gather(Shape(input), 0)
//     ss   = Gather(Shape(input), 1)
//     mul  = bs * ss
//     out  = Range(0, mul, 1)
//
// collapses to host arith over tensor.dim of input, avoiding the
// SEGV-on-device-store-of-shape-vector issue on gfx1151.  But when the
// SAME Shape(input) value is ALSO directly fed to Reshape (as in
// Qwen3.5 mrope: `Reshape(Range_output, Shape(input))`), that use is
// not a Gather — GatherShapeFold leaves it alone, and `onnx.Shape`
// survives into ConvertOnnxToHip.  ReshapeConversion ignores its
// second operand (it derives output shape from result type alone),
// so for a `<?> -> <?, ?>` reshape with both output dims dynamic it
// emits `output_shape [tensor.dim(out, 0), tensor.dim(out, 0)]` — the
// SAME SSA value twice, semantically [N, N] backed by an N-element
// buffer.  No downstream pass recovers the correct per-dim sizes once
// the [N, N] expand_shape has been emitted.  Net effect: a 1×1 input
// passes by coincidence (1×1 = 1²), every asymmetric shape silently
// corrupts.
//
// This fold rewrites the shape operand BEFORE ConvertOnnxToHip so the
// resulting `tensor.from_elements` is exactly what ReshapeConversion's
// existing multi-dyn-per-group branch picks up.  Bug fix is structural
// at the operand level; no change to ReshapeConversion required.
//
// Implementation notes
// --------------------
//   * Pattern roots on `onnx.Reshape` (greedy applier visits ops in
//     post-order; we only need to fire once per Reshape).
//   * Uses `modifyOpInPlace` to swap operand 1.  Under
//     `GreedyRewriteStrictness::ExistingOps` (the OnnxToHipPass's
//     pre-lowering set strictness) the modified op is re-considered
//     but our match guard `operand(1).getDefiningOp() == onnx.Shape`
//     fails on the second visit, so no infinite loop.
//   * Honors `onnx.Shape`'s ONNX-15 `start`/`end` attributes (matches
//     GatherShapeFold's normalization).  When the sub-range is empty
//     we leave the op alone (a downstream verifier will reject it).
//   * Leaves the original `onnx.Shape` op in place — it may have other
//     uses (e.g. another Gather still pending fold).  DCE removes it
//     later if it becomes unused.
//   * Result-type check: the Reshape's existing result type is the
//     ground truth ONNX shape inference computed; we don't touch it.
//     The shape operand's RankedTensorType is `tensor<rangeLen x i64>`
//     by construction.
//
// Non-goals
// ---------
//   * Folding `Reshape(_, Concat(Gather(Shape, 0), Gather(Shape, 1)))`:
//     after GatherShapeFold each Gather becomes a 1-element
//     tensor.from_elements, and ConcatConversion (or ReshapeConversion
//     itself) handles the Concat case directly.  This fold only
//     addresses the `Reshape(_, Shape)` direct-use case.
//   * Folding `Range(_, Shape[*], _)` or `Slice(_, Shape, ...)`:
//     analogous patterns; out of scope for this fix.  Add sibling
//     folds when those uses arise.
//
//===----------------------------------------------------------------------===//

#include "OnnxToHipUtils.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "reshape-shape-fold"

STATISTIC(NumReshapeShapeFolds,
          "Number of Reshape(_, Shape(x)) idioms whose shape operand was "
          "rewritten to tensor.from_elements(tensor.dim(x, *))");

namespace mlir {
namespace hip {

namespace {

struct ReshapeOfShapeToFromElements : public mlir::RewritePattern {
  ReshapeOfShapeToFromElements(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Reshape", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *reshapeOp,
                  mlir::PatternRewriter &rewriter) const override {
    if (reshapeOp->getNumOperands() < 2)
      return rewriter.notifyMatchFailure(reshapeOp, "reshape.arity");

    mlir::Value shapeOperand = reshapeOp->getOperand(1);
    mlir::Operation *shapeOp = shapeOperand.getDefiningOp();
    if (!shapeOp || shapeOp->getName().getStringRef() != "onnx.Shape")
      return rewriter.notifyMatchFailure(reshapeOp, "operand1.not_onnx_shape");

    // onnx.Shape has a single operand (the input whose shape is being
    // queried).  Defensive against malformed IR.
    if (shapeOp->getNumOperands() != 1)
      return rewriter.notifyMatchFailure(reshapeOp, "shape.arity");

    mlir::Value src = shapeOp->getOperand(0);
    auto srcType = mlir::dyn_cast<mlir::RankedTensorType>(src.getType());
    if (!srcType)
      return rewriter.notifyMatchFailure(reshapeOp, "shape.src_unranked");
    int64_t rank = srcType.getRank();

    // ONNX-15 Shape start/end attributes select a sub-range of the shape
    // vector.  Normalize the same way GatherShapeFold does.
    int64_t start = 0;
    int64_t end = rank;
    if (auto startAttr = shapeOp->getAttrOfType<mlir::IntegerAttr>("start"))
      start = startAttr.getSInt();
    if (auto endAttr = shapeOp->getAttrOfType<mlir::IntegerAttr>("end"))
      end = endAttr.getSInt();
    if (start < 0)
      start += rank;
    if (end < 0)
      end += rank;
    start = std::max(start, int64_t(0));
    end = std::min(end, rank);
    int64_t rangeLen = std::max(end - start, int64_t(0));
    if (rangeLen == 0)
      return rewriter.notifyMatchFailure(reshapeOp, "shape.empty_range");

    // The shape operand's tensor type must be tensor<rangeLen x i64>.
    // If the existing operand type disagrees the IR is malformed; bail.
    auto shapeOperandType =
        mlir::dyn_cast<mlir::RankedTensorType>(shapeOperand.getType());
    if (!shapeOperandType || shapeOperandType.getRank() != 1 ||
        !shapeOperandType.getElementType().isInteger(64) ||
        (!shapeOperandType.isDynamicDim(0) &&
         shapeOperandType.getDimSize(0) != rangeLen))
      return rewriter.notifyMatchFailure(reshapeOp,
                                         "shape.operand_type_mismatch");

    mlir::Location loc = reshapeOp->getLoc();
    auto i64Type = rewriter.getI64Type();

    // Materialize per-dim i64 SSA values.  For dynamic dims emit
    // `arith.index_cast %tensor.dim`, for static dims a plain
    // `arith.constant` so canonicalization can fold downstream.
    llvm::SmallVector<mlir::Value> elems;
    elems.reserve(rangeLen);
    for (int64_t i = 0; i < rangeLen; ++i) {
      int64_t absDim = start + i;
      mlir::Value elem;
      if (srcType.isDynamicDim(absDim)) {
        mlir::Value dimVal =
            mlir::tensor::DimOp::create(rewriter, loc, src, absDim);
        elem = mlir::arith::IndexCastOp::create(rewriter, loc, i64Type, dimVal);
      } else {
        elem = mlir::arith::ConstantOp::create(
            rewriter, loc,
            rewriter.getI64IntegerAttr(srcType.getDimSize(absDim)));
      }
      elems.push_back(elem);
    }

    // Reconstruct the shape operand's tensor type with the now-static
    // length (rangeLen).  Even if the original op carried tensor<?xi64>,
    // tensor.from_elements demands a static rank-1 type with size ==
    // elems.size().
    auto newShapeType = mlir::RankedTensorType::get({rangeLen}, i64Type);
    mlir::Value newShape =
        mlir::tensor::FromElementsOp::create(rewriter, loc, newShapeType, elems)
            .getResult();

    LLVM_DEBUG(llvm::dbgs()
               << "[" DEBUG_TYPE "] rewrote " << reshapeOp->getName()
               << " shape operand from " << shapeOp->getName()
               << " to from_elements over " << rangeLen << " dim(s) of src\n");

    rewriter.modifyOpInPlace(reshapeOp,
                             [&] { reshapeOp->setOperand(1, newShape); });
    ++NumReshapeShapeFolds;
    return mlir::success();
  }
};

} // namespace

void populateReshapeShapeFoldPatterns(mlir::RewritePatternSet &patterns,
                                      mlir::MLIRContext *ctx) {
  patterns.add<ReshapeOfShapeToFromElements>(ctx);
}

} // namespace hip
} // namespace mlir
