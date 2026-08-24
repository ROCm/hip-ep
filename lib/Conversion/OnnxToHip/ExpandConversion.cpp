/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

/// Read element \p idx of the 1-D `shape` operand to a host `index`. Folds a
/// constant shape; otherwise reads it back with a stream sync (see
/// ReadbackScalar.h) -- a bare host load of the GPU-computed extent races the
/// producing kernel and sizes the output from stale memory (a zero collapses to
/// a null buffer and `wrap_expand` fails; a wrong size collapses the result).
static mlir::Value readShapeEntryToIndex(mlir::PatternRewriter &rewriter,
                                         mlir::Location loc, mlir::Value ctx,
                                         mlir::Value shape, int64_t idx) {
  mlir::Value scalar = readbackShapeEntryToHost(rewriter, loc, ctx, shape, idx);
  return mlir::arith::IndexCastOp::create(rewriter, loc,
                                          rewriter.getIndexType(), scalar);
}

/// onnx.Expand -> hip.expand
///
/// We trust the result type produced by ONNX shape inference. For dynamic
/// dims in the result we extract the corresponding entry from the `shape`
/// input tensor (right-aligned with the result rank, NumPy-style); leading
/// dims that are absent from `shape` fall back to the matching input dim.
///
/// `Expand`'s shape operand is a BROADCAST target, not a literal resize: per
/// ONNX/NumPy broadcasting, the output extent at a position is
/// `max(input_dim, shape_dim)`, and a `1` in `shape` at a position where the
/// input already has a real (possibly >1) extent means "do not broadcast
/// here", i.e. keep the input's extent. Using the shape entry verbatim
/// silently collapses any such input dim to 1 whenever the shape operand
/// says 1 there -- see docs/nemotron-mamba-shape-collapse.md for the mamba
/// `Expand(_, [1,1,1,16,1])` fallout this caused (batch/seq collapsed to 1,
/// corrupting the residual stream via a stale-pool-memory read one op later).
struct ExpandToHip : public mlir::RewritePattern {
  ExpandToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Expand", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto ctxOrFailure = getContextArg(op, rewriter);
    if (mlir::failed(ctxOrFailure))
      return mlir::failure();
    mlir::Value context = *ctxOrFailure;

    mlir::Location loc = op->getLoc();
    mlir::Value input = op->getOperand(0);
    mlir::Value shape = op->getOperand(1);

    auto resultType =
        mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());
    auto inputType = mlir::cast<mlir::RankedTensorType>(input.getType());

    int64_t resultRank = resultType.getRank();
    int64_t inputRank = inputType.getRank();

    // shape is a 1-D int tensor; for dynamic result dims we extract entries
    // from it (right-aligned with the result rank). When `shape` itself has
    // a dynamic length we cannot reason about it here, so bail out.
    auto shapeType = mlir::cast<mlir::RankedTensorType>(shape.getType());
    if (shapeType.getRank() != 1 || shapeType.isDynamicDim(0))
      return rewriter.notifyMatchFailure(
          op, "expand shape input must have static rank-1 type");
    int64_t shapeLen = shapeType.getDimSize(0);

    llvm::SmallVector<mlir::Value> dynSizes;
    for (int64_t i = 0; i < resultRank; ++i) {
      if (!resultType.isDynamicDim(i))
        continue;
      int64_t shapeIdx = i - (resultRank - shapeLen);
      int64_t inputIdx = i - (resultRank - inputRank);
      mlir::Value dim;
      if (shapeIdx >= 0) {
        // The shape entry may be GPU-computed; read it back with a stream sync
        // (constants fold) instead of a bare host load of device memory. See
        // readShapeEntryToIndex for the correctness rationale.
        mlir::Value shapeDim =
            readShapeEntryToIndex(rewriter, loc, context, shape, shapeIdx);
        if (inputIdx >= 0) {
          // Broadcast semantics (see the struct doc comment above): a `1`
          // here means "keep the input's extent", not "resize to 1".
          mlir::Value inputDim =
              mlir::tensor::DimOp::create(rewriter, loc, input, inputIdx);
          mlir::Value one =
              mlir::arith::ConstantIndexOp::create(rewriter, loc, 1);
          mlir::Value shapeSaysOne = mlir::arith::CmpIOp::create(
              rewriter, loc, mlir::arith::CmpIPredicate::eq, shapeDim, one);
          dim = mlir::arith::SelectOp::create(rewriter, loc, shapeSaysOne,
                                              inputDim, shapeDim);
        } else {
          // No corresponding input dim (this is a purely-broadcast leading
          // dim absent from the input rank): the shape entry is the literal
          // extent, nothing to compare it against.
          dim = shapeDim;
        }
      } else {
        if (inputIdx < 0)
          return rewriter.notifyMatchFailure(
              op, "cannot resolve dynamic dim from input or shape");
        dim = mlir::tensor::DimOp::create(rewriter, loc, input, inputIdx);
      }
      dynSizes.push_back(dim);
    }

    mlir::Value init =
        mlir::tensor::EmptyOp::create(rewriter, loc, resultType.getShape(),
                                      resultType.getElementType(), dynSizes);

    auto hipOp =
        mlir::hip::ExpandOp::create(rewriter, loc, context, input, shape, init);
    rewriter.replaceOp(op, hipOp->getResult(0));
    return mlir::success();
  }
};

} // namespace

void populateExpandConversionPatterns(RewritePatternSet &patterns,
                                      MLIRContext *ctx) {
  patterns.add<ExpandToHip>(ctx);
}

} // namespace hip
} // namespace mlir
