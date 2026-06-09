/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

/// onnx.Tile -> hip.tile
///
/// For each dynamic output dim, the init alloc size must be
/// `input.dim[i] * repeats[i]`, NOT `input.dim[i]` alone -- the latter is
/// what `createEmptyTensor(resultType, input)` would do and it under-sizes
/// the output buffer by the tile factor along every axis the model expects
/// to grow. Under-size symptom downstream: the tiled-axis dim seen by
/// consumers stays at the pre-tile value, and a subsequent broadcast Mul
/// fails the MIOpen rank-broadcast check (`BTensor dim != 1 && BTensor dim
/// != CTensor dim`).
///
/// Before:
///   %0 = "onnx.Tile"(%in, %r) :
///          (tensor<?x1152xf16>, tensor<2xi64>) -> tensor<?x1152xf16>
///   // init alloc would size dim 0 from %in.dim[0] (pre-tile)
///
/// After:
///   %c0 = arith.constant 0 : index
///   %d0 = tensor.dim %in, %c0
///   %r0_i64 = tensor.extract %r[%c0] : tensor<2xi64>
///   %r0_idx = arith.index_cast %r0_i64 : i64 to index
///   %d0_out = arith.muli %d0, %r0_idx
///   %init   = tensor.empty(%d0_out) : tensor<?x1152xf16>
///   %0 = hip.tile(%ctx) ins(%in, %r : ..., tensor<2xi64>)
///                       outs(%init : tensor<?x1152xf16>)
struct TileToHip : public mlir::RewritePattern {
  TileToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Tile", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto ctxOrFailure = getContextArg(op, rewriter);
    if (mlir::failed(ctxOrFailure))
      return mlir::failure();
    mlir::Value context = *ctxOrFailure;

    mlir::Location loc = op->getLoc();
    mlir::Value input = op->getOperand(0);
    mlir::Value repeats = op->getOperand(1);

    auto resultType =
        mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());
    auto inputType = mlir::cast<mlir::RankedTensorType>(input.getType());
    int64_t rank = resultType.getRank();

    // For each dynamic output dim, compute input.dim[i] * repeats[i].
    llvm::SmallVector<mlir::Value> dynSizes;
    for (int64_t i : llvm::seq<int64_t>(rank)) {
      if (!resultType.isDynamicDim(i))
        continue;
      // input dim (static or runtime).
      mlir::Value inDim;
      if (inputType.isDynamicDim(i)) {
        inDim = mlir::tensor::DimOp::create(rewriter, loc, input, i);
      } else {
        inDim = mlir::arith::ConstantIndexOp::create(rewriter, loc,
                                                     inputType.getDimSize(i));
      }
      // repeats[i] is i64 in a 1-D tensor; extract and cast to index.
      mlir::Value iIdx = mlir::arith::ConstantIndexOp::create(rewriter, loc, i);
      mlir::Value rI64 = mlir::tensor::ExtractOp::create(
          rewriter, loc, repeats, mlir::ValueRange{iIdx});
      mlir::Value rIdx = mlir::arith::IndexCastOp::create(
          rewriter, loc, rewriter.getIndexType(), rI64);
      mlir::Value outDim =
          mlir::arith::MulIOp::create(rewriter, loc, inDim, rIdx);
      dynSizes.push_back(outDim);
    }
    mlir::Value init =
        mlir::tensor::EmptyOp::create(rewriter, loc, resultType.getShape(),
                                      resultType.getElementType(), dynSizes);

    auto hipOp =
        mlir::hip::TileOp::create(rewriter, loc, context, input, repeats, init);
    rewriter.replaceOp(op, hipOp->getResult(0));
    return mlir::success();
  }
};

} // namespace

void populateTileConversionPatterns(RewritePatternSet &patterns,
                                    MLIRContext *ctx) {
  patterns.add<TileToHip>(ctx);
}

} // namespace hip
} // namespace mlir
