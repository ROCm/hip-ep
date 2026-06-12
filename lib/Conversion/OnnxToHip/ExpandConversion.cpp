/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

/// Read element \p idx of the 1-D `shape` operand to a host `index`.
///
/// The `shape` operand is frequently computed on the GPU (the canonical case:
/// a meshgrid where each result dim is a `Range` length read back from
/// `image_grid_thw` and packed via `tensor.from_elements` +
/// `tensor.insert_slice` into the shape vector). A bare `tensor.extract
/// %shape[idx]` lowers to a host `memref.load` of that device buffer with no
/// stream synchronization — it reads stale/garbage on targets where the pool is
/// true device memory (it accidentally works where the pool is UMA-mapped
/// host-accessible memory). The garbage extent then sizes the `tensor.empty`
/// output: a zero size yields a zero-byte (null) output buffer and a wrong size
/// collapses the result, so `wrap_expand` fails with a null-output /
/// wrong-shape and the downstream rope cos/sin tensors collapse.
///
/// When `shape` is a compile-time constant we fold the entry to an
/// `arith.constant` (no device traffic). Otherwise we slice the entry to a
/// rank-0 tensor and read it through `hip.readback_scalar` (D2H + stream sync)
/// so the host observes the value the producing kernel actually wrote. Mirrors
/// the same fix in RangeConversion / ReshapeConversion.
///
/// Before (runtime shape, incorrect):
///   %d = tensor.extract %shape[%idx] : tensor<4xi64>   // host load of dev mem
///   %i = arith.index_cast %d : i64 to index
/// After:
///   %s   = tensor.extract_slice %shape[%idx] [1] [1] : tensor<4xi64> to
///   tensor<1xi64> %s0  = tensor.collapse_shape %s [] : tensor<1xi64> into
///   tensor<i64> %v   = hip.readback_scalar(%ctx, %s0 : tensor<i64>) -> i64 %i
///   = arith.index_cast %v : i64 to index
static mlir::Value readShapeEntryToIndex(mlir::PatternRewriter &rewriter,
                                         mlir::Location loc, mlir::Value ctx,
                                         mlir::Value shape, int64_t idx) {
  auto shapeType = mlir::cast<mlir::RankedTensorType>(shape.getType());
  mlir::Type elemTy = shapeType.getElementType();

  // Compile-time constant shape: fold the entry to a host constant.
  mlir::DenseElementsAttr dense;
  if (auto cst = shape.getDefiningOp<mlir::arith::ConstantOp>())
    dense = mlir::dyn_cast<mlir::DenseElementsAttr>(cst.getValue());
  else if (mlir::Operation *def = shape.getDefiningOp())
    if (def->getName().getStringRef() == "onnx.Constant")
      dense = def->getAttrOfType<mlir::DenseElementsAttr>("value");
  if (dense && idx < dense.getNumElements()) {
    if (mlir::isa<mlir::IntegerType>(elemTy)) {
      auto it = dense.getValues<llvm::APInt>().begin();
      int64_t v = (*(it + idx)).getSExtValue();
      return mlir::arith::ConstantIndexOp::create(rewriter, loc, v);
    }
  }

  // Runtime (possibly GPU-computed) shape: slice to rank-0, synchronized
  // readback, then cast to index.
  llvm::SmallVector<mlir::OpFoldResult> offsets{rewriter.getIndexAttr(idx)};
  llvm::SmallVector<mlir::OpFoldResult> sizes{rewriter.getIndexAttr(1)};
  llvm::SmallVector<mlir::OpFoldResult> strides{rewriter.getIndexAttr(1)};
  mlir::Value entry1d = mlir::tensor::ExtractSliceOp::create(
      rewriter, loc, shape, offsets, sizes, strides);
  mlir::Value entry0d = mlir::tensor::CollapseShapeOp::create(
      rewriter, loc, mlir::RankedTensorType::get({}, elemTy), entry1d,
      llvm::ArrayRef<mlir::ReassociationIndices>{});
  mlir::Value scalar =
      ReadbackScalarOp::create(rewriter, loc, elemTy, ctx, entry0d).getResult();
  return mlir::arith::IndexCastOp::create(rewriter, loc,
                                          rewriter.getIndexType(), scalar);
}

/// onnx.Expand -> hip.expand
///
/// We trust the result type produced by ONNX shape inference. For dynamic
/// dims in the result we extract the corresponding entry from the `shape`
/// input tensor (right-aligned with the result rank, NumPy-style); leading
/// dims that are absent from `shape` fall back to the matching input dim.
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
      mlir::Value dim;
      if (shapeIdx >= 0) {
        // The shape entry may be GPU-computed; read it back with a stream sync
        // (constants fold) instead of a bare host load of device memory. See
        // readShapeEntryToIndex for the correctness rationale.
        dim = readShapeEntryToIndex(rewriter, loc, context, shape, shapeIdx);
      } else {
        int64_t inputIdx = i - (resultRank - inputRank);
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
