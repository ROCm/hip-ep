/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

// Highest output rank the HIP expand kernel accepts.
constexpr int64_t kMaxBroadcastRank = 8;

// Integer values of a constant `repeats`, as an inline constant or a carrier
// that still holds its payload.
static bool readConstantRepeats(Value repeats, SmallVectorImpl<int64_t> &out) {
  DenseElementsAttr dense = getConstantDense(repeats);
  if (auto carrier = repeats.getDefiningOp<ConstantOp>())
    dense = dyn_cast_or_null<DenseElementsAttr>(carrier.getValueAttr());
  auto ints = dyn_cast_or_null<DenseIntElementsAttr>(dense);
  if (!ints)
    return false;
  for (const APInt &value : ints.getValues<APInt>())
    out.push_back(value.getSExtValue());
  return true;
}

/// A static tile with constant repeats is a broadcast. Each repeated extent
/// above one gets a unit dim in front of it, every repeated unit dim is
/// broadcast to its repeat count, and each repeat is merged back with the
/// extent it repeats:
///
///   %y = onnx.Tile(%x, repeats = [2, 4, 1])
///          : (tensor<2x1x3xf32>) -> tensor<4x4x3xf32>
///
/// becomes
///
///   %e = tensor.expand_shape %x [[0, 1], [2], [3]]
///          : tensor<2x1x3xf32> into tensor<1x2x1x3xf32>
///   %b = hip.expand(%ctx) ins(%e, %shape = [2, 2, 4, 3])
///          -> tensor<2x2x4x3xf32>
///   %y = tensor.collapse_shape %b [[0, 1], [2], [3]]
///          : tensor<2x2x4x3xf32> into tensor<4x4x3xf32>
///
/// The reshapes and the broadcast are views that a fused rocMLIR kernel folds
/// into its index maps; outside one, the broadcast runs as the expand kernel.
/// Returns failure, with the IR unchanged, for anything else.
static FailureOr<Value> tileAsBroadcast(PatternRewriter &rewriter, Location loc,
                                        Value context, Value input,
                                        Value repeats,
                                        RankedTensorType resultType) {
  auto inputType = dyn_cast<RankedTensorType>(input.getType());
  if (!inputType || !inputType.hasStaticShape() ||
      !resultType.hasStaticShape() ||
      inputType.getRank() != resultType.getRank() ||
      inputType.getElementType() != resultType.getElementType())
    return failure();
  SmallVector<int64_t> counts;
  if (!readConstantRepeats(repeats, counts) ||
      static_cast<int64_t>(counts.size()) != inputType.getRank())
    return failure();

  SmallVector<int64_t> expandedShape;
  SmallVector<int64_t> broadcastShape;
  SmallVector<ReassociationIndices> groups;
  for (auto [extent, count, out] :
       llvm::zip_equal(inputType.getShape(), counts, resultType.getShape())) {
    if (extent < 1 || count < 1 || extent * count != out)
      return failure();
    ReassociationIndices group;
    if (extent > 1 && count > 1) {
      group.push_back(expandedShape.size());
      expandedShape.push_back(1);
      broadcastShape.push_back(count);
    }
    group.push_back(expandedShape.size());
    expandedShape.push_back(extent);
    broadcastShape.push_back(extent == 1 ? count : extent);
    groups.push_back(std::move(group));
  }
  if (static_cast<int64_t>(broadcastShape.size()) > kMaxBroadcastRank)
    return failure();

  Type elementType = inputType.getElementType();
  bool insertsDims = expandedShape.size() != counts.size();
  Value source = input;
  if (insertsDims)
    source = tensor::ExpandShapeOp::create(
        rewriter, loc, RankedTensorType::get(expandedShape, elementType), input,
        groups);

  auto shapeType = RankedTensorType::get(
      {static_cast<int64_t>(broadcastShape.size())}, rewriter.getI64Type());
  Value shape = ConstantOp::create(
      rewriter, loc, shapeType,
      DenseIntElementsAttr::get(shapeType, ArrayRef<int64_t>(broadcastShape)));
  Value init =
      tensor::EmptyOp::create(rewriter, loc, broadcastShape, elementType);
  Value result = ExpandOp::create(rewriter, loc, context, source, shape, init)
                     ->getResult(0);
  if (insertsDims)
    result = tensor::CollapseShapeOp::create(rewriter, loc, resultType, result,
                                             groups);
  return result;
}

/// onnx.Tile -> hip.expand between reshapes when static (see tileAsBroadcast),
/// otherwise hip.tile.
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

    FailureOr<Value> broadcast =
        tileAsBroadcast(rewriter, loc, context, input, repeats, resultType);
    if (succeeded(broadcast)) {
      rewriter.replaceOp(op, *broadcast);
      return mlir::success();
    }

    // Trust shape inference: output rank == input rank, dims may be dynamic.
    // Use input as the source for any dynamic dim sizes.
    mlir::Value init = createEmptyTensor(rewriter, loc, resultType, input);

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
