/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

// The single in-range index of a constant scalar or one-element `indices`,
// normalized against `axisSize`, if it is that.
static std::optional<int64_t> singleConstantIndex(mlir::Value indices,
                                                  int64_t axisSize) {
  mlir::DenseElementsAttr dense = getConstantDense(indices);
  if (auto carrier = indices.getDefiningOp<ConstantOp>())
    dense =
        mlir::dyn_cast_or_null<mlir::DenseElementsAttr>(carrier.getValueAttr());
  auto ints = mlir::dyn_cast_or_null<mlir::DenseIntElementsAttr>(dense);
  if (!ints || ints.getNumElements() != 1)
    return std::nullopt;
  int64_t index = (*ints.getValues<llvm::APInt>().begin()).getSExtValue();
  if (index < 0)
    index += axisSize;
  if (index < 0 || index >= axisSize)
    return std::nullopt;
  return index;
}

/// A Gather of one constant index on static floating-point data reads a
/// size-1 slice of the gathered axis, which is a view rather than a kernel:
///
///   %y = onnx.Gather(%x : tensor<2x3x5x2xf32>, %c1 : tensor<i64>) {axis = 3}
///
/// becomes
///
///   %s = tensor.extract_slice %x[0, 0, 0, 1] [2, 3, 5, 1] [1, 1, 1, 1]
///   %y = tensor.collapse_shape %s [[0], [1], [2, 3]]  // tensor<2x3x5xf32>
///
/// A one-element 1-D index keeps the unit dim, so only the slice is emitted.
/// Integer data is left to hip.gather, so shape-vector arithmetic keeps the
/// folds that match it. A slice no kernel absorbs becomes a copy, and for a
/// small result a device-to-device copy costs more than the gather launch, so
/// results under kMinSliceElements keep hip.gather.
static constexpr int64_t kMinSliceElements = 4096;

static mlir::FailureOr<mlir::Value>
gatherAsSlice(mlir::PatternRewriter &rewriter, mlir::Location loc,
              mlir::Value data, mlir::Value indices, int64_t axis,
              mlir::RankedTensorType resultType) {
  auto dataType = mlir::dyn_cast<mlir::RankedTensorType>(data.getType());
  auto indicesType = mlir::dyn_cast<mlir::RankedTensorType>(indices.getType());
  if (!dataType || !indicesType || !dataType.hasStaticShape() ||
      !indicesType.hasStaticShape() || !resultType.hasStaticShape() ||
      resultType.getNumElements() < kMinSliceElements ||
      !mlir::isa<mlir::FloatType>(dataType.getElementType()) ||
      resultType.getElementType() != dataType.getElementType())
    return mlir::failure();
  int64_t rank = dataType.getRank();
  if (axis < 0 || axis >= rank || indicesType.getNumElements() != 1 ||
      indicesType.getRank() > 1)
    return mlir::failure();
  std::optional<int64_t> index =
      singleConstantIndex(indices, dataType.getDimSize(axis));
  if (!index)
    return mlir::failure();

  llvm::SmallVector<int64_t> sliceShape(dataType.getShape());
  sliceShape[axis] = 1;
  bool keepsUnitDim = indicesType.getRank() == 1;
  llvm::SmallVector<int64_t> expectedShape(sliceShape);
  if (!keepsUnitDim)
    expectedShape.erase(expectedShape.begin() + axis);
  if (llvm::ArrayRef<int64_t>(expectedShape) != resultType.getShape())
    return mlir::failure();

  llvm::SmallVector<mlir::OpFoldResult> offsets(rank, rewriter.getIndexAttr(0));
  offsets[axis] = rewriter.getIndexAttr(*index);
  llvm::SmallVector<mlir::OpFoldResult> sizes;
  for (int64_t d : sliceShape)
    sizes.push_back(rewriter.getIndexAttr(d));
  llvm::SmallVector<mlir::OpFoldResult> strides(rank, rewriter.getIndexAttr(1));
  auto sliceType =
      mlir::RankedTensorType::get(sliceShape, dataType.getElementType());
  mlir::Value slice = mlir::tensor::ExtractSliceOp::create(
      rewriter, loc, sliceType, data, offsets, sizes, strides);
  if (keepsUnitDim)
    return slice;

  // Fold the unit dim into a neighbour; a rank-1 slice collapses to rank 0.
  llvm::SmallVector<mlir::ReassociationIndices> groups;
  if (rank > 1) {
    for (int64_t d = 0; d < rank; ++d) {
      if (d == axis)
        continue;
      groups.push_back({d});
    }
    if (axis == rank - 1)
      groups.back().push_back(axis);
    else
      groups[axis].insert(groups[axis].begin(), axis);
  }
  return mlir::Value(mlir::tensor::CollapseShapeOp::create(
      rewriter, loc, resultType, slice, groups));
}

//===----------------------------------------------------------------------===//
// ONNX Gather -> HIP Gather
//===----------------------------------------------------------------------===//

struct GatherToHip : public mlir::RewritePattern {
  GatherToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Gather", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto ctxOrFailure = getContextArg(op, rewriter);
    if (mlir::failed(ctxOrFailure))
      return rewriter.notifyMatchFailure(op, "missing context argument");
    mlir::Value context = *ctxOrFailure;

    mlir::Location loc = op->getLoc();
    mlir::Value data = op->getOperand(0);
    mlir::Value indices = op->getOperand(1);

    // Get axis attribute from ONNX Gather operation
    int64_t axis = op->getAttrOfType<mlir::IntegerAttr>("axis").getSInt();
    auto axisAttr = rewriter.getI64IntegerAttr(axis);

    // UnrankedOpGuard lets through a ranked result over unranked operands
    // (e.g. `tensor<*xi64>` indices from NonZero/TopK), whose rank the dim
    // loops below would read as garbage.
    auto resultType =
        mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
    auto dataType = mlir::dyn_cast<mlir::RankedTensorType>(data.getType());
    auto indicesType =
        mlir::dyn_cast<mlir::RankedTensorType>(indices.getType());
    if (!resultType || !dataType || !indicesType)
      return rewriter.notifyMatchFailure(op, "gather.unranked");

    // Normalize negative axis for dimension calculations only
    int64_t normalizedAxis = axis < 0 ? axis + dataType.getRank() : axis;

    mlir::FailureOr<mlir::Value> slice =
        gatherAsSlice(rewriter, loc, data, indices, normalizedAxis, resultType);
    if (mlir::succeeded(slice)) {
      rewriter.replaceOp(op, *slice);
      return mlir::success();
    }

    // Create output tensor with dynamic shape support
    // Output shape: [data[0:axis], indices.shape, data[axis+1:]]
    llvm::SmallVector<mlir::Value> dynSizes;
    int64_t outDimIdx = 0;

    // Copy dimensions before axis from data
    for (auto i : llvm::seq<int64_t>(0, normalizedAxis)) {
      if (outDimIdx < resultType.getRank() &&
          resultType.isDynamicDim(outDimIdx))
        dynSizes.push_back(mlir::tensor::DimOp::create(rewriter, loc, data, i));
      outDimIdx++;
    }
    // Copy all dimensions from indices
    for (auto i : llvm::seq<int64_t>(0, indicesType.getRank())) {
      if (outDimIdx < resultType.getRank() &&
          resultType.isDynamicDim(outDimIdx))
        dynSizes.push_back(
            mlir::tensor::DimOp::create(rewriter, loc, indices, i));
      outDimIdx++;
    }
    // Copy dimensions after axis from data
    for (auto i : llvm::seq<int64_t>(normalizedAxis + 1, dataType.getRank())) {
      if (outDimIdx < resultType.getRank() &&
          resultType.isDynamicDim(outDimIdx))
        dynSizes.push_back(mlir::tensor::DimOp::create(rewriter, loc, data, i));
      outDimIdx++;
    }

    mlir::Value init =
        mlir::tensor::EmptyOp::create(rewriter, loc, resultType.getShape(),
                                      resultType.getElementType(), dynSizes);

    // Create hip.gather operation
    auto gatherOp = mlir::hip::GatherOp::create(rewriter, loc, context, data,
                                                indices, init, axisAttr);

    rewriter.replaceOp(op, gatherOp->getResult(0));
    return mlir::success();
  }
};

} // namespace

void populateGatherConversionPatterns(RewritePatternSet &patterns,
                                      MLIRContext *ctx) {
  patterns.add<GatherToHip>(ctx);
}

} // namespace hip
} // namespace mlir
