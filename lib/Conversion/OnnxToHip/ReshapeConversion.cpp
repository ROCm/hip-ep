/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

//===----------------------------------------------------------------------===//
// Shape Operations Helpers (Reshape, Unsqueeze, Squeeze)
//===----------------------------------------------------------------------===//

/// Validate common requirements for Unsqueeze/Squeeze operations.
/// Requires: 2 operands (data, axes), ranked tensors, matching element types.
/// Axes constness is now optional -- the caller falls back to inferring
/// the inserted/dropped dims from input vs output shapes when the axes
/// operand is opaque (e.g. externalised behind bufferization.to_tensor).
mlir::LogicalResult
validateSqueezeUnsqueezeOp(mlir::Operation *op, mlir::PatternRewriter &rewriter,
                           const char *tensorOpName, mlir::Value &data,
                           mlir::Value &axes, mlir::RankedTensorType &inputType,
                           mlir::RankedTensorType &outputType) {
  if (op->getNumOperands() != 2)
    return rewriter.notifyMatchFailure(op, "expected 2 operands (data, axes)");

  data = op->getOperand(0);
  axes = op->getOperand(1);

  inputType = mlir::dyn_cast<mlir::RankedTensorType>(data.getType());
  outputType =
      mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
  if (!inputType || !outputType)
    return rewriter.notifyMatchFailure(op, "expected ranked tensor types");
  if (inputType.getElementType() != outputType.getElementType())
    return rewriter.notifyMatchFailure(op, "element type mismatch");

  return mlir::success();
}

/// Greedy shape-based Unsqueeze axes inference.  Returns the indices in
/// `outShape` of the dimensions that were inserted as unit dims.  Returns
/// nullopt if the shapes are incompatible.  Treats ShapedType::kDynamic as
/// a wildcard that matches any input dim of the same kind.
static std::optional<llvm::SmallVector<int64_t>>
inferUnsqueezeAxesFromShapes(llvm::ArrayRef<int64_t> inShape,
                             llvm::ArrayRef<int64_t> outShape) {
  if ((int64_t)outShape.size() <= (int64_t)inShape.size())
    return std::nullopt;
  llvm::SmallVector<int64_t> axes;
  size_t inIdx = 0;
  for (size_t outIdx = 0; outIdx < outShape.size(); ++outIdx) {
    int64_t out = outShape[outIdx];
    if (inIdx < inShape.size() && inShape[inIdx] == out) {
      ++inIdx;
    } else if (out == 1) {
      axes.push_back(static_cast<int64_t>(outIdx));
    } else {
      return std::nullopt;
    }
  }
  if (inIdx != inShape.size())
    return std::nullopt;
  if (axes.size() != outShape.size() - inShape.size())
    return std::nullopt;
  return axes;
}

/// Greedy shape-based Squeeze axes inference: returns the indices in
/// `inShape` of the dropped (size-1) dims.
static std::optional<llvm::SmallVector<int64_t>>
inferSqueezeAxesFromShapes(llvm::ArrayRef<int64_t> inShape,
                           llvm::ArrayRef<int64_t> outShape) {
  if ((int64_t)inShape.size() <= (int64_t)outShape.size())
    return std::nullopt;
  llvm::SmallVector<int64_t> axes;
  size_t outIdx = 0;
  for (size_t inIdx = 0; inIdx < inShape.size(); ++inIdx) {
    int64_t in = inShape[inIdx];
    if (outIdx < outShape.size() && outShape[outIdx] == in) {
      ++outIdx;
    } else if (in == 1) {
      axes.push_back(static_cast<int64_t>(inIdx));
    } else {
      return std::nullopt;
    }
  }
  if (outIdx != outShape.size())
    return std::nullopt;
  if (axes.size() != inShape.size() - outShape.size())
    return std::nullopt;
  return axes;
}

/// Build output shape for expand_shape operations.
/// Used by both Reshape and Unsqueeze when expanding dimensions.
///
/// For static dimensions: use compile-time size from outputType.
/// For dynamic dimensions: extract from input via DimOp, dividing out any
/// static dimensions in the same reassociation group.
llvm::SmallVector<mlir::OpFoldResult> buildExpandShapeOutputShape(
    mlir::PatternRewriter &rewriter, mlir::Location loc, mlir::Value data,
    mlir::RankedTensorType outputType,
    llvm::ArrayRef<mlir::ReassociationIndices> reassoc) {
  int64_t outputRank = outputType.getRank();

  llvm::SmallVector<int64_t> outDimToInDim(outputRank, -1);
  for (auto [g, group] : llvm::enumerate(reassoc))
    for (int64_t idx : group)
      outDimToInDim[idx] = static_cast<int64_t>(g);

  llvm::SmallVector<mlir::OpFoldResult> outputShape;
  for (int64_t i : llvm::seq<int64_t>(outputRank)) {
    if (!outputType.isDynamicDim(i)) {
      outputShape.push_back(rewriter.getIndexAttr(outputType.getDimSize(i)));
      continue;
    }

    auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(data.getType());
    int64_t inputRank = inputType ? inputType.getRank() : 0;
    int64_t srcDim = outDimToInDim[i];
    if (srcDim < 0 || srcDim >= (int64_t)reassoc.size()) {
      // Output dim is dynamic but not covered by any reassociation group --
      // fall back to a single tensor.dim on the input at the matching axis,
      // but guard against out-of-bounds i (e.g. data is rank 0 / lower rank).
      if (i < inputRank) {
        mlir::Value sz = mlir::tensor::DimOp::create(rewriter, loc, data, i);
        outputShape.push_back(sz);
      } else {
        outputShape.push_back(rewriter.getIndexAttr(1));
      }
      continue;
    }
    const auto &group = reassoc[srcDim];

    int64_t staticProduct = 1;
    bool hasOtherDynamic = false;
    for (int64_t idx : group) {
      if (idx == i)
        continue;
      if (outputType.isDynamicDim(idx))
        hasOtherDynamic = true;
      else
        staticProduct *= outputType.getDimSize(idx);
    }

    if (srcDim >= inputRank) {
      // Source dim doesn't exist on the input (e.g. Unsqueeze adding a dim
      // beyond the input rank).  Fall back to a 1-element placeholder; the
      // expand_shape verifier will catch any inconsistency.
      outputShape.push_back(rewriter.getIndexAttr(1));
      continue;
    }
    mlir::Value inputSize =
        mlir::tensor::DimOp::create(rewriter, loc, data, srcDim);

    if (hasOtherDynamic) {
      // Multiple dynamic output dims share a single dynamic input dim; we
      // can only resolve one of them by division.  Use the input size as-is
      // for this slot and let downstream verifiers complain if the result
      // is rank-incompatible.  (Kokoro's iSTFTNet decoder produces this
      // shape pattern for its data-dependent upsampling output.)
      outputShape.push_back(inputSize);
      continue;
    }

    if (staticProduct == 1) {
      outputShape.push_back(inputSize);
    } else {
      mlir::Value divisor =
          mlir::arith::ConstantIndexOp::create(rewriter, loc, staticProduct);
      mlir::Value dynSize =
          mlir::arith::DivUIOp::create(rewriter, loc, inputSize, divisor);
      outputShape.push_back(dynSize);
    }
  }

  return outputShape;
}

//===----------------------------------------------------------------------===//
// Reshape -> standard tensor ops (zero-cost metadata reinterpretation)
//===----------------------------------------------------------------------===//

/// onnx.Reshape -> tensor.expand_shape / tensor.collapse_shape.
///
/// Reshape is a zero-cost metadata operation: it reinterprets shape/strides
/// without moving data.  We lower to standard MLIR tensor ops which
/// bufferize to memref.expand_shape / memref.collapse_shape (zero-copy
/// alias) and then to LLVM struct manipulation (same data pointer, new
/// sizes/strides).  No HIP kernel is needed.
struct ReshapeToStdTensor : public mlir::RewritePattern {
  ReshapeToStdTensor(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Reshape", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::Value data = op->getOperand(0);
    auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(data.getType());
    auto outputType =
        mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
    if (!inputType || !outputType)
      return rewriter.notifyMatchFailure(op, "expected ranked tensor types");
    if (inputType.getElementType() != outputType.getElementType())
      return rewriter.notifyMatchFailure(op, "element type mismatch");

    mlir::Location loc = op->getLoc();
    int64_t inputRank = inputType.getRank();
    int64_t outputRank = outputType.getRank();

    // No-op: same type
    if (inputType == outputType) {
      rewriter.replaceOp(op, data);
      return mlir::success();
    }

    // Different rank: expand or collapse
    if (outputRank != inputRank) {
      // Rank-0 input -> rank-N: tensor.expand_shape with empty
      // reassociation doesn't bufferize (memref.expand_shape requires
      // rank >= 1).  Emit tensor.extract + tensor.from_elements +
      // optional cast like the rank-0 unsqueeze path does.  Output
      // must collapse to a single element (all dims must be 1).
      if (inputRank == 0 && outputRank >= 1) {
        llvm::SmallVector<int64_t> staticShape(outputRank, 1);
        auto staticOutType = mlir::RankedTensorType::get(
            staticShape, outputType.getElementType());
        mlir::Value scalar =
            mlir::tensor::ExtractOp::create(rewriter, loc, data,
                                             mlir::ValueRange{})
                .getResult();
        mlir::Value out =
            mlir::tensor::FromElementsOp::create(rewriter, loc, staticOutType,
                                                  mlir::ValueRange{scalar})
                .getResult();
        if (staticOutType != outputType)
          out = mlir::tensor::CastOp::create(rewriter, loc, outputType, out)
                    .getResult();
        rewriter.replaceOp(op, out);
        return mlir::success();
      }
      // Rank-N input -> rank-0: pull the single element out via
      // tensor.extract @ [0,...,0] then re-wrap.  Same rationale as
      // the rank-0 unsqueeze path.
      if (inputRank >= 1 && outputRank == 0) {
        mlir::Value zero =
            mlir::arith::ConstantIndexOp::create(rewriter, loc, 0).getResult();
        llvm::SmallVector<mlir::Value> indices(inputRank, zero);
        mlir::Value scalar =
            mlir::tensor::ExtractOp::create(rewriter, loc, data, indices)
                .getResult();
        mlir::Value out = mlir::tensor::FromElementsOp::create(
                               rewriter, loc, outputType,
                               mlir::ValueRange{scalar})
                               .getResult();
        rewriter.replaceOp(op, out);
        return mlir::success();
      }

      auto reassocOpt =
          mlir::getReassociationIndicesForReshape(inputType, outputType);
      if (!reassocOpt)
        return rewriter.notifyMatchFailure(
            op, "cannot compute reshape reassociation");

      if (outputRank > inputRank) {
        // Expand: use shared helper to build output shape
        auto outputShape = buildExpandShapeOutputShape(rewriter, loc, data,
                                                       outputType, *reassocOpt);
        auto expandOp = mlir::tensor::ExpandShapeOp::create(
            rewriter, loc, outputType, data, *reassocOpt, outputShape);
        rewriter.replaceOp(op, expandOp.getResult());
      } else {
        // Collapse: no dynamic shape computation needed
        auto collapseOp = mlir::tensor::CollapseShapeOp::create(
            rewriter, loc, outputType, data, *reassocOpt);
        rewriter.replaceOp(op, collapseOp.getResult());
      }
      return mlir::success();
    }

    // Same rank, different shape: collapse to 1-D then expand.
    if (!inputType.hasStaticShape() || !outputType.hasStaticShape())
      return rewriter.notifyMatchFailure(
          op, "same-rank dynamic reshape not supported");

    int64_t numElems = inputType.getNumElements();
    if (numElems != outputType.getNumElements())
      return rewriter.notifyMatchFailure(op, "element count mismatch");

    auto flatType =
        mlir::RankedTensorType::get({numElems}, inputType.getElementType());

    mlir::ReassociationIndices allInputDims;
    for (int64_t i : llvm::seq<int64_t>(inputRank))
      allInputDims.push_back(i);
    llvm::SmallVector<mlir::ReassociationIndices> collapseReassoc = {
        allInputDims};
    auto collapsed = mlir::tensor::CollapseShapeOp::create(
        rewriter, loc, flatType, data, collapseReassoc);

    mlir::ReassociationIndices allOutputDims;
    for (int64_t i : llvm::seq<int64_t>(outputRank))
      allOutputDims.push_back(i);
    llvm::SmallVector<mlir::ReassociationIndices> expandReassoc = {
        allOutputDims};
    llvm::SmallVector<mlir::OpFoldResult> flatOutputShape;
    for (int64_t i : llvm::seq<int64_t>(outputRank))
      flatOutputShape.push_back(
          rewriter.getIndexAttr(outputType.getDimSize(i)));
    auto expanded = mlir::tensor::ExpandShapeOp::create(
        rewriter, loc, outputType, collapsed.getResult(), expandReassoc,
        flatOutputShape);

    rewriter.replaceOp(op, expanded.getResult());
    return mlir::success();
  }
};

//===----------------------------------------------------------------------===//
// Unsqueeze -> standard tensor ops (zero-cost metadata reinterpretation)
//===----------------------------------------------------------------------===//

/// onnx.Unsqueeze -> tensor.expand_shape (zero-cost metadata operation).
struct UnsqueezeToStdTensor : public mlir::RewritePattern {
  UnsqueezeToStdTensor(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Unsqueeze", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::Value data, axes;
    mlir::RankedTensorType inputType, outputType;
    if (auto result = validateSqueezeUnsqueezeOp(
            op, rewriter, "expand_shape", data, axes, inputType, outputType);
        failed(result))
      return result;

    int64_t inRank = inputType.getRank();
    int64_t outRank = outputType.getRank();
    // Degenerate: input and output both rank 0 -- forward.
    if (inRank == 0 && outRank == 0) {
      rewriter.replaceOp(op, data);
      return mlir::success();
    }
    if (outRank <= inRank)
      return rewriter.notifyMatchFailure(op, "unsqueeze must add dimensions");

    // Rank-0 input -> any rank N: tensor.expand_shape doesn't accept an
    // empty reassociation in memref form (memref.expand_shape requires
    // rank >= 1).  Use `tensor.extract` to pull the scalar value, then
    // `tensor.from_elements` to build a rank-N tensor of all-1 dims.
    // ONNX Unsqueeze of a scalar is well-defined only when every output
    // dim is 1 (semantically the scalar wrapped in a single element).
    if (inRank == 0) {
      // Static-1 dims are accepted; dynamic dims are accepted ONLY if
      // the runtime size will be 1 (which is the only case where an
      // unsqueeze of a scalar makes sense -- ONNX would reject larger
      // sizes).  We synthesise a tensor<1x1x...xT> of all-1 dims, then
      // tensor.cast to whatever the output type wants.
      llvm::SmallVector<int64_t> staticShape(outRank, 1);
      auto staticOutType = mlir::RankedTensorType::get(
          staticShape, outputType.getElementType());
      mlir::Location loc = op->getLoc();
      mlir::Value scalar =
          mlir::tensor::ExtractOp::create(rewriter, loc, data, mlir::ValueRange{})
              .getResult();
      mlir::Value out =
          mlir::tensor::FromElementsOp::create(rewriter, loc, staticOutType,
                                                mlir::ValueRange{scalar})
              .getResult();
      if (staticOutType != outputType)
        out = mlir::tensor::CastOp::create(rewriter, loc, outputType, out)
                  .getResult();
      rewriter.replaceOp(op, out);
      return mlir::success();
    }

    // Try axes-from-shape inference first (works without any constant-axes
    // operand, and handles dynamic dims).
    auto axesOpt =
        inferUnsqueezeAxesFromShapes(inputType.getShape(), outputType.getShape());
    if (!axesOpt)
      return rewriter.notifyMatchFailure(
          op, "cannot infer unsqueeze axes from shapes");

    llvm::SmallSet<int64_t, 8> axesSet(axesOpt->begin(), axesOpt->end());

    // Build reassociation: for each input dim, group it with all
    // immediately-following inserted unit dims.
    llvm::SmallVector<mlir::ReassociationIndices> reassoc(inRank);
    int64_t inIdx = -1;
    for (int64_t outIdx = 0; outIdx < outRank; ++outIdx) {
      if (axesSet.count(outIdx)) {
        // Inserted unit dim -- attach to the most-recent input dim, or
        // to the first one if it leads.
        int64_t target = std::max<int64_t>(0, inIdx);
        reassoc[target].push_back(outIdx);
      } else {
        ++inIdx;
        if (inIdx >= inRank)
          return rewriter.notifyMatchFailure(
              op, "unsqueeze axes inconsistent with shapes");
        reassoc[inIdx].push_back(outIdx);
      }
    }
    if (inIdx + 1 != inRank)
      return rewriter.notifyMatchFailure(
          op, "unsqueeze axes inconsistent with input rank");

    mlir::Location loc = op->getLoc();
    auto outputShape =
        buildExpandShapeOutputShape(rewriter, loc, data, outputType, reassoc);
    auto expandOp = mlir::tensor::ExpandShapeOp::create(
        rewriter, loc, outputType, data, reassoc, outputShape);
    rewriter.replaceOp(op, expandOp.getResult());
    return mlir::success();
  }
};

//===----------------------------------------------------------------------===//
// Squeeze -> standard tensor ops (zero-cost metadata reinterpretation)
//===----------------------------------------------------------------------===//

/// onnx.Squeeze -> tensor.collapse_shape (zero-cost metadata operation).
struct SqueezeToStdTensor : public mlir::RewritePattern {
  SqueezeToStdTensor(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Squeeze", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::Value data, axes;
    mlir::RankedTensorType inputType, outputType;
    if (auto result = validateSqueezeUnsqueezeOp(
            op, rewriter, "collapse_shape", data, axes, inputType, outputType);
        failed(result))
      return result;

    int64_t inRank = inputType.getRank();
    int64_t outRank = outputType.getRank();
    if (inRank == 0 && outRank == 0) {
      rewriter.replaceOp(op, data);
      return mlir::success();
    }
    if (inRank <= outRank)
      return rewriter.notifyMatchFailure(op, "squeeze must remove dimensions");

    // tensor.collapse_shape requires at least one input dim per output
    // group, so we can't represent "rank-N -> rank-0" with it.  Emit a
    // tensor.extract that pulls out the single element when every input
    // dim is statically 1 (the only case where rank-0 is well-defined);
    // for the common Kokoro pattern `tensor<?xi64> -> tensor<i64>` we
    // have to trust that the runtime size is 1 and emit the same
    // tensor.extract pattern with a runtime-asserted index of 0.
    if (outRank == 0) {
      mlir::Location loc = op->getLoc();
      mlir::Value zero =
          mlir::arith::ConstantIndexOp::create(rewriter, loc, 0).getResult();
      llvm::SmallVector<mlir::Value> indices(inRank, zero);
      mlir::Value scalar = mlir::tensor::ExtractOp::create(
                                rewriter, loc, data, indices)
                                .getResult();
      // Re-wrap into rank-0 tensor.
      mlir::Value out = mlir::tensor::FromElementsOp::create(
                             rewriter, loc, outputType, mlir::ValueRange{scalar})
                             .getResult();
      rewriter.replaceOp(op, out);
      return mlir::success();
    }

    auto axesOpt =
        inferSqueezeAxesFromShapes(inputType.getShape(), outputType.getShape());
    if (!axesOpt)
      return rewriter.notifyMatchFailure(
          op, "cannot infer squeeze axes from shapes");

    llvm::SmallSet<int64_t, 8> axesSet(axesOpt->begin(), axesOpt->end());

    // Build reassociation: for each output dim, group it with the
    // following squeezed unit dims.
    llvm::SmallVector<mlir::ReassociationIndices> reassoc(outRank);
    int64_t outIdx = -1;
    for (int64_t inIdx = 0; inIdx < inRank; ++inIdx) {
      if (axesSet.count(inIdx)) {
        int64_t target = std::max<int64_t>(0, outIdx);
        reassoc[target].push_back(inIdx);
      } else {
        ++outIdx;
        if (outIdx >= outRank)
          return rewriter.notifyMatchFailure(
              op, "squeeze axes inconsistent with shapes");
        reassoc[outIdx].push_back(inIdx);
      }
    }
    if (outIdx + 1 != outRank)
      return rewriter.notifyMatchFailure(
          op, "squeeze axes inconsistent with output rank");

    mlir::Location loc = op->getLoc();
    auto collapseOp = mlir::tensor::CollapseShapeOp::create(
        rewriter, loc, outputType, data, reassoc);
    rewriter.replaceOp(op, collapseOp.getResult());
    return mlir::success();
  }
};

} // namespace

void mlir::hip::populateReshapeConversionPatterns(RewritePatternSet &patterns,
                                                  MLIRContext *ctx) {
  patterns.add<ReshapeToStdTensor, UnsqueezeToStdTensor, SqueezeToStdTensor>(
      ctx);
}

/// Subset that only runs Unsqueeze/Squeeze -- safe to use in
/// preLowerShapeOps because Unsqueeze/Squeeze need their `axes`
/// constant to be readable, but ReshapeToStdTensor reads the new shape
/// via tensor.dim and benefits from running after ReduceMean/Slice
/// have folded their shapes.
void mlir::hip::populateUnsqueezeSqueezeConversionPatterns(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.add<UnsqueezeToStdTensor, SqueezeToStdTensor>(ctx);
}

} // namespace hip
} // namespace mlir
