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
/// Requires: 2 operands (data, axes), ranked tensors, matching element types,
/// and constant axes (dynamic axes would require runtime shape computation).
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

  auto axesDefOp = axes.getDefiningOp();
  if (!axesDefOp) {
    std::string msg =
        "axes must be defined by a constant operation (block "
        "argument not supported). Dynamic axes would require "
        "runtime shape computation and cannot use zero-cost tensor.";
    msg += tensorOpName;
    return rewriter.notifyMatchFailure(op, msg);
  }

  bool isConstant = mlir::isa<mlir::arith::ConstantOp>(axesDefOp) ||
                    axesDefOp->hasAttr("value");
  if (!isConstant) {
    std::string msg =
        "axes must be constant (arith.constant or onnx.Constant). "
        "Dynamic axes would require runtime shape computation and "
        "cannot use zero-cost tensor.";
    msg += tensorOpName;
    msg += " approach";
    return rewriter.notifyMatchFailure(op, msg);
  }

  return mlir::success();
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
      outDimToInDim[idx] = g;

  llvm::SmallVector<mlir::OpFoldResult> outputShape;
  for (int64_t i : llvm::seq<int64_t>(outputRank)) {
    if (!outputType.isDynamicDim(i)) {
      outputShape.push_back(rewriter.getIndexAttr(outputType.getDimSize(i)));
      continue;
    }

    int64_t srcDim = outDimToInDim[i];
    const auto &group = reassoc[srcDim];

    int64_t staticProduct = 1;
    for (int64_t idx : group)
      if (!outputType.isDynamicDim(idx))
        staticProduct *= outputType.getDimSize(idx);

    mlir::Value inputSize =
        mlir::tensor::DimOp::create(rewriter, loc, data, srcDim);
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

    auto reassocOpt =
        mlir::getReassociationIndicesForReshape(inputType, outputType);
    if (!reassocOpt)
      return rewriter.notifyMatchFailure(
          op, "cannot compute unsqueeze reassociation");

    mlir::Location loc = op->getLoc();
    auto outputShape = buildExpandShapeOutputShape(rewriter, loc, data,
                                                   outputType, *reassocOpt);
    auto expandOp = mlir::tensor::ExpandShapeOp::create(
        rewriter, loc, outputType, data, *reassocOpt, outputShape);
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

    auto reassocOpt =
        mlir::getReassociationIndicesForReshape(inputType, outputType);
    if (!reassocOpt)
      return rewriter.notifyMatchFailure(
          op, "cannot compute squeeze reassociation");

    mlir::Location loc = op->getLoc();
    auto collapseOp = mlir::tensor::CollapseShapeOp::create(
        rewriter, loc, outputType, data, *reassocOpt);
    rewriter.replaceOp(op, collapseOp.getResult());
    return mlir::success();
  }
};

//===----------------------------------------------------------------------===//
// Split -> standard tensor ops (zero-cost slice views)
//===----------------------------------------------------------------------===//

/// onnx.Split -> tensor.extract_slice operations (zero-cost metadata
/// operation).
///
/// Split divides a tensor into multiple chunks along a specified axis.
/// This is a zero-cost operation: it creates views into the input tensor
/// without copying data. We lower to standard MLIR tensor.extract_slice ops
/// which bufferize to memref.subview (zero-copy alias) and then to LLVM
/// pointer arithmetic. No HIP kernel is needed.
struct SplitToStdTensor : public mlir::RewritePattern {
  SplitToStdTensor(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Split", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::Value input = op->getOperand(0);
    auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(input.getType());
    if (!inputType)
      return rewriter.notifyMatchFailure(op, "expected ranked tensor type");

    mlir::Location loc = op->getLoc();
    int64_t inputRank = inputType.getRank();
    unsigned numOutputs = op->getNumResults();

    // Edge case: single output is identity operation
    if (numOutputs == 1) {
      rewriter.replaceOp(op, input);
      return mlir::success();
    }

    // Extract axis attribute (default 0)
    int64_t axis = 0;
    if (auto axisAttr = op->getAttrOfType<mlir::IntegerAttr>("axis"))
      axis = axisAttr.getSInt();

    // Normalize negative axis
    if (axis < 0)
      axis += inputRank;
    if (axis < 0 || axis >= inputRank)
      return rewriter.notifyMatchFailure(op, "axis out of range");

    // Determine split mode: equal splits or custom splits
    llvm::SmallVector<mlir::OpFoldResult> splitLengths;
    bool isEqualSplit = true;

    if (op->getNumOperands() == 2) {
      // Check if second operand is onnx.NoValue (representing none/optional)
      mlir::Value splitInput = op->getOperand(1);
      auto splitDefOp = splitInput.getDefiningOp();

      // Handle onnx.NoValue: treat as equal split
      if (splitDefOp &&
          splitDefOp->getName().getStringRef() == "onnx.NoValue") {
        isEqualSplit = true;
      } else {
        // Custom splits: extract split lengths from second input (must be
        // constant)
        if (!splitDefOp)
          return rewriter.notifyMatchFailure(
              op, "split input must be defined by a constant operation");

        mlir::DenseElementsAttr splitAttr;
        if (auto constOp = mlir::dyn_cast<mlir::arith::ConstantOp>(splitDefOp))
          splitAttr =
              mlir::dyn_cast<mlir::DenseElementsAttr>(constOp.getValue());
        else if (splitDefOp->hasAttr("value"))
          splitAttr = mlir::dyn_cast<mlir::DenseElementsAttr>(
              splitDefOp->getAttr("value"));

        if (!splitAttr)
          return rewriter.notifyMatchFailure(
              op, "split input must be a constant dense tensor");

        // Extract split lengths as index attributes
        for (auto val : splitAttr.getValues<int64_t>())
          splitLengths.push_back(rewriter.getIndexAttr(val));

        if (splitLengths.size() != numOutputs)
          return rewriter.notifyMatchFailure(
              op, "split lengths count must match number of outputs");

        // Validate sum of splits equals axis dimension (if axis is static)
        if (!inputType.isDynamicDim(axis)) {
          int64_t axisDimSize = inputType.getDimSize(axis);
          int64_t splitSum = 0;
          for (const auto &length : splitLengths) {
            if (auto attr =
                    llvm::dyn_cast_if_present<mlir::Attribute>(length)) {
              auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(attr);
              if (intAttr) {
                splitSum += intAttr.getInt();
              } else {
                // Non-integer attribute, can't validate
                splitSum = -1;
                break;
              }
            } else {
              // Dynamic split length, can't validate at compile time
              splitSum = -1;
              break;
            }
          }
          if (splitSum >= 0 && splitSum != axisDimSize) {
            return rewriter.notifyMatchFailure(
                op, "sum of split lengths must equal axis dimension size");
          }
        }

        isEqualSplit = false;
      }
    }

    // Handle equal splits
    if (isEqualSplit) {
      mlir::Value axisDim =
          rewriter.create<mlir::tensor::DimOp>(loc, input, axis);
      mlir::Value numOutputsVal =
          rewriter.create<mlir::arith::ConstantIndexOp>(loc, numOutputs);
      mlir::Value chunkSize =
          rewriter.create<mlir::arith::DivUIOp>(loc, axisDim, numOutputsVal);

      // For equal splits: all outputs except possibly the last have size
      // chunkSize The last output gets the remainder: axis_size -
      // (num_outputs-1) * chunkSize
      for (unsigned i = 0; i < numOutputs - 1; ++i)
        splitLengths.push_back(chunkSize);

      // Last chunk size = axis_size - sum(previous chunks)
      // = axis_size - (num_outputs - 1) * chunkSize
      // Note: numOutputs is always >= 2 here (single output case returns early)
      mlir::Value numPrevChunks =
          rewriter.create<mlir::arith::ConstantIndexOp>(loc, numOutputs - 1);
      mlir::Value prevTotal =
          rewriter.create<mlir::arith::MulIOp>(loc, chunkSize, numPrevChunks);
      mlir::Value lastChunkSize =
          rewriter.create<mlir::arith::SubIOp>(loc, axisDim, prevTotal);
      splitLengths.push_back(lastChunkSize);
    }

    // Generate extract_slice for each output
    llvm::SmallVector<mlir::Value> replacements;
    mlir::OpFoldResult currentOffset = rewriter.getIndexAttr(0);

    for (unsigned i = 0; i < numOutputs; ++i) {
      auto outputType =
          mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(i).getType());
      if (!outputType)
        return rewriter.notifyMatchFailure(op, "expected ranked output type");

      // Track the actual slice size used for this output (for offset calculation)
      mlir::OpFoldResult actualSliceSize;

      // Build offsets, sizes, strides arrays
      llvm::SmallVector<mlir::OpFoldResult> offsets, sizes, strides;
      for (int64_t dim = 0; dim < inputRank; ++dim) {
        if (dim == axis) {
          // Axis dimension: use computed offset and split length
          offsets.push_back(currentOffset);

          // For the size: if the output dimension is static, use static attr;
          // otherwise use the dynamic value
          if (!outputType.isDynamicDim(dim)) {
            int64_t staticSize = outputType.getDimSize(dim);
            sizes.push_back(rewriter.getIndexAttr(staticSize));
            actualSliceSize = rewriter.getIndexAttr(staticSize);

            // Validate consistency: static output size must match split length
            if (!isEqualSplit) {
              // For custom splits, verify the split length matches
              if (auto attr = llvm::dyn_cast_if_present<mlir::Attribute>(splitLengths[i])) {
                if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(attr)) {
                  if (intAttr.getInt() != staticSize) {
                    return rewriter.notifyMatchFailure(
                        op, "custom split length does not match static output dimension");
                  }
                }
              }
            }
          } else {
            sizes.push_back(splitLengths[i]);
            actualSliceSize = splitLengths[i];
          }
        } else {
          // Other dimensions: identity (offset=0, size=dim_size, stride=1)
          offsets.push_back(rewriter.getIndexAttr(0));
          if (outputType.isDynamicDim(dim)) {
            mlir::Value dimSize =
                rewriter.create<mlir::tensor::DimOp>(loc, input, dim);
            sizes.push_back(dimSize);
          } else {
            sizes.push_back(rewriter.getIndexAttr(outputType.getDimSize(dim)));
          }
        }
        strides.push_back(rewriter.getIndexAttr(1));
      }

      // Create extract_slice operation using OpBuilder
      // This will properly decompose OpFoldResults into static attrs and
      // dynamic operands
      mlir::OperationState state(
          loc, mlir::tensor::ExtractSliceOp::getOperationName());
      mlir::tensor::ExtractSliceOp::build(rewriter, state, outputType, input,
                                          offsets, sizes, strides);
      auto sliceOp = rewriter.create(state);
      replacements.push_back(sliceOp->getResult(0));

      // Update offset for next slice
      if (i < numOutputs - 1) {
        // IMPORTANT: Use the actual slice size (which may differ from splitLengths[i]
        // when output type has static dimension). This ensures correct offset calculation.
        mlir::Value offsetVal =
            mlir::getValueOrCreateConstantIndexOp(rewriter, loc, currentOffset);
        mlir::Value lengthVal =
            mlir::getValueOrCreateConstantIndexOp(rewriter, loc, actualSliceSize);
        mlir::Value newOffsetVal =
            rewriter.create<mlir::arith::AddIOp>(loc, offsetVal, lengthVal);
        currentOffset = newOffsetVal;
      }
    }

    rewriter.replaceOp(op, replacements);
    return mlir::success();
  }
};

} // namespace

void mlir::hip::populateReshapeConversionPatterns(RewritePatternSet &patterns,
                                                  MLIRContext *ctx) {
  patterns.add<ReshapeToStdTensor, UnsqueezeToStdTensor, SqueezeToStdTensor,
               SplitToStdTensor>(ctx);
}

} // namespace hip
} // namespace mlir
