//===- ReshapeConversion.cpp - ONNX-to-HIP Reshape conversion - *- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.  All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

//===----------------------------------------------------------------------===//
// Shape Operations Helpers (Reshape, Unsqueeze, Squeeze)
//===----------------------------------------------------------------------===//

/// True when the defining op of the axes value is compile-time known.
///
/// After constant externalization, small onnx.Constant / arith.constant tensors
/// become memref.get_global + bufferization.to_tensor (see OnnxToHip.cpp);
/// those must still count as constant axes. Arbitrary ToTensor(alloc) is not.
static bool axesDefOpIsCompileTimeKnown(Operation* axesDefOp) {
  if (isa<arith::ConstantOp>(axesDefOp) || axesDefOp->hasAttr("value"))
    return true;
  auto toTensor = dyn_cast<bufferization::ToTensorOp>(axesDefOp);
  if (!toTensor)
    return false;
  Operation* bufDef = toTensor.getBuffer().getDefiningOp();
  return bufDef && isa<memref::GetGlobalOp>(bufDef);
}

/// Validate common requirements for Unsqueeze/Squeeze operations.
/// Requires: 2 operands (data, axes), ranked tensors, matching element types,
/// and constant axes (dynamic axes would require runtime shape computation).
LogicalResult validateSqueezeUnsqueezeOp(Operation* op,
                                         PatternRewriter& rewriter,
                                         const char* tensorOpName, Value& data,
                                         Value& axes,
                                         RankedTensorType& inputType,
                                         RankedTensorType& outputType) {
  if (op->getNumOperands() != 2)
    return rewriter.notifyMatchFailure(op, "expected 2 operands (data, axes)");

  data = op->getOperand(0);
  axes = op->getOperand(1);

  inputType = dyn_cast<RankedTensorType>(data.getType());
  outputType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
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

  if (!axesDefOpIsCompileTimeKnown(axesDefOp)) {
    std::string msg =
        "axes must be compile-time constant (arith.constant, onnx.Constant, or "
        "memref.get_global + bufferization.to_tensor from constant "
        "externalization). Dynamic axes need runtime shape computation and "
        "cannot use zero-cost tensor.";
    msg += tensorOpName;
    msg += " approach";
    return rewriter.notifyMatchFailure(op, msg);
  }

  return success();
}

/// Build output shape for expand_shape operations.
/// Used by both Reshape and Unsqueeze when expanding dimensions.
///
/// For static dimensions: use compile-time size from outputType.
/// For dynamic dimensions: extract from input via DimOp, dividing out any
/// static dimensions in the same reassociation group.
llvm::SmallVector<OpFoldResult>
buildExpandShapeOutputShape(PatternRewriter& rewriter, Location loc, Value data,
                            RankedTensorType outputType,
                            llvm::ArrayRef<ReassociationIndices> reassoc) {
  int64_t outputRank = outputType.getRank();

  llvm::SmallVector<int64_t> outDimToInDim(outputRank, -1);
  for (auto [g, group] : llvm::enumerate(reassoc))
    for (int64_t idx : group)
      outDimToInDim[idx] = g;

  llvm::SmallVector<OpFoldResult> outputShape;
  for (int64_t i : llvm::seq<int64_t>(outputRank)) {
    if (!outputType.isDynamicDim(i)) {
      outputShape.push_back(rewriter.getIndexAttr(outputType.getDimSize(i)));
      continue;
    }

    int64_t srcDim = outDimToInDim[i];
    const auto& group = reassoc[srcDim];

    int64_t staticProduct = 1;
    for (int64_t idx : group)
      if (!outputType.isDynamicDim(idx))
        staticProduct *= outputType.getDimSize(idx);

    Value inputSize = tensor::DimOp::create(rewriter, loc, data, srcDim);
    if (staticProduct == 1) {
      outputShape.push_back(inputSize);
    } else {
      Value divisor =
          arith::ConstantIndexOp::create(rewriter, loc, staticProduct);
      Value dynSize = arith::DivUIOp::create(rewriter, loc, inputSize, divisor);
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
struct ReshapeToStdTensor : public RewritePattern {
  ReshapeToStdTensor(MLIRContext* ctx)
      : RewritePattern("onnx.Reshape", /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation* op,
                                PatternRewriter& rewriter) const override {
    Value data = op->getOperand(0);
    auto inputType = dyn_cast<RankedTensorType>(data.getType());
    auto outputType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!inputType || !outputType)
      return rewriter.notifyMatchFailure(op, "expected ranked tensor types");
    if (inputType.getElementType() != outputType.getElementType())
      return rewriter.notifyMatchFailure(op, "element type mismatch");

    Location loc = op->getLoc();
    int64_t inputRank = inputType.getRank();
    int64_t outputRank = outputType.getRank();

    // No-op: same type
    if (inputType == outputType) {
      rewriter.replaceOp(op, data);
      return success();
    }

    // Different rank: expand or collapse
    if (outputRank != inputRank) {
      auto reassocOpt =
          getReassociationIndicesForReshape(inputType, outputType);
      if (!reassocOpt)
        return rewriter.notifyMatchFailure(
            op, "cannot compute reshape reassociation");

      if (outputRank > inputRank) {
        // Expand: use shared helper to build output shape
        auto outputShape = buildExpandShapeOutputShape(rewriter, loc, data,
                                                       outputType, *reassocOpt);
        auto expandOp = tensor::ExpandShapeOp::create(
            rewriter, loc, outputType, data, *reassocOpt, outputShape);
        rewriter.replaceOp(op, expandOp.getResult());
      } else {
        // Collapse: no dynamic shape computation needed
        auto collapseOp = tensor::CollapseShapeOp::create(
            rewriter, loc, outputType, data, *reassocOpt);
        rewriter.replaceOp(op, collapseOp.getResult());
      }
      return success();
    }

    // Same rank, different shape: collapse to 1-D then expand.
    if (!inputType.hasStaticShape() || !outputType.hasStaticShape())
      return rewriter.notifyMatchFailure(
          op, "same-rank dynamic reshape not supported");

    int64_t numElems = inputType.getNumElements();
    if (numElems != outputType.getNumElements())
      return rewriter.notifyMatchFailure(op, "element count mismatch");

    auto flatType =
        RankedTensorType::get({numElems}, inputType.getElementType());

    ReassociationIndices allInputDims;
    for (int64_t i : llvm::seq<int64_t>(inputRank))
      allInputDims.push_back(i);
    llvm::SmallVector<ReassociationIndices> collapseReassoc = {allInputDims};
    auto collapsed = tensor::CollapseShapeOp::create(rewriter, loc, flatType,
                                                     data, collapseReassoc);

    ReassociationIndices allOutputDims;
    for (int64_t i : llvm::seq<int64_t>(outputRank))
      allOutputDims.push_back(i);
    llvm::SmallVector<ReassociationIndices> expandReassoc = {allOutputDims};
    llvm::SmallVector<OpFoldResult> flatOutputShape;
    for (int64_t i : llvm::seq<int64_t>(outputRank))
      flatOutputShape.push_back(
          rewriter.getIndexAttr(outputType.getDimSize(i)));
    auto expanded = tensor::ExpandShapeOp::create(
        rewriter, loc, outputType, collapsed.getResult(), expandReassoc,
        flatOutputShape);

    rewriter.replaceOp(op, expanded.getResult());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Unsqueeze -> standard tensor ops (zero-cost metadata reinterpretation)
//===----------------------------------------------------------------------===//

/// onnx.Unsqueeze -> tensor.expand_shape (zero-cost metadata operation).
struct UnsqueezeToStdTensor : public RewritePattern {
  UnsqueezeToStdTensor(MLIRContext* ctx)
      : RewritePattern("onnx.Unsqueeze", /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation* op,
                                PatternRewriter& rewriter) const override {
    Value data, axes;
    RankedTensorType inputType, outputType;
    if (auto result = validateSqueezeUnsqueezeOp(
            op, rewriter, "expand_shape", data, axes, inputType, outputType);
        failed(result))
      return result;

    auto reassocOpt = getReassociationIndicesForReshape(inputType, outputType);
    if (!reassocOpt)
      return rewriter.notifyMatchFailure(
          op, "cannot compute unsqueeze reassociation");

    Location loc = op->getLoc();
    auto outputShape = buildExpandShapeOutputShape(rewriter, loc, data,
                                                   outputType, *reassocOpt);
    auto expandOp = tensor::ExpandShapeOp::create(
        rewriter, loc, outputType, data, *reassocOpt, outputShape);
    rewriter.replaceOp(op, expandOp.getResult());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Squeeze -> standard tensor ops (zero-cost metadata reinterpretation)
//===----------------------------------------------------------------------===//

/// onnx.Squeeze -> tensor.collapse_shape (zero-cost metadata operation).
struct SqueezeToStdTensor : public RewritePattern {
  SqueezeToStdTensor(MLIRContext* ctx)
      : RewritePattern("onnx.Squeeze", /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation* op,
                                PatternRewriter& rewriter) const override {
    Value data, axes;
    RankedTensorType inputType, outputType;
    if (auto result = validateSqueezeUnsqueezeOp(
            op, rewriter, "collapse_shape", data, axes, inputType, outputType);
        failed(result))
      return result;

    auto reassocOpt = getReassociationIndicesForReshape(inputType, outputType);
    if (!reassocOpt)
      return rewriter.notifyMatchFailure(
          op, "cannot compute squeeze reassociation");

    Location loc = op->getLoc();
    auto collapseOp = tensor::CollapseShapeOp::create(rewriter, loc, outputType,
                                                      data, *reassocOpt);
    rewriter.replaceOp(op, collapseOp.getResult());
    return success();
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
struct SplitToStdTensor : public RewritePattern {
  SplitToStdTensor(MLIRContext* ctx)
      : RewritePattern("onnx.Split", /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation* op,
                                PatternRewriter& rewriter) const override {
    Value input = op->getOperand(0);
    auto inputType = dyn_cast<RankedTensorType>(input.getType());
    if (!inputType)
      return rewriter.notifyMatchFailure(op, "expected ranked tensor type");

    Location loc = op->getLoc();
    int64_t inputRank = inputType.getRank();
    unsigned numOutputs = op->getNumResults();

    // Edge case: single output is identity operation
    if (numOutputs == 1) {
      rewriter.replaceOp(op, input);
      return success();
    }

    // Extract axis attribute (default 0)
    int64_t axis = 0;
    if (auto axisAttr = op->getAttrOfType<IntegerAttr>("axis"))
      axis = axisAttr.getSInt();

    // Normalize negative axis
    if (axis < 0)
      axis += inputRank;
    if (axis < 0 || axis >= inputRank)
      return rewriter.notifyMatchFailure(op, "axis out of range");

    // Determine split mode: equal splits or custom splits
    llvm::SmallVector<OpFoldResult> splitLengths;
    bool isEqualSplit = true;

    if (op->getNumOperands() == 2) {
      // Check if second operand is onnx.NoValue (representing none/optional)
      Value splitInput = op->getOperand(1);
      auto splitDefOp = splitInput.getDefiningOp();

      // Handle onnx.NoValue: treat as equal split
      if (splitDefOp &&
          splitDefOp->getName().getStringRef() == "onnx.NoValue") {
        isEqualSplit = true;
      } else {
        // Custom splits: prefer compile-time constants when available, but
        // also support runtime split-length tensors (e.g. externalized
        // constants loaded via globals).
        DenseElementsAttr splitAttr;
        if (splitDefOp && isa<arith::ConstantOp>(splitDefOp))
          if (auto constOp = dyn_cast<arith::ConstantOp>(splitDefOp))
            splitAttr = dyn_cast<DenseElementsAttr>(constOp.getValue());
        if (!splitAttr && splitDefOp && splitDefOp->hasAttr("value"))
          splitAttr = dyn_cast<DenseElementsAttr>(splitDefOp->getAttr("value"));

        if (splitAttr) {
          // Extract split lengths as index attributes
          for (auto val : splitAttr.getValues<APInt>())
            splitLengths.push_back(rewriter.getIndexAttr(val.getSExtValue()));

          if (splitLengths.size() != numOutputs)
            return rewriter.notifyMatchFailure(
                op, "split lengths count must match number of outputs");

          // Validate sum of splits equals axis dimension (if axis is static)
          if (!inputType.isDynamicDim(axis)) {
            int64_t axisDimSize = inputType.getDimSize(axis);
            int64_t splitSum = 0;
            for (const auto& length : splitLengths) {
              if (auto attr = llvm::dyn_cast_if_present<Attribute>(length)) {
                auto intAttr = dyn_cast<IntegerAttr>(attr);
                if (intAttr) {
                  splitSum += intAttr.getInt();
                } else {
                  splitSum = -1;
                  break;
                }
              } else {
                splitSum = -1;
                break;
              }
            }
            if (splitSum >= 0 && splitSum != axisDimSize) {
              return rewriter.notifyMatchFailure(
                  op, "sum of split lengths must equal axis dimension size");
            }
          }
        } else {
          // Runtime path: read split lengths element-by-element.
          auto splitType = dyn_cast<RankedTensorType>(splitInput.getType());
          if (!splitType || splitType.getRank() != 1)
            return rewriter.notifyMatchFailure(
                op, "split input must be a rank-1 tensor");
          if (!splitType.getElementType().isIntOrIndex())
            return rewriter.notifyMatchFailure(
                op, "split input element type must be integer or index");
          if (!splitType.isDynamicDim(0) &&
              splitType.getDimSize(0) != static_cast<int64_t>(numOutputs))
            return rewriter.notifyMatchFailure(
                op, "split lengths count must match number of outputs");

          auto indexType = rewriter.getIndexType();
          for (unsigned i = 0; i < numOutputs; ++i) {
            Value idx = rewriter.create<arith::ConstantIndexOp>(loc, i);
            Value len = rewriter.create<tensor::ExtractOp>(loc, splitInput,
                                                           ValueRange{idx});
            if (len.getType() != indexType)
              len = rewriter.create<arith::IndexCastOp>(loc, indexType, len);
            splitLengths.push_back(len);
          }
        }

        isEqualSplit = false;
      }
    }

    // Handle equal splits
    if (isEqualSplit) {
      Value axisDim = rewriter.create<tensor::DimOp>(loc, input, axis);
      Value numOutputsVal =
          rewriter.create<arith::ConstantIndexOp>(loc, numOutputs);
      Value chunkSize =
          rewriter.create<arith::DivUIOp>(loc, axisDim, numOutputsVal);

      // For equal splits: all outputs except possibly the last have size
      // chunkSize The last output gets the remainder: axis_size -
      // (num_outputs-1) * chunkSize
      for (unsigned i = 0; i < numOutputs - 1; ++i)
        splitLengths.push_back(chunkSize);

      // Last chunk size = axis_size - sum(previous chunks)
      // = axis_size - (num_outputs - 1) * chunkSize
      // Note: numOutputs is always >= 2 here (single output case returns early)
      Value numPrevChunks =
          rewriter.create<arith::ConstantIndexOp>(loc, numOutputs - 1);
      Value prevTotal =
          rewriter.create<arith::MulIOp>(loc, chunkSize, numPrevChunks);
      Value lastChunkSize =
          rewriter.create<arith::SubIOp>(loc, axisDim, prevTotal);
      splitLengths.push_back(lastChunkSize);
    }

    // Generate extract_slice for each output
    llvm::SmallVector<Value> replacements;
    OpFoldResult currentOffset = rewriter.getIndexAttr(0);

    for (unsigned i = 0; i < numOutputs; ++i) {
      auto outputType = dyn_cast<RankedTensorType>(op->getResult(i).getType());
      if (!outputType)
        return rewriter.notifyMatchFailure(op, "expected ranked output type");

      // Track the actual slice size used for this output (for offset
      // calculation)
      OpFoldResult actualSliceSize;

      // Build offsets, sizes, strides arrays
      llvm::SmallVector<OpFoldResult> offsets, sizes, strides;
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
              if (auto attr =
                      llvm::dyn_cast_if_present<Attribute>(splitLengths[i])) {
                if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
                  if (intAttr.getInt() != staticSize) {
                    return rewriter.notifyMatchFailure(
                        op, "custom split length does not match static output "
                            "dimension");
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
            Value dimSize = rewriter.create<tensor::DimOp>(loc, input, dim);
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
      OperationState state(loc, tensor::ExtractSliceOp::getOperationName());
      tensor::ExtractSliceOp::build(rewriter, state, outputType, input, offsets,
                                    sizes, strides);
      auto sliceOp = rewriter.create(state);
      replacements.push_back(sliceOp->getResult(0));

      // Update offset for next slice
      if (i < numOutputs - 1) {
        // IMPORTANT: Use the actual slice size (which may differ from
        // splitLengths[i] when output type has static dimension). This ensures
        // correct offset calculation.
        Value offsetVal =
            getValueOrCreateConstantIndexOp(rewriter, loc, currentOffset);
        Value lengthVal =
            getValueOrCreateConstantIndexOp(rewriter, loc, actualSliceSize);
        Value newOffsetVal =
            rewriter.create<arith::AddIOp>(loc, offsetVal, lengthVal);
        currentOffset = newOffsetVal;
      }
    }

    rewriter.replaceOp(op, replacements);
    return success();
  }
};

} // namespace

void mlir::hip::populateReshapeConversionPatterns(RewritePatternSet& patterns,
                                                  MLIRContext* ctx) {
  patterns.add<ReshapeToStdTensor, UnsqueezeToStdTensor, SqueezeToStdTensor,
               SplitToStdTensor>(ctx);
}

} // namespace hip
} // namespace mlir
