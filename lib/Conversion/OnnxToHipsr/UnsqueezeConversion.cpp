/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- UnsqueezeConversion.cpp - Convert onnx.Unsqueeze to hipsr.compute -===//
//
// onnx.Unsqueeze inserts unit-length axes without moving data, so the body is a
// tensor.expand_shape. It bufferizes to a memref that aliases its source, so
// nothing runs on the device.
//
// The result type comes from the operands, not the type ONNX declared: the axes
// place the unit dimensions, the input gives the rest along with the element
// type and memory space. Reading the axes also resolves a dimension the
// declared type leaves dynamic.
//
// Before, with %axes the constant [-1]:
//   %r = "onnx.Unsqueeze"(%x, %axes)
//       : (tensor<2x3xf16, #hipsr.mem<device>>, tensor<1xi64>)
//       -> tensor<2x3x?xf16, #hipsr.mem<device>>
//
// After, with region types left out. The axes put the unit last, so the
// dimension the declared type left dynamic comes out as 1:
//   %init = hipsr.placeholder(%ctx) ins(%x)
//       {placeholder_type = #hipsr.placeholder_type<normal>}
//       : tensor<2x3x1xf16, #hipsr.mem<device>> shape_region {
//   ^bb0(%x_shape):
//     %c2 = arith.constant 2 : index
//     %c3 = arith.constant 3 : index
//     %c1 = arith.constant 1 : index
//     %s = shape.from_extents %c2, %c3, %c1
//     hipsr.shape_yield %s
//   }
//   %r = hipsr.compute(%ctx) ins(%x) outs(%init) {
//   ^bb0(%body_ctx, %in, %dest):
//     %out = tensor.expand_shape %in [[0], [1, 2]] output_shape [2, 3, 1]
//     hipsr.compute_yield %out
//   } : tensor<2x3x1xf16, #hipsr.mem<device>>
//
// A dynamic input dimension is read off %x_shape in the region.
//
//===----------------------------------------------------------------------===//

#include "OnnxToHipsrUtils.h"

#include "hip/Conversion/OnnxToHipsr/OnnxToHipsr.h"
#include "hip/Dialect/Hipsr/IR/HipsrOps.h"
#include "hip/Dialect/Hipsr/IR/HipsrShapeRegionPopulationUtils.h"
#include "hip/Dialect/Onnx/IR/OnnxOps.h"

#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallVectorExtras.h"

#include <cassert>
#include <optional>

namespace mlir {
namespace hipsr {
namespace {

//===----------------------------------------------------------------------===//
// The result type
//===----------------------------------------------------------------------===//

// The result type: the axes place the unit dimensions, the input gives the rest
// along with the element type and memory space.
FailureOr<RankedTensorType>
inferResultType(onnx::UnsqueezeOp op, RankedTensorType inputType, Value axes,
                ConversionPatternRewriter &rewriter) {
  // TODO: Support axes that only hold their value at runtime.
  DenseIntElementsAttr folded;
  if (!matchPattern(axes, m_Constant(&folded))) {
    return rewriter.notifyMatchFailure(op,
                                       "expected axes that fold to a constant");
  }
  int64_t resultRank = inputType.getRank() + folded.size();

  // ONNX counts a negative axis from the end of the result and leaves the order
  // of the axes open, so only which positions they name matters.
  llvm::SmallBitVector inserted(resultRank);
  for (const APInt &entry : folded.getValues<APInt>()) {
    int64_t axis = entry.getSExtValue();
    if (axis < 0) {
      axis += resultRank;
    }
    if (axis < 0 || axis >= resultRank) {
      return rewriter.notifyMatchFailure(
          op, "expected every axis within the result rank");
    }
    if (inserted.test(axis)) {
      return rewriter.notifyMatchFailure(op, "expected no repeated axis");
    }
    inserted.set(axis);
  }

  // An inserted position holds a unit, and the input's dimensions fill the rest
  // in order.
  SmallVector<int64_t> dims;
  dims.reserve(resultRank);
  int64_t inputAxis = 0;
  for (int64_t axis : llvm::seq<int64_t>(0, resultRank)) {
    dims.push_back(inserted[axis] ? 1 : inputType.getDimSize(inputAxis++));
  }
  return RankedTensorType::get(dims, inputType.getElementType(),
                               inputType.getEncoding());
}

//===----------------------------------------------------------------------===//
// The destination
//===----------------------------------------------------------------------===//

// The input's dimensions for the expand's inference: a stated one is a
// constant, a dynamic one is read off the input's shape.
SmallVector<OpFoldResult> buildInputDims(OpBuilder &builder, Location loc,
                                         Value inputShape,
                                         RankedTensorType inputType) {
  return llvm::map_to_vector(
      llvm::seq<int64_t>(0, inputType.getRank()),
      [&](int64_t axis) -> OpFoldResult {
        if (!inputType.isDynamicDim(axis)) {
          return builder.getIndexAttr(inputType.getDimSize(axis));
        }
        // shape.get_extent returns a `!shape.size`, which converts to index
        // because it can hold an error.
        Value size = shape::GetExtentOp::create(builder, loc, inputShape, axis);
        Value dim = shape::SizeToIndexOp::create(builder, loc,
                                                 builder.getIndexType(), size);
        return dim;
      });
}

// One dimension per result axis, from tensor.expand_shape's own inference: it
// places the input's dimensions and takes the rest from the result type. The
// division it would need is by the unit axes, so it folds away and leaves an
// unused constant for canonicalization.
void populateShapeRegion(OpBuilder &builder, PlaceholderOp placeholder,
                         RankedTensorType inputType,
                         RankedTensorType resultType,
                         ArrayRef<ReassociationIndices> reassociation) {
  OpBuilder::InsertionGuard guard(builder);
  Block &block = createPlaceholderShapeBlock(builder, placeholder);
  builder.setInsertionPointToStart(&block);

  Location loc = placeholder.getLoc();
  Value inputShape = PlaceholderShapeRegionArgs{block}.in(0);
  SmallVector<OpFoldResult> inputDims =
      buildInputDims(builder, loc, inputShape, inputType);
  FailureOr<SmallVector<OpFoldResult>> resultDims =
      tensor::ExpandShapeOp::inferOutputShape(builder, loc, resultType,
                                              reassociation, inputDims);
  assert(succeeded(resultDims) &&
         "a group holds one input dimension, so only it can be dynamic");
  Value shape = shape::FromExtentsOp::create(
      builder, loc, shape::ShapeType::get(builder.getContext()),
      getValueOrCreateConstantIndexOp(builder, loc, *resultDims));
  ShapeYieldOp::create(builder, loc, ValueRange{shape});
}

//===----------------------------------------------------------------------===//
// The compute body
//===----------------------------------------------------------------------===//

void populateComputeBody(OpBuilder &builder, ComputeOp computeOp,
                         ArrayRef<ReassociationIndices> reassociation) {
  OpBuilder::InsertionGuard guard(builder);
  Location loc = computeOp.getLoc();

  // The body's arguments are the operands: ctx, the inputs, then the outputs.
  TypeRange argTypes = computeOp->getOperandTypes();
  SmallVector<Location> argLocs(argTypes.size(), loc);
  Block *body =
      builder.createBlock(&computeOp.getBody(), {}, argTypes, argLocs);
  builder.setInsertionPointToStart(body);

  Value input = body->getArgument(1);
  auto resultType =
      cast<RankedTensorType>(body->getArguments().back().getType());
  FailureOr<SmallVector<OpFoldResult>> resultDims =
      tensor::ExpandShapeOp::inferOutputShape(
          builder, loc, resultType, reassociation,
          tensor::getMixedSizes(builder, loc, input));
  assert(succeeded(resultDims) &&
         "a group holds one input dimension, so only it can be dynamic");
  Value expanded = tensor::ExpandShapeOp::create(
      builder, loc, resultType, input, reassociation, *resultDims);
  ComputeYieldOp::create(builder, loc, ValueRange{expanded});
}

//===----------------------------------------------------------------------===//
// The pattern
//===----------------------------------------------------------------------===//

struct UnsqueezeToHipsr : public OpConversionPattern<onnx::UnsqueezeOp> {
  // The converter goes unused: the result type depends on the axes operand,
  // which a type conversion never sees. Taken for a uniform setup.
  UnsqueezeToHipsr(const TypeConverter &, MLIRContext *ctx)
      : OpConversionPattern(ctx) {}

  LogicalResult
  matchAndRewrite(onnx::UnsqueezeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getData();
    auto inputType = dyn_cast<RankedTensorType>(input.getType());
    if (!inputType) {
      return rewriter.notifyMatchFailure(op, "expected a ranked tensor input");
    }
    FailureOr<RankedTensorType> inferred =
        inferResultType(op, inputType, adaptor.getAxes(), rewriter);
    if (failed(inferred)) {
      return failure();
    }
    RankedTensorType resultType = *inferred;

    // An unsqueeze with no axes inserts nothing and needs no destination.
    if (inputType == resultType) {
      rewriter.replaceOp(op, input);
      return success();
    }
    std::optional<SmallVector<ReassociationIndices>> reassociation =
        getReassociationIndicesForReshape(inputType, resultType);
    assert(reassociation &&
           "the result is the input's dimensions with unit axes inserted");
    FailureOr<Value> ctx = getHipsrContextArg(op, rewriter);
    if (failed(ctx)) {
      return failure();
    }

    // The region needs only constants and the input's shape, so a normal
    // placeholder is enough.
    Location loc = op.getLoc();
    auto init =
        PlaceholderOp::create(rewriter, loc, TypeRange{resultType}, *ctx,
                              ValueRange{input}, PlaceholderType::Normal);
    populateShapeRegion(rewriter, init, inputType, resultType, *reassociation);

    auto computeOp =
        ComputeOp::create(rewriter, loc, TypeRange{resultType}, *ctx,
                          ValueRange{input}, ValueRange{init.getResult(0)});
    populateComputeBody(rewriter, computeOp, *reassociation);
    rewriter.replaceOp(op, computeOp.getResult(0));
    return success();
  }
};

} // namespace

void populateUnsqueezeConversionPatterns(const TypeConverter &typeConverter,
                                         RewritePatternSet &patterns,
                                         MLIRContext *ctx) {
  patterns.add<UnsqueezeToHipsr>(typeConverter, ctx);
}

} // namespace hipsr
} // namespace mlir
