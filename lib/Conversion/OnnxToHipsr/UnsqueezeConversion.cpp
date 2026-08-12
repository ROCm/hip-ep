/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- UnsqueezeConversion.cpp - Convert onnx.Unsqueeze to hipsr.compute -===//
//
// onnx.Unsqueeze inserts unit axes without moving data, so the compute body is
// a tensor.expand_shape, which bufferizes to a memref that aliases its source.
//
// The `axes` operand goes unread. Its values are already reflected in the
// result type, and it is often not a constant at this point: a graph slice can
// route a folded axes tensor across a part boundary, leaving a block argument
// behind. Matching the result extents against the input extents in order
// recovers the same grouping and works for both the opset-13 operand form and
// the older attribute form.
//
// Before:
//   %r = "onnx.Unsqueeze"(%x, %axes)
//       : (tensor<?x?xui8>, tensor<1xi64>) -> tensor<?x?x1xui8>
//
// After:
//   %init = hipsr.placeholder(%ctx) ins(%x : tensor<?x?xui8>)
//       {placeholder_type = #hipsr.placeholder_type<normal>} :
//       tensor<?x?x1xui8> shape_region {
//   ^bb0(%x_shape: !shape.shape):
//     %d0 = shape.get_extent %x_shape, 0
//     %i0 = shape.size_to_index %d0
//     %d1 = shape.get_extent %x_shape, 1
//     %i1 = shape.size_to_index %d1
//     %c1 = arith.constant 1 : index
//     %shape = shape.from_extents %i0, %i1, %c1 : index, index, index
//     hipsr.shape_yield %shape : !shape.shape
//   }
//   %r = hipsr.compute(%ctx) ins(%x : tensor<?x?xui8>)
//                            outs(%init : tensor<?x?x1xui8>) {
//   ^bb0(%body_ctx: !hipsr.context, %in: tensor<?x?xui8>,
//        %dest: tensor<?x?x1xui8>):
//     %e = tensor.expand_shape %in [[0], [1, 2]] output_shape [%d0, %d1, 1]
//         : tensor<?x?xui8> into tensor<?x?x1xui8>
//     hipsr.compute_yield %e : tensor<?x?x1xui8>
//   } : tensor<?x?x1xui8>
//
//===----------------------------------------------------------------------===//

#include "OnnxToHipsrUtils.h"

#include "hip/Conversion/OnnxToHipsr/OnnxToHipsr.h"
#include "hip/Dialect/Hipsr/IR/HipsrOps.h"
#include "hip/Dialect/Hipsr/IR/HipsrShapeRegionPopulationUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/PatternMatch.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>

namespace mlir {
namespace hipsr {
namespace {

// How the result axes line up with the input axes. `sourceAxis` holds the
// input axis each result axis came from, or `kInsertedAxis` for an axis the
// unsqueeze added.
constexpr int64_t kInsertedAxis = -1;

struct AxisMapping {
  ::llvm::SmallVector<::mlir::ReassociationIndices> reassociation;
  ::llvm::SmallVector<int64_t> sourceAxis;
};

// Walks the result axes and consumes an input axis whenever the two extents
// agree, treating every other result axis as inserted. Unsqueeze never
// reorders axes, so this in-order match recovers the grouping the `axes`
// operand describes. Fails when a result axis is neither a match nor a static
// unit extent, which means the operation is not an unsqueeze at all.
std::optional<AxisMapping>
mapUnsqueezeAxes(::mlir::RankedTensorType inputType,
                 ::mlir::RankedTensorType resultType) {
  AxisMapping mapping;
  mapping.reassociation.resize(inputType.getRank());
  mapping.sourceAxis.assign(resultType.getRank(), kInsertedAxis);

  int64_t inputAxis = 0;
  for (int64_t resultAxis : ::llvm::seq<int64_t>(0, resultType.getRank())) {
    if (inputAxis < inputType.getRank() &&
        inputType.getDimSize(inputAxis) == resultType.getDimSize(resultAxis)) {
      mapping.reassociation[inputAxis].push_back(resultAxis);
      mapping.sourceAxis[resultAxis] = inputAxis;
      ++inputAxis;
      continue;
    }
    if (resultType.getDimSize(resultAxis) != 1) {
      return std::nullopt;
    }
    // An inserted axis joins the group of the input axis it follows, or of the
    // first one when it leads. A rank-0 input has no group to join, which is
    // the empty reassociation tensor.expand_shape uses for that case.
    if (!mapping.reassociation.empty()) {
      mapping.reassociation[inputAxis == 0 ? 0 : inputAxis - 1].push_back(
          resultAxis);
    }
  }
  if (inputAxis != inputType.getRank()) {
    return std::nullopt;
  }
  return mapping;
}

// The result extents are the input's with unit extents spliced in, so each one
// is either a literal or a copy of the input extent it came from.
::llvm::SmallVector<::mlir::OpFoldResult> buildDestinationExtents(
    ::mlir::OpBuilder &builder, ::mlir::RankedTensorType resultType,
    ::llvm::ArrayRef<int64_t> sourceAxis,
    ::llvm::function_ref<::mlir::Value(int64_t)> getInputExtent) {
  return ::llvm::map_to_vector(::llvm::seq<int64_t>(0, resultType.getRank()),
                               [&](int64_t axis) -> ::mlir::OpFoldResult {
                                 if (!resultType.isDynamicDim(axis)) {
                                   return builder.getIndexAttr(
                                       resultType.getDimSize(axis));
                                 }
                                 return getInputExtent(sourceAxis[axis]);
                               });
}

void populateUnsqueezeShapeRegion(::mlir::OpBuilder &builder,
                                  PlaceholderOp placeholder,
                                  ::mlir::RankedTensorType resultType,
                                  ::llvm::ArrayRef<int64_t> sourceAxis) {
  ::mlir::OpBuilder::InsertionGuard guard(builder);
  ::mlir::Block &block = createPlaceholderShapeBlock(builder, placeholder);
  builder.setInsertionPointToStart(&block);

  ::mlir::Location loc = placeholder.getLoc();
  ::mlir::Value inputShape = PlaceholderShapeRegionArgs{block}.in(0);
  ::llvm::SmallVector<::mlir::OpFoldResult> extents = buildDestinationExtents(
      builder, resultType, sourceAxis, [&](int64_t axis) -> ::mlir::Value {
        return shapeExtentAsIndex(builder, loc, inputShape, axis);
      });

  ::llvm::SmallVector<::mlir::Value> extentValues = ::llvm::map_to_vector(
      extents, [&](::mlir::OpFoldResult extent) -> ::mlir::Value {
        return ::mlir::getValueOrCreateConstantIndexOp(builder, loc, extent);
      });
  ::mlir::Value shape = ::mlir::shape::FromExtentsOp::create(
      builder, loc, ::mlir::shape::ShapeType::get(builder.getContext()),
      extentValues);
  ShapeYieldOp::create(builder, loc, ::mlir::ValueRange{shape});
}

void populateUnsqueezeComputeBody(::mlir::OpBuilder &builder,
                                  ComputeOp computeOp,
                                  ::mlir::RankedTensorType resultType,
                                  const AxisMapping &mapping) {
  ::mlir::OpBuilder::InsertionGuard guard(builder);
  ::mlir::Location loc = computeOp.getLoc();
  ::mlir::Block &body = createComputeBodyBlock(builder, computeOp);
  ::mlir::Value input = computeBodyInput(body, 0);

  ::llvm::SmallVector<::mlir::OpFoldResult> outputShape =
      buildDestinationExtents(
          builder, resultType, mapping.sourceAxis,
          [&](int64_t axis) -> ::mlir::Value {
            ::mlir::Value index =
                ::mlir::arith::ConstantIndexOp::create(builder, loc, axis);
            return ::mlir::tensor::DimOp::create(builder, loc, input, index);
          });
  ::mlir::Value expanded = ::mlir::tensor::ExpandShapeOp::create(
      builder, loc, resultType, input, mapping.reassociation, outputShape);
  ComputeYieldOp::create(builder, loc, ::mlir::ValueRange{expanded});
}

struct UnsqueezeToHipsr : public ::mlir::RewritePattern {
  UnsqueezeToHipsr(::mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Unsqueeze", /*benefit=*/1, ctx) {}

  ::mlir::LogicalResult
  matchAndRewrite(::mlir::Operation *op,
                  ::mlir::PatternRewriter &rewriter) const override {
    if (op->getNumOperands() < 1 || op->getNumResults() != 1) {
      return rewriter.notifyMatchFailure(
          op, "expected at least one operand and a single result");
    }

    ::mlir::Value input = op->getOperand(0);
    auto inputType =
        ::mlir::dyn_cast<::mlir::RankedTensorType>(input.getType());
    auto resultType =
        ::mlir::dyn_cast<::mlir::RankedTensorType>(op->getResult(0).getType());
    if (!inputType || !resultType) {
      return rewriter.notifyMatchFailure(op, "expected ranked tensor types");
    }
    if (inputType.getElementType() != resultType.getElementType()) {
      return rewriter.notifyMatchFailure(
          op, "expected matching input and result element types");
    }
    // An unsqueeze that inserts nothing leaves the input untouched.
    if (inputType == resultType) {
      rewriter.replaceOp(op, input);
      return ::mlir::success();
    }
    if (resultType.getRank() <= inputType.getRank()) {
      return rewriter.notifyMatchFailure(op,
                                         "expected the result to gain a rank");
    }

    std::optional<AxisMapping> mapping =
        mapUnsqueezeAxes(inputType, resultType);
    if (!mapping) {
      return rewriter.notifyMatchFailure(
          op, "expected the result extents to be the input's plus unit axes");
    }

    ::mlir::FailureOr<::mlir::Value> ctx = getHipsrContextArg(op, rewriter);
    if (::mlir::failed(ctx)) {
      return ::mlir::failure();
    }

    ::mlir::Location loc = op->getLoc();
    auto init = PlaceholderOp::create(
        rewriter, loc, ::mlir::TypeRange{resultType}, *ctx,
        ::mlir::ValueRange{input}, PlaceholderType::Normal);
    populateUnsqueezeShapeRegion(rewriter, init, resultType,
                                 mapping->sourceAxis);

    auto computeOp = ComputeOp::create(
        rewriter, loc, ::mlir::TypeRange{resultType}, *ctx,
        ::mlir::ValueRange{input}, ::mlir::ValueRange{init.getResult(0)});
    populateUnsqueezeComputeBody(rewriter, computeOp, resultType, *mapping);

    rewriter.replaceOp(op, computeOp.getResult(0));
    return ::mlir::success();
  }
};

} // namespace

void populateUnsqueezeConversionPatterns(::mlir::RewritePatternSet &patterns,
                                         ::mlir::MLIRContext *ctx) {
  patterns.add<UnsqueezeToHipsr>(ctx);
}

} // namespace hipsr
} // namespace mlir
