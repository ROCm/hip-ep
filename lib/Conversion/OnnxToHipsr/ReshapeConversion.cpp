/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- ReshapeConversion.cpp - Convert onnx.Reshape to hipsr.compute -----===//
//
// onnx.Reshape reinterprets extents in row-major order without moving data.
// The conversion emits tensor.collapse_shape or tensor.expand_shape inside a
// hipsr.compute; both bufferize to memref forms that alias their source, so no
// kernel runs.
//
// The target shape operand is not read. Shape inference has already folded it
// into the result type, including any -1 entry, and deriving the destination
// from the result type keeps the whole shape a function of the input shape.
// That is what makes a normal placeholder sufficient: no host readback of the
// shape tensor, so no barrier.
//
// Because a reshape preserves the element count, a single dynamic result
// extent is the input's element count divided by the static result extents.
// Two dynamic result extents would not be recoverable that way, and the
// reassociation search cannot disambiguate them either, so they are rejected.
//
// Before:
//   %r = "onnx.Reshape"(%x, %shape)
//       : (tensor<?x4096xf16>, tensor<1xi64>) -> tensor<?xf16>
//
// After:
//   %init = hipsr.placeholder(%ctx) ins(%x : tensor<?x4096xf16>)
//       {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<?xf16>
//       shape_region {
//   ^bb0(%x_shape: !shape.shape):
//     %d0 = shape.get_extent %x_shape, 0
//     %c4096 = arith.constant 4096 : index
//     %n = arith.muli %d0, %c4096 : index
//     %shape = shape.from_extents %n : index
//     hipsr.shape_yield %shape : !shape.shape
//   }
//   %r = hipsr.compute(%ctx) ins(%x : tensor<?x4096xf16>)
//                            outs(%init : tensor<?xf16>) {
//   ^bb0(%body_ctx: !hipsr.context, %in: tensor<?x4096xf16>,
//        %dest: tensor<?xf16>):
//     %flat = tensor.collapse_shape %in [[0, 1]]
//         : tensor<?x4096xf16> into tensor<?xf16>
//     hipsr.compute_yield %flat : tensor<?xf16>
//   } : tensor<?xf16>
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
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>

namespace mlir {
namespace hipsr {
namespace {

// Builds one extent per result axis. `getInputExtent` supplies a dynamic input
// extent, and differs between the two callers: the shape region reads a
// `!shape.shape` block argument, the compute body reads the input tensor.
::llvm::SmallVector<::mlir::OpFoldResult> buildDestinationExtents(
    ::mlir::OpBuilder &builder, ::mlir::Location loc,
    ::mlir::RankedTensorType inputType, ::mlir::RankedTensorType resultType,
    ::llvm::function_ref<::mlir::Value(int64_t)> getInputExtent) {
  ::llvm::SmallVector<::mlir::OpFoldResult> extents;
  extents.reserve(resultType.getRank());

  int64_t staticResultProduct = 1;
  std::optional<unsigned> dynamicAxis;
  for (int64_t axis : ::llvm::seq<int64_t>(0, resultType.getRank())) {
    if (resultType.isDynamicDim(axis)) {
      dynamicAxis = extents.size();
      extents.push_back(::mlir::OpFoldResult{});
      continue;
    }
    staticResultProduct *= resultType.getDimSize(axis);
    extents.push_back(builder.getIndexAttr(resultType.getDimSize(axis)));
  }
  if (!dynamicAxis) {
    return extents;
  }

  // Fold the static input extents into a single factor so only the dynamic
  // ones cost a multiply.
  int64_t staticInputProduct = 1;
  ::mlir::Value elementCount;
  for (int64_t axis : ::llvm::seq<int64_t>(0, inputType.getRank())) {
    if (!inputType.isDynamicDim(axis)) {
      staticInputProduct *= inputType.getDimSize(axis);
      continue;
    }
    ::mlir::Value extent = getInputExtent(axis);
    elementCount =
        elementCount
            ? ::mlir::arith::MulIOp::create(builder, loc, elementCount, extent)
            : extent;
  }
  if (staticInputProduct != 1 || !elementCount) {
    ::mlir::Value factor = ::mlir::arith::ConstantIndexOp::create(
        builder, loc, staticInputProduct);
    elementCount =
        elementCount
            ? ::mlir::arith::MulIOp::create(builder, loc, elementCount, factor)
            : factor;
  }
  if (staticResultProduct != 1) {
    ::mlir::Value divisor = ::mlir::arith::ConstantIndexOp::create(
        builder, loc, staticResultProduct);
    elementCount =
        ::mlir::arith::DivUIOp::create(builder, loc, elementCount, divisor);
  }
  extents[*dynamicAxis] = elementCount;
  return extents;
}

void populateReshapeShapeRegion(::mlir::OpBuilder &builder,
                                PlaceholderOp placeholder,
                                ::mlir::RankedTensorType inputType,
                                ::mlir::RankedTensorType resultType) {
  ::mlir::OpBuilder::InsertionGuard guard(builder);
  ::mlir::Block &block = createPlaceholderShapeBlock(builder, placeholder);
  builder.setInsertionPointToStart(&block);

  ::mlir::Location loc = placeholder.getLoc();
  ::mlir::Value inputShape = PlaceholderShapeRegionArgs{block}.in(0);
  ::llvm::SmallVector<::mlir::OpFoldResult> extents = buildDestinationExtents(
      builder, loc, inputType, resultType, [&](int64_t axis) -> ::mlir::Value {
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

void populateReshapeComputeBody(
    ::mlir::OpBuilder &builder, ComputeOp computeOp,
    ::mlir::RankedTensorType inputType, ::mlir::RankedTensorType resultType,
    ::llvm::ArrayRef<::mlir::ReassociationIndices> reassociation) {
  ::mlir::OpBuilder::InsertionGuard guard(builder);
  ::mlir::Location loc = computeOp.getLoc();
  ::mlir::Block &body = createComputeBodyBlock(builder, computeOp);
  ::mlir::Value input = computeBodyInput(body, 0);

  ::mlir::Value reshaped;
  if (inputType.getRank() > resultType.getRank()) {
    // The grouping alone determines the collapsed extents, so no extra
    // operands are needed.
    reshaped = ::mlir::tensor::CollapseShapeOp::create(builder, loc, resultType,
                                                       input, reassociation);
  } else {
    // Expanding splits one input extent over a group, which the input type
    // cannot pin down, so the op carries the destination extents.
    ::llvm::SmallVector<::mlir::OpFoldResult> outputShape =
        buildDestinationExtents(
            builder, loc, inputType, resultType,
            [&](int64_t axis) -> ::mlir::Value {
              ::mlir::Value index =
                  ::mlir::arith::ConstantIndexOp::create(builder, loc, axis);
              return ::mlir::tensor::DimOp::create(builder, loc, input, index);
            });
    reshaped = ::mlir::tensor::ExpandShapeOp::create(
        builder, loc, resultType, input, reassociation, outputShape);
  }
  ComputeYieldOp::create(builder, loc, ::mlir::ValueRange{reshaped});
}

struct ReshapeToHipsr : public ::mlir::RewritePattern {
  ReshapeToHipsr(::mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Reshape", /*benefit=*/1, ctx) {}

  ::mlir::LogicalResult
  matchAndRewrite(::mlir::Operation *op,
                  ::mlir::PatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 2 || op->getNumResults() != 1) {
      return rewriter.notifyMatchFailure(
          op, "expected two operands and a single result");
    }
    // With allowzero set, a 0 in the target shape is a literal extent instead
    // of a copy of the input's, which gives a different result shape.
    if (auto allowZero = op->getAttrOfType<::mlir::IntegerAttr>("allowzero")) {
      if (!allowZero.getValue().isZero()) {
        return rewriter.notifyMatchFailure(op, "expected allowzero = 0");
      }
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

    // An identity reshape reinterprets nothing, so it needs no destination.
    if (inputType == resultType) {
      rewriter.replaceOp(op, input);
      return ::mlir::success();
    }
    if (resultType.getNumDynamicDims() > 1) {
      return rewriter.notifyMatchFailure(
          op, "expected at most one dynamic result extent");
    }

    std::optional<::llvm::SmallVector<::mlir::ReassociationIndices>>
        reassociation =
            ::mlir::getReassociationIndicesForReshape(inputType, resultType);
    if (!reassociation) {
      return rewriter.notifyMatchFailure(
          op, "expected a collapsing or expanding reshape");
    }

    ::mlir::FailureOr<::mlir::Value> ctx = getHipsrContextArg(op, rewriter);
    if (::mlir::failed(ctx)) {
      return ::mlir::failure();
    }

    ::mlir::Location loc = op->getLoc();
    auto init = PlaceholderOp::create(
        rewriter, loc, ::mlir::TypeRange{resultType}, *ctx,
        ::mlir::ValueRange{input}, PlaceholderType::Normal);
    populateReshapeShapeRegion(rewriter, init, inputType, resultType);

    auto computeOp = ComputeOp::create(
        rewriter, loc, ::mlir::TypeRange{resultType}, *ctx,
        ::mlir::ValueRange{input}, ::mlir::ValueRange{init.getResult(0)});
    populateReshapeComputeBody(rewriter, computeOp, inputType, resultType,
                               *reassociation);

    rewriter.replaceOp(op, computeOp.getResult(0));
    return ::mlir::success();
  }
};

} // namespace

void populateReshapeConversionPatterns(::mlir::RewritePatternSet &patterns,
                                       ::mlir::MLIRContext *ctx) {
  patterns.add<ReshapeToHipsr>(ctx);
}

} // namespace hipsr
} // namespace mlir
