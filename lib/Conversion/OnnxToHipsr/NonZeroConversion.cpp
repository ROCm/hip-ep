/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- NonZeroConversion.cpp - Convert onnx.NonZero ----------------------===//
//
// onnx.NonZero reports the position of every non-zero element, so how many
// columns its result holds follows the values in the input rather than its
// shape. That splits the conversion in two:
//
// - hipsr.nonzero writes into a destination sized for the worst case, where
//   every element is non-zero. Those extents come from the input shape, so a
//   normal placeholder covers them.
// - a hipsr.compute narrows that destination to the positions found. Its
//   column count is the number hipsr.nonzero published, which only a host
//   readback resolves, so its placeholder is a barrier.
//
// The count stays on the shape graph. The compute body recovers it from its
// own destination with tensor.dim, which the barrier region already resolved,
// so it never becomes a compute operand -- hipsr.compute takes device values
// and the count is host-resident.
//
// Before:
//   %p = "onnx.NonZero"(%mask) : (tensor<?x?xi8>) -> tensor<2x?xi64>
//
// After:
//   %cap_init, %count_init = hipsr.placeholder(%ctx)
//       ins(%mask : tensor<?x?xi8>)
//       {placeholder_type = #hipsr.placeholder_type<normal>}
//       : tensor<2x?xi64>, tensor<1xi64>
//   %cap, %count = hipsr.nonzero(%ctx) ins(%mask : tensor<?x?xi8>)
//       outs(%cap_init, %count_init : tensor<2x?xi64>, tensor<1xi64>)
//       : tensor<2x?xi64>, tensor<1xi64>
//   %init = hipsr.placeholder(%ctx) ins(%count_init : tensor<1xi64>)
//       {placeholder_type = #hipsr.placeholder_type<barrier>} : tensor<2x?xi64>
//       shape_region {
//   ^bb0(%shape_ctx: !hipsr.context, %c: tensor<1xi64>):
//     %c0 = arith.constant 0 : index
//     %found = tensor.extract %c[%c0] : tensor<1xi64>
//     %n = arith.index_cast %found : i64 to index
//     %rows = arith.constant 2 : index
//     %shape = shape.from_extents %rows, %n : index, index
//     hipsr.shape_yield %shape : !shape.shape
//   }
//   %p = hipsr.compute(%ctx) ins(%cap : tensor<2x?xi64>)
//                            outs(%init : tensor<2x?xi64>) {
//   ^bb0(%body_ctx: !hipsr.context, %in: tensor<2x?xi64>,
//        %dest: tensor<2x?xi64>):
//     %c1 = arith.constant 1 : index
//     %n = tensor.dim %dest, %c1 : tensor<2x?xi64>
//     %positions = tensor.extract_slice %in[0, 0] [2, %n] [1, 1]
//         : tensor<2x?xi64> to tensor<2x?xi64>
//     hipsr.compute_yield %positions : tensor<2x?xi64>
//   } : tensor<2x?xi64>
//
//===----------------------------------------------------------------------===//

#include "OnnxToHipsrUtils.h"

#include "hip/Conversion/OnnxToHipsr/OnnxToHipsr.h"
#include "hip/Dialect/Hipsr/IR/HipsrOps.h"
#include "hip/Dialect/Hipsr/IR/HipsrShapeRegionPopulationUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"

#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace hipsr {
namespace {

// The destination has to hold the worst case, where every element is
// non-zero, so its column count is the input's element count -- a number only
// a static input shape pins down.
::mlir::RankedTensorType capacityType(::mlir::RankedTensorType inputType,
                                      ::mlir::Type elementType) {
  int64_t capacity = inputType.hasStaticShape() ? inputType.getNumElements()
                                                : ::mlir::ShapedType::kDynamic;
  return ::mlir::RankedTensorType::get({inputType.getRank(), capacity},
                                       elementType);
}

// Reads the published count back on the host and pairs it with the row count,
// which the input rank fixes at compile time.
void populateNonZeroShapeRegion(::mlir::OpBuilder &builder,
                                PlaceholderOp placeholder, int64_t rows) {
  ::mlir::OpBuilder::InsertionGuard guard(builder);
  ::mlir::Block &block = createPlaceholderShapeBlock(builder, placeholder);
  builder.setInsertionPointToStart(&block);

  ::mlir::Location loc = placeholder.getLoc();
  ::mlir::Value count = PlaceholderShapeRegionArgs{block}.in(0);
  ::mlir::Value zero = ::mlir::arith::ConstantIndexOp::create(builder, loc, 0);
  ::mlir::Value found = ::mlir::tensor::ExtractOp::create(
      builder, loc, count, ::mlir::ValueRange{zero});
  ::mlir::Value columns = ::mlir::arith::IndexCastOp::create(
      builder, loc, builder.getIndexType(), found);
  ::mlir::Value rowCount =
      ::mlir::arith::ConstantIndexOp::create(builder, loc, rows);
  ::mlir::Value shape = ::mlir::shape::FromExtentsOp::create(
      builder, loc, ::mlir::shape::ShapeType::get(builder.getContext()),
      ::mlir::ValueRange{rowCount, columns});
  ShapeYieldOp::create(builder, loc, ::mlir::ValueRange{shape});
}

// Keeps the columns that hold a position and drops the rest of the capacity.
void populateNonZeroComputeBody(::mlir::OpBuilder &builder, ComputeOp computeOp,
                                ::mlir::RankedTensorType resultType) {
  ::mlir::OpBuilder::InsertionGuard guard(builder);
  ::mlir::Location loc = computeOp.getLoc();
  ::mlir::Block &body = createComputeBodyBlock(builder, computeOp);

  // The destination carries the count the barrier region resolved, so taking
  // the extent from it spares the body a second readback.
  ::mlir::Value destination = computeBodyOutput(body, computeOp, 0);
  ::mlir::Value one = ::mlir::arith::ConstantIndexOp::create(builder, loc, 1);
  ::mlir::Value columns =
      ::mlir::tensor::DimOp::create(builder, loc, destination, one);

  ::llvm::SmallVector<::mlir::OpFoldResult> offsets(2, builder.getIndexAttr(0));
  ::llvm::SmallVector<::mlir::OpFoldResult> sizes{
      builder.getIndexAttr(resultType.getDimSize(0)),
      ::mlir::OpFoldResult{columns}};
  ::llvm::SmallVector<::mlir::OpFoldResult> strides(2, builder.getIndexAttr(1));
  ::mlir::Value positions = ::mlir::tensor::ExtractSliceOp::create(
      builder, loc, resultType, computeBodyInput(body, 0), offsets, sizes,
      strides);
  ComputeYieldOp::create(builder, loc, ::mlir::ValueRange{positions});
}

struct NonZeroToHipsr : public ::mlir::RewritePattern {
  NonZeroToHipsr(::mlir::MLIRContext *ctx)
      : RewritePattern("onnx.NonZero", /*benefit=*/1, ctx) {}

  ::mlir::LogicalResult
  matchAndRewrite(::mlir::Operation *op,
                  ::mlir::PatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 1 || op->getNumResults() != 1) {
      return rewriter.notifyMatchFailure(
          op, "expected one operand and a single result");
    }

    ::mlir::Value input = op->getOperand(0);
    auto inputType =
        ::mlir::dyn_cast<::mlir::RankedTensorType>(input.getType());
    // A rank-0 input has no axis to name a position by, leaving no rows.
    if (!inputType || inputType.getRank() == 0) {
      return rewriter.notifyMatchFailure(
          op, "expected a ranked tensor input with at least one axis");
    }
    // The search compares against an integer zero, which covers the masks
    // this converts; a float input needs a different test.
    if (!inputType.getElementType().isIntOrIndex()) {
      return rewriter.notifyMatchFailure(
          op, "expected an integer or boolean input");
    }

    auto resultType =
        ::mlir::dyn_cast<::mlir::RankedTensorType>(op->getResult(0).getType());
    if (!resultType || resultType.getRank() != 2 ||
        !resultType.getElementType().isInteger(64)) {
      return rewriter.notifyMatchFailure(op, "expected a rank-2 i64 result");
    }
    if (resultType.getDimSize(0) != inputType.getRank()) {
      return rewriter.notifyMatchFailure(
          op, "result must have one row per input axis");
    }
    // How many positions there are follows the values, so a static column
    // count would assert something the conversion cannot honor.
    if (!resultType.isDynamicDim(1)) {
      return rewriter.notifyMatchFailure(
          op, "expected a dynamic result column count");
    }

    ::mlir::FailureOr<::mlir::Value> ctx = getHipsrContextArg(op, rewriter);
    if (::mlir::failed(ctx)) {
      return ::mlir::failure();
    }

    ::mlir::Location loc = op->getLoc();
    ::mlir::Type i64Type = rewriter.getI64Type();
    ::mlir::RankedTensorType positionsType = capacityType(inputType, i64Type);
    auto countType = ::mlir::RankedTensorType::get({1}, i64Type);

    auto searchInits = PlaceholderOp::create(
        rewriter, loc, ::mlir::TypeRange{positionsType, countType}, *ctx,
        ::mlir::ValueRange{input}, PlaceholderType::Normal);
    auto searchOp = NonZeroOp::create(
        rewriter, loc, ::mlir::TypeRange{positionsType, countType}, *ctx, input,
        searchInits.getResult(0), searchInits.getResult(1));

    auto init = PlaceholderOp::create(
        rewriter, loc, ::mlir::TypeRange{resultType}, *ctx,
        ::mlir::ValueRange{searchOp.getResult(1)}, PlaceholderType::Barrier);
    populateNonZeroShapeRegion(rewriter, init, inputType.getRank());

    auto computeOp =
        ComputeOp::create(rewriter, loc, ::mlir::TypeRange{resultType}, *ctx,
                          ::mlir::ValueRange{searchOp.getResult(0)},
                          ::mlir::ValueRange{init.getResult(0)});
    populateNonZeroComputeBody(rewriter, computeOp, resultType);

    rewriter.replaceOp(op, computeOp.getResult(0));
    return ::mlir::success();
  }
};

} // namespace

void populateNonZeroConversionPatterns(::mlir::RewritePatternSet &patterns,
                                       ::mlir::MLIRContext *ctx) {
  patterns.add<NonZeroToHipsr>(ctx);
}

} // namespace hipsr
} // namespace mlir
