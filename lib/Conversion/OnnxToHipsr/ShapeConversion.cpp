/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- ShapeConversion.cpp - Convert onnx.Shape to hipsr.compute ---------===//
//
// onnx.Shape returns its input's extents as a 1-D i64 tensor. No hipsr op
// matches it, because the work is a handful of tensor and arith ops rather
// than a library call, so the conversion groups them in a hipsr.compute.
//
// A compute op is not DPS and its result shape depends on whatever its body
// does, so hipsr-populate-shape-region has no recipe to dispatch on. This
// conversion therefore fills the placeholder's shape region itself, which the
// pass then leaves alone.
//
// Before:
//   %s = "onnx.Shape"(%x) : (tensor<?x128xf16>) -> tensor<2xi64>
//
// After:
//   %init = hipsr.placeholder(%ctx)
//       {placeholder_type = #hipsr.placeholder_type<normal>} : tensor<2xi64>
//       shape_region {
//     %c2 = arith.constant 2 : index
//     %shape = shape.from_extents %c2 : index
//     hipsr.shape_yield %shape : !shape.shape
//   }
//   %s = hipsr.compute(%ctx) ins(%x : tensor<?x128xf16>)
//                            outs(%init : tensor<2xi64>) {
//   ^bb0(%body_ctx: !hipsr.context, %in: tensor<?x128xf16>,
//        %dest: tensor<2xi64>):
//     %c0 = arith.constant 0 : index
//     %d0 = tensor.dim %in, %c0 : tensor<?x128xf16>
//     %e0 = arith.index_cast %d0 : index to i64
//     %e1 = arith.constant 128 : i64
//     %shape = tensor.from_elements %e0, %e1 : tensor<2xi64>
//     hipsr.compute_yield %shape : tensor<2xi64>
//   } : tensor<2xi64>
//
//===----------------------------------------------------------------------===//

#include "OnnxToHipsrUtils.h"

#include "hip/Conversion/OnnxToHipsr/OnnxToHipsr.h"
#include "hip/Dialect/Hipsr/IR/HipsrOps.h"
#include "hip/Dialect/Hipsr/IR/HipsrShapeRegionPopulationUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <utility>

namespace mlir {
namespace hipsr {
namespace {

// Resolves ONNX Shape's optional start/end attributes against the input rank:
// a negative bound counts back from the end, then both clamp into [0, rank].
std::pair<int64_t, int64_t> resolveAxisRange(::mlir::Operation *op,
                                             int64_t rank) {
  int64_t start = 0;
  int64_t end = rank;
  // getSInt/getInt each assert on a signedness this attribute may not carry,
  // so read the raw APInt.
  if (auto startAttr = op->getAttrOfType<::mlir::IntegerAttr>("start")) {
    start = startAttr.getValue().getSExtValue();
  }
  if (auto endAttr = op->getAttrOfType<::mlir::IntegerAttr>("end")) {
    end = endAttr.getValue().getSExtValue();
  }
  if (start < 0) {
    start += rank;
  }
  if (end < 0) {
    end += rank;
  }
  return {std::clamp<int64_t>(start, 0, rank),
          std::clamp<int64_t>(end, 0, rank)};
}

// The destination holds one extent per selected axis, a count fixed by the
// input's rank rather than by its extents, so the region is a constant and the
// placeholder needs no shape-graph inputs.
void populateShapeRegion(::mlir::OpBuilder &builder, PlaceholderOp placeholder,
                         int64_t numExtents) {
  ::mlir::OpBuilder::InsertionGuard guard(builder);
  ::mlir::Block &block = createPlaceholderShapeBlock(builder, placeholder);
  builder.setInsertionPointToStart(&block);

  ::mlir::Location loc = placeholder.getLoc();
  ::mlir::Value extent =
      ::mlir::arith::ConstantIndexOp::create(builder, loc, numExtents);
  ::mlir::Value shape = ::mlir::shape::FromExtentsOp::create(
      builder, loc, ::mlir::shape::ShapeType::get(builder.getContext()),
      ::mlir::ValueRange{extent});
  ShapeYieldOp::create(builder, loc, ::mlir::ValueRange{shape});
}

// A static axis becomes a constant, so only the dynamic ones cost a query.
void populateComputeBody(::mlir::OpBuilder &builder, ComputeOp computeOp,
                         ::mlir::RankedTensorType inputType,
                         ::mlir::RankedTensorType resultType, int64_t start,
                         int64_t end) {
  ::mlir::OpBuilder::InsertionGuard guard(builder);
  ::mlir::Location loc = computeOp.getLoc();

  ::llvm::SmallVector<::mlir::Type> argTypes{computeOp.getCtx().getType()};
  ::llvm::append_range(argTypes, computeOp.getInputs().getTypes());
  ::llvm::append_range(argTypes, computeOp.getOutputs().getTypes());
  ::llvm::SmallVector<::mlir::Location> argLocs(argTypes.size(), loc);
  ::mlir::Block *body =
      builder.createBlock(&computeOp.getBody(), {}, argTypes, argLocs);
  builder.setInsertionPointToStart(body);

  // Entry-block arguments are ctx, then the inputs, then the outputs.
  ::mlir::Value input = body->getArgument(1);
  ::mlir::Type i64Type = builder.getI64Type();
  ::llvm::SmallVector<::mlir::Value> extents;
  extents.reserve(end - start);
  for (int64_t axis : ::llvm::seq(start, end)) {
    if (!inputType.isDynamicDim(axis)) {
      extents.push_back(::mlir::arith::ConstantOp::create(
          builder, loc, builder.getI64IntegerAttr(inputType.getDimSize(axis))));
      continue;
    }
    ::mlir::Value index =
        ::mlir::arith::ConstantIndexOp::create(builder, loc, axis);
    ::mlir::Value dim =
        ::mlir::tensor::DimOp::create(builder, loc, input, index);
    extents.push_back(
        ::mlir::arith::IndexCastOp::create(builder, loc, i64Type, dim));
  }

  ::mlir::Value shape =
      ::mlir::tensor::FromElementsOp::create(builder, loc, resultType, extents);
  ComputeYieldOp::create(builder, loc, ::mlir::ValueRange{shape});
}

struct ShapeToHipsr : public ::mlir::RewritePattern {
  ShapeToHipsr(::mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Shape", /*benefit=*/1, ctx) {}

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
    if (!inputType) {
      return rewriter.notifyMatchFailure(op, "expected ranked tensor input");
    }

    auto resultType =
        ::mlir::dyn_cast<::mlir::RankedTensorType>(op->getResult(0).getType());
    if (!resultType || resultType.getRank() != 1 ||
        !resultType.getElementType().isInteger(64)) {
      return rewriter.notifyMatchFailure(op, "expected a rank-1 i64 result");
    }

    auto [start, end] = resolveAxisRange(op, inputType.getRank());
    // ONNX returns a zero-element tensor for an empty range. Rejecting it
    // reports the operation instead of handing a zero-extent destination to
    // the allocator.
    if (start >= end) {
      return rewriter.notifyMatchFailure(op, "expected a non-empty axis range");
    }
    if (resultType.getDimSize(0) != end - start) {
      return rewriter.notifyMatchFailure(
          op, "result length must equal the number of selected axes");
    }

    ::mlir::FailureOr<::mlir::Value> ctx = getHipsrContextArg(op, rewriter);
    if (::mlir::failed(ctx)) {
      return ::mlir::failure();
    }

    ::mlir::Location loc = op->getLoc();
    auto init = PlaceholderOp::create(
        rewriter, loc, ::mlir::TypeRange{resultType}, *ctx,
        ::mlir::ValueRange{}, PlaceholderType::Normal);
    populateShapeRegion(rewriter, init, resultType.getDimSize(0));

    auto computeOp = ComputeOp::create(
        rewriter, loc, ::mlir::TypeRange{resultType}, *ctx,
        ::mlir::ValueRange{input}, ::mlir::ValueRange{init.getResult(0)});
    populateComputeBody(rewriter, computeOp, inputType, resultType, start, end);

    rewriter.replaceOp(op, computeOp.getResult(0));
    return ::mlir::success();
  }
};

} // namespace

void populateShapeConversionPatterns(::mlir::RewritePatternSet &patterns,
                                     ::mlir::MLIRContext *ctx) {
  patterns.add<ShapeToHipsr>(ctx);
}

} // namespace hipsr
} // namespace mlir
