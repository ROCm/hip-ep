/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- ShapeConversion.cpp - Convert onnx.Shape to hipsr.compute ---------===//
//
// onnx.Shape returns its input's extents as a 1-D i64 tensor. That is a few
// tensor and arith ops rather than a library call, so no hipsr op matches it
// and the conversion groups them in a hipsr.compute.
//
// It also fills the placeholder's shape region, which
// hipsr-populate-shape-region cannot: a compute's result shape follows its
// body, so that pass has nothing to dispatch on. It leaves a filled region
// alone.
//
// Extents are read on the host, so the chain is built in #hipsr.mem<host>, the
// space hipsr.expand takes its shape operand from, whatever the ONNX result
// names. Data keeps the #hipsr.mem<device> the pass gives a tensor naming no
// space.
//
// Before:
//   %s = "onnx.Shape"(%x) : (tensor<?x128xf16, #hipsr.mem<device>>)
//       -> tensor<2xi64>
//
// After:
//   %init = hipsr.placeholder(%ctx)
//       ins(%x : tensor<?x128xf16, #hipsr.mem<device>>)
//       {placeholder_type = #hipsr.placeholder_type<normal>}
//       : tensor<2xi64, #hipsr.mem<host>> shape_region {
//   ^bb0(%x_shape: !shape.shape):
//     %c2 = arith.constant 2 : index
//     %shape = shape.from_extents %c2 : index
//     hipsr.shape_yield %shape : !shape.shape
//   }
//   %s = hipsr.compute(%ctx) ins(%x : tensor<?x128xf16, #hipsr.mem<device>>)
//                            outs(%init : tensor<2xi64, #hipsr.mem<host>>) {
//   ^bb0(%body_ctx: !hipsr.context, %in: tensor<?x128xf16, #hipsr.mem<device>>,
//        %dest: tensor<2xi64, #hipsr.mem<host>>):
//     %c0 = arith.constant 0 : index
//     %d0 = tensor.dim %in, %c0 : tensor<?x128xf16, #hipsr.mem<device>>
//     %e0 = arith.index_cast %d0 : index to i64
//     %e1 = arith.constant 128 : i64
//     %shape = tensor.from_elements %e0, %e1 : tensor<2xi64, #hipsr.mem<host>>
//     hipsr.compute_yield %shape : tensor<2xi64, #hipsr.mem<host>>
//   } : tensor<2xi64, #hipsr.mem<host>>
//
//===----------------------------------------------------------------------===//

#include "OnnxToHipsrUtils.h"

#include "hip/Conversion/OnnxToHipsr/OnnxToHipsr.h"
#include "hip/Dialect/Hipsr/IR/HipsrOps.h"
#include "hip/Dialect/Hipsr/IR/HipsrShapeRegionPopulationUtils.h"
#include "hip/Dialect/Hipsr/IR/HipsrTypes.h"
#include "hip/Dialect/Onnx/IR/OnnxOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallVectorExtras.h"

#include <algorithm>
#include <utility>

namespace mlir {
namespace hipsr {
namespace {

// Resolves the start/end bounds against the input rank. `start` carries the
// schema's default of 0, while `end` has none: an absent one means the rank.
// ONNX adds the rank to a negative bound and clamps the result into [0, rank],
// so a bound past either end is legal and names that end rather than being an
// error.
std::pair<int64_t, int64_t> resolveAxisRange(onnx::ShapeOp op, int64_t rank) {
  int64_t start = op.getStart();
  int64_t end = op.getEnd().value_or(rank);
  if (start < 0) {
    start += rank;
  }
  if (end < 0) {
    end += rank;
  }
  return {std::clamp<int64_t>(start, 0, rank),
          std::clamp<int64_t>(end, 0, rank)};
}

// The input's rank fixes the result length, so the region is constant and needs
// no shape-graph inputs.
void populateShapeRegion(OpBuilder &builder, PlaceholderOp placeholder,
                         int64_t numExtents) {
  OpBuilder::InsertionGuard guard(builder);
  Block &block = createPlaceholderShapeBlock(builder, placeholder);
  builder.setInsertionPointToStart(&block);

  Location loc = placeholder.getLoc();
  Value extent = arith::ConstantIndexOp::create(builder, loc, numExtents);
  Value shape = createExtentTensor(builder, loc, ValueRange{extent});
  ShapeYieldOp::create(builder, loc, ValueRange{shape});
}

// A static axis becomes a constant, so only a dynamic one needs a tensor.dim.
void populateComputeBody(OpBuilder &builder, ComputeOp computeOp,
                         RankedTensorType inputType,
                         RankedTensorType extentsType, int64_t start,
                         int64_t end) {
  OpBuilder::InsertionGuard guard(builder);
  Location loc = computeOp.getLoc();

  // The body's arguments are the operands: ctx, then the inputs, then the
  // outputs.
  TypeRange argTypes = computeOp->getOperandTypes();
  SmallVector<Location> argLocs(argTypes.size(), loc);
  Block *body =
      builder.createBlock(&computeOp.getBody(), {}, argTypes, argLocs);
  builder.setInsertionPointToStart(body);

  Value input = body->getArgument(1);
  auto extentOf = [&](int64_t axis) -> Value {
    if (!inputType.isDynamicDim(axis)) {
      return arith::ConstantOp::create(
          builder, loc, builder.getI64IntegerAttr(inputType.getDimSize(axis)));
    }
    Value dim = tensor::DimOp::create(builder, loc, input, axis);
    return arith::IndexCastOp::create(builder, loc, builder.getI64Type(), dim);
  };
  SmallVector<Value> extents =
      llvm::map_to_vector(llvm::seq(start, end), extentOf);

  Value shape =
      tensor::FromElementsOp::create(builder, loc, extentsType, extents);
  ComputeYieldOp::create(builder, loc, ValueRange{shape});
}

// Converts no type itself: the result is the extents vector, whose length the
// input's rank fixes.
struct ShapeToHipsr : public OpConversionPattern<onnx::ShapeOp> {
  // Takes the pass converter to match the other patterns.
  ShapeToHipsr(const TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern(ctx) {}

  LogicalResult
  matchAndRewrite(onnx::ShapeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getData();
    auto inputType = dyn_cast<RankedTensorType>(input.getType());
    if (!inputType) {
      return rewriter.notifyMatchFailure(op, "expected ranked tensor input");
    }

    auto resultType = dyn_cast<RankedTensorType>(op.getShape().getType());
    if (!resultType || resultType.getRank() != 1 ||
        !resultType.getElementType().isInteger(64)) {
      return rewriter.notifyMatchFailure(op, "expected a rank-1 i64 result");
    }

    // An ONNX result names no space, so this names host. Another space would
    // need a copy this does not emit.
    if (resultType.getEncoding() && !isHostRankedTensor(resultType)) {
      return rewriter.notifyMatchFailure(
          op, "expected a result in the host memory space");
    }
    RankedTensorType extentsType =
        tensorTypeInSpace(resultType, MemorySpace::Host);

    auto [start, end] = resolveAxisRange(op, inputType.getRank());
    // ONNX returns a zero-element tensor here, and a zero-extent destination
    // has nothing to allocate.
    if (start >= end) {
      return rewriter.notifyMatchFailure(op, "expected a non-empty axis range");
    }
    if (extentsType.getDimSize(0) != end - start) {
      return rewriter.notifyMatchFailure(
          op, "result length must equal the number of selected axes");
    }

    FailureOr<Value> ctx = getHipsrContextArg(op, rewriter);
    if (failed(ctx)) {
      return failure();
    }

    // The extent count fixes the shape; the input is there to match the
    // compute.
    Location loc = op.getLoc();
    auto init =
        PlaceholderOp::create(rewriter, loc, TypeRange{extentsType}, *ctx,
                              ValueRange{input}, PlaceholderType::Normal);
    populateShapeRegion(rewriter, init, extentsType.getDimSize(0));

    auto computeOp =
        ComputeOp::create(rewriter, loc, TypeRange{extentsType}, *ctx,
                          ValueRange{input}, ValueRange{init.getResult(0)});
    populateComputeBody(rewriter, computeOp, inputType, extentsType, start,
                        end);

    rewriter.replaceOp(op, computeOp.getResult(0));
    return success();
  }
};

} // namespace

void populateShapeConversionPatterns(const TypeConverter &typeConverter,
                                     RewritePatternSet &patterns,
                                     MLIRContext *ctx) {
  patterns.add<ShapeToHipsr>(typeConverter, ctx);
}

} // namespace hipsr
} // namespace mlir
