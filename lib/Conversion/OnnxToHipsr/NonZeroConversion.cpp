/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- NonZeroConversion.cpp - Convert onnx.NonZero ----------------------===//
//
// How many positions onnx.NonZero reports follows the values in the input, not
// its shape, so the conversion takes three ops:
//
// - hipsr.nonzero searches on the device. It writes the positions into a
//   destination sized for the worst case, where every element is non-zero, and
//   how many it found into a second one. Both sizes follow the input shape, so
//   hipsr-populate-shape-region fills their placeholder.
// - hipsr.copy_d2h brings the count to the host, the only place that can
//   resolve the published column count.
// - hipsr.compute narrows the search destination to the positions found. Its
//   placeholder is a barrier reading that host count, so the conversion fills
//   that shape region itself. The barrier and the compute list the same two
//   values, each reading the one it needs, so they share a pool domain.
//
// Before:
//   %p = "onnx.NonZero"(%mask) : (tensor<?x?xi8, #hipsr.mem<device>>)
//       -> tensor<2x?xi64, #hipsr.mem<device>>
//
// After:
//   %cap_init, %count_init = hipsr.placeholder(%ctx)
//       ins(%mask : tensor<?x?xi8, #hipsr.mem<device>>)
//       {placeholder_type = #hipsr.placeholder_type<normal>}
//       : tensor<2x?xi64, #hipsr.mem<device>>,
//         tensor<1xi64, #hipsr.mem<device>>
//   %cap, %count = hipsr.nonzero(%ctx)
//       ins(%mask : tensor<?x?xi8, #hipsr.mem<device>>)
//       outs(%cap_init, %count_init
//            : tensor<2x?xi64, #hipsr.mem<device>>,
//              tensor<1xi64, #hipsr.mem<device>>)
//       : tensor<2x?xi64, #hipsr.mem<device>>,
//         tensor<1xi64, #hipsr.mem<device>>
//   %host_init = hipsr.placeholder(%ctx)
//       ins(%count_init : tensor<1xi64, #hipsr.mem<device>>)
//       {placeholder_type = #hipsr.placeholder_type<normal>}
//       : tensor<1xi64, #hipsr.mem<host>> shape_region {
//   ^bb0(%count_shape: !shape.shape):
//     hipsr.shape_yield %count_shape : !shape.shape
//   }
//   %host_count = hipsr.copy_d2h(%ctx)
//       ins(%count : tensor<1xi64, #hipsr.mem<device>>)
//       outs(%host_init : tensor<1xi64, #hipsr.mem<host>>)
//       : tensor<1xi64, #hipsr.mem<host>>
//   %init = hipsr.placeholder(%ctx)
//       ins(%host_init, %cap_init
//           : tensor<1xi64, #hipsr.mem<host>>,
//             tensor<2x?xi64, #hipsr.mem<device>>)
//       {placeholder_type = #hipsr.placeholder_type<barrier>}
//       : tensor<2x?xi64, #hipsr.mem<device>> shape_region {
//   ^bb0(%shape_ctx: !hipsr.context, %host: tensor<1xi64, #hipsr.mem<host>>,
//        %positions: tensor<2x?xi64, #hipsr.mem<device>>):
//     %c0 = arith.constant 0 : index
//     %found = tensor.extract %host[%c0] : tensor<1xi64, #hipsr.mem<host>>
//     %columns = arith.index_cast %found : i64 to index
//     %c2 = arith.constant 2 : index
//     %shape = shape.from_extents %c2, %columns : index, index
//     hipsr.shape_yield %shape : !shape.shape
//   }
//   %p = hipsr.compute(%ctx)
//       ins(%host_count, %cap
//           : tensor<1xi64, #hipsr.mem<host>>,
//             tensor<2x?xi64, #hipsr.mem<device>>)
//       outs(%init : tensor<2x?xi64, #hipsr.mem<device>>) {
//   ^bb0(%body_ctx: !hipsr.context,
//        %count: tensor<1xi64, #hipsr.mem<host>>,
//        %in: tensor<2x?xi64, #hipsr.mem<device>>,
//        %dest: tensor<2x?xi64, #hipsr.mem<device>>):
//     %c1 = arith.constant 1 : index
//     %n = tensor.dim %dest, %c1 : tensor<2x?xi64, #hipsr.mem<device>>
//     %positions = tensor.extract_slice %in[0, 0] [2, %n] [1, 1]
//         : tensor<2x?xi64, #hipsr.mem<device>>
//           to tensor<2x?xi64, #hipsr.mem<device>>
//     hipsr.compute_yield %positions : tensor<2x?xi64, #hipsr.mem<device>>
//   } : tensor<2x?xi64, #hipsr.mem<device>>
//
// Both placeholders around the copy are built on data results but print the
// destinations holding them: placeholder inputs belong to the shape graph, so
// rewirePlaceholderInputs resolves a data result onto its destination.
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
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace hipsr {
namespace {

struct NarrowPlaceholderShapeArgs : PlaceholderShapeRegionArgs {
  Value getCount() const { return in(0); }
};

// Where the search writes: one row per input axis, and a column per input
// element, the worst case where every one of them is non-zero.
RankedTensorType capacityType(RankedTensorType inputType, Type elementType) {
  int64_t capacity = inputType.hasStaticShape() ? inputType.getNumElements()
                                                : ShapedType::kDynamic;
  return tensorTypeInSpace(
      RankedTensorType::get({inputType.getRank(), capacity}, elementType),
      MemorySpace::Device);
}

// The narrowed shape: one row per input axis, and a column for each position
// the host count says the search found.
void populateNarrowShapeRegion(OpBuilder &builder, PlaceholderOp placeholder,
                               int64_t rows) {
  OpBuilder::InsertionGuard guard(builder);
  Block &block = createPlaceholderShapeBlock(builder, placeholder);
  builder.setInsertionPointToStart(&block);

  Location loc = placeholder.getLoc();
  NarrowPlaceholderShapeArgs args{block};
  Value count = args.getCount();
  Value zero = arith::ConstantIndexOp::create(builder, loc, 0);
  Value found =
      tensor::ExtractOp::create(builder, loc, count, ValueRange{zero});
  Value columns =
      arith::IndexCastOp::create(builder, loc, builder.getIndexType(), found);
  Value rowCount = arith::ConstantIndexOp::create(builder, loc, rows);
  Value shape = shape::FromExtentsOp::create(
      builder, loc, shape::ShapeType::get(builder.getContext()),
      ValueRange{rowCount, columns});
  ShapeYieldOp::create(builder, loc, ValueRange{shape});
}

// Keeps the columns that hold a position and drops the rest of the capacity.
void populateNarrowBody(OpBuilder &builder, ComputeOp computeOp) {
  OpBuilder::InsertionGuard guard(builder);
  Location loc = computeOp.getLoc();

  // The body's arguments are the operands: ctx, then the inputs, then the
  // outputs.
  TypeRange argTypes = computeOp->getOperandTypes();
  SmallVector<Location> argLocs(argTypes.size(), loc);
  Block *body =
      builder.createBlock(&computeOp.getBody(), {}, argTypes, argLocs);
  builder.setInsertionPointToStart(body);

  // Argument 1 is the count, read only by the barrier's shape region. The
  // destination it sized gives the slice its extents.
  Value capacity = body->getArgument(2);
  Value destination = body->getArguments().back();
  SmallVector<OpFoldResult> sizes =
      tensor::getMixedSizes(builder, loc, destination);
  SmallVector<OpFoldResult> offsets(sizes.size(), builder.getIndexAttr(0));
  SmallVector<OpFoldResult> strides(sizes.size(), builder.getIndexAttr(1));
  Value positions = tensor::ExtractSliceOp::create(builder, loc, capacity,
                                                   offsets, sizes, strides);
  ComputeYieldOp::create(builder, loc, ValueRange{positions});
}

struct NonZeroToHipsr : public OpConversionPattern<onnx::NonZeroOp> {
  // The type converter is unused: converting the operands would overwrite the
  // memory space their producers chose. It stays in the signature so every
  // pattern is built the same way.
  NonZeroToHipsr(const TypeConverter &, MLIRContext *ctx)
      : OpConversionPattern(ctx) {}

  LogicalResult
  matchAndRewrite(onnx::NonZeroOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getX();
    auto inputType = dyn_cast<RankedTensorType>(input.getType());
    if (!inputType || inputType.getRank() == 0) {
      return rewriter.notifyMatchFailure(
          op, "expected a ranked tensor input with at least one axis");
    }
    // The search compares against an integer zero, so a float would need a
    // different test.
    if (!inputType.getElementType().isIntOrIndex()) {
      return rewriter.notifyMatchFailure(
          op, "expected an integer or boolean input");
    }
    // The only copy this emits is the count coming back.
    if (!isDeviceRankedTensor(inputType)) {
      return rewriter.notifyMatchFailure(op,
                                         "expected a device-resident input");
    }

    auto resultType = dyn_cast<RankedTensorType>(op.getY().getType());
    if (!resultType || resultType.getRank() != 2) {
      return rewriter.notifyMatchFailure(op, "expected a rank-2 result");
    }
    if (resultType.getDimSize(0) != inputType.getRank()) {
      return rewriter.notifyMatchFailure(
          op, "result must have one row per input axis");
    }
    // How many positions there are follows the values, so no result type can
    // claim to know it.
    if (!resultType.isDynamicDim(1)) {
      return rewriter.notifyMatchFailure(
          op, "expected a dynamic result column count");
    }
    resultType = tensorTypeInSpace(resultType, MemorySpace::Device);

    FailureOr<Value> ctx = getHipsrContextArg(op, rewriter);
    if (failed(ctx)) {
      return failure();
    }

    Location loc = op.getLoc();
    Type i64 = rewriter.getI64Type();
    // The search writes both destinations, so the count lands on the device
    // beside the positions.
    SmallVector<Type, 2> searchTypes = {
        capacityType(inputType, i64),
        tensorTypeInSpace(RankedTensorType::get({1}, i64),
                          MemorySpace::Device)};

    auto searchInits =
        PlaceholderOp::create(rewriter, loc, searchTypes, *ctx,
                              ValueRange{input}, PlaceholderType::Normal);
    auto searchOp =
        NonZeroOp::create(rewriter, loc, searchTypes, *ctx, input,
                          searchInits.getResult(0), searchInits.getResult(1));

    auto copyOp = createCopyD2H(rewriter, loc, *ctx, searchOp.getResult(1));

    // One list for both keeps them in one pool domain.
    SmallVector<Value> narrowInputs{copyOp.getResult(0), searchOp.getResult(0)};
    auto init =
        PlaceholderOp::create(rewriter, loc, TypeRange{resultType}, *ctx,
                              narrowInputs, PlaceholderType::Barrier);
    populateNarrowShapeRegion(rewriter, init, inputType.getRank());

    auto computeOp =
        ComputeOp::create(rewriter, loc, TypeRange{resultType}, *ctx,
                          narrowInputs, ValueRange{init.getResult(0)});
    populateNarrowBody(rewriter, computeOp);

    rewriter.replaceOp(op, computeOp.getResult(0));
    return success();
  }
};

} // namespace

void populateNonZeroConversionPatterns(const TypeConverter &typeConverter,
                                       RewritePatternSet &patterns,
                                       MLIRContext *ctx) {
  patterns.add<NonZeroToHipsr>(typeConverter, ctx);
}

} // namespace hipsr
} // namespace mlir
