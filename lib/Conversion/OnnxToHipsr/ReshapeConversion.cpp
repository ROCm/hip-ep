/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- ReshapeConversion.cpp - Convert onnx.Reshape to hipsr.compute -----===//
//
// onnx.Reshape regroups dimensions in row-major order without moving data, so
// the hipsr.compute body collapses the input to its 1-D form and expands the
// result out of it, skipping either step when it is a no-op. Both ops bufferize
// to memrefs that alias their source, so nothing runs on the device, and
// canonicalization composes the pair back into one op where one can express the
// regroup.
//
// The result type is inferred from the operands, not taken from the type ONNX
// declared, which often says less. The target shape states the dimensions, and
// the input states the element type and memory space the result aliases. The
// shape region then works off the input's shape and reads no memory.
//
// Before, with %shape the constant [-1, 32]:
//   %r = "onnx.Reshape"(%x, %shape)
//       : (tensor<?x8x4xf16, #hipsr.mem<device>>, tensor<2xi64>)
//       -> tensor<?x?xf16, #hipsr.mem<device>>
//
// After, with the types inside the regions left out:
//   %init = hipsr.placeholder(%ctx) ins(%x)
//       {placeholder_type = #hipsr.placeholder_type<normal>}
//       : tensor<?x32xf16, #hipsr.mem<device>> shape_region {
//   ^bb0(%x_shape):
//     %count = shape.num_elements %x_shape
//     %n = shape.size_to_index %count
//     %c32 = arith.constant 32 : index
//     %rows = arith.divui %n, %c32
//     %cols = arith.constant 32 : index
//     %s = shape.from_extents %rows, %cols
//     hipsr.shape_yield %s
//   }
//   %r = hipsr.compute(%ctx) ins(%x) outs(%init) {
//   ^bb0(%body_ctx, %in, %dest):
//     %flat = tensor.collapse_shape %in [[0, 1, 2]]
//     %c0 = arith.constant 0 : index
//     %dim = tensor.dim %dest, %c0
//     %out = tensor.expand_shape %flat [[0, 1]] output_shape [%dim, 32]
//     hipsr.compute_yield %out
//   } : tensor<?x32xf16, #hipsr.mem<device>>
//
//===----------------------------------------------------------------------===//

#include "OnnxToHipsrUtils.h"

#include "hip/Conversion/OnnxToHipsr/OnnxToHipsr.h"
#include "hip/Dialect/Hipsr/IR/HipsrOps.h"
#include "hip/Dialect/Hipsr/IR/HipsrShapeRegionPopulationUtils.h"
#include "hip/Dialect/Onnx/IR/OnnxOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"

#include <cassert>

namespace mlir {
namespace hipsr {
namespace {

//===----------------------------------------------------------------------===//
// Inferring the result type
//===----------------------------------------------------------------------===//

// The product of the static dimensions, with the dynamic ones left out.
int64_t productOfStaticDims(ArrayRef<int64_t> dims) {
  return llvm::product_of(llvm::make_filter_range(
      dims, [](int64_t dim) { return ShapedType::isStatic(dim); }));
}

// The result type, read off the operands: the target shape states the
// dimensions, the one written as -1 comes from the preserved element count, and
// the input gives the element type and memory space the result aliases.
FailureOr<RankedTensorType>
inferResultType(onnx::ReshapeOp op, RankedTensorType inputType, Value shape,
                ConversionPatternRewriter &rewriter) {
  // TODO: Support a target shape that only holds its value at runtime.
  DenseIntElementsAttr folded;
  if (!matchPattern(shape, m_Constant(&folded))) {
    return rewriter.notifyMatchFailure(
        op, "expected a target shape that folds to a constant");
  }
  SmallVector<int64_t> dims = llvm::map_to_vector(
      folded.getValues<APInt>(), [](const APInt &entry) -> int64_t {
        int64_t value = entry.getSExtValue();
        return value < 0 ? ShapedType::kDynamic : value;
      });

  // The -1 is resolved below from the element count, but a 0 is not: with
  // allowzero clear it copies the input's dimension at that axis instead.
  // TODO: Resolve a 0 too.
  if (llvm::is_contained(dims, 0)) {
    return rewriter.notifyMatchFailure(op, "expected no 0 in the target shape");
  }
  // The element count fills in one -1. ONNX allows no more than one, and a
  // second would have no answer.
  if (llvm::count(dims, ShapedType::kDynamic) > 1) {
    return rewriter.notifyMatchFailure(
        op, "expected at most one dimension left to infer");
  }

  // The -1 stays dynamic unless the input pins the element count down. A
  // reshape preserves that count, so the division below comes out even.
  if (auto *unknown = llvm::find(dims, ShapedType::kDynamic);
      unknown != dims.end() && inputType.hasStaticShape()) {
    *unknown = inputType.getNumElements() / productOfStaticDims(dims);
  }
  return RankedTensorType::get(dims, inputType.getElementType(),
                               inputType.getEncoding());
}

//===----------------------------------------------------------------------===//
// The destination
//===----------------------------------------------------------------------===//

// The dimension ONNX wrote as -1: the input's element count over the dimensions
// the target shape does state. `shape.num_elements` is the product of every
// extent, so the input's rank never comes into it. A `!shape.size` can hold an
// error value that arith has no way to carry, so the count converts to index.
Value buildInferredDim(OpBuilder &builder, Location loc, Value inputShape,
                       int64_t resultStaticCount) {
  assert(resultStaticCount > 0 &&
         "a 0 result dimension is rejected before this point");

  Value count = shape::NumElementsOp::create(builder, loc, inputShape);
  Value dividend =
      shape::SizeToIndexOp::create(builder, loc, builder.getIndexType(), count);
  Value divisor =
      arith::ConstantIndexOp::create(builder, loc, resultStaticCount);
  return arith::DivUIOp::create(builder, loc, dividend, divisor);
}

// One value per result axis: a stated dimension is a constant, the inferred one
// is computed off the input's shape.
void populateShapeRegion(OpBuilder &builder, PlaceholderOp placeholder,
                         ArrayRef<int64_t> resultShape) {
  assert(llvm::count_if(resultShape, ShapedType::isDynamic) <= 1 &&
         "a second inferred dimension is rejected before this point");

  OpBuilder::InsertionGuard guard(builder);
  Block &block = createPlaceholderShapeBlock(builder, placeholder);
  builder.setInsertionPointToStart(&block);

  Location loc = placeholder.getLoc();
  Value inputShape = PlaceholderShapeRegionArgs{block}.in(0);
  SmallVector<Value> dims =
      llvm::map_to_vector(resultShape, [&](int64_t dim) -> Value {
        if (ShapedType::isDynamic(dim)) {
          return buildInferredDim(builder, loc, inputShape,
                                  productOfStaticDims(resultShape));
        }
        return arith::ConstantIndexOp::create(builder, loc, dim);
      });
  Value shape = shape::FromExtentsOp::create(
      builder, loc, shape::ShapeType::get(builder.getContext()), dims);
  ShapeYieldOp::create(builder, loc, ValueRange{shape});
}

//===----------------------------------------------------------------------===//
// The compute body
//===----------------------------------------------------------------------===//

// The 1-D form `type` flattens to, dynamic when the element count is not a
// compile-time constant.
RankedTensorType flatTensorType(RankedTensorType type) {
  int64_t count =
      type.hasStaticShape() ? type.getNumElements() : ShapedType::kDynamic;
  return RankedTensorType::get({count}, type.getElementType(),
                               type.getEncoding());
}

// Every axis in one group, which is the grouping a collapse or an expand takes
// to reach the 1-D form.
SmallVector<ReassociationIndices> flatReassociation(int64_t rank) {
  return {llvm::to_vector<2>(llvm::seq<int64_t>(0, rank))};
}

// Every regroup goes through the 1-D form: one group collapses every input
// axis, one expands the result out of it. A single collapse or expand would fit
// some regroups, but `ComposeExpandOfCollapseOp` already finds those. `dest` is
// the placeholder, so it carries the result type and an expansion reads the
// dimensions it resolved.
Value createReshape(OpBuilder &builder, Location loc, Value input, Value dest) {
  auto inputType = cast<RankedTensorType>(input.getType());
  auto resultType = cast<RankedTensorType>(dest.getType());
  assert(inputType != resultType &&
         "an identity reshape is dropped before it is given a body");

  RankedTensorType flatType = flatTensorType(inputType);
  Value flat = input;
  if (inputType.getRank() > 1) {
    flat = tensor::CollapseShapeOp::create(
        builder, loc, flatType, input, flatReassociation(inputType.getRank()));
  }

  if (RankedTensorType resultFlatType = flatTensorType(resultType);
      flatType != resultFlatType) {
    // Collapse and expand cannot make a dynamic dimension static; a cast can.
    flat = tensor::CastOp::create(builder, loc, resultFlatType, flat);
    flatType = resultFlatType;
  }
  if (flatType == resultType) {
    return flat;
  }

  return tensor::ExpandShapeOp::create(
             builder, loc, resultType, flat,
             flatReassociation(resultType.getRank()),
             tensor::getMixedSizes(builder, loc, dest))
      .getResult();
}

// The compute and the destination both carry the inferred type, so the body
// takes its types from its own arguments and never widens back to the one ONNX
// declared.
void populateComputeBody(OpBuilder &builder, ComputeOp computeOp) {
  OpBuilder::InsertionGuard guard(builder);
  Location loc = computeOp.getLoc();

  // The body's arguments are the operands: ctx, the inputs, then the outputs.
  TypeRange argTypes = computeOp->getOperandTypes();
  SmallVector<Location> argLocs(argTypes.size(), loc);
  Block *body =
      builder.createBlock(&computeOp.getBody(), {}, argTypes, argLocs);
  builder.setInsertionPointToStart(body);

  Value reshaped = createReshape(builder, loc, body->getArgument(1),
                                 body->getArguments().back());
  ComputeYieldOp::create(builder, loc, ValueRange{reshaped});
}

//===----------------------------------------------------------------------===//
// The pattern
//===----------------------------------------------------------------------===//

struct ReshapeToHipsr : public OpConversionPattern<onnx::ReshapeOp> {
  // The converter goes unused: the result type depends on the target shape
  // operand, which a type conversion never sees. Taken so the pass adds every
  // pattern the same way.
  ReshapeToHipsr(const TypeConverter &, MLIRContext *ctx)
      : OpConversionPattern(ctx) {}

  LogicalResult
  matchAndRewrite(onnx::ReshapeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // allowzero makes a 0 a literal dimension instead of a copy of the input's.
    // TODO: Support it once a 0 entry is resolved.
    if (op.getAllowzero() != 0) {
      return rewriter.notifyMatchFailure(op, "expected allowzero = 0");
    }
    Value input = adaptor.getData();
    auto inputType = dyn_cast<RankedTensorType>(input.getType());
    if (!inputType) {
      return rewriter.notifyMatchFailure(op, "expected a ranked tensor input");
    }
    FailureOr<RankedTensorType> inferred =
        inferResultType(op, inputType, adaptor.getShape(), rewriter);
    if (failed(inferred)) {
      return failure();
    }
    RankedTensorType resultType = *inferred;

    // An identity reshape needs no destination. Inferring the type first also
    // catches the identities the declared type hides.
    if (inputType == resultType) {
      rewriter.replaceOp(op, input);
      return success();
    }
    FailureOr<Value> ctx = getHipsrContextArg(op, rewriter);
    if (failed(ctx)) {
      return failure();
    }

    // The region builds the dimensions out of constants and the input's shape,
    // so a normal placeholder is enough.
    // TODO: A target shape that only holds its value at runtime would have the
    // region read that operand instead, and a host read takes a barrier
    // placeholder.
    Location loc = op.getLoc();
    auto init =
        PlaceholderOp::create(rewriter, loc, TypeRange{resultType}, *ctx,
                              ValueRange{input}, PlaceholderType::Normal);
    populateShapeRegion(rewriter, init, resultType.getShape());

    auto computeOp =
        ComputeOp::create(rewriter, loc, TypeRange{resultType}, *ctx,
                          ValueRange{input}, ValueRange{init.getResult(0)});
    populateComputeBody(rewriter, computeOp);
    rewriter.replaceOp(op, computeOp.getResult(0));
    return success();
  }
};

} // namespace

void populateReshapeConversionPatterns(const TypeConverter &typeConverter,
                                       RewritePatternSet &patterns,
                                       MLIRContext *ctx) {
  patterns.add<ReshapeToHipsr>(typeConverter, ctx);
}

} // namespace hipsr
} // namespace mlir
