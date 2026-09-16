/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- ReshapeConversion.cpp - Convert onnx.Reshape to hipsr.compute -----===//
//
// onnx.Reshape regroups dimensions in row-major order without moving data, so
// the body collapses the input to its 1-D form and expands the result out of
// it, skipping either step when it is a no-op. Both bufferize to memrefs that
// alias their source, so nothing runs on the device, and canonicalization
// composes the pair back where one op can express the regroup.
//
// The result type comes from the operands, not the type ONNX declared: the
// target shape states the dimensions, the input gives the element type and
// memory space. The shape region reads no memory.
//
// Before, with %shape the constant [-1, 4]:
//   %r = "onnx.Reshape"(%x, %shape)
//       : (tensor<2x3x4xf16, #hipsr.mem<device>>, tensor<2xi64>)
//       -> tensor<?x?xf16, #hipsr.mem<device>>
//
// After, with region types left out. The input states its element count, so the
// -1 resolves to 6 while the type is inferred:
//   %init = hipsr.placeholder(%ctx) ins(%x)
//       {placeholder_type = #hipsr.placeholder_type<normal>}
//       : tensor<6x4xf16, #hipsr.mem<device>> shape_region {
//   ^bb0(%x_shape):
//     %c6 = arith.constant 6 : index
//     %c4 = arith.constant 4 : index
//     %s = shape.from_extents %c6, %c4
//     hipsr.shape_yield %s
//   }
//   %r = hipsr.compute(%ctx) ins(%x) outs(%init) {
//   ^bb0(%body_ctx, %in, %dest):
//     %flat = tensor.collapse_shape %in [[0, 1, 2]]
//     %out = tensor.expand_shape %flat [[0, 1]] output_shape [6, 4]
//     hipsr.compute_yield %out
//   } : tensor<6x4xf16, #hipsr.mem<device>>
//
// A dynamic input leaves the -1 to runtime: the region divides
// shape.num_elements by the dimensions the operand states, and the expand reads
// that dimension back off %dest.
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

// The result type: the target shape states the dimensions, the -1 comes from
// the element count, and the input gives the element type and memory space.
FailureOr<RankedTensorType>
inferResultType(onnx::ReshapeOp op, RankedTensorType inputType, Value shape,
                ConversionPatternRewriter &rewriter) {
  // TODO: Support a target shape that only holds its value at runtime.
  DenseIntElementsAttr folded;
  if (!matchPattern(shape, m_Constant(&folded))) {
    return rewriter.notifyMatchFailure(
        op, "expected a target shape that folds to a constant");
  }
  // ONNX writes the dimension to infer as -1, which is kDynamic here.
  SmallVector<int64_t> dims = llvm::map_to_vector(
      folded.getValues<APInt>(), [](const APInt &entry) -> int64_t {
        int64_t value = entry.getSExtValue();
        return value < 0 ? ShapedType::kDynamic : value;
      });

  // A 0 copies the input's dimension at that axis when allowzero is clear.
  // TODO: Resolve a 0 too.
  if (llvm::is_contained(dims, 0)) {
    return rewriter.notifyMatchFailure(op, "expected no 0 in the target shape");
  }
  // The element count fills in one -1; a second would have no answer.
  if (llvm::count(dims, ShapedType::kDynamic) > 1) {
    return rewriter.notifyMatchFailure(
        op, "expected at most one dimension left to infer");
  }

  // A reshape preserves the element count, so this division comes out even.
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

// The dimension ONNX wrote as -1: the input's element count over what the
// target shape does state. `shape.num_elements` keeps the rank out of it, and
// the `!shape.size` it returns converts to index because it can hold an error.
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

// The 1-D form `type` flattens to, dynamic unless the element count is known.
RankedTensorType flatTensorType(RankedTensorType type) {
  int64_t count =
      type.hasStaticShape() ? type.getNumElements() : ShapedType::kDynamic;
  return RankedTensorType::get({count}, type.getElementType(),
                               type.getEncoding());
}

// One group holding every axis, which is what reaches the 1-D form.
SmallVector<ReassociationIndices> flatReassociation(int64_t rank) {
  return {llvm::to_vector<2>(llvm::seq<int64_t>(0, rank))};
}

// Every regroup goes through the 1-D form. A single op would fit some, but
// `ComposeExpandOfCollapseOp` already finds those. `dest` is the placeholder,
// so its type is the result type.
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
  // A 1-D result is already the flat form.
  if (flatType == resultType) {
    return flat;
  }

  // The flat source cannot state a dynamic dimension; the destination can.
  return tensor::ExpandShapeOp::create(
             builder, loc, resultType, flat,
             flatReassociation(resultType.getRank()),
             tensor::getMixedSizes(builder, loc, dest))
      .getResult();
}

// The compute and the destination carry the inferred type, so the body takes
// its types from its own arguments and never widens back to the declared one.
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
  // operand, which a type conversion never sees. Taken for a uniform setup.
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

    // The region needs only constants and the input's shape, so a normal
    // placeholder is enough.
    // TODO: A runtime-only target shape would need a host read, so a barrier.
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
