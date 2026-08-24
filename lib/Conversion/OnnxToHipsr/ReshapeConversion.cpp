/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- ReshapeConversion.cpp - Convert onnx.Reshape to hipsr.compute -----===//
//
// onnx.Reshape regroups extents in row-major order without moving data. The
// conversion emits a tensor.collapse_shape and a tensor.expand_shape through
// the 1-D form, inside a hipsr.compute. Both bufferize to memrefs that alias
// their source, so nothing runs on the device, and canonicalization composes
// the pair into one op when a single op can express the regroup.
//
// The result type is inferred from the operands rather than taken from the one
// ONNX declared, which is often the weaker of the two: the target shape states
// every extent, bar the one written as -1 and left to the preserved element
// count, and the input states the element type and memory space it aliases. So
// the shape region works from the input's shape alone and reads nothing out of
// memory.
//
// Three forms are rejected. A target shape that does not fold, because nothing
// else states the extents. A second inferred extent, because one equation
// recovers one unknown and the rest would take a host read of the operand. And
// a 0, which with allowzero clear copies the input's extent at that axis;
// resolving that is a separate step from the arithmetic here.
//
// Before, with %shape the constant [-1]:
//   %r = "onnx.Reshape"(%x, %shape)
//       : (tensor<?x4096xf16, #hipsr.mem<device>>, tensor<1xi64>)
//       -> tensor<?xf16, #hipsr.mem<device>>
//
// After, with the types inside the regions left out:
//   %init = hipsr.placeholder(%ctx) ins(%x)
//       {placeholder_type = #hipsr.placeholder_type<normal>}
//       : tensor<?xf16, #hipsr.mem<device>> shape_region {
//   ^bb0(%x_shape):
//     %d0 = shape.get_extent %x_shape, 0
//     %c = arith.constant 4096 : index
//     %n = arith.muli %d0, %c
//     %s = shape.from_extents %n
//     hipsr.shape_yield %s
//   }
//   %r = hipsr.compute(%ctx) ins(%x) outs(%init) {
//   ^bb0(%body_ctx, %in, %dest):
//     %flat = tensor.collapse_shape %in [[0, 1]]
//     hipsr.compute_yield %flat
//   } : tensor<?xf16, #hipsr.mem<device>>
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

// The product of a shape's static extents. Over the target extents it is the
// divisor the inferred hole divides the element count by; over the input's
// shape it is the part of that count already known.
int64_t staticExtentProduct(ArrayRef<int64_t> extents) {
  return llvm::product_of(llvm::make_filter_range(
      extents, [](int64_t extent) { return ShapedType::isStatic(extent); }));
}

// What the reshape produces, read off its own operands rather than the type
// ONNX declared: the target shape states every extent, and the one it writes as
// -1 comes from the element count a reshape preserves. The result aliases the
// input, so it keeps the input's element type and memory space.
FailureOr<RankedTensorType>
inferResultType(onnx::ReshapeOp op, RankedTensorType inputType, Value shape,
                ConversionPatternRewriter &rewriter) {
  DenseIntElementsAttr folded;
  if (!matchPattern(shape, m_Constant(&folded))) {
    return rewriter.notifyMatchFailure(
        op, "expected a target shape that folds to a constant");
  }
  SmallVector<int64_t> extents = llvm::map_to_vector(
      folded.getValues<APInt>(), [](const APInt &entry) -> int64_t {
        int64_t value = entry.getSExtValue();
        return value < 0 ? ShapedType::kDynamic : value;
      });

  // With allowzero clear a 0 copies the input's extent at that axis, which is a
  // separate step, and it would leave the inferred extent nothing to divide by.
  if (llvm::is_contained(extents, 0)) {
    return rewriter.notifyMatchFailure(op, "expected no 0 in the target shape");
  }
  // One equation recovers one unknown, and the rest would take a host read of
  // the operand.
  if (llvm::count(extents, ShapedType::kDynamic) > 1) {
    return rewriter.notifyMatchFailure(
        op, "expected at most one extent left to infer");
  }

  // The hole stays dynamic unless the input pins the element count down.
  if (auto *hole = llvm::find(extents, ShapedType::kDynamic);
      hole != extents.end() && inputType.hasStaticShape()) {
    *hole = inputType.getNumElements() / staticExtentProduct(extents);
  }
  return RankedTensorType::get(extents, inputType.getElementType(),
                               inputType.getEncoding());
}

//===----------------------------------------------------------------------===//
// The destination
//===----------------------------------------------------------------------===//

// `!shape.size` carries an error state that arith has no room for, so an
// extent read off a `!shape.shape` has to be converted to index.
Value buildShapeExtent(OpBuilder &builder, Location loc, Value shape,
                       int64_t axis) {
  Value size = shape::GetExtentOp::create(builder, loc, shape, axis);
  return shape::SizeToIndexOp::create(builder, loc, builder.getIndexType(),
                                      size);
}

// The extent ONNX left to the element count: the input's count over `divisor`.
// Both sides are products, so the input's static extents and the divisor cancel
// where they divide evenly, which is what leaves a plain flatten with no
// division at all. What is left is one multiply per dynamic input extent.
Value buildInferredExtent(OpBuilder &builder, Location loc,
                          RankedTensorType inputType, int64_t divisor,
                          Value inputShape) {
  assert(divisor > 0 && "a 0 result extent is rejected before this point");

  SmallVector<Value> factors;
  for (int64_t axis : llvm::seq<int64_t>(0, inputType.getRank())) {
    if (inputType.isDynamicDim(axis)) {
      factors.push_back(buildShapeExtent(builder, loc, inputShape, axis));
    }
  }

  int64_t staticCount = staticExtentProduct(inputType.getShape());
  if (staticCount % divisor == 0) {
    staticCount /= divisor;
    divisor = 1;
  }
  if (staticCount != 1 || factors.empty()) {
    factors.push_back(
        arith::ConstantIndexOp::create(builder, loc, staticCount));
  }

  Value extent = factors.front();
  for (Value factor : llvm::drop_begin(factors)) {
    extent = arith::MulIOp::create(builder, loc, extent, factor);
  }
  if (divisor == 1) {
    return extent;
  }
  Value divisorValue = arith::ConstantIndexOp::create(builder, loc, divisor);
  return arith::DivUIOp::create(builder, loc, extent, divisorValue);
}

// One value per result axis. At most one extent is left to infer, so the stated
// ones become constants and that single hole is filled from the input's shape.
SmallVector<Value> buildResultExtents(OpBuilder &builder, Location loc,
                                      RankedTensorType inputType,
                                      RankedTensorType resultType,
                                      Value inputShape) {
  SmallVector<Value> values =
      llvm::map_to_vector(resultType.getShape(), [&](int64_t extent) -> Value {
        if (ShapedType::isDynamic(extent)) {
          return nullptr;
        }
        return arith::ConstantIndexOp::create(builder, loc, extent);
      });
  if (auto *inferred = llvm::find(values, Value()); inferred != values.end()) {
    *inferred = buildInferredExtent(builder, loc, inputType,
                                    staticExtentProduct(resultType.getShape()),
                                    inputShape);
  }
  return values;
}

void populateShapeRegion(OpBuilder &builder, PlaceholderOp placeholder,
                         RankedTensorType inputType,
                         RankedTensorType resultType) {
  OpBuilder::InsertionGuard guard(builder);
  Block &block = createPlaceholderShapeBlock(builder, placeholder);
  builder.setInsertionPointToStart(&block);

  Location loc = placeholder.getLoc();
  SmallVector<Value> values =
      buildResultExtents(builder, loc, inputType, resultType,
                         PlaceholderShapeRegionArgs{block}.in(0));
  Value shape = shape::FromExtentsOp::create(
      builder, loc, shape::ShapeType::get(builder.getContext()), values);
  ShapeYieldOp::create(builder, loc, ValueRange{shape});
}

//===----------------------------------------------------------------------===//
// The compute body
//===----------------------------------------------------------------------===//

// The 1-D form `type` flattens to, dynamic when the element count is not a
// compile-time constant.
RankedTensorType flatTensorType(RankedTensorType type, Attribute space) {
  int64_t count =
      type.hasStaticShape() ? type.getNumElements() : ShapedType::kDynamic;
  return RankedTensorType::get({count}, type.getElementType(), space);
}

// Every axis in one group: the grouping that reaches the 1-D form.
SmallVector<ReassociationIndices> flatReassociation(int64_t rank) {
  return {llvm::to_vector<2>(llvm::seq<int64_t>(0, rank))};
}

// Every regroup goes through the 1-D form: one group collapses every input
// axis, one expands the result out of it. A single collapse or expand would
// fit some regroups, but `ComposeExpandOfCollapseOp` already finds those.
//
// `dest` is the placeholder, so an expansion reads the extents it resolved.
Value createReshape(OpBuilder &builder, Location loc, Value input,
                    RankedTensorType inputType, RankedTensorType resultType,
                    Value dest) {
  assert(inputType != resultType &&
         "an identity reshape is dropped before it is given a body");

  // The result took the input's space, so both sides agree on it.
  Attribute space = resultType.getEncoding();
  RankedTensorType flatType = flatTensorType(inputType, space);
  Value flat = input;
  if (inputType.getRank() > 1) {
    flat = tensor::CollapseShapeOp::create(
        builder, loc, flatType, input, flatReassociation(inputType.getRank()));
  }

  // A collapsed extent has to agree with its group on being dynamic, so the
  // collapse cannot pin down an element count only the result side knows.
  if (RankedTensorType resultFlatType = flatTensorType(resultType, space);
      flatType != resultFlatType) {
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
// never widens back to the type ONNX declared.
void populateComputeBody(OpBuilder &builder, ComputeOp computeOp,
                         RankedTensorType inputType,
                         RankedTensorType resultType) {
  OpBuilder::InsertionGuard guard(builder);
  Location loc = computeOp.getLoc();

  // The body's arguments are the operands: ctx, then the inputs, then the
  // outputs.
  TypeRange argTypes = computeOp->getOperandTypes();
  SmallVector<Location> argLocs(argTypes.size(), loc);
  Block *body =
      builder.createBlock(&computeOp.getBody(), {}, argTypes, argLocs);
  builder.setInsertionPointToStart(body);

  Value reshaped = createReshape(builder, loc, body->getArgument(1), inputType,
                                 resultType, body->getArguments().back());
  ComputeYieldOp::create(builder, loc, ValueRange{reshaped});
}

//===----------------------------------------------------------------------===//
// The pattern
//===----------------------------------------------------------------------===//

struct ReshapeToHipsr : public OpConversionPattern<onnx::ReshapeOp> {
  // The converter goes unused: the result type depends on the target shape
  // operand, which a type conversion never sees. Taken so that the pass adds
  // every pattern the same way.
  ReshapeToHipsr(const TypeConverter &, MLIRContext *ctx)
      : OpConversionPattern(ctx) {}

  LogicalResult
  matchAndRewrite(onnx::ReshapeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // allowzero flips what a 0 entry means, giving a different result shape.
    if (op.getAllowzero() != 0) {
      return rewriter.notifyMatchFailure(op, "expected allowzero = 0");
    }
    Value input = adaptor.getData();
    auto inputType = dyn_cast<RankedTensorType>(input.getType());
    if (!inputType) {
      return rewriter.notifyMatchFailure(op, "expected a ranked tensor input");
    }
    FailureOr<RankedTensorType> resultType =
        inferResultType(op, inputType, adaptor.getShape(), rewriter);
    if (failed(resultType)) {
      return failure();
    }

    // An identity reshape needs no destination. Inferring first also catches
    // the ones only the target shape operand reveals to be identities.
    if (inputType == *resultType) {
      rewriter.replaceOp(op, input);
      return success();
    }
    // A static input leaves no extent to infer, so both counts are known.
    if (inputType.hasStaticShape() &&
        inputType.getNumElements() != resultType->getNumElements()) {
      return rewriter.notifyMatchFailure(
          op, "expected the element count to be preserved");
    }
    FailureOr<Value> ctx = getHipsrContextArg(op, rewriter);
    if (failed(ctx)) {
      return failure();
    }

    // The shape region reads no memory, so a normal placeholder is enough.
    Location loc = op.getLoc();
    auto init =
        PlaceholderOp::create(rewriter, loc, TypeRange{*resultType}, *ctx,
                              ValueRange{input}, PlaceholderType::Normal);
    populateShapeRegion(rewriter, init, inputType, *resultType);

    auto computeOp =
        ComputeOp::create(rewriter, loc, TypeRange{*resultType}, *ctx,
                          ValueRange{input}, ValueRange{init.getResult(0)});
    populateComputeBody(rewriter, computeOp, inputType, *resultType);
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
