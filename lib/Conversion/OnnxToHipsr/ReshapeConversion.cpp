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
// Every result extent has to be a compile-time constant, bar the one ONNX
// writes as -1 and leaves to the preserved element count. The result type
// states the static extents and a target shape operand that folds to a
// constant states the rest, so the shape region works from the input's shape
// alone and reads nothing out of memory.
//
// Two forms are rejected rather than lowered. A second inferred extent, since
// one equation recovers one unknown and the rest are only in the operand,
// which it would take a host read to reach. And a 0 entry, which with
// allowzero clear copies the input's extent at that axis: no graph has needed
// it yet, and resolving it is a separate step from the arithmetic here.
//
// The pattern runs in the order below, and the helpers are grouped to match:
//
//   1. Reject the unsupported forms and put the result type in the input's
//      memory space, since the result aliases the input.
//   2. Resolve every result extent, which refines the result type.
//   3. Drop the reshape when the refined type is the input's own.
//   4. Build the placeholder whose shape region resolves the destination.
//   5. Build the hipsr.compute whose body regroups the input into it.
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
//     %n = arith.muli %d0, %c        // the -1 takes the whole element count
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
#include <optional>

namespace mlir {
namespace hipsr {
namespace {

//===----------------------------------------------------------------------===//
// Resolving the result extents
//===----------------------------------------------------------------------===//

// One extent per result axis, with ShapedType::kDynamic marking the axis ONNX
// left to the element count. The result type states the static extents; for a
// dynamic one, a target shape operand that folds to a constant decides.
FailureOr<SmallVector<int64_t>>
resolveResultExtents(onnx::ReshapeOp op, RankedTensorType resultType,
                     Value shape, ConversionPatternRewriter &rewriter) {
  SmallVector<int64_t> entries;
  DenseIntElementsAttr folded;
  if (matchPattern(shape, m_Constant(&folded)) &&
      folded.getNumElements() == resultType.getRank()) {
    entries =
        llvm::map_to_vector(folded.getValues<APInt>(), [](const APInt &entry) {
          return entry.getSExtValue();
        });
  }

  SmallVector<int64_t> extents;
  extents.reserve(resultType.getRank());
  for (int64_t axis : llvm::seq<int64_t>(0, resultType.getRank())) {
    if (!resultType.isDynamicDim(axis)) {
      extents.push_back(resultType.getDimSize(axis));
      continue;
    }
    // An operand that stayed a runtime value says nothing about any axis,
    // which is what a -1 entry already means to ONNX.
    int64_t entry = entries.empty() ? -1 : entries[axis];
    if (entry == 0) {
      return rewriter.notifyMatchFailure(op,
                                         "expected no 0 in the target shape");
    }
    extents.push_back(entry > 0 ? entry : ShapedType::kDynamic);
  }

  if (llvm::count(extents, ShapedType::kDynamic) > 1) {
    return rewriter.notifyMatchFailure(
        op, "expected at most one result extent left to infer");
  }
  return extents;
}

// The divisor for the inferred extent: the product of the extents the result
// already states.
int64_t constantExtentProduct(ArrayRef<int64_t> extents) {
  return llvm::product_of(llvm::make_filter_range(
      extents, [](int64_t extent) { return !ShapedType::isDynamic(extent); }));
}

// Folds the inferred extent when the input's element count is a compile-time
// constant. A dynamic input extent could be anything, and a 0 among the stated
// extents leaves nothing to divide by.
std::optional<int64_t> foldInferredExtent(RankedTensorType inputType,
                                          ArrayRef<int64_t> extents) {
  int64_t divisor = constantExtentProduct(extents);
  if (!inputType.hasStaticShape() || divisor == 0) {
    return std::nullopt;
  }
  return inputType.getNumElements() / divisor;
}

// Writes the resolved extents back into the result type, so that no consumer
// has to rediscover one the target shape operand already gave away.
RankedTensorType refineResultType(RankedTensorType resultType,
                                  RankedTensorType inputType,
                                  ArrayRef<int64_t> extents) {
  int64_t inferred =
      foldInferredExtent(inputType, extents).value_or(ShapedType::kDynamic);
  SmallVector<int64_t> refined =
      llvm::map_to_vector(extents, [&](int64_t extent) {
        return ShapedType::isDynamic(extent) ? inferred : extent;
      });
  return resultType.clone(refined);
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
// Both sides are products, so the input's static extents and the divisor
// cancel where they divide evenly, which is what leaves a plain flatten with
// no division at all. The remaining static extents fold into one constant,
// leaving one multiply per dynamic extent.
Value buildInferredExtent(OpBuilder &builder, Location loc,
                          RankedTensorType inputType, int64_t divisor,
                          Value inputShape) {
  int64_t staticCount = 1;
  SmallVector<Value> factors;
  for (int64_t axis : llvm::seq<int64_t>(0, inputType.getRank())) {
    if (inputType.isDynamicDim(axis)) {
      factors.push_back(buildShapeExtent(builder, loc, inputShape, axis));
      continue;
    }
    staticCount *= inputType.getDimSize(axis);
  }
  if (divisor > 0 && staticCount % divisor == 0) {
    staticCount /= divisor;
    divisor = 1;
  }
  if (staticCount != 1) {
    factors.push_back(
        arith::ConstantIndexOp::create(builder, loc, staticCount));
  }

  Value extent = factors.empty()
                     ? arith::ConstantIndexOp::create(builder, loc, 1)
                     : factors.front();
  for (Value factor : llvm::drop_begin(factors)) {
    extent = arith::MulIOp::create(builder, loc, extent, factor);
  }
  if (divisor == 1) {
    return extent;
  }
  Value divisorValue = arith::ConstantIndexOp::create(builder, loc, divisor);
  return arith::DivUIOp::create(builder, loc, extent, divisorValue);
}

// One value per result axis. Only the inferred extent can still be dynamic,
// and it comes out of the input's shape.
SmallVector<Value> buildResultExtents(OpBuilder &builder, Location loc,
                                      RankedTensorType inputType,
                                      ArrayRef<int64_t> extents,
                                      Value inputShape) {
  SmallVector<Value> values(extents.size());
  std::optional<size_t> inferredAxis;
  for (auto [axis, extent] : llvm::enumerate(extents)) {
    if (ShapedType::isDynamic(extent)) {
      inferredAxis = axis;
      continue;
    }
    values[axis] = arith::ConstantIndexOp::create(builder, loc, extent);
  }
  if (inferredAxis) {
    values[*inferredAxis] = buildInferredExtent(
        builder, loc, inputType, constantExtentProduct(extents), inputShape);
  }
  return values;
}

void populateShapeRegion(OpBuilder &builder, PlaceholderOp placeholder,
                         RankedTensorType inputType,
                         ArrayRef<int64_t> extents) {
  OpBuilder::InsertionGuard guard(builder);
  Block &block = createPlaceholderShapeBlock(builder, placeholder);
  builder.setInsertionPointToStart(&block);

  Location loc = placeholder.getLoc();
  SmallVector<Value> values =
      buildResultExtents(builder, loc, inputType, extents,
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

// A rank-0 side has no axis to collapse, so one op covers the whole reshape.
// A collapse takes its extents from the grouping; an expansion splits one
// source extent over a group, so it carries the target extents instead.
FailureOr<Value> createRank0Regroup(OpBuilder &builder, Location loc,
                                    Value source, RankedTensorType sourceType,
                                    RankedTensorType targetType) {
  std::optional<SmallVector<ReassociationIndices>> reassociation =
      getReassociationIndicesForReshape(sourceType, targetType);
  if (!reassociation) {
    return failure();
  }
  if (sourceType.getRank() > targetType.getRank()) {
    return tensor::CollapseShapeOp::create(builder, loc, targetType, source,
                                           *reassociation)
        .getResult();
  }
  FailureOr<SmallVector<OpFoldResult>> outputShape =
      tensor::ExpandShapeOp::inferOutputShape(
          builder, loc, targetType, *reassociation,
          tensor::getMixedSizes(builder, loc, source));
  if (failed(outputShape)) {
    return failure();
  }
  return tensor::ExpandShapeOp::create(builder, loc, targetType, source,
                                       *reassociation, *outputShape)
      .getResult();
}

// Every regroup goes through the 1-D form: one group collapses every input
// axis, one expands the result out of it. A single collapse or expand would
// fit some regroups, but `ComposeExpandOfCollapseOp` already finds those.
//
// `dest` is the placeholder, so an expansion reads the extents it resolved.
FailureOr<Value> createReshape(OpBuilder &builder, Location loc, Value input,
                               RankedTensorType inputType,
                               RankedTensorType resultType, Value dest) {
  assert(inputType != resultType &&
         "an identity reshape is dropped before it is given a body");
  if (inputType.getRank() == 0 || resultType.getRank() == 0) {
    return createRank0Regroup(builder, loc, input, inputType, resultType);
  }

  // Both sides share a space: the result took the input's, and only rank 0
  // names none.
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

// The compute and the destination both carry the refined type, so the body
// never widens back to the type ONNX declared.
LogicalResult populateComputeBody(OpBuilder &builder, ComputeOp computeOp,
                                  RankedTensorType inputType,
                                  RankedTensorType refinedType) {
  OpBuilder::InsertionGuard guard(builder);
  Location loc = computeOp.getLoc();

  // The ODS builder only reserves the region. Its arguments are the operands:
  // ctx, then the input, then the destination the compute writes to.
  TypeRange argTypes = computeOp->getOperandTypes();
  SmallVector<Location> argLocs(argTypes.size(), loc);
  Block *body =
      builder.createBlock(&computeOp.getBody(), {}, argTypes, argLocs);
  builder.setInsertionPointToStart(body);

  FailureOr<Value> reshaped =
      createReshape(builder, loc, body->getArgument(1), inputType, refinedType,
                    body->getArguments().back());
  if (failed(reshaped)) {
    return failure();
  }
  ComputeYieldOp::create(builder, loc, ValueRange{*reshaped});
  return success();
}

//===----------------------------------------------------------------------===//
// The pattern
//===----------------------------------------------------------------------===//

// The result aliases the input, so it lives in the input's space whatever ONNX
// declared. Rejects the forms that would need a copy or a layout change.
FailureOr<RankedTensorType>
resultTypeInInputSpace(onnx::ReshapeOp op, RankedTensorType inputType,
                       ConversionPatternRewriter &rewriter) {
  auto resultType = dyn_cast<RankedTensorType>(op.getReshaped().getType());
  if (!resultType) {
    return rewriter.notifyMatchFailure(op, "expected ranked tensor types");
  }
  if (inputType.getElementType() != resultType.getElementType()) {
    return rewriter.notifyMatchFailure(
        op, "expected matching input and result element types");
  }
  // Only a rank-0 input names no space: the type converter leaves scalar host
  // roots alone so that arith.constant stays legal. Default those to device.
  Attribute space = inputType.getEncoding();
  if (!space) {
    space = MemorySpaceAttr::get(rewriter.getContext(), MemorySpace::Device);
  }
  if (Attribute declared = resultType.getEncoding();
      declared && declared != space) {
    return rewriter.notifyMatchFailure(
        op, "expected the result in the input's memory space");
  }
  return resultType.cloneWithEncoding(space);
}

struct ReshapeToHipsr : public OpConversionPattern<onnx::ReshapeOp> {
  // The converter goes unused: the result type depends on the target shape
  // operand, which a type conversion never sees. Taken so that the pass adds
  // every pattern the same way.
  ReshapeToHipsr(const TypeConverter &typeConverter, MLIRContext *ctx)
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
      return rewriter.notifyMatchFailure(op, "expected ranked tensor types");
    }
    FailureOr<RankedTensorType> resultType =
        resultTypeInInputSpace(op, inputType, rewriter);
    if (failed(resultType)) {
      return failure();
    }
    FailureOr<SmallVector<int64_t>> extents =
        resolveResultExtents(op, *resultType, adaptor.getShape(), rewriter);
    if (failed(extents)) {
      return failure();
    }
    RankedTensorType refinedType =
        refineResultType(*resultType, inputType, *extents);

    // An identity reshape needs no destination. Refining first also catches
    // the ones only the target shape operand reveals to be identities.
    if (inputType == refinedType) {
      rewriter.replaceOp(op, input);
      return success();
    }
    // Two static shapes that disagree on the element count cannot describe the
    // same data.
    if (inputType.hasStaticShape() && refinedType.hasStaticShape() &&
        inputType.getNumElements() != refinedType.getNumElements()) {
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
        PlaceholderOp::create(rewriter, loc, TypeRange{refinedType}, *ctx,
                              ValueRange{input}, PlaceholderType::Normal);
    populateShapeRegion(rewriter, init, inputType, *extents);

    auto computeOp =
        ComputeOp::create(rewriter, loc, TypeRange{refinedType}, *ctx,
                          ValueRange{input}, ValueRange{init.getResult(0)});
    if (failed(
            populateComputeBody(rewriter, computeOp, inputType, refinedType))) {
      return failure();
    }
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
