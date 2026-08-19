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
// The target shape operand is not simply a list of extents. With allowzero
// clear, ONNX gives two entries a meaning of their own: 0 copies the input's
// extent at that axis, and -1 asks for the one extent that the preserved
// element count can recover. Resolving those two is most of the work here.
//
// The destination's shape then comes from one of two places.
//
// Derived. Every extent follows from the result type's static dims, a target
// shape operand that folds to a constant, and the input's shape. Nothing has
// to be read out of memory, so a normal placeholder is enough.
//
// Host. Otherwise two or more extents are left inferred, and the element count
// recovers only one, so the shape region has to read the target shape operand
// itself. That is a host read: the operand must be in host memory and the
// placeholder a barrier. A shape operand anywhere else is rejected; nothing
// here copies it to the host.
//
// The pattern runs in the order below, and the helpers are grouped to match:
//
//   1. Reject the unsupported forms and put the result type in the input's
//      memory space, since the result aliases the input.
//   2. Classify every result extent. That refines the result type and picks
//      which of the two destinations follows.
//   3. Drop the reshape when the refined type is the input's own.
//   4. Build the placeholder whose shape region resolves the destination.
//   5. Build the hipsr.compute whose body regroups the input into it.
//
// Before:
//   %r = "onnx.Reshape"(%x, %shape)
//       : (tensor<?x?x4xf16, #hipsr.mem<device>>,
//          tensor<2xi64, #hipsr.mem<host>>)
//       -> tensor<?x?xf16, #hipsr.mem<device>>
//
// After, with the types inside the regions left out:
//   %init = hipsr.placeholder(%ctx) ins(%x, %shape)
//       {placeholder_type = #hipsr.placeholder_type<barrier>}
//       : tensor<?x?xf16, #hipsr.mem<device>> shape_region {
//   ^bb0(%shape_ctx, %x_arg, %shape_arg):
//     %n = arith.muli ...            // the input's element count
//     %e0 = arith.index_cast tensor.extract %shape_arg[0]
//     %e1 = arith.index_cast tensor.extract %shape_arg[1]
//     ...                            // resolve a 0 or a -1 entry
//     %s = shape.from_extents %e0, %e1
//     hipsr.shape_yield %s
//   }
//   %r = hipsr.compute(%ctx) ins(%x) outs(%init) {
//   ^bb0(%body_ctx, %in, %dest):
//     %flat = tensor.collapse_shape %in [[0, 1, 2]]
//     %d0 = tensor.dim %dest, 0      // the extents the placeholder resolved
//     %d1 = tensor.dim %dest, 1
//     %out = tensor.expand_shape %flat [[0, 1]] output_shape [%d0, %d1]
//     hipsr.compute_yield %out
//   } : tensor<?x?xf16, #hipsr.mem<device>>
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

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"

#include <cassert>
#include <functional>
#include <optional>

namespace mlir {
namespace hipsr {
namespace {

//===----------------------------------------------------------------------===//
// Classifying the result extents
//===----------------------------------------------------------------------===//

// Where one result extent comes from. The three kinds are the three meanings a
// target shape entry can have: an extent, a copy of the input's, or the -1 the
// element count has to recover.
struct TargetExtent {
  enum class Kind { Constant, CopiesInput, Inferred };
  Kind kind = Kind::Inferred;
  // The extent for Constant, the input axis copied for CopiesInput.
  int64_t value = 0;

  static TargetExtent constant(int64_t extent) {
    return {Kind::Constant, extent};
  }
  static TargetExtent copiesInput(int64_t axis) {
    return {Kind::CopiesInput, axis};
  }
  static TargetExtent inferred() { return {}; }
};

auto isKind(TargetExtent::Kind kind) {
  return [kind](const TargetExtent &extent) { return extent.kind == kind; };
}

// Classifies every result extent without reading the target shape operand at
// runtime. A static result dim gives its own extent; for a dynamic one, an
// operand that folds to a constant decides.
//
// Nothing means two or more extents are left inferred, which the element count
// cannot recover. That is not an error but the choice of destination: those
// extents have to be read on the host instead.
std::optional<SmallVector<TargetExtent>>
classifyTargetExtents(RankedTensorType resultType, RankedTensorType inputType,
                      Value shape) {
  SmallVector<int64_t> entries;
  DenseIntElementsAttr folded;
  if (matchPattern(shape, m_Constant(&folded)) &&
      folded.getNumElements() == resultType.getRank()) {
    entries =
        llvm::map_to_vector(folded.getValues<APInt>(), [](const APInt &entry) {
          return entry.getSExtValue();
        });
  }

  SmallVector<TargetExtent> classified;
  classified.reserve(resultType.getRank());
  for (int64_t axis : llvm::seq<int64_t>(0, resultType.getRank())) {
    if (!resultType.isDynamicDim(axis)) {
      classified.push_back(TargetExtent::constant(resultType.getDimSize(axis)));
      continue;
    }
    // An operand that stayed a runtime value says nothing about any axis,
    // which is what a -1 entry already means to ONNX.
    int64_t entry = entries.empty() ? -1 : entries[axis];
    if (entry > 0) {
      classified.push_back(TargetExtent::constant(entry));
    } else if (entry == 0 && axis < inputType.getRank()) {
      classified.push_back(
          inputType.isDynamicDim(axis)
              ? TargetExtent::copiesInput(axis)
              : TargetExtent::constant(inputType.getDimSize(axis)));
    } else {
      classified.push_back(TargetExtent::inferred());
    }
  }

  if (llvm::count_if(classified, isKind(TargetExtent::Kind::Inferred)) > 1) {
    return std::nullopt;
  }
  return classified;
}

// The axes whose extent the result copies from the input.
llvm::SmallDenseSet<int64_t>
copiedInputAxes(ArrayRef<TargetExtent> classified) {
  auto copies = llvm::make_filter_range(
      classified, isKind(TargetExtent::Kind::CopiesInput));
  llvm::SmallDenseSet<int64_t> copied;
  copied.insert_range(
      llvm::map_range(copies, std::mem_fn(&TargetExtent::value)));
  return copied;
}

// The divisor for an inferred extent, which is the element count over this.
int64_t constantExtentProduct(ArrayRef<TargetExtent> classified) {
  auto constants =
      llvm::make_filter_range(classified, isKind(TargetExtent::Kind::Constant));
  return llvm::product_of(
      llvm::map_range(constants, std::mem_fn(&TargetExtent::value)));
}

// Folds the inferred extent when every axis it divides is static. A copied
// axis cancels between the element count and the divisor, which is how a
// dynamic input axis beside an inferred extent still folds.
std::optional<int64_t> foldInferredExtent(RankedTensorType inputType,
                                          ArrayRef<TargetExtent> classified) {
  llvm::SmallDenseSet<int64_t> copied = copiedInputAxes(classified);
  auto axes = llvm::seq<int64_t>(0, inputType.getRank());
  auto divided = llvm::make_filter_range(
      axes, [&](int64_t axis) { return !copied.contains(axis); });
  if (llvm::any_of(divided, [&](int64_t axis) {
        return inputType.isDynamicDim(axis);
      })) {
    return std::nullopt;
  }
  int64_t divisor = constantExtentProduct(classified);
  if (divisor == 0) {
    return std::nullopt;
  }
  auto extents = llvm::map_range(
      divided, [&](int64_t axis) { return inputType.getDimSize(axis); });
  return llvm::product_of(extents) / divisor;
}

// Writes the classified extents back into the result type, so that no consumer
// has to rediscover one the target shape operand already gave away.
RankedTensorType refineResultType(RankedTensorType resultType,
                                  RankedTensorType inputType,
                                  ArrayRef<TargetExtent> classified) {
  SmallVector<int64_t> refined = llvm::map_to_vector(
      classified, [&](const TargetExtent &extent) -> int64_t {
        switch (extent.kind) {
        case TargetExtent::Kind::Constant:
          return extent.value;
        case TargetExtent::Kind::CopiesInput:
          return inputType.getDimSize(extent.value);
        case TargetExtent::Kind::Inferred:
          return foldInferredExtent(inputType, classified)
              .value_or(ShapedType::kDynamic);
        }
        llvm_unreachable("unhandled target extent kind");
      });
  return resultType.clone(refined);
}

//===----------------------------------------------------------------------===//
// Element-count arithmetic, emitted by both shape regions
//===----------------------------------------------------------------------===//

// Multiplies `factors`. An empty list gives 1.
Value buildProduct(OpBuilder &builder, Location loc, ArrayRef<Value> factors) {
  if (factors.empty()) {
    return arith::ConstantIndexOp::create(builder, loc, 1);
  }
  Value product = factors.front();
  for (Value factor : factors.drop_front()) {
    product = arith::MulIOp::create(builder, loc, product, factor);
  }
  return product;
}

// The product of the input's extents outside `skipped`. The static ones fold
// into a single constant, leaving one multiply per dynamic extent.
Value buildElementCount(OpBuilder &builder, Location loc,
                        RankedTensorType inputType,
                        const llvm::SmallDenseSet<int64_t> &skipped,
                        function_ref<Value(int64_t)> extentAt) {
  int64_t staticCount = 1;
  SmallVector<Value> factors;
  for (int64_t axis : llvm::seq<int64_t>(0, inputType.getRank())) {
    if (skipped.contains(axis)) {
      continue;
    }
    if (!inputType.isDynamicDim(axis)) {
      staticCount *= inputType.getDimSize(axis);
      continue;
    }
    factors.push_back(extentAt(axis));
  }
  if (staticCount != 1) {
    factors.push_back(
        arith::ConstantIndexOp::create(builder, loc, staticCount));
  }
  return buildProduct(builder, loc, factors);
}

//===----------------------------------------------------------------------===//
// Destination whose shape is derived from the input's
//===----------------------------------------------------------------------===//

// `!shape.size` carries an error state that arith has no room for, so an
// extent read off a `!shape.shape` has to be converted to index.
Value buildShapeExtent(OpBuilder &builder, Location loc, Value shape,
                       int64_t axis) {
  Value size = shape::GetExtentOp::create(builder, loc, shape, axis);
  return shape::SizeToIndexOp::create(builder, loc, builder.getIndexType(),
                                      size);
}

// One extent per result axis. Every one is a constant or comes out of
// `inputShape`, so the extents can still be dynamic even though the
// classification is complete.
SmallVector<Value> buildResultExtents(OpBuilder &builder, Location loc,
                                      RankedTensorType inputType,
                                      ArrayRef<TargetExtent> classified,
                                      Value inputShape) {
  SmallVector<Value> extents(classified.size());
  std::optional<int64_t> inferredAxis;
  for (auto [axis, extent] : llvm::enumerate(classified)) {
    switch (extent.kind) {
    case TargetExtent::Kind::Constant:
      extents[axis] =
          arith::ConstantIndexOp::create(builder, loc, extent.value);
      break;
    case TargetExtent::Kind::CopiesInput:
      extents[axis] = buildShapeExtent(builder, loc, inputShape, extent.value);
      break;
    case TargetExtent::Kind::Inferred:
      inferredAxis = axis;
      break;
    }
  }
  if (!inferredAxis) {
    return extents;
  }

  Value elementCount = buildElementCount(
      builder, loc, inputType, copiedInputAxes(classified), [&](int64_t axis) {
        return buildShapeExtent(builder, loc, inputShape, axis);
      });
  if (int64_t divisor = constantExtentProduct(classified); divisor != 1) {
    Value divisorValue = arith::ConstantIndexOp::create(builder, loc, divisor);
    elementCount =
        arith::DivUIOp::create(builder, loc, elementCount, divisorValue);
  }
  extents[*inferredAxis] = elementCount;
  return extents;
}

void populateDerivedShapeRegion(OpBuilder &builder, PlaceholderOp placeholder,
                                RankedTensorType inputType,
                                ArrayRef<TargetExtent> classified) {
  OpBuilder::InsertionGuard guard(builder);
  Block &block = createPlaceholderShapeBlock(builder, placeholder);
  builder.setInsertionPointToStart(&block);

  Location loc = placeholder.getLoc();
  SmallVector<Value> extents =
      buildResultExtents(builder, loc, inputType, classified,
                         PlaceholderShapeRegionArgs{block}.in(0));
  yieldShapeFromExtents(builder, loc, extents);
}

// The region needs only the input's shape, so a normal placeholder does.
PlaceholderOp createDerivedShapePlaceholder(OpBuilder &builder, Location loc,
                                            Value ctx, Value input,
                                            RankedTensorType inputType,
                                            RankedTensorType destType,
                                            ArrayRef<TargetExtent> classified) {
  auto init = PlaceholderOp::create(builder, loc, TypeRange{destType}, ctx,
                                    ValueRange{input}, PlaceholderType::Normal);
  populateDerivedShapeRegion(builder, init, inputType, classified);
  return init;
}

//===----------------------------------------------------------------------===//
// Destination whose shape is read on the host
//===----------------------------------------------------------------------===//

// One extent per result axis, read out of the target shape operand. Resolving
// the 0 and -1 entries falls to the region here, since only the runtime holds
// the values they depend on.
void populateHostShapeRegion(OpBuilder &builder, PlaceholderOp placeholder,
                             RankedTensorType inputType, int64_t resultRank) {
  OpBuilder::InsertionGuard guard(builder);
  Block &block = createPlaceholderShapeBlock(builder, placeholder);
  builder.setInsertionPointToStart(&block);

  Location loc = placeholder.getLoc();
  // A barrier region takes the operands themselves rather than their shapes,
  // which is what makes the target extents readable at all.
  PlaceholderShapeRegionArgs args{block};
  Value input = args.in(0);
  Value target = args.in(1);

  // The region holds the input tensor, not a `!shape.shape`, so a dynamic
  // extent comes from tensor.dim.
  auto inputExtent = [&](int64_t axis) -> Value {
    if (inputType.isDynamicDim(axis)) {
      return tensor::DimOp::create(builder, loc, input, axis);
    }
    return arith::ConstantIndexOp::create(builder, loc,
                                          inputType.getDimSize(axis));
  };

  Value zero = arith::ConstantIndexOp::create(builder, loc, 0);
  Value one = arith::ConstantIndexOp::create(builder, loc, 1);
  Value count =
      buildElementCount(builder, loc, inputType, /*skipped=*/{}, inputExtent);

  // An inferred axis contributes 1 to the divisor rather than its entry, so
  // the element count divides by the extents that are actually stated.
  SmallVector<Value> stated(resultRank);
  SmallVector<Value> isInferred(resultRank);
  SmallVector<Value> divisorFactors;
  for (int64_t axis : llvm::seq<int64_t>(0, resultRank)) {
    Value position = arith::ConstantIndexOp::create(builder, loc, axis);
    Value entry =
        tensor::ExtractOp::create(builder, loc, target, ValueRange{position});
    Value extent =
        arith::IndexCastOp::create(builder, loc, builder.getIndexType(), entry);
    if (axis < inputType.getRank()) {
      Value copies = arith::CmpIOp::create(
          builder, loc, arith::CmpIPredicate::eq, extent, zero);
      extent = arith::SelectOp::create(builder, loc, copies, inputExtent(axis),
                                       extent);
    }
    stated[axis] = extent;
    isInferred[axis] = arith::CmpIOp::create(
        builder, loc, arith::CmpIPredicate::slt, extent, zero);
    divisorFactors.push_back(
        arith::SelectOp::create(builder, loc, isInferred[axis], one, extent));
  }

  Value inferred = arith::DivUIOp::create(
      builder, loc, count, buildProduct(builder, loc, divisorFactors));
  SmallVector<Value> extents = llvm::map_to_vector(
      llvm::seq<int64_t>(0, resultRank), [&](int64_t axis) -> Value {
        return arith::SelectOp::create(builder, loc, isInferred[axis], inferred,
                                       stated[axis]);
      });
  yieldShapeFromExtents(builder, loc, extents);
}

// The region has to read the shape operand, which is a host read: the operand
// must be in host memory and the placeholder a barrier to order it.
FailureOr<PlaceholderOp>
createHostShapePlaceholder(onnx::ReshapeOp op, Value target,
                           ConversionPatternRewriter &rewriter, Value ctx,
                           Value input, RankedTensorType inputType,
                           RankedTensorType destType) {
  auto targetType = dyn_cast<RankedTensorType>(target.getType());
  if (!targetType || targetType.getRank() != 1 ||
      !targetType.getElementType().isInteger(64)) {
    return rewriter.notifyMatchFailure(
        op, "expected a rank-1 i64 target shape operand");
  }
  if (targetType.getDimSize(0) != destType.getRank()) {
    return rewriter.notifyMatchFailure(
        op, "expected one target shape entry per result axis");
  }
  auto space = dyn_cast_if_present<MemorySpaceAttr>(targetType.getEncoding());
  if (!space || space.getValue() != MemorySpace::Host) {
    return rewriter.notifyMatchFailure(
        op, "expected the target shape operand in host memory");
  }

  Location loc = op.getLoc();
  auto init = PlaceholderOp::create(rewriter, loc, TypeRange{destType}, ctx,
                                    ValueRange{input, target},
                                    PlaceholderType::Barrier);
  populateHostShapeRegion(rewriter, init, inputType, destType.getRank());
  return init;
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
  Block &body = createComputeBodyBlock(builder, computeOp);
  Value input = computeBodyInput(body, 0);
  // The lone output argument, which is the placeholder the compute writes to.
  Value dest = body.getArguments().back();

  FailureOr<Value> reshaped =
      createReshape(builder, loc, input, inputType, refinedType, dest);
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

    // The classification refines the result type and picks the destination.
    std::optional<SmallVector<TargetExtent>> classified =
        classifyTargetExtents(*resultType, inputType, adaptor.getShape());
    RankedTensorType refinedType =
        classified ? refineResultType(*resultType, inputType, *classified)
                   : *resultType;

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

    Location loc = op.getLoc();
    FailureOr<PlaceholderOp> init =
        classified
            ? createDerivedShapePlaceholder(rewriter, loc, *ctx, input,
                                            inputType, refinedType, *classified)
            : createHostShapePlaceholder(op, adaptor.getShape(), rewriter, *ctx,
                                         input, inputType, refinedType);
    if (failed(init)) {
      return failure();
    }

    auto computeOp =
        ComputeOp::create(rewriter, loc, TypeRange{refinedType}, *ctx,
                          ValueRange{input}, ValueRange{init->getResult(0)});
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
