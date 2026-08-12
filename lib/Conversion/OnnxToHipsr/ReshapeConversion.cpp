/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- ReshapeConversion.cpp - Convert onnx.Reshape to hipsr.compute -----===//
//
// onnx.Reshape regroups extents in row-major order without moving data. The
// conversion emits a tensor.collapse_shape and a tensor.expand_shape through
// the 1-D form, inside a hipsr.compute. Both ops bufferize to memrefs that
// alias their source, so nothing runs on the device, and canonicalization
// composes the pair into a single op when one can express the regroup.
//
// The destination has to be known one of two ways; nothing else is supported.
//
// Compile time. The result type gives its static extents and a target shape
// operand that folds to a constant gives the rest. One extent may still be
// unknown: a reshape preserves the element count, so it is that count divided
// by the others. The shape region only needs the input's shape here, so a
// normal placeholder is enough.
//
// Runtime. Otherwise the extents are only in the target shape operand, which
// must be a host tensor the shape region can read. That needs a barrier
// placeholder, whose shape region takes the operands rather than their shapes.
// A shape operand anywhere else is rejected; nothing here copies it to the
// host.
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

#include <functional>
#include <optional>

namespace mlir {
namespace hipsr {
namespace {

//===----------------------------------------------------------------------===//
// Naming the result extents at compile time
//===----------------------------------------------------------------------===//

// Where a result extent comes from when the target shape needs no runtime
// read. ONNX allows one -1 entry, which `Inferred` marks; the rest are either
// known outright or copied from the input.
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

// Names every result extent without reading the target shape operand at
// runtime. A static result dim gives its own extent. For a dynamic one, a
// target shape operand that folds to a constant decides: with allowzero clear,
// a positive entry is the extent, a 0 copies the input's extent at that axis,
// and a -1 says nothing. Fails when more than one extent is left unnamed,
// since the preserved element count recovers only one.
FailureOr<SmallVector<TargetExtent>>
nameTargetExtents(RankedTensorType resultType, RankedTensorType inputType,
                  Value shape) {
  SmallVector<int64_t> stated;
  DenseIntElementsAttr entries;
  if (matchPattern(shape, m_Constant(&entries)) &&
      entries.getNumElements() == resultType.getRank()) {
    stated =
        llvm::map_to_vector(entries.getValues<APInt>(), [](const APInt &entry) {
          return entry.getSExtValue();
        });
  }

  SmallVector<TargetExtent> named;
  named.reserve(resultType.getRank());
  for (int64_t axis : llvm::seq<int64_t>(0, resultType.getRank())) {
    if (!resultType.isDynamicDim(axis)) {
      named.push_back(TargetExtent::constant(resultType.getDimSize(axis)));
      continue;
    }
    // An operand that stayed a runtime value says nothing about any axis,
    // which is what a -1 entry already means to ONNX.
    int64_t entry = stated.empty() ? -1 : stated[axis];
    if (entry > 0) {
      named.push_back(TargetExtent::constant(entry));
    } else if (entry == 0 && axis < inputType.getRank()) {
      named.push_back(inputType.isDynamicDim(axis)
                          ? TargetExtent::copiesInput(axis)
                          : TargetExtent::constant(inputType.getDimSize(axis)));
    } else {
      named.push_back(TargetExtent::inferred());
    }
  }

  if (llvm::count_if(named, isKind(TargetExtent::Kind::Inferred)) > 1) {
    return failure();
  }
  return named;
}

// The axes whose extent the result copies from the input.
llvm::SmallDenseSet<int64_t> copiedInputAxes(ArrayRef<TargetExtent> named) {
  auto copies =
      llvm::make_filter_range(named, isKind(TargetExtent::Kind::CopiesInput));
  llvm::SmallDenseSet<int64_t> copied;
  copied.insert_range(
      llvm::map_range(copies, std::mem_fn(&TargetExtent::value)));
  return copied;
}

// The product of the known extents. The inferred extent is the element count
// divided by this.
int64_t statedProduct(ArrayRef<TargetExtent> named) {
  auto stated =
      llvm::make_filter_range(named, isKind(TargetExtent::Kind::Constant));
  return llvm::product_of(
      llvm::map_range(stated, std::mem_fn(&TargetExtent::value)));
}

// Returns the inferred extent when nothing it divides is dynamic. A copied
// axis appears in the element count and in the divisor, so the two cancel and
// only the input's other extents are left over the divisor. That is how a
// dynamic axis beside an inferred one still folds.
std::optional<int64_t> foldInferredExtent(RankedTensorType inputType,
                                          ArrayRef<TargetExtent> named) {
  llvm::SmallDenseSet<int64_t> copied = copiedInputAxes(named);
  auto axes = llvm::seq<int64_t>(0, inputType.getRank());
  auto divided = llvm::make_filter_range(
      axes, [&](int64_t axis) { return !copied.contains(axis); });
  if (llvm::any_of(divided, [&](int64_t axis) {
        return inputType.isDynamicDim(axis);
      })) {
    return std::nullopt;
  }
  int64_t divisor = statedProduct(named);
  if (divisor == 0) {
    return std::nullopt;
  }
  auto extents = llvm::map_range(
      divided, [&](int64_t axis) { return inputType.getDimSize(axis); });
  return llvm::product_of(extents) / divisor;
}

// Puts the named extents into the result type. Everything downstream takes
// that type, so an extent recovered here is one no consumer has to rediscover.
RankedTensorType refineResultType(RankedTensorType resultType,
                                  RankedTensorType inputType,
                                  ArrayRef<TargetExtent> named) {
  SmallVector<int64_t> refined =
      llvm::map_to_vector(named, [&](const TargetExtent &extent) -> int64_t {
        switch (extent.kind) {
        case TargetExtent::Kind::Constant:
          return extent.value;
        case TargetExtent::Kind::CopiesInput:
          return inputType.getDimSize(extent.value);
        case TargetExtent::Kind::Inferred:
          return foldInferredExtent(inputType, named)
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
// Destination whose extents are named at compile time
//===----------------------------------------------------------------------===//

// Extents of a `!shape.shape` are `!shape.size`, which carries an error state
// that arith has no room for, so reading one means converting it to index.
Value getShapeExtent(OpBuilder &builder, Location loc, Value shape,
                     int64_t axis) {
  Value size = shape::GetExtentOp::create(builder, loc, shape, axis);
  return shape::SizeToIndexOp::create(builder, loc, builder.getIndexType(),
                                      size);
}

// Returns one extent per result axis, reading the input's dynamic extents out
// of `inputShape`.
SmallVector<Value> buildResultExtents(OpBuilder &builder, Location loc,
                                      RankedTensorType inputType,
                                      ArrayRef<TargetExtent> named,
                                      Value inputShape) {
  SmallVector<Value> extents(named.size());
  std::optional<int64_t> inferredAxis;
  for (auto [axis, extent] : llvm::enumerate(named)) {
    switch (extent.kind) {
    case TargetExtent::Kind::Constant:
      extents[axis] =
          arith::ConstantIndexOp::create(builder, loc, extent.value);
      break;
    case TargetExtent::Kind::CopiesInput:
      extents[axis] = getShapeExtent(builder, loc, inputShape, extent.value);
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
      builder, loc, inputType, copiedInputAxes(named), [&](int64_t axis) {
        return getShapeExtent(builder, loc, inputShape, axis);
      });
  if (int64_t divisor = statedProduct(named); divisor != 1) {
    Value divisorValue = arith::ConstantIndexOp::create(builder, loc, divisor);
    elementCount =
        arith::DivUIOp::create(builder, loc, elementCount, divisorValue);
  }
  extents[*inferredAxis] = elementCount;
  return extents;
}

void populateKnownShapeRegion(OpBuilder &builder, PlaceholderOp placeholder,
                              RankedTensorType inputType,
                              ArrayRef<TargetExtent> named) {
  OpBuilder::InsertionGuard guard(builder);
  Block &block = createPlaceholderShapeBlock(builder, placeholder);
  builder.setInsertionPointToStart(&block);

  Location loc = placeholder.getLoc();
  SmallVector<Value> extents = buildResultExtents(
      builder, loc, inputType, named, PlaceholderShapeRegionArgs{block}.in(0));
  yieldShapeFromExtents(builder, loc, extents);
}

// Every extent is named at compile time, so the shape region needs nothing but
// the input's shape and the placeholder stays normal.
PlaceholderOp createKnownShapePlaceholder(OpBuilder &builder, Location loc,
                                          Value ctx, Value input,
                                          RankedTensorType inputType,
                                          RankedTensorType destType,
                                          ArrayRef<TargetExtent> named) {
  auto init = PlaceholderOp::create(builder, loc, TypeRange{destType}, ctx,
                                    ValueRange{input}, PlaceholderType::Normal);
  populateKnownShapeRegion(builder, init, inputType, named);
  return init;
}

//===----------------------------------------------------------------------===//
// Destination whose extents are read on the host
//===----------------------------------------------------------------------===//

// Reads one extent per result axis out of the target shape operand. An entry
// is not always the extent: with allowzero clear a 0 copies the input's extent
// at that axis, and a -1 is the element count divided by the rest. Both are
// resolved here, against values only the runtime has.
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

  // The region holds the input tensor here, not a `!shape.shape`, so a dynamic
  // extent comes from tensor.dim. A static one is already known.
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

// More than one extent is left unnamed, so the shape region has to read the
// shape operand. That is a host read, which needs the operand in host memory
// and a barrier placeholder to order it.
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

// Returns the 1-D form `type` flattens to, dynamic when the element count is
// not a compile-time constant.
RankedTensorType flatTensorType(RankedTensorType type, Attribute space) {
  int64_t count =
      type.hasStaticShape() ? type.getNumElements() : ShapedType::kDynamic;
  return RankedTensorType::get({count}, type.getElementType(), space);
}

// The grouping the 1-D form uses on both sides: every axis in one group.
SmallVector<ReassociationIndices> wholeRank(int64_t rank) {
  return {llvm::to_vector<2>(llvm::seq<int64_t>(0, rank))};
}

// Regroups a rank-0 side, the one case with no 1-D form to go through. A
// collapse takes its extents from the grouping, but an expansion splits one
// source extent over a group, so it carries the target extents instead.
FailureOr<Value> createRegroup(OpBuilder &builder, Location loc, Value source,
                               RankedTensorType sourceType,
                               RankedTensorType targetType,
                               ArrayRef<ReassociationIndices> reassociation) {
  if (sourceType.getRank() > targetType.getRank()) {
    return tensor::CollapseShapeOp::create(builder, loc, targetType, source,
                                           reassociation)
        .getResult();
  }
  FailureOr<SmallVector<OpFoldResult>> outputShape =
      tensor::ExpandShapeOp::inferOutputShape(
          builder, loc, targetType, reassociation,
          tensor::getMixedSizes(builder, loc, source));
  if (failed(outputShape)) {
    return failure();
  }
  return tensor::ExpandShapeOp::create(builder, loc, targetType, source,
                                       reassociation, *outputShape)
      .getResult();
}

// Every regroup goes the same way through the 1-D form: one group collapses
// every input axis, one expands the result out of it. Some regroups would fit
// a single collapse or expand instead, but `ComposeExpandOfCollapseOp` finds
// those, so looking for them here would repeat its work. Both halves alias
// either way.
//
// A rank-0 side has no axis to collapse, so it regroups directly.
//
// `dest` is the placeholder, typed as `resultType`, so an expansion reads the
// extents it resolved back off it.
FailureOr<Value> createReshape(OpBuilder &builder, Location loc, Value input,
                               RankedTensorType inputType,
                               RankedTensorType resultType, Value dest) {
  if (inputType == resultType) {
    return input;
  }
  if (inputType.getRank() == 0 || resultType.getRank() == 0) {
    std::optional<SmallVector<ReassociationIndices>> reassociation =
        getReassociationIndicesForReshape(inputType, resultType);
    if (!reassociation) {
      return failure();
    }
    return createRegroup(builder, loc, input, inputType, resultType,
                         *reassociation);
  }

  Attribute space = resultType.getEncoding();
  RankedTensorType flatType = flatTensorType(inputType, space);
  Value flat = input;
  if (inputType.getRank() > 1) {
    flat = tensor::CollapseShapeOp::create(builder, loc, flatType, input,
                                           wholeRank(inputType.getRank()));
  }

  // One side can pin the element count down while the other leaves it dynamic.
  // A collapsed extent has to agree with its group on being dynamic, so the
  // collapse cannot carry the refinement itself and a cast follows it.
  if (RankedTensorType resultFlatType = flatTensorType(resultType, space);
      flatType != resultFlatType) {
    flat = tensor::CastOp::create(builder, loc, resultFlatType, flat);
    flatType = resultFlatType;
  }
  if (flatType == resultType) {
    return flat;
  }

  return tensor::ExpandShapeOp::create(
             builder, loc, resultType, flat, wholeRank(resultType.getRank()),
             tensor::getMixedSizes(builder, loc, dest))
      .getResult();
}

// The refined type is what the compute and the destination both carry, so an
// expansion reads its extents off `dest` and the body needs no widening back
// to the type ONNX declared.
LogicalResult populateComputeBody(OpBuilder &builder, ComputeOp computeOp,
                                  RankedTensorType inputType,
                                  RankedTensorType refinedType) {
  OpBuilder::InsertionGuard guard(builder);
  Location loc = computeOp.getLoc();
  Block &body = createComputeBodyBlock(builder, computeOp);
  Value input = computeBodyInput(body, 0);

  FailureOr<Value> reshaped = createReshape(
      builder, loc, input, inputType, refinedType, body.getArguments().back());
  if (failed(reshaped)) {
    return failure();
  }
  ComputeYieldOp::create(builder, loc, ValueRange{*reshaped});
  return success();
}

//===----------------------------------------------------------------------===//
// The pattern
//===----------------------------------------------------------------------===//

// Rejects the forms the conversion does not handle and returns the result type
// in the memory space the data will actually live in.
FailureOr<RankedTensorType>
resolveResultType(onnx::ReshapeOp op, RankedTensorType inputType,
                  ConversionPatternRewriter &rewriter) {
  auto resultType = dyn_cast<RankedTensorType>(op.getReshaped().getType());
  if (!resultType) {
    return rewriter.notifyMatchFailure(op, "expected ranked tensor types");
  }
  if (inputType.getElementType() != resultType.getElementType()) {
    return rewriter.notifyMatchFailure(
        op, "expected matching input and result element types");
  }
  // The result aliases the input, so it lives wherever the input does. Only a
  // rank-0 input names no space, since the type converter leaves scalar host
  // roots alone, and the tensor it expands to has to live somewhere: device.
  // A result naming a different space would need a copy this does not emit.
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
  // Takes the pass converter to match the other patterns.
  ReshapeToHipsr(const TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern(ctx) {}

  LogicalResult
  matchAndRewrite(onnx::ReshapeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // With allowzero set, a 0 in the target shape is a literal extent instead
    // of a copy of the input's, which gives a different result shape.
    if (op.getAllowzero() != 0) {
      return rewriter.notifyMatchFailure(op, "expected allowzero = 0");
    }
    Value input = adaptor.getData();
    auto inputType = dyn_cast<RankedTensorType>(input.getType());
    if (!inputType) {
      return rewriter.notifyMatchFailure(op, "expected ranked tensor types");
    }
    FailureOr<RankedTensorType> resultType =
        resolveResultType(op, inputType, rewriter);
    if (failed(resultType)) {
      return failure();
    }

    // Naming the extents both refines the type everything downstream takes and
    // decides which of the two destinations the reshape gets.
    FailureOr<SmallVector<TargetExtent>> named =
        nameTargetExtents(*resultType, inputType, adaptor.getShape());
    RankedTensorType refinedType =
        succeeded(named) ? refineResultType(*resultType, inputType, *named)
                         : *resultType;

    // An identity reshape reinterprets nothing, so it needs no destination.
    // Refining first also catches the ones only the target shape operand shows
    // to be identities.
    if (inputType == refinedType) {
      rewriter.replaceOp(op, input);
      return success();
    }
    // A reshape preserves the element count, so two static shapes that
    // disagree on it cannot describe the same data.
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
        succeeded(named)
            ? createKnownShapePlaceholder(rewriter, loc, *ctx, input, inputType,
                                          refinedType, *named)
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
