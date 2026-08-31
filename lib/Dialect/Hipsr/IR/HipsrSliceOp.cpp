/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Conversion/HipsrToLLVM/HipsrToLLVM.h"
#include "hip/Dialect/Hipsr/IR/HipsrOps.h"

#include "hip/Dialect/Hipsr/IR/HipsrLLVMLoweringUtils.h"
#include "hip/Dialect/Hipsr/IR/HipsrShapeRegionPopulationUtils.h"

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/RegionUtils.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>

using namespace mlir;
using namespace mlir::hipsr;

namespace {
// The entries a window slot holds at compile time, or null when it carries a
// value holding them instead.
DenseI64ArrayAttr constantEntries(OpFoldResult slot) {
  return dyn_cast_if_present<DenseI64ArrayAttr>(
      dyn_cast_if_present<Attribute>(slot));
}

// The window as a shape region sees it: each slot is the attribute holding its
// entries or the argument carrying them, and is null where the region cannot
// reach them. The op keeps a slot in one form or the other, so a slot without
// an attribute has an operand, and a barrier region gets those in operand order
// after the data, while any other region gets only the data's shape.
struct SlicePlaceholderShapeArgs : PlaceholderShapeRegionArgs {
  SlicePlaceholderShapeArgs(Block &block, SliceOp op)
      : PlaceholderShapeRegionArgs(block) {
    bool isBarrier =
        cast<PlaceholderOp>(block.getParentOp()).getPlaceholderType() ==
        PlaceholderType::Barrier;
    DenseI64ArrayAttr attrs[] = {op.getStartsAttrAttr(), op.getEndsAttrAttr(),
                                 op.getAxesAttrAttr(), op.getStepsAttrAttr()};
    unsigned next = 1;
    for (auto [slot, attr] : llvm::zip_equal(window, attrs)) {
      if (attr) {
        slot = attr;
      } else if (isBarrier) {
        slot = in(next++);
      }
    }
  }

  // The data itself in a barrier region, only its shape in any other.
  Value getData() const { return in(0); }
  OpFoldResult getStarts() const { return window[0]; }
  OpFoldResult getEnds() const { return window[1]; }
  OpFoldResult getAxes() const { return window[2]; }
  OpFoldResult getSteps() const { return window[3]; }

private:
  OpFoldResult window[4];
};

// How many entries the window takes out of an axis `axisSize` long, following
// ONNX Slice: a negative bound counts back from the end, both bounds clamp to
// the axis, and the step strides what is left. `step` must not be zero.
Value windowSize(OpBuilder &builder, Location loc, Value start, Value end,
                 int64_t step, Value axisSize) {
  auto index = [&](int64_t value) -> Value {
    return arith::ConstantIndexOp::create(builder, loc, value);
  };
  Value zero = index(0);
  auto resolve = [&](Value bound) -> Value {
    Value isNegative = builder.createOrFold<arith::CmpIOp>(
        loc, arith::CmpIPredicate::slt, bound, zero);
    Value counted = builder.createOrFold<arith::AddIOp>(loc, bound, axisSize);
    return builder.createOrFold<arith::SelectOp>(loc, isNegative, counted,
                                                 bound);
  };
  auto clamp = [&](Value bound, Value low, Value high) -> Value {
    return builder.createOrFold<arith::MinSIOp>(
        loc, builder.createOrFold<arith::MaxSIOp>(loc, bound, low), high);
  };

  Value first = resolve(start);
  Value last = resolve(end);
  Value span;
  if (step > 0) {
    span = builder.createOrFold<arith::SubIOp>(loc, clamp(last, zero, axisSize),
                                               clamp(first, zero, axisSize));
  } else {
    // A reverse window runs down from `first`, and ONNX spells "through the
    // first entry" as -1.
    Value top = builder.createOrFold<arith::SubIOp>(loc, axisSize, index(1));
    span = builder.createOrFold<arith::SubIOp>(loc, clamp(first, zero, top),
                                               clamp(last, index(-1), top));
    step = -step;
  }
  Value taken = builder.createOrFold<arith::MaxSIOp>(loc, span, zero);
  return builder.createOrFold<arith::CeilDivSIOp>(loc, taken, index(step));
}
} // namespace

MutableOperandRange SliceOp::getDpsInitsMutable() { return getInitMutable(); }

LogicalResult SliceOp::verify() {
  constexpr StringLiteral names[] = {"starts", "ends", "axes", "steps"};
  DenseI64ArrayAttr attrs[] = {getStartsAttrAttr(), getEndsAttrAttr(),
                               getAxesAttrAttr(), getStepsAttrAttr()};
  Value operands[] = {getStarts(), getEnds(), getAxes(), getSteps()};
  std::optional<int64_t> entries;
  for (auto [name, attr, operand] : llvm::zip_equal(names, attrs, operands)) {
    if (attr && operand) {
      return emitOpError() << name << " cannot have both an operand and "
                           << name << "_attr";
    }
    if (!attr && !operand) {
      return emitOpError() << name << " needs an operand or " << name
                           << "_attr";
    }
    // How many axes narrow follows from the graph, not from a value: only the
    // entries themselves are left to run time.
    int64_t length = attr ? static_cast<int64_t>(attr.size())
                          : cast<ShapedType>(operand.getType()).getDimSize(0);
    if (ShapedType::isDynamic(length)) {
      return emitOpError() << name << " must hold a known number of entries";
    }
    if (entries && *entries != length) {
      return emitOpError("starts, ends, axes and steps must hold one entry per "
                         "sliced axis");
    }
    entries = length;
  }

  // A window narrows axes; an axis it leaves alone passes through whole.
  auto dataType = cast<ShapedType>(getData().getType());
  auto outputType = cast<ShapedType>(getInit().getType());
  for (auto [axis, dataSize, outputSize] :
       llvm::enumerate(dataType.getShape(), outputType.getShape())) {
    if (ShapedType::isStatic(dataSize) && ShapedType::isStatic(outputSize) &&
        outputSize > dataSize) {
      return emitOpError("an output axis must not be longer than the data "
                         "axis; axis ")
             << axis << " is " << dataSize << " long in the data and "
             << outputSize << " in the output";
    }
  }
  return success();
}

namespace mlir {
namespace hipsr {

LogicalResult populateSliceShapeRegion(OpBuilder &builder, Block &shapeBlock,
                                       SliceOp op) {
  auto dataType = cast<ShapedType>(op.getData().getType());
  int64_t rank = dataType.getRank();
  SlicePlaceholderShapeArgs args{shapeBlock, op};

  OpFoldResult starts = args.getStarts();
  OpFoldResult ends = args.getEnds();
  if (!starts || !ends) {
    return op.emitOpError("a bound the graph computes needs a barrier "
                          "placeholder, whose shape region reads the operands");
  }
  DenseI64ArrayAttr axesAttr = constantEntries(args.getAxes());
  DenseI64ArrayAttr stepsAttr = constantEntries(args.getSteps());
  if (!axesAttr || !stepsAttr) {
    return op.emitOpError(
        "needs compile-time axes and steps to tell which axes "
        "narrow and which way each window runs");
  }
  ArrayRef<int64_t> axes = axesAttr.asArrayRef();
  ArrayRef<int64_t> steps = stepsAttr.asArrayRef();

  // Which window entry narrows each axis. An axis the window leaves out keeps
  // the data's size.
  SmallVector<std::optional<int64_t>> narrowedBy(rank);
  for (auto [entry, axis, step] : llvm::enumerate(axes, steps)) {
    int64_t resolvedAxis = axis < 0 ? axis + rank : axis;
    if (resolvedAxis < 0 || resolvedAxis >= rank) {
      return op.emitOpError("axes must be in [-data rank, data rank); got ")
             << axis;
    }
    if (narrowedBy[resolvedAxis]) {
      return op.emitOpError("axes must be distinct; got axis ")
             << resolvedAxis << " twice";
    }
    if (step == 0) {
      return op.emitOpError("steps must not be zero");
    }
    narrowedBy[resolvedAxis] = entry;
  }

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(&shapeBlock);
  Location loc = op.getLoc();

  // A bound an attribute holds is a constant; one behind an argument is read
  // out of it.
  auto boundOf = [&](OpFoldResult slot, int64_t entry) -> Value {
    if (DenseI64ArrayAttr bounds = constantEntries(slot)) {
      return arith::ConstantIndexOp::create(builder, loc, bounds[entry]);
    }
    Value index = arith::ConstantIndexOp::create(builder, loc, entry);
    Value bound = tensor::ExtractOp::create(builder, loc, cast<Value>(slot),
                                            ValueRange{index});
    return arith::IndexCastOp::create(builder, loc, builder.getIndexType(),
                                      bound);
  };

  // How long the data is along an axis, asked of the data where the region
  // holds it and of its shape otherwise.
  auto sizeOf = [&](int64_t axis) -> Value {
    if (!dataType.isDynamicDim(axis)) {
      return arith::ConstantIndexOp::create(builder, loc,
                                            dataType.getDimSize(axis));
    }
    Value data = args.getData();
    if (args.holdsDataValues()) {
      return tensor::DimOp::create(builder, loc, data, axis);
    }
    return shape::GetExtentOp::create(builder, loc, data, axis);
  };

  SmallVector<Value> sizes =
      llvm::map_to_vector(llvm::seq<int64_t>(rank), [&](int64_t axis) -> Value {
        std::optional<int64_t> entry = narrowedBy[axis];
        if (!entry) {
          return sizeOf(axis);
        }
        return windowSize(builder, loc, boundOf(starts, *entry),
                          boundOf(ends, *entry), steps[*entry], sizeOf(axis));
      });
  Value outputShape = createExtentTensor(builder, loc, sizes);
  ShapeYieldOp::create(builder, loc, ValueRange{outputShape});

  // Folding on creation leaves dead constants behind.
  IRRewriter rewriter(builder);
  (void)runRegionDCE(rewriter, *shapeBlock.getParent());
  return success();
}

} // namespace hipsr
} // namespace mlir

namespace {

constexpr const char *kWrapSlice = "wrap_slice";

// `wrap_slice` resolves the window against the data shape, so the call passes
// both shapes as host arrays plus how many entries the window holds. The
// runtime also takes a null window pointer, reading it as ONNX's default, but
// this op spells every slot out and so never sends one.
struct SliceLowering : ConvertOpToLLVMPattern<SliceOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(SliceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();

    // Only the shapes are needed by type; a window slot just hands a pointer
    // over.
    auto dataType = dyn_cast<MemRefType>(op.getData().getType());
    auto outputType = dyn_cast<MemRefType>(op.getInit().getType());
    if (!dataType || !outputType) {
      return rewriter.notifyMatchFailure(
          op, "operands must be memrefs (run bufferization first)");
    }
    int64_t elementType = getHipdnnDataType(dataType.getElementType());
    if (elementType < 0) {
      return rewriter.notifyMatchFailure(op, "unsupported element type");
    }

    Type i64Type = rewriter.getI64Type();
    auto i64Const = [&](int64_t value) -> Value {
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      rewriter.getI64IntegerAttr(value));
    };
    Value dataShape = emitHostI64Array(
        extractShape(dataType, adaptor.getData(), rewriter, loc, i64Type),
        rewriter, loc);
    Value outputShape = emitHostI64Array(
        extractShape(outputType, adaptor.getInit(), rewriter, loc, i64Type),
        rewriter, loc);

    // Each slot reaches the runtime as a host pointer: an attribute becomes an
    // array on the stack and an operand hands over the buffer it already has.
    auto windowPtr = [&](DenseI64ArrayAttr attr, Value memref) -> Value {
      if (attr) {
        return emitHostI64Array(
            llvm::map_to_vector(attr.asArrayRef(), i64Const), rewriter, loc);
      }
      return extractContiguousMemRefPtr(memref, rewriter, loc);
    };
    DenseI64ArrayAttr startsAttr = op.getStartsAttrAttr();
    Value startsPtr = windowPtr(startsAttr, adaptor.getStarts());
    Value endsPtr = windowPtr(op.getEndsAttrAttr(), adaptor.getEnds());
    Value axesPtr = windowPtr(op.getAxesAttrAttr(), adaptor.getAxes());
    Value stepsPtr = windowPtr(op.getStepsAttrAttr(), adaptor.getSteps());

    // The runtime counts each slot's entries separately, but every slot is the
    // same length here and that length is known, so one constant covers them.
    Value entries = i64Const(
        startsAttr ? static_cast<int64_t>(startsAttr.size())
                   : cast<ShapedType>(op.getStarts().getType()).getDimSize(0));

    using SliceCall =
        RuntimeFunc<i32, hostPtr, devicePtr, hostPtr, hostPtr, hostPtr, hostPtr,
                    devicePtr, hostPtr, i64, hostPtr, i64, i64, i64, i64, i64>;
    auto sliceFunc =
        SliceCall::lookupOrCreateFn(rewriter, loc, module, kWrapSlice);
    if (failed(sliceFunc)) {
      return failure();
    }
    if (failed(sliceFunc->call(adaptor.getCtx(), adaptor.getData(), startsPtr,
                               endsPtr, axesPtr, stepsPtr, adaptor.getInit(),
                               dataShape, dataType.getRank(), outputShape,
                               outputType.getRank(), entries, entries, entries,
                               elementType))) {
      return failure();
    }
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void mlir::hipsr::populateHipsrSliceLoweringPatterns(
    const LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  patterns.add<SliceLowering>(converter);
}
