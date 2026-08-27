/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- SliceConversion.cpp - Convert onnx.Slice to hipsr.slice ------------===//
//
// hipsr.slice takes the same window as onnx.Slice, so each slot carries over,
// and which form it takes is settled here: entries the compiler can read become
// an attribute, leaving the constant the constant conversion made to dead-code
// elimination, and entries the graph computes stay the host operand that
// carries them. hipsr.slice spells the whole window out, so an `axes` or
// `steps` the graph leaves out gets ONNX's default written into its attribute
// rather than being passed on as omitted.
//
// hipsr-populate-shape-region fills the placeholder's shape region in later,
// but the layout follows from the forms: with the whole window in attributes a
// normal placeholder can work the output shape out from the data's, while an
// operand has to be read at run time, which takes a barrier placeholder
// holding the data and every operand left.
//
// Before, with `steps` omitted and types left out:
//   %starts = "onnx.Constant"() {value = dense<1> : tensor<1xi64>}
//   %ends = "onnx.Constant"() {value = dense<7> : tensor<1xi64>}
//   %axes = "onnx.Constant"() {value = dense<0> : tensor<1xi64>}
//   %none = "onnx.NoValue"() {value}
//   %0 = "onnx.Slice"(%data, %starts, %ends, %axes, %none)
//
// After, leaving the three converted constants unused:
//   %init = hipsr.placeholder(%ctx) ins(%data)
//       {placeholder_type = #hipsr.placeholder_type<normal>}
//   %0 = hipsr.slice(%ctx) ins(%data) outs(%init)
//       {axes_attr = array<i64: 0>, ends_attr = array<i64: 7>,
//        starts_attr = array<i64: 1>, steps_attr = array<i64: 1>}
//
//===----------------------------------------------------------------------===//

#include "OnnxToHipsrUtils.h"

#include "hip/Conversion/OnnxToHipsr/OnnxToHipsr.h"
#include "hip/Dialect/Hipsr/IR/HipsrOps.h"
#include "hip/Dialect/Hipsr/IR/HipsrTypes.h"
#include "hip/Dialect/Onnx/IR/OnnxOps.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace hipsr {
namespace {

// What hipsr.slice takes in a window slot. Entries the compiler can read become
// an attribute, wherever the constant itself lives, and entries the graph
// computes stay the host operand that carries them. A slot the graph leaves out
// is null. Anything else would take a copy this conversion does not emit.
// TODO: Copy a device window operand to the host instead.
FailureOr<OpFoldResult> windowSlot(Value operand,
                                   ConversionPatternRewriter &rewriter) {
  // An importer spells an omitted slot as an onnx.NoValue, typed `none`, and
  // leaves the default to the reader.
  if (isa<NoneType>(operand.getType())) {
    return OpFoldResult();
  }
  auto type = dyn_cast<RankedTensorType>(operand.getType());
  if (!type || type.getRank() != 1 || !type.getElementType().isInteger(64)) {
    return failure();
  }
  DenseIntElementsAttr entries;
  if (matchPattern(operand, m_Constant(&entries))) {
    return OpFoldResult(rewriter.getDenseI64ArrayAttr(
        llvm::to_vector(entries.getValues<int64_t>())));
  }
  if (isHostRankedTensor(type)) {
    return OpFoldResult(operand);
  }
  return failure();
}

// The entries a slot holds at compile time, or null when the graph computes
// them.
DenseI64ArrayAttr constantEntries(OpFoldResult slot) {
  return dyn_cast_if_present<DenseI64ArrayAttr>(
      dyn_cast_if_present<Attribute>(slot));
}

// The operand carrying a slot's entries, or null when the compiler has them.
Value computedEntries(OpFoldResult slot) {
  return dyn_cast_if_present<Value>(slot);
}

struct SliceToHipsr : public OpConversionPattern<onnx::SliceOp> {
  // The type converter is unused: converting the operands would overwrite the
  // memory space their producers chose. It stays in the signature so every
  // pattern is built the same way.
  SliceToHipsr(const TypeConverter &, MLIRContext *ctx)
      : OpConversionPattern(ctx) {}

  LogicalResult
  matchAndRewrite(onnx::SliceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value data = adaptor.getData();
    auto dataType = dyn_cast<RankedTensorType>(data.getType());
    auto resultType = dyn_cast<RankedTensorType>(op.getOutput().getType());
    if (!dataType || !resultType) {
      return rewriter.notifyMatchFailure(op, "expected ranked tensor types");
    }
    // The data is sliced on the device and this emits no copy, so host data is
    // rejected.
    // TODO: Slice a host tensor on the host instead.
    if (!isDeviceRankedTensor(dataType)) {
      return rewriter.notifyMatchFailure(op, "expected device-resident data");
    }

    FailureOr<Value> ctx = getHipsrContextArg(op, rewriter);
    if (failed(ctx)) {
      return failure();
    }

    FailureOr<OpFoldResult> starts = windowSlot(adaptor.getStarts(), rewriter);
    FailureOr<OpFoldResult> ends = windowSlot(adaptor.getEnds(), rewriter);
    FailureOr<OpFoldResult> axes = windowSlot(adaptor.getAxes(), rewriter);
    FailureOr<OpFoldResult> steps = windowSlot(adaptor.getSteps(), rewriter);
    if (failed(starts) || failed(ends) || failed(axes) || failed(steps)) {
      return rewriter.notifyMatchFailure(
          op, "expected a rank-1 i64 window the host can read");
    }
    // ONNX Slice has no default for the bounds, and neither does hipsr.slice.
    if (!*starts || !*ends) {
      return rewriter.notifyMatchFailure(op, "expected starts and ends");
    }
    // Writing a default in needs the count, which hipsr.slice takes as known
    // anyway. Whichever form the slot took, its operand's type still counts it.
    int64_t entries =
        cast<RankedTensorType>(adaptor.getStarts().getType()).getDimSize(0);
    if (ShapedType::isDynamic(entries)) {
      return rewriter.notifyMatchFailure(op,
                                         "expected a window of a known length");
    }

    // hipsr.slice spells every slot out, so an `axes` or `steps` the graph
    // leaves out takes ONNX's default: the leading axes and unit steps.
    if (!*axes) {
      *axes = rewriter.getDenseI64ArrayAttr(
          llvm::to_vector(llvm::seq<int64_t>(entries)));
    }
    if (!*steps) {
      *steps = rewriter.getDenseI64ArrayAttr(SmallVector<int64_t>(entries, 1));
    }

    // Entries left in an operand can only be read at run time, which takes a
    // barrier placeholder: its region gets the operands themselves. The barrier
    // takes every operand the op has, so the region finds each where the op
    // keeps it.
    SmallVector<Value> inputs{data};
    for (OpFoldResult slot : {*starts, *ends, *axes, *steps}) {
      if (Value operand = computedEntries(slot)) {
        inputs.push_back(operand);
      }
    }
    bool runtimeWindow = inputs.size() > 1;

    Location loc = op.getLoc();
    resultType = tensorTypeInSpace(resultType, MemorySpace::Device);
    Value init = PlaceholderOp::create(rewriter, loc, TypeRange{resultType},
                                       *ctx, inputs,
                                       runtimeWindow ? PlaceholderType::Barrier
                                                     : PlaceholderType::Normal)
                     .getResult(0);
    auto sliceOp =
        SliceOp::create(rewriter, loc, TypeRange{resultType}, *ctx, data,
                        computedEntries(*starts), computedEntries(*ends),
                        computedEntries(*axes), computedEntries(*steps), init,
                        constantEntries(*starts), constantEntries(*ends),
                        constantEntries(*axes), constantEntries(*steps));
    rewriter.replaceOp(op, sliceOp.getResult(0));
    return success();
  }
};

} // namespace

void populateSliceConversionPatterns(const TypeConverter &typeConverter,
                                     RewritePatternSet &patterns,
                                     MLIRContext *ctx) {
  patterns.add<SliceToHipsr>(typeConverter, ctx);
}

} // namespace hipsr
} // namespace mlir
