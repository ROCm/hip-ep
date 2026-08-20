/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"
#include "hip/Support/SliceUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/STLExtras.h"

#include <limits>
#include <optional>

namespace mlir {
namespace hip {
namespace {

static Value normalizeOptional(Value value) {
  if (!value)
    return value;
  Operation *definingOp = value.getDefiningOp();
  if (definingOp && definingOp->getName().getStringRef() == "onnx.NoValue")
    return Value();
  return value;
}

struct SliceInput {
  Value value;
  int64_t length = 0;
  std::optional<SmallVector<int64_t>> constant;
  int64_t readbackOffset = -1;
};

static FailureOr<SliceInput> inspectSliceInput(Value value) {
  auto type = dyn_cast<RankedTensorType>(value.getType());
  if (!type || type.getRank() != 1 ||
      (!type.getElementType().isInteger(32) &&
       !type.getElementType().isInteger(64)) ||
      type.isDynamicDim(0))
    return failure();

  SliceInput input;
  input.value = value;
  input.length = type.getDimSize(0);
  SmallVector<int64_t> values;
  if (extractConstantIntVector(value, values)) {
    if (static_cast<int64_t>(values.size()) != input.length)
      return failure();
    input.constant = std::move(values);
  }
  return input;
}

static LogicalResult validateResultShape(RankedTensorType resultType,
                                         ArrayRef<int64_t> expected) {
  return success(isResultTypeCompatibleWithPayloadShape(resultType, expected));
}

static SmallVector<Value> materializeValues(PatternRewriter &rewriter,
                                            Location loc, SliceInput &input,
                                            ReadbackControlOp readback) {
  SmallVector<Value> values;
  values.reserve(input.length);
  if (input.constant) {
    for (int64_t value : *input.constant) {
      values.push_back(
          arith::ConstantOp::create(rewriter, loc, rewriter.getI64Type(),
                                    rewriter.getI64IntegerAttr(value)));
    }
    return values;
  }
  for (int64_t i : llvm::seq<int64_t>(input.length))
    values.push_back(readback.getValues()[input.readbackOffset + i]);
  return values;
}

struct SliceDecompose : public RewritePattern {
  SliceDecompose(MLIRContext *context)
      : RewritePattern("onnx.Slice", /*benefit=*/2, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    // Complete structural and conservative result validation before creating
    // the first constant, readback, dim query, or destination.
    if (op->getNumOperands() < 3 || op->getNumOperands() > 5 ||
        op->getNumResults() != 1)
      return rewriter.notifyMatchFailure(op, "expected 3-5 inputs, 1 output");

    Value data = op->getOperand(0);
    auto dataType = dyn_cast<RankedTensorType>(data.getType());
    auto resultType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!dataType || !resultType ||
        dataType.getRank() != resultType.getRank() ||
        dataType.getElementType() != resultType.getElementType())
      return rewriter.notifyMatchFailure(
          op, "data/result must be ranked tensors with matching rank and type");

    FailureOr<SliceInput> starts = inspectSliceInput(op->getOperand(1));
    FailureOr<SliceInput> ends = inspectSliceInput(op->getOperand(2));
    if (failed(starts) || failed(ends))
      return rewriter.notifyMatchFailure(
          op, "starts and ends must be statically-sized rank-1 i32/i64");
    if (starts->length != ends->length)
      return rewriter.notifyMatchFailure(
          op, "starts and ends must have equal static lengths");
    int64_t count = starts->length;

    std::optional<SliceInput> axes;
    if (op->getNumOperands() >= 4) {
      Value value = normalizeOptional(op->getOperand(3));
      if (value) {
        FailureOr<SliceInput> inspected = inspectSliceInput(value);
        if (failed(inspected) || inspected->length != count)
          return rewriter.notifyMatchFailure(
              op, "axes must be statically-sized rank-1 i32/i64 with the "
                  "same length as starts");
        axes = std::move(*inspected);
      }
    }
    std::optional<SliceInput> steps;
    if (op->getNumOperands() == 5) {
      Value value = normalizeOptional(op->getOperand(4));
      if (value) {
        FailureOr<SliceInput> inspected = inspectSliceInput(value);
        if (failed(inspected) || inspected->length != count)
          return rewriter.notifyMatchFailure(
              op, "steps must be statically-sized rank-1 i32/i64 with the "
                  "same length as starts");
        steps = std::move(*inspected);
      }
    }

    int64_t rank = dataType.getRank();
    if (rank > std::numeric_limits<int32_t>::max())
      return rewriter.notifyMatchFailure(
          op, "Slice rank exceeds operand-segment representation");
    bool allConstant = starts->constant && ends->constant &&
                       (!axes || axes->constant) && (!steps || steps->constant);
    if (steps && steps->constant &&
        llvm::is_contained(*steps->constant, int64_t{0}))
      return rewriter.notifyMatchFailure(op, "Slice steps must be non-zero");

    SmallVector<int64_t> knownAxes;
    bool axesKnown = !axes || axes->constant.has_value();
    if (axesKnown) {
      SmallVector<int64_t> rawAxes;
      if (axes) {
        rawAxes.assign(axes->constant->begin(), axes->constant->end());
      } else {
        rawAxes = llvm::to_vector(llvm::seq<int64_t>(0, count));
      }
      std::vector<int64_t> normalized;
      if (!hipdnn_ep::slice::normalizeAxes(rank, rawAxes.data(), count,
                                           normalized))
        return rewriter.notifyMatchFailure(
            op, "Slice axes must be unique and within the data rank");
      knownAxes.assign(normalized.begin(), normalized.end());
    }

    SmallVector<int64_t> conservativeShape(dataType.getShape());
    SmallVector<int64_t> exactStaticShape;
    if (allConstant) {
      std::optional<ArrayRef<int64_t>> staticAxes;
      if (axes)
        staticAxes = *axes->constant;
      std::optional<ArrayRef<int64_t>> staticSteps;
      if (steps)
        staticSteps = *steps->constant;
      FailureOr<SmallVector<int64_t>> inferred =
          inferSliceShape(dataType.getShape(), *starts->constant,
                          *ends->constant, staticAxes, staticSteps);
      if (failed(inferred))
        return rewriter.notifyMatchFailure(
            op, "constant Slice parameters are invalid");
      exactStaticShape = std::move(*inferred);
      conservativeShape = exactStaticShape;
    } else if (axesKnown) {
      for (int64_t axis : knownAxes)
        conservativeShape[axis] =
            dataType.getDimSize(axis) == 0 ? 0 : ShapedType::kDynamic;
    } else {
      for (int64_t axis : llvm::seq<int64_t>(rank))
        conservativeShape[axis] =
            dataType.getDimSize(axis) == 0 ? 0 : ShapedType::kDynamic;
    }
    if (failed(validateResultShape(resultType, conservativeShape)))
      return rewriter.notifyMatchFailure(
          op, "Slice result type contradicts the validated semantic shape");

    // The HIP fallback needs a context. Positive all-static slices remain a
    // zero-copy extract_slice and therefore do not require one.
    SmallVector<int64_t> resolvedStaticSteps(count, 1);
    if (steps && steps->constant)
      resolvedStaticSteps.assign(steps->constant->begin(),
                                 steps->constant->end());
    bool decompose =
        allConstant && llvm::all_of(resolvedStaticSteps,
                                    [](int64_t step) { return step > 0; });
    if (!decompose)
      return failure();
    std::optional<hipdnn_ep::slice::NormalizedParameters> staticParameters;
    if (allConstant &&
        llvm::none_of(dataType.getShape(), ShapedType::isDynamic)) {
      hipdnn_ep::slice::NormalizedParameters normalized;
      const int64_t *staticAxes = axes ? axes->constant->data() : nullptr;
      const int64_t *staticSteps = steps ? steps->constant->data() : nullptr;
      if (!hipdnn_ep::slice::normalizeParameters(
              dataType.getShape().data(), rank, starts->constant->data(),
              ends->constant->data(), staticAxes, staticSteps, count,
              normalized))
        return rewriter.notifyMatchFailure(
            op, "constant Slice parameters are invalid");
      staticParameters = std::move(normalized);
    }

    Location loc = op->getLoc();
    Value context;
    if (!decompose) {
      FailureOr<Value> contextOr = getContextArg(op, rewriter);
      if (failed(contextOr))
        return failure();
      context = *contextOr;
    }

    SmallVector<Value> runtimeSources;
    SmallVector<SliceInput *> orderedInputs = {&*starts, &*ends};
    if (axes)
      orderedInputs.push_back(&*axes);
    if (steps)
      orderedInputs.push_back(&*steps);
    for (SliceInput *input : orderedInputs) {
      if (!input->constant) {
        input->readbackOffset = 0;
        runtimeSources.push_back(input->value);
      }
    }

    ReadbackControlOp readback;
    Value readbackValid;
    if (!runtimeSources.empty()) {
      SmallVector<Type> runtimeTypes;
      llvm::transform(runtimeSources, std::back_inserter(runtimeTypes),
                      [](Value value) { return value.getType(); });
      FailureOr<ReadbackControlLayout> layout =
          getReadbackControlLayout(runtimeTypes);
      assert(succeeded(layout) && "validated Slice sources must group");
      unsigned runtimeIndex = 0;
      for (SliceInput *input : orderedInputs) {
        if (input->constant)
          continue;
        input->readbackOffset = layout->resultOffsets[runtimeIndex++];
      }
      SmallVector<Type> resultTypes = {rewriter.getI1Type()};
      resultTypes.append(layout->totalCount, rewriter.getI64Type());
      readback = ReadbackControlOp::create(rewriter, loc, resultTypes, context,
                                           runtimeSources);
      readbackValid = readback.getValid();
    } else {
      readbackValid = arith::ConstantOp::create(
          rewriter, loc, rewriter.getI1Type(), rewriter.getBoolAttr(true));
    }

    SmallVector<Value> startValues =
        materializeValues(rewriter, loc, *starts, readback);
    SmallVector<Value> endValues =
        materializeValues(rewriter, loc, *ends, readback);
    SmallVector<Value> axisValues;
    std::optional<ArrayRef<Value>> axisValuesRef;
    if (axes) {
      axisValues = materializeValues(rewriter, loc, *axes, readback);
      axisValuesRef = axisValues;
    }
    SmallVector<Value> stepValues;
    std::optional<ArrayRef<Value>> stepValuesRef;
    if (steps) {
      stepValues = materializeValues(rewriter, loc, *steps, readback);
      stepValuesRef = stepValues;
    }

    MaterializedSliceParameters parameters;
    if (failed(materializeSliceParameters(
            rewriter, loc, data, startValues, endValues, axisValuesRef,
            stepValuesRef, readbackValid, parameters)))
      return failure();

    SmallVector<Value> extentIndices;
    extentIndices.reserve(rank);
    for (Value extent : parameters.extents) {
      extentIndices.push_back(arith::IndexCastOp::create(
          rewriter, loc, rewriter.getIndexType(), extent));
    }

    if (decompose) {
      SmallVector<OpFoldResult> offsets, sizes, strides;
      offsets.reserve(rank);
      sizes.reserve(rank);
      strides.reserve(rank);
      for (int64_t axis : llvm::seq<int64_t>(rank)) {
        if (staticParameters) {
          offsets.push_back(
              rewriter.getIndexAttr(staticParameters->starts[axis]));
          sizes.push_back(
              rewriter.getIndexAttr(staticParameters->extents[axis]));
          strides.push_back(
              rewriter.getIndexAttr(staticParameters->steps[axis]));
          continue;
        }
        offsets.push_back(OpFoldResult(
            arith::IndexCastOp::create(rewriter, loc, rewriter.getIndexType(),
                                       parameters.starts[axis])
                .getResult()));
        if (resultType.isDynamicDim(axis))
          sizes.push_back(extentIndices[axis]);
        else
          sizes.push_back(rewriter.getIndexAttr(resultType.getDimSize(axis)));
        strides.push_back(OpFoldResult(
            arith::IndexCastOp::create(rewriter, loc, rewriter.getIndexType(),
                                       parameters.steps[axis])
                .getResult()));
      }
      Value slice = tensor::ExtractSliceOp::create(
          rewriter, loc, resultType, data, offsets, sizes, strides);
      rewriter.replaceOp(op, slice);
      return success();
    }

    return failure();
  }
};

struct SliceToHip : public mlir::RewritePattern {
  SliceToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Slice", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    if (op->getNumOperands() < 3 || op->getNumOperands() > 5 ||
        op->getNumResults() != 1)
      return rewriter.notifyMatchFailure(op, "expected 3-5 inputs, 1 output");

    auto ctxOrFailure = getContextArg(op, rewriter);
    if (mlir::failed(ctxOrFailure))
      return mlir::failure();
    mlir::Value context = *ctxOrFailure;

    mlir::Location loc = op->getLoc();
    mlir::Value data = op->getOperand(0);
    mlir::Value starts = op->getOperand(1);
    mlir::Value ends = op->getOperand(2);
    mlir::Value axes = op->getNumOperands() >= 4
                           ? normalizeOptional(op->getOperand(3))
                           : mlir::Value();
    mlir::Value steps = op->getNumOperands() == 5
                            ? normalizeOptional(op->getOperand(4))
                            : mlir::Value();

    auto resultType =
        mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(op, "result must be ranked tensor");

    auto dataType = mlir::cast<mlir::RankedTensorType>(data.getType());

    // For each dynamic output dim, source an upper bound from the data
    // dim at the same position. This is sound because ONNX Slice can
    // never widen any axis -- output dim i is at most data dim i. The
    // runtime stub honours `output_shape[i]` as the actual extent so the
    // over-allocation is benign.
    llvm::SmallVector<mlir::Value> dynSizes;
    for (int64_t i = 0; i < resultType.getRank(); ++i) {
      if (!resultType.isDynamicDim(i))
        continue;
      if (i >= dataType.getRank())
        return rewriter.notifyMatchFailure(
            op, "result rank exceeds data rank — invalid Slice");
      if (dataType.isDynamicDim(i))
        dynSizes.push_back(mlir::tensor::DimOp::create(rewriter, loc, data, i));
      else
        dynSizes.push_back(mlir::arith::ConstantIndexOp::create(
            rewriter, loc, dataType.getDimSize(i)));
    }
    mlir::Value init =
        mlir::tensor::EmptyOp::create(rewriter, loc, resultType.getShape(),
                                      resultType.getElementType(), dynSizes);

    auto hipOp = mlir::hip::SliceOp::create(rewriter, loc, context, data,
                                            starts, ends, axes, steps, init);
    rewriter.replaceOp(op, hipOp->getResult(0));
    return mlir::success();
  }
};

} // namespace

void populateSliceConversionPatterns(RewritePatternSet &patterns,
                                     MLIRContext *context) {
  patterns.add<SliceDecompose, SliceToHip>(context);
}

} // namespace hip
} // namespace mlir
