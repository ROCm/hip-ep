/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/APInt.h"

namespace mlir {
namespace hip {
namespace {

//===----------------------------------------------------------------------===//
// Slice lowering
//===----------------------------------------------------------------------===//
//
// Two patterns are registered (in benefit order):
//
//   * SliceDecompose (benefit=2) — when starts/ends/axes/steps are all
//     compile-time constants AND every effective step is positive, the op
//     is rewritten to a `tensor.extract_slice`, which bufferizes to a
//     zero-copy `memref.subview`. This is the by-far most common case in
//     transformer models (slicing a fixed prefix off a static-shape KV / mask
//     tensor) and avoids any runtime call. Dynamic input/output dims are
//     supported as long as either the dim is NOT touched by `axes` (in
//     which case we forward the data dim via `tensor.dim`), or the input
//     dim is static so the per-axis ONNX clamping rules can be evaluated
//     at compile time.
//
//   * SliceToHip (benefit=1) — fallback for non-constant indices or negative
//     steps. Produces a native `hip.slice` DPS op whose runtime function is
//     a stub today (throws); models that hit this path are unsupported until
//     the runtime is implemented, but the conversion / bufferization pipeline
//     can still verify and link the IR. Dynamic output dims are sourced from
//     `tensor.dim` on `data` (an upper bound — Slice cannot widen any axis).

/// Return the dense-elements attribute backing \p value if it can be
/// determined at compile time. Matches the patterns produced by
/// `lowerOnnxConstants` (which runs before `convertComputeOps`).
static mlir::DenseElementsAttr getCompileTimeConstantTensor(mlir::Value value) {
  mlir::Operation *defOp = value.getDefiningOp();
  if (!defOp)
    return nullptr;
  if (auto cst = mlir::dyn_cast<mlir::arith::ConstantOp>(defOp))
    return mlir::dyn_cast<mlir::DenseElementsAttr>(cst.getValue());
  if (auto attr = defOp->getAttr("value"))
    if (auto dense = mlir::dyn_cast<mlir::DenseElementsAttr>(attr))
      return dense;
  if (auto toTensor = mlir::dyn_cast<mlir::bufferization::ToTensorOp>(defOp)) {
    auto bufDef =
        toTensor.getBuffer().getDefiningOp<mlir::memref::GetGlobalOp>();
    if (!bufDef)
      return nullptr;
    auto module = bufDef->getParentOfType<mlir::ModuleOp>();
    if (!module)
      return nullptr;
    auto global =
        module.lookupSymbol<mlir::memref::GlobalOp>(bufDef.getNameAttr());
    if (!global)
      return nullptr;
    return mlir::dyn_cast_or_null<mlir::DenseElementsAttr>(
        global.getInitialValueAttr());
  }
  return nullptr;
}

/// Populate \p out from a dense 1-D integer tensor attribute.
static mlir::LogicalResult
denseIntVectorToSmallVector(mlir::DenseElementsAttr dense,
                            llvm::SmallVectorImpl<int64_t> &out) {
  if (!dense)
    return mlir::failure();
  auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(dense.getType());
  if (!tensorType || tensorType.getRank() != 1)
    return mlir::failure();
  auto elemTy = tensorType.getElementType();
  if (!elemTy.isInteger(64) && !elemTy.isInteger(32))
    return mlir::failure();
  out.clear();
  for (mlir::APInt entry : dense.getValues<mlir::APInt>())
    out.push_back(entry.getSExtValue());
  return mlir::success();
}

/// Extract a 1-D integer tensor constant into a SmallVector<int64_t>.
/// Returns failure if the tensor is missing, not 1-D, or not int32/int64.
static mlir::LogicalResult
extractIntVector(mlir::Value v, llvm::SmallVectorImpl<int64_t> &out) {
  if (!v)
    return mlir::failure();
  return denseIntVectorToSmallVector(getCompileTimeConstantTensor(v), out);
}

/// Prefer compile-time slice params stamped by SliceShapeFold (captured
/// before constant externalization); fall back to reading inline operands.
static mlir::LogicalResult
extractSliceParamVector(mlir::Operation *op, llvm::StringRef attrName,
                        mlir::Value operand,
                        llvm::SmallVectorImpl<int64_t> &out) {
  if (auto attr = op->getAttrOfType<mlir::DenseI64ArrayAttr>(attrName)) {
    out.assign(attr.asArrayRef().begin(), attr.asArrayRef().end());
    return mlir::success();
  }
  return extractIntVector(operand, out);
}

/// Normalise an ONNX Slice operand reference (`v`): if it is an `onnx.NoValue`
/// placeholder (used for absent optional inputs), returns null Value.
static mlir::Value normaliseOptional(mlir::Value v) {
  if (!v)
    return v;
  auto defOp = v.getDefiningOp();
  if (defOp && defOp->getName().getStringRef() == "onnx.NoValue")
    return mlir::Value();
  return v;
}

struct SliceDecompose : public mlir::RewritePattern {
  SliceDecompose(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Slice", /*benefit=*/2, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    if (op->getNumOperands() < 3 || op->getNumOperands() > 5 ||
        op->getNumResults() != 1)
      return rewriter.notifyMatchFailure(op, "expected 3-5 inputs, 1 output");

    mlir::Value data = op->getOperand(0);
    auto dataType = mlir::dyn_cast<mlir::RankedTensorType>(data.getType());
    auto outType =
        mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
    if (!dataType || !outType)
      return rewriter.notifyMatchFailure(op, "expected ranked tensor types");

    int64_t rank = dataType.getRank();

    llvm::SmallVector<int64_t> startsVec, endsVec;
    if (mlir::failed(extractSliceParamVector(op, "hipdnn.slice_starts",
                                             op->getOperand(1), startsVec)) ||
        mlir::failed(extractSliceParamVector(op, "hipdnn.slice_ends",
                                             op->getOperand(2), endsVec)))
      return rewriter.notifyMatchFailure(
          op, "starts/ends are not compile-time constants");

    llvm::SmallVector<int64_t> axesVec;
    if (op->getNumOperands() >= 4) {
      mlir::Value axes = normaliseOptional(op->getOperand(3));
      if (axes) {
        if (mlir::failed(extractSliceParamVector(op, "hipdnn.slice_axes", axes,
                                                 axesVec)))
          return rewriter.notifyMatchFailure(
              op, "axes is not a compile-time constant");
      } else if (auto attr = op->getAttrOfType<mlir::DenseI64ArrayAttr>(
                     "hipdnn.slice_axes")) {
        axesVec.assign(attr.asArrayRef().begin(), attr.asArrayRef().end());
      }
    }
    if (axesVec.empty())
      for (int64_t i : llvm::seq<int64_t>(rank))
        axesVec.push_back(i);

    llvm::SmallVector<int64_t> stepsVec;
    if (op->getNumOperands() == 5) {
      mlir::Value steps = normaliseOptional(op->getOperand(4));
      if (steps) {
        if (mlir::failed(extractSliceParamVector(op, "hipdnn.slice_steps",
                                                 steps, stepsVec)))
          return rewriter.notifyMatchFailure(
              op, "steps is not a compile-time constant");
      } else if (auto attr = op->getAttrOfType<mlir::DenseI64ArrayAttr>(
                     "hipdnn.slice_steps")) {
        stepsVec.assign(attr.asArrayRef().begin(), attr.asArrayRef().end());
      }
    }
    if (stepsVec.empty())
      stepsVec.assign(axesVec.size(), 1);

    if (axesVec.size() != startsVec.size() ||
        axesVec.size() != endsVec.size() || axesVec.size() != stepsVec.size())
      return rewriter.notifyMatchFailure(op, "starts/ends/axes/steps mismatch");

    // tensor.extract_slice only supports positive strides; negative-step
    // slices need a separate reverse pass and fall through to the native op.
    for (int64_t s : stepsVec)
      if (s <= 0)
        return rewriter.notifyMatchFailure(
            op, "negative or zero step is not supported by extract_slice");

    mlir::Location loc = op->getLoc();

    // Build the set of axes touched by slice and validate the input is
    // static on each of those axes (we need the static dim size to apply
    // the ONNX clamping rules at compile time).
    llvm::SmallSet<int64_t, 8> seenAxes;
    for (size_t k = 0; k < axesVec.size(); ++k) {
      int64_t axis = axesVec[k];
      if (axis < 0)
        axis += rank;
      if (axis < 0 || axis >= rank)
        return rewriter.notifyMatchFailure(op, "axis out of range");
      if (!seenAxes.insert(axis).second)
        return rewriter.notifyMatchFailure(op, "duplicate axis");
      // ONNX clamps start/end against `dim`. If dim is dynamic we cannot
      // resolve the slice size at compile time -- fall through to the
      // hip.slice runtime op.
      if (dataType.isDynamicDim(axis))
        return rewriter.notifyMatchFailure(
            op, "slice axis has dynamic input dim; "
                "cannot apply ONNX clamping at compile time");
    }

    // Default: full range, unit stride on every dim. Untouched dynamic
    // dims forward through as `tensor.dim` so the extract_slice's size
    // operands are well-defined; untouched static dims become attrs.
    llvm::SmallVector<mlir::OpFoldResult> offsets, sizes, strides;
    offsets.assign(rank, rewriter.getIndexAttr(0));
    sizes.reserve(rank);
    strides.assign(rank, rewriter.getIndexAttr(1));
    for (int64_t i : llvm::seq<int64_t>(rank)) {
      if (dataType.isDynamicDim(i)) {
        mlir::Value dimVal =
            mlir::tensor::DimOp::create(rewriter, loc, data, i);
        sizes.push_back(dimVal);
      } else {
        sizes.push_back(rewriter.getIndexAttr(dataType.getDimSize(i)));
      }
    }

    // Apply per-axis (start, end, step), implementing the ONNX spec's
    // negative-index and clamping rules.
    for (size_t k = 0; k < axesVec.size(); ++k) {
      int64_t axis = axesVec[k];
      if (axis < 0)
        axis += rank;

      int64_t dim = dataType.getDimSize(axis);
      int64_t start = startsVec[k];
      int64_t end = endsVec[k];
      int64_t step = stepsVec[k];

      if (start < 0)
        start += dim;
      if (end < 0)
        end += dim;
      // Positive-step clamp per spec: start in [0, dim], end in [0, dim].
      start = std::clamp<int64_t>(start, 0, dim);
      end = std::clamp<int64_t>(end, 0, dim);
      if (end < start)
        end = start; // empty slice

      int64_t size = (end - start + step - 1) / step;
      if (size < 0)
        size = 0;

      offsets[axis] = rewriter.getIndexAttr(start);
      sizes[axis] = rewriter.getIndexAttr(size);
      strides[axis] = rewriter.getIndexAttr(step);
    }

    // Sanity check (only meaningful for static output dims): computed
    // sizes must match the IR-inferred output type. If they diverge we
    // have an unsupported corner case and should fall through to the
    // native op rather than silently produce wrong shapes. Dynamic output
    // dims are intentionally skipped -- the IR cannot tell us what value
    // to compare against.
    for (int64_t i : llvm::seq<int64_t>(rank)) {
      if (outType.isDynamicDim(i))
        continue;
      auto attr = llvm::dyn_cast_if_present<mlir::Attribute>(sizes[i]);
      auto intAttr = mlir::dyn_cast_or_null<mlir::IntegerAttr>(attr);
      if (!intAttr || intAttr.getInt() != outType.getDimSize(i))
        return rewriter.notifyMatchFailure(
            op, "computed slice size does not match inferred output");
    }

    mlir::OperationState state(
        loc, mlir::tensor::ExtractSliceOp::getOperationName());
    mlir::tensor::ExtractSliceOp::build(rewriter, state, outType, data, offsets,
                                        sizes, strides);
    mlir::Operation *sliceOp = rewriter.create(state);
    rewriter.replaceOp(op, sliceOp->getResult(0));
    return mlir::success();
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
                           ? normaliseOptional(op->getOperand(3))
                           : mlir::Value();
    mlir::Value steps = op->getNumOperands() == 5
                            ? normaliseOptional(op->getOperand(4))
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
                                     MLIRContext *ctx) {
  patterns.add<SliceDecompose, SliceToHip>(ctx);
}

} // namespace hip
} // namespace mlir
