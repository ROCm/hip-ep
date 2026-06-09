/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/APInt.h"

#include <algorithm>
#include <limits>

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
//     steps. Produces a native `hip.slice` DPS op. Dynamic output dims are
//     sized as exactly as compile-time information allows (see below); only
//     the irrecoverable case falls back to the `tensor.dim` upper bound.
//
// Dynamic-output-dim sizing in SliceToHip — three cases per dynamic axis:
//
//   (A) Touched axis, STATIC input dim  → IntegerAttr with the compile-time
//       extent (ONNX neg-index + clamp evaluated at compile time).
//   (B) Touched axis, DYNAMIC input dim, CONSTANT starts/ends/steps → host
//       `index` arith on `tensor.dim` (descriptor-only, no GPU read), via
//       `emitRuntimeExtent`. This is the headline fix: main previously sized
//       these dims to `data.dim[i]` (an upper bound), so downstream reductions
//       / matmuls folded the kernel's zero-padded tail into the result and the
//       output cosine collapsed on dynamic-shape vision encoders. Sizing to
//       the exact logical extent removes the padding from the live region.
//   (C) Untouched axis, OR anything we cannot fold → `data.dim[i]` upper bound
//       (the legacy contract; Slice can never widen an axis).
//
// Empty-slice (logical extent == 0) policy: the result dim is sized to the
// data-dim upper bound, NOT 0. Two self-contained reasons: (1) ONNX Slice can
// never widen an axis, so `data.dim(axis)` is always a sound over-allocation
// and the runtime honours the real extent via `output_shape[i]`; (2) a true
// dim-0 buffer is rejected by MIOpen broadcast tensor descriptors. (A
// degenerate empty slice feeding a buffer-backed accumulator must likewise
// stay at the upper bound rather than collapse to 0.)
//
// NOTE (deferred): a fourth case — RUNTIME starts/ends with constant axes (the
// per-token attention slice inside an outlined `hip.loop` body) — is NOT
// handled here. It requires following loop-body block-args back to the
// enclosing loop's captures and a `hipdnn.fold_value` constant sidecar that is
// not present on this branch; it lands with the dynamic-vision loop infra.

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

/// Extract a 1-D integer tensor constant into a SmallVector<int64_t>.
/// Returns failure if the tensor is missing, not 1-D, or not int32/int64.
static mlir::LogicalResult
extractIntVector(mlir::Value v, llvm::SmallVectorImpl<int64_t> &out) {
  if (!v)
    return mlir::failure();
  auto dense = getCompileTimeConstantTensor(v);
  if (!dense)
    return mlir::failure();
  auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(dense.getType());
  if (!tensorType || tensorType.getRank() != 1)
    return mlir::failure();
  auto elemTy = tensorType.getElementType();
  if (!elemTy.isInteger(64) && !elemTy.isInteger(32))
    return mlir::failure();
  for (mlir::APInt entry : dense.getValues<mlir::APInt>())
    out.push_back(entry.getSExtValue());
  return mlir::success();
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

/// (start, end, step) for one sliced axis, pre-normalisation.
struct SliceParams {
  int64_t start;
  int64_t end;
  int64_t step;
};

/// Compute the logical slice extent for one axis under ONNX semantics on a
/// statically-known input dim. Mirrors `emitRuntimeExtent`'s runtime arith.
static int64_t clampSlice(int64_t dim, SliceParams p) {
  int64_t start = p.start, end = p.end, step = p.step;
  if (start < 0)
    start += dim;
  if (end < 0)
    end += dim;
  start = std::clamp<int64_t>(start, 0, dim);
  end = std::clamp<int64_t>(end, 0, dim);
  if (end < start)
    end = start;
  int64_t sz = (end - start + step - 1) / step;
  return sz < 0 ? 0 : sz;
}

/// Per-axis SliceToHip output-size hint. Priority order:
///   * `staticSize >= 0`: exact compile-time extent (use IntegerAttr).
///   * `runtimeFromDim = true` with `runtimeStart/runtimeEnd/step`: compute the
///     extent at host runtime from `tensor.dim` (constant indices, dynamic
///     input dim). Pure host-side `index` arith — no GPU read.
///   * `useUpperBound = true`: fall back to `data.dim[i]`.
struct SliceAxisInfo {
  int64_t staticSize = -1;
  bool runtimeFromDim = false;
  bool useUpperBound = false;
  int64_t runtimeStart = 0;
  int64_t runtimeEnd = 0;
  int64_t step = 1;
};

/// Resolve which input axes a Slice op touches and their (start, end, step).
/// Returns `true` only when starts/ends/axes/steps ALL fold to compile-time
/// constants and every touched axis maps to a unique in-range input axis.
/// Untouched axes get `useUpperBound`; touched axes get `staticSize` (static
/// input dim) or `runtimeFromDim` (dynamic input dim, constant indices).
static bool resolveSliceExtents(mlir::Operation *op,
                                mlir::RankedTensorType dataType,
                                llvm::SmallVectorImpl<SliceAxisInfo> &info) {
  if (op->getNumOperands() < 3)
    return false;
  llvm::SmallVector<int64_t> startsVec, endsVec, axesVec, stepsVec;
  if (mlir::failed(extractIntVector(op->getOperand(1), startsVec)) ||
      mlir::failed(extractIntVector(op->getOperand(2), endsVec)))
    return false;
  int64_t rank = dataType.getRank();
  if (op->getNumOperands() >= 4) {
    mlir::Value axes = normaliseOptional(op->getOperand(3));
    if (axes)
      if (mlir::failed(extractIntVector(axes, axesVec)))
        return false;
  }
  if (axesVec.empty())
    for (int64_t i : llvm::seq<int64_t>(rank))
      axesVec.push_back(i);
  if (op->getNumOperands() == 5) {
    mlir::Value steps = normaliseOptional(op->getOperand(4));
    if (steps)
      if (mlir::failed(extractIntVector(steps, stepsVec)))
        return false;
  }
  if (stepsVec.empty())
    stepsVec.assign(axesVec.size(), 1);
  if (axesVec.size() != startsVec.size() || axesVec.size() != endsVec.size() ||
      axesVec.size() != stepsVec.size())
    return false;

  info.assign(rank, SliceAxisInfo{});
  for (int64_t i : llvm::seq<int64_t>(rank))
    info[i].useUpperBound = true;
  llvm::SmallSet<int64_t, 8> seenAxes;
  for (size_t k = 0; k < axesVec.size(); ++k) {
    int64_t axis = axesVec[k];
    if (axis < 0)
      axis += rank;
    if (axis < 0 || axis >= rank)
      return false;
    if (!seenAxes.insert(axis).second)
      return false;
    int64_t step = stepsVec[k];
    // tensor extents are non-negative; negative/zero step (reverse slices)
    // is unsupported by this sizing path — fall back to the upper bound.
    if (step <= 0)
      return false;
    info[axis].useUpperBound = false;
    info[axis].step = step;
    if (dataType.isDynamicDim(axis)) {
      info[axis].runtimeFromDim = true;
      info[axis].runtimeStart = startsVec[k];
      info[axis].runtimeEnd = endsVec[k];
    } else {
      int64_t dim = dataType.getDimSize(axis);
      info[axis].staticSize =
          clampSlice(dim, SliceParams{startsVec[k], endsVec[k], step});
    }
  }
  return true;
}

/// Emit host-side `index` arithmetic computing the ONNX logical slice extent
/// on a dynamic-input-dim axis given constant start/end/step. `dim` is the
/// `tensor.dim` Value. INT64_MAX/MIN sentinels (ONNX "to end" / "from begin")
/// resolve to `dim` / `0`.
///
/// Before:  axis i dynamic-output, data.dim(i) = %d (runtime), start=2,
/// end=MAX, step=1 After:   %lo = max(2, 0); %s = min(%lo, %d)
///          %e  = %d                       // end==MAX -> dim
///          %eGeS = max(%e, %s); %diff = %eGeS - %s
///          %extent = (%diff + step-1) / step
static mlir::Value emitRuntimeExtent(mlir::OpBuilder &b, mlir::Location loc,
                                     mlir::Value dim, int64_t startC,
                                     int64_t endC, int64_t step) {
  using mlir::arith::AddIOp;
  using mlir::arith::ConstantIndexOp;
  using mlir::arith::DivSIOp;
  using mlir::arith::MaxSIOp;
  using mlir::arith::MinSIOp;
  using mlir::arith::SubIOp;
  mlir::Value zero = ConstantIndexOp::create(b, loc, 0);

  auto clampOnDim = [&](int64_t raw) -> mlir::Value {
    if (raw == std::numeric_limits<int64_t>::max())
      return dim;
    if (raw == std::numeric_limits<int64_t>::min())
      return zero;
    if (raw >= 0) {
      mlir::Value v = ConstantIndexOp::create(b, loc, raw);
      return MinSIOp::create(b, loc, v, dim);
    }
    mlir::Value v = ConstantIndexOp::create(b, loc, raw);
    mlir::Value sum = AddIOp::create(b, loc, dim, v);
    mlir::Value lo = MaxSIOp::create(b, loc, sum, zero);
    return MinSIOp::create(b, loc, lo, dim);
  };

  mlir::Value sIdx = clampOnDim(startC);
  mlir::Value eIdx = clampOnDim(endC);
  mlir::Value eGeS = MaxSIOp::create(b, loc, eIdx, sIdx);
  mlir::Value diff = SubIOp::create(b, loc, eGeS, sIdx);
  mlir::Value stepM1 = ConstantIndexOp::create(b, loc, step - 1);
  mlir::Value stepC_ = ConstantIndexOp::create(b, loc, step);
  mlir::Value diffPlusM = AddIOp::create(b, loc, diff, stepM1);
  return DivSIOp::create(b, loc, diffPlusM, stepC_);
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
    if (mlir::failed(extractIntVector(op->getOperand(1), startsVec)) ||
        mlir::failed(extractIntVector(op->getOperand(2), endsVec)))
      return rewriter.notifyMatchFailure(
          op, "starts/ends are not compile-time constants");

    llvm::SmallVector<int64_t> axesVec;
    if (op->getNumOperands() >= 4) {
      mlir::Value axes = normaliseOptional(op->getOperand(3));
      if (axes) {
        if (mlir::failed(extractIntVector(axes, axesVec)))
          return rewriter.notifyMatchFailure(
              op, "axes is not a compile-time constant");
      }
    }
    if (axesVec.empty())
      for (int64_t i : llvm::seq<int64_t>(rank))
        axesVec.push_back(i);

    llvm::SmallVector<int64_t> stepsVec;
    if (op->getNumOperands() == 5) {
      mlir::Value steps = normaliseOptional(op->getOperand(4));
      if (steps) {
        if (mlir::failed(extractIntVector(steps, stepsVec)))
          return rewriter.notifyMatchFailure(
              op, "steps is not a compile-time constant");
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

    // Size each dynamic output dim as exactly as compile-time information
    // allows (see the per-axis cases in the file header). When the params do
    // not all fold (`haveExtents == false`) or an axis is untouched, fall back
    // to the data-dim upper bound: ONNX Slice can never widen an axis, so
    // `data.dim[i]` is a sound over-bound and the runtime honours the actual
    // logical extent via `output_shape[i]`.
    llvm::SmallVector<SliceAxisInfo> sliceInfo;
    bool haveExtents = resolveSliceExtents(op, dataType, sliceInfo);

    // Empty-slice (logical extent == 0) is sized to the upper bound, not 0:
    // Slice never widens an axis so `data.dim(axis)` is a sound over-bound, and
    // MIOpen broadcast descriptors reject dim 0. `upper` = `data.dim(axis)`.
    auto guardWithUpper = [&](mlir::Value extent, mlir::Value upper) {
      mlir::Value zero = mlir::arith::ConstantIndexOp::create(rewriter, loc, 0);
      mlir::Value isZero = mlir::arith::CmpIOp::create(
          rewriter, loc, mlir::arith::CmpIPredicate::eq, extent, zero);
      return mlir::arith::SelectOp::create(rewriter, loc, isZero, upper, extent)
          .getResult();
    };

    llvm::SmallVector<mlir::Value> dynSizes;
    for (int64_t i = 0; i < resultType.getRank(); ++i) {
      if (!resultType.isDynamicDim(i))
        continue;
      if (i >= dataType.getRank())
        return rewriter.notifyMatchFailure(
            op, "result rank exceeds data rank — invalid Slice");

      mlir::Value upper =
          dataType.isDynamicDim(i)
              ? mlir::tensor::DimOp::create(rewriter, loc, data, i).getResult()
              : mlir::arith::ConstantIndexOp::create(rewriter, loc,
                                                     dataType.getDimSize(i))
                    .getResult();

      if (!haveExtents || sliceInfo[i].useUpperBound) {
        dynSizes.push_back(upper); // case (C)
        continue;
      }
      if (sliceInfo[i].staticSize >= 0) { // case (A)
        if (sliceInfo[i].staticSize == 0)
          dynSizes.push_back(upper); // empty -> upper bound
        else
          dynSizes.push_back(mlir::arith::ConstantIndexOp::create(
              rewriter, loc, sliceInfo[i].staticSize));
        continue;
      }
      if (sliceInfo[i].runtimeFromDim) { // case (B)
        // `runtimeFromDim` is only set for a dynamic input dim, so `upper`
        // (= `tensor.dim(data, i)`) is exactly the dim the extent arith needs.
        mlir::Value extent =
            emitRuntimeExtent(rewriter, loc, upper, sliceInfo[i].runtimeStart,
                              sliceInfo[i].runtimeEnd, sliceInfo[i].step);
        dynSizes.push_back(guardWithUpper(extent, upper));
        continue;
      }
      dynSizes.push_back(upper);
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
