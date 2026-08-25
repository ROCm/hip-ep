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
//     steps. Produces a native `hip.slice` DPS op, executed by `wrap_slice` in
//     Runtime/real/slice.cpp since #284 (this comment claimed a throwing stub
//     long after that landed). Worth knowing when reading the extent logic
//     below: that runtime D2Hs starts/ends/axes/steps and synchronizes the
//     stream on every call, so this op already costs a host sync per
//     execution. Dynamic output dims are computed from
//     the slice bounds when those are host-resolvable, and otherwise fall back
//     to `tensor.dim` on `data` (an upper bound — Slice cannot widen any axis).
//
// That fallback is only an upper bound, and an upper bound is not a safe
// stand-in for the extent: the extent this pattern puts on the `tensor.empty`
// init IS the output shape as far as everything downstream is concerned, since
// `tensor.dim` of a DPS result resolves through to the init. A Slice that keeps
// one row of a 12k-row tensor therefore hands every consumer a 12k-row shape.
// Gemma-4 26B-A4B decode does exactly this — `Slice(CumSum(attention_mask),
// Shape(attn)[1] - Shape(ids)[1], Shape(attn)[1])` takes the single current
// query position — and the inflated extent turned the causal mask from
// [1, 1, S] into [1, S, S], 60% of a decode step. Hence resolveHostIndex:
// `ends - starts` is host arithmetic over `onnx.Shape` entries, so the true
// extent is computable without any device readback.
//
// The one extent that must NOT shrink is zero. The extent here is also the
// allocation, and one consumer depends on that allocation being the upper
// bound: FixLoopAccumulatorOffset.cpp rewrites a growing-Concat accumulator in
// an outlined `hip.loop` body and, as its header states, "assumes ... v_init is
// pre-sized to full capacity (the canonical `Slice(x, k, k, axis) -> Loop`
// pattern)". `Slice(x, k, k, axis)` is the ONNX empty-tensor idiom used to seed
// such an accumulator; its exact extent is 0, and a zero-capacity init leaves
// the loop appending chunks into nothing -- on Qwen3.6-VL's windowed attention
// that surfaced as `memrefCopy(rank-2 strided) failed: invalid pitch argument`,
// because the append subview's destination pitch was 0. So an extent that
// evaluates to zero falls back to the data dim; see buildAllocCapacity.

/// Return the dense-elements attribute backing \p value if it can be
/// determined at compile time. Recognizes arith constants, inspectable
/// hip.constant value carriers produced before `convertComputeOps`, and a
/// legacy initialized-global bridge.
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

/// Prefer compile-time slice params stamped by SliceShapeFold while producers
/// were generic ONNX constants; fall back to inspecting inline carriers.
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

/// Producer-chain depth walked by `resolveHostIndex`. Shape arithmetic in
/// exported graphs is a handful of ops deep; the bound only stops a
/// pathological walk.
static constexpr int kHostIndexMaxDepth = 8;

/// Resolve element \p idx of an integer shape tensor \p v to a host `index`
/// SSA value, materializing `arith` ops as needed. Returns a null Value when
/// the element is not host-computable.
///
/// Both the pre- and post-conversion spelling of ONNX shape arithmetic are
/// accepted, because the greedy driver gives no ordering guarantee between this
/// pattern and the ones rewriting the producers: `onnx.Shape` may still be
/// present, or may already be
/// `tensor.from_elements(arith.index_cast(tensor.dim))`, and `onnx.Sub` may
/// already be `hip.sub`. On Gemma-4 26B-A4B it is in practice the ONNX
/// spelling that arrives here, this pattern running first; the post-conversion
/// spelling is covered by test 10 in test_slice.mlir rather than by a model,
/// since nothing pins the order and losing either branch silently costs the
/// extent.
static mlir::Value resolveHostIndex(mlir::OpBuilder &b, mlir::Location loc,
                                    mlir::Value v, int64_t idx, int depth) {
  if (!v || idx < 0 || depth > kHostIndexMaxDepth)
    return {};
  auto vType = mlir::dyn_cast<mlir::RankedTensorType>(v.getType());
  if (!vType || !vType.getElementType().isSignlessInteger())
    return {};
  // Bound the request against the value's own extent once, here, rather than in
  // each producer branch: an out-of-range element must fail rather than resolve
  // to something plausible. Without this an `onnx.Shape` narrowed by its `end`
  // attribute would hand back dim(start + idx) for an element it does not have,
  // which is the same class of silently-wrong extent this file exists to fix.
  if (vType.hasStaticShape() && idx >= vType.getNumElements())
    return {};

  if (mlir::DenseElementsAttr dense = getCompileTimeConstantTensor(v)) {
    if (idx >= dense.getNumElements())
      return {};
    return mlir::arith::ConstantIndexOp::create(
        b, loc,
        (*(dense.getValues<mlir::APInt>().begin() + idx)).getSExtValue());
  }

  mlir::Operation *def = v.getDefiningOp();
  if (!def)
    return {};

  if (auto fromElems = mlir::dyn_cast<mlir::tensor::FromElementsOp>(def)) {
    if (idx >= static_cast<int64_t>(fromElems.getElements().size()))
      return {};
    mlir::Value elem = fromElems.getElements()[idx];
    // ShapeToTensorDims index_casts every tensor.dim to i64 to pack it; take
    // the index back rather than casting a second time.
    if (auto cast = elem.getDefiningOp<mlir::arith::IndexCastOp>())
      if (cast.getIn().getType().isIndex())
        return cast.getIn();
    return mlir::arith::IndexCastOp::create(b, loc, b.getIndexType(), elem);
  }

  if (auto cast = mlir::dyn_cast<mlir::tensor::CastOp>(def))
    return resolveHostIndex(b, loc, cast.getSource(), idx, depth + 1);

  llvm::StringRef opName = def->getName().getStringRef();

  // Unconverted onnx.Shape: element idx is dim (start + idx) of the operand.
  // `start` is normalized as ShapeToTensorDims normalizes it; `end` only bounds
  // how many elements exist, which the caller-side extent check above covers.
  if (opName == "onnx.Shape") {
    mlir::Value input = def->getOperand(0);
    auto inType = mlir::dyn_cast<mlir::RankedTensorType>(input.getType());
    if (!inType)
      return {};
    int64_t rank = inType.getRank();
    int64_t start = 0;
    if (auto startAttr = def->getAttrOfType<mlir::IntegerAttr>("start"))
      start = startAttr.getSInt();
    if (start < 0)
      start += rank;
    start = std::max(start, int64_t(0));
    int64_t dimIdx = start + idx;
    if (dimIdx < 0 || dimIdx >= rank)
      return {};
    if (!inType.isDynamicDim(dimIdx))
      return mlir::arith::ConstantIndexOp::create(b, loc,
                                                  inType.getDimSize(dimIdx));
    return mlir::tensor::DimOp::create(b, loc, input, dimIdx);
  }

  // Binary shape arithmetic. hip elementwise ops are DPS, so their operands are
  // (ctx, lhs, rhs, out); the ONNX forms are plain (lhs, rhs).
  unsigned lhsPos = 0;
  bool isHip =
      mlir::isa<mlir::hip::SubOp, mlir::hip::AddOp, mlir::hip::MulOp>(def);
  if (isHip)
    lhsPos = 1;
  else if (opName != "onnx.Sub" && opName != "onnx.Add" && opName != "onnx.Mul")
    return {};

  mlir::Value lhs = def->getOperand(lhsPos);
  mlir::Value rhs = def->getOperand(lhsPos + 1);
  // A rank-0 or single-element operand is broadcast against the other.
  auto elemIdx = [&](mlir::Value operand) -> int64_t {
    auto t = mlir::dyn_cast<mlir::RankedTensorType>(operand.getType());
    return (t && t.hasStaticShape() && t.getNumElements() == 1) ? 0 : idx;
  };
  mlir::Value l = resolveHostIndex(b, loc, lhs, elemIdx(lhs), depth + 1);
  mlir::Value r = resolveHostIndex(b, loc, rhs, elemIdx(rhs), depth + 1);
  if (!l || !r)
    return {};

  bool isSub = isHip ? mlir::isa<mlir::hip::SubOp>(def) : opName == "onnx.Sub";
  bool isAdd = isHip ? mlir::isa<mlir::hip::AddOp>(def) : opName == "onnx.Add";
  if (isSub)
    return mlir::arith::SubIOp::create(b, loc, l, r);
  if (isAdd)
    return mlir::arith::AddIOp::create(b, loc, l, r);
  return mlir::arith::MulIOp::create(b, loc, l, r);
}

/// Emit the number of elements ONNX Slice produces on one axis, given runtime
/// `start`/`end` bounds and a positive compile-time \p step. Mirrors the
/// compile-time arithmetic in SliceDecompose; `arith` ops are needed here only
/// because the bounds are not known until run time.
static mlir::Value buildSliceExtent(mlir::OpBuilder &b, mlir::Location loc,
                                    mlir::Value dim, mlir::Value start,
                                    mlir::Value end, int64_t step) {
  mlir::Value zero = mlir::arith::ConstantIndexOp::create(b, loc, 0);
  // A negative bound counts from the end, and the sign is not known until run
  // time, so both forms are computed and selected between. Neither sentinel
  // exporters use needs special handling: INT64_MAX ("to the end") stays
  // positive, so `v + dim` overflows but is never selected, and the clamp
  // returns `dim`; INT64_MIN ("from the beginning") cannot overflow, since
  // adding a non-negative `dim` only moves it toward zero.
  auto normalize = [&](mlir::Value v) -> mlir::Value {
    mlir::Value shifted = mlir::arith::AddIOp::create(b, loc, v, dim);
    mlir::Value isNeg = mlir::arith::CmpIOp::create(
        b, loc, mlir::arith::CmpIPredicate::slt, v, zero);
    mlir::Value picked =
        mlir::arith::SelectOp::create(b, loc, isNeg, shifted, v);
    picked = mlir::arith::MaxSIOp::create(b, loc, picked, zero);
    return mlir::arith::MinSIOp::create(b, loc, picked, dim);
  };
  mlir::Value len =
      mlir::arith::SubIOp::create(b, loc, normalize(end), normalize(start));
  len = mlir::arith::MaxSIOp::create(b, loc, len, zero);
  if (step == 1)
    return len;
  return mlir::arith::CeilDivSIOp::create(
      b, loc, len, mlir::arith::ConstantIndexOp::create(b, loc, step));
}

/// Capacity to put on the `tensor.empty` init for a slice length \p len: the
/// length itself, or \p dim when the length is zero. A zero-length init is a
/// zero-byte buffer, and the `Slice(x, k, k, axis) -> Loop` accumulator seed
/// (see the file header) needs full capacity to append into. The compare is
/// against a run-time value because a bound pair can be equal only at run time;
/// bounds that are never equal are unaffected, so Gemma-4 -- whose length is
/// `Shape(attn)[1] - Shape(ids)[1]`, 1 at decode and S at prefill -- still gets
/// its exact extent.
static mlir::Value buildAllocCapacity(mlir::OpBuilder &b, mlir::Location loc,
                                      mlir::Value dim, mlir::Value len) {
  mlir::Value zero = mlir::arith::ConstantIndexOp::create(b, loc, 0);
  mlir::Value isEmpty = mlir::arith::CmpIOp::create(
      b, loc, mlir::arith::CmpIPredicate::eq, len, zero);
  return mlir::arith::SelectOp::create(b, loc, isEmpty, dim, len);
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

    // Which axes are sliced, and with what step. Both must be known to compute
    // an extent; a non-constant axes/steps operand leaves every dim on the
    // conservative upper bound below.
    llvm::SmallVector<int64_t> axesVec, stepsVec;
    bool haveAxes = true;
    if (axes)
      haveAxes = mlir::succeeded(
          extractSliceParamVector(op, "hipdnn.slice_axes", axes, axesVec));
    if (haveAxes && axesVec.empty())
      for (int64_t i : llvm::seq<int64_t>(dataType.getRank()))
        axesVec.push_back(i);
    if (steps && haveAxes)
      haveAxes = mlir::succeeded(
          extractSliceParamVector(op, "hipdnn.slice_steps", steps, stepsVec));
    if (haveAxes && stepsVec.empty())
      stepsVec.assign(axesVec.size(), 1);
    if (stepsVec.size() != axesVec.size())
      haveAxes = false;

    // Compute each dynamic output dim from the slice bounds where possible;
    // see the file header for why the data dim is not an acceptable substitute.
    // Ops materialized by a resolution that then fails are pure and get DCE'd.
    //
    // Where the bounds do not resolve, the fallback keeps the upper bound and
    // therefore keeps the over-sized shape. It could be exact instead:
    // CompressConversion solves the same shrinking-extent problem by reading
    // the true count back from the device with `hip.ReadbackDimOp`. That is not
    // done here, but not because a readback is unaffordable on this op --
    // `wrap_slice` already D2Hs its own bounds and syncs the stream every call,
    // so the sync is paid regardless. It is that a readback would add a second
    // sync point, in the shape computation ahead of the slice, to serve a case
    // no model in scope reaches: on Gemma-4 every sliced axis resolves from its
    // bounds, and host arithmetic is strictly better than a readback wherever
    // it is available. If a model does land here, revisit it -- the cost is
    // lower than it looks. Hence the debug line below: the fallback is a known
    // performance cliff, and it should be findable without an RGP capture.
    llvm::SmallVector<mlir::Value> dynSizes;
    for (int64_t i = 0; i < resultType.getRank(); ++i) {
      if (!resultType.isDynamicDim(i))
        continue;
      if (i >= dataType.getRank())
        return rewriter.notifyMatchFailure(
            op, "result rank exceeds data rank — invalid Slice");

      mlir::Value dim =
          dataType.isDynamicDim(i)
              ? mlir::tensor::DimOp::create(rewriter, loc, data, i).getResult()
              : mlir::arith::ConstantIndexOp::create(rewriter, loc,
                                                     dataType.getDimSize(i))
                    .getResult();

      mlir::Value extent;
      // An axis `axes` does not name is not sliced at all, so the data dim is
      // its exact extent rather than a fallback. Only an axis this op really
      // slices can land on the upper bound, so only that is worth reporting --
      // and if `axes` itself was unreadable, any axis might be sliced.
      bool axisIsSliced = !haveAxes;
      bool emptyAccumulatorSeed = false;
      if (haveAxes) {
        for (size_t k = 0; k < axesVec.size(); ++k) {
          int64_t axis =
              axesVec[k] < 0 ? axesVec[k] + dataType.getRank() : axesVec[k];
          if (axis != i)
            continue;
          axisIsSliced = true;
          if (stepsVec[k] <= 0)
            break;
          // One `Value` for both bounds is `Slice(x, k, k, axis)` spelled the
          // way exporters emit it, so the length is zero without resolving
          // anything. Take the data dim directly rather than emitting the
          // arithmetic and the select that would fold back to it.
          if (starts == ends) {
            emptyAccumulatorSeed = true;
            break;
          }
          mlir::Value s = resolveHostIndex(rewriter, loc, starts, k, 0);
          mlir::Value e = resolveHostIndex(rewriter, loc, ends, k, 0);
          if (s && e)
            extent = buildAllocCapacity(
                rewriter, loc, dim,
                buildSliceExtent(rewriter, loc, dim, s, e, stepsVec[k]));
          break;
        }
      }
      if (!extent && axisIsSliced && !emptyAccumulatorSeed)
        LLVM_DEBUG(llvm::dbgs()
                   << "[" DEBUG_TYPE "] slice extent for dim " << i
                   << " falls back to the data dim: no host-resolvable bounds "
                      "(or a non-positive step), so consumers see an upper "
                      "bound rather than the slice length -- "
                   << *op << "\n");
      dynSizes.push_back(extent ? extent : dim);
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
