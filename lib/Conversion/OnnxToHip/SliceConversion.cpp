/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"

#include "llvm/ADT/APInt.h"

namespace mlir {
namespace hip {
namespace {

//===----------------------------------------------------------------------===//
// Slice lowering
//===----------------------------------------------------------------------===//
//
// One pattern is registered: SliceToHip lowers `onnx.Slice` to a `hip.slice`
// DPS op whose runtime kernel writes the logical prefix into the output
// buffer (commit 72b17ba added the per-axis `logical_extent` contract so
// any over-allocated tail is deterministically zero-padded).
//
// The contribution here is exact output-buffer sizing on every dynamic
// output dim where we can prove the logical extent at compile time. Four
// cases per dynamic axis:
//
//   1. Constant starts/ends/steps + static input dim
//      → IntegerAttr with the compile-time extent.
//   2. Constant starts/ends/steps + dynamic input dim
//      → host `index` arith on `tensor.dim` (descriptor-only, no GPU read).
//   3. Runtime starts/ends with constant axes/steps (e.g. per-token slices
//      inside Qwen vision's outlined loop body, with starts/ends produced
//      by an upstream Gather over `image_grid_thw`)
//      → host `index` arith on `tensor.extract` of the runtime 1-D index
//        operand, then clamped via the same ONNX semantics. Backed by the
//        MaterializeHostScalars pass routing the index source through host
//        scratch — no D2H, no sync.
//   4. Anything else → fall back to the data-dim upper bound (the legacy
//      over-alloc contract; the kernel's zero-padded tail keeps it safe).
//
// Empty-slice (extent == 0) policy:
// The TRUE logical extent (possibly 0) is always used for the output
// buffer dim. Earlier code wrapped the result in
// `select(extent==0, upper_bound, extent)` to keep downstream MIOpen
// broadcast descriptors (which reject dim 0) happy. That patch silently
// over-allocated the empty-slice output to the data dim and cascaded
// padding garbage: in `Slice(x, k, k, axis) -> Loop` (an empty slice
// used as a Loop accumulator's v_init), `tensor.dim(acc, axis)` inside
// the body returns the upper bound instead of 0, so the Concat-grown
// accumulator starts at upper_bound + chunk and the downstream Reshape
// chain operates on a wildly oversized buffer. Tolerating dim 0 in the
// MIOpen wrapper (`wrap_miopenOpTensor` identity-copies LHS->OUT when
// one operand has any dim 0 and the other matches OUT) is cheaper than
// silently corrupting the accumulator. Past_kv-on-first-decode-token
// (the original motivator) also benefits: downstream Concat/Reshape now
// see the correct dim 0 and short-circuit naturally.
//
// Constants are resolved via `tryFoldToDenseAttr`, which peels
// `bufferization.to_tensor` / `memref.cast` / `tensor.cast` wrappers, reads
// a `hipdnn.fold_value` sidecar attr on `memref.global` (attached by
// `lowerOnnxConstants` for small int constants — see OnnxToHip.cpp), and
// follows `BlockArgument`s captured into outlined `hip.loop` body funcs
// back to their parent-scope value via `followCaptureToParent`.
//
// Before / After (loop-body slice with runtime starts/ends, const axes,
// dynamic input dim — case 3):
//
//   Before:
//     %y = "onnx.Slice"(%x, %s, %e, %axes_cap)
//        : (tensor<2x?x1152xf16>, tensor<1xi64>(runtime), ...) ->
//          tensor<2x?x1152xf16>
//     // tensor.empty dyn-size for axis 1 = tensor.dim(%x, 1)
//     //                                    (upper bound — leaks padding)
//
//   After:
//     %s0 = tensor.extract %s[%c0] : tensor<1xi64>
//     %e0 = tensor.extract %e[%c0] : tensor<1xi64>
//     // ONNX neg-index + clamp arith on index
//     %extent = ...   // possibly 0 at runtime
//     // tensor.empty dyn-size for axis 1 = %extent (no zero-guard)

// ---------------------------------------------------------------------------
// Constant resolution
// ---------------------------------------------------------------------------

/// If `value` is a BlockArgument of an outlined `hip.loop` body func, follow
/// it back to the corresponding `captures` operand of the enclosing loop op.
///
/// Body func argument layout produced by `OnnxLoopOutline`:
///   arg 0:                          !hip.context
///   arg 1:                          iter index
///   arg 2:                          cond_in
///   args [3 .. 3+num_loop_carried): loop-carried
///   args [3+num_loop_carried .. ):  captures (in `loopOp.getCaptures()` order)
static mlir::Value followCaptureToParent(mlir::Value value) {
  auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(value);
  if (!blockArg)
    return {};
  auto funcOp = mlir::dyn_cast_or_null<mlir::func::FuncOp>(
      blockArg.getOwner()->getParentOp());
  if (!funcOp)
    return {};
  if (blockArg.getOwner() != &funcOp.getBody().front())
    return {};
  auto module = funcOp->getParentOfType<mlir::ModuleOp>();
  if (!module)
    return {};

  mlir::hip::LoopOp loopOp;
  llvm::StringRef bodyName = funcOp.getName();
  module.walk([&](mlir::hip::LoopOp candidate) {
    if (candidate.getBodyFunc() == bodyName) {
      loopOp = candidate;
      return mlir::WalkResult::interrupt();
    }
    return mlir::WalkResult::advance();
  });
  if (!loopOp)
    return {};

  unsigned numLoopCarried =
      static_cast<unsigned>(loopOp.getNumLoopCarried());
  unsigned argNum = blockArg.getArgNumber();
  unsigned captureStart = 3u + numLoopCarried;
  if (argNum < captureStart)
    return {};

  unsigned captureIdx = argNum - captureStart;
  auto captures = loopOp.getCaptures();
  if (captureIdx >= captures.size())
    return {};
  return captures[captureIdx];
}

/// Walk through value-preserving wrappers until either a compile-time
/// `DenseElementsAttr` is found, or the chain ends in a value we can't see
/// through. Bounded iteration count guards against malformed IR.
static mlir::DenseElementsAttr tryFoldToDenseAttr(mlir::Value value) {
  for (int iter = 0; iter < 16; ++iter) {
    if (!value)
      return nullptr;

    {
      mlir::Attribute attr;
      if (mlir::matchPattern(value, mlir::m_Constant(&attr))) {
        if (auto dense = mlir::dyn_cast<mlir::DenseElementsAttr>(attr))
          return dense;
        return nullptr;
      }
    }

    if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(value)) {
      mlir::Value parent = followCaptureToParent(blockArg);
      if (!parent)
        return nullptr;
      value = parent;
      continue;
    }

    mlir::Operation *defOp = value.getDefiningOp();
    if (!defOp)
      return nullptr;

    if (auto toTensor =
            mlir::dyn_cast<mlir::bufferization::ToTensorOp>(defOp)) {
      value = toTensor.getBuffer();
      continue;
    }
    if (auto cast = mlir::dyn_cast<mlir::memref::CastOp>(defOp)) {
      value = cast.getSource();
      continue;
    }
    if (auto cast = mlir::dyn_cast<mlir::tensor::CastOp>(defOp)) {
      value = cast.getSource();
      continue;
    }
    if (auto getGlobal = mlir::dyn_cast<mlir::memref::GetGlobalOp>(defOp)) {
      auto module = defOp->getParentOfType<mlir::ModuleOp>();
      if (!module)
        return nullptr;
      auto global = module.lookupSymbol<mlir::memref::GlobalOp>(
          getGlobal.getNameAttr());
      if (!global)
        return nullptr;
      if (auto fold = global->getAttrOfType<mlir::DenseElementsAttr>(
              "hipdnn.fold_value"))
        return fold;
      return mlir::dyn_cast_or_null<mlir::DenseElementsAttr>(
          global.getInitialValueAttr());
    }

    if (defOp->getName().getStringRef() == "onnx.Constant") {
      if (auto attr = defOp->getAttr("value"))
        if (auto dense = mlir::dyn_cast<mlir::DenseElementsAttr>(attr))
          return dense;
      return nullptr;
    }

    return nullptr;
  }
  return nullptr;
}

/// Extract a 1-D integer tensor constant into a SmallVector<int64_t>.
static mlir::LogicalResult
extractIntVector(mlir::Value v, llvm::SmallVectorImpl<int64_t> &out) {
  if (!v)
    return mlir::failure();
  auto dense = tryFoldToDenseAttr(v);
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

static mlir::Value normaliseOptional(mlir::Value v) {
  if (!v)
    return v;
  auto defOp = v.getDefiningOp();
  if (defOp && defOp->getName().getStringRef() == "onnx.NoValue")
    return mlir::Value();
  return v;
}

/// Compute the logical slice extent for one axis under ONNX semantics, on a
/// statically-known input dim. Returns nullopt if any input is irrecoverable.
struct SliceParams {
  int64_t start;
  int64_t end;
  int64_t step;
};

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

/// Per-axis SliceToHip output-size hint. Four states (priority order):
///   * `staticSize >= 0`: exact compile-time extent (use IntegerAttr).
///   * `runtimeFromDim = true` with `runtimeStart`, `runtimeEnd`, `step`:
///     compute extent at host runtime as
///       clamp(min(end, dim) - max(start_norm, 0), 0, dim) / step (ceildiv).
///     Pure host-side index arith on `tensor.dim`.
///   * `runtimeFromOperand = true` with `paramIndex`, `step`: extract
///     start/end from the slice's runtime operand-1 / operand-2 (1-D int
///     tensor) at `paramIndex`, then run the same clamp arith. Requires the
///     starts/ends source to be host-readable (either already pinned via
///     `MaterializeHostScalars` or a chain that the pass will redirect once
///     the host loads we emit here become its triggers). Wrap the result
///     with a zero-guard `select(extent==0, upper_bound, extent)` —
///     downstream MIOpen broadcast tensor descriptors reject dim 0, and
///     genuine empty slices (e.g. past_kv on first decode token) need the
///     legacy upper-bound contract.
///   * `useUpperBound = true`: fall back to `data.dim[i]`.
struct SliceAxisInfo {
  int64_t staticSize = -1;
  bool runtimeFromDim = false;
  bool runtimeFromOperand = false;
  bool useUpperBound = false;
  int64_t runtimeStart = 0;
  int64_t runtimeEnd = 0;
  int64_t paramIndex = -1; // index into starts/ends/steps 1-D tensors
  int64_t step = 1;
};

/// Resolve which input axes a Slice op touches and what their (start, end,
/// step) are. Returns `true` only when EVERY param folds and EVERY touched
/// axis maps to a unique input axis in range. Untouched axes get
/// `useUpperBound = true` (i.e. use `data.dim[i]` verbatim); touched axes
/// get either `staticSize` (static input dim) or `runtimeFromDim` (dynamic
/// input dim, constant indices).
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
    if (axes) {
      if (mlir::failed(extractIntVector(axes, axesVec)))
        return false;
    }
  }
  if (axesVec.empty())
    for (int64_t i : llvm::seq<int64_t>(rank))
      axesVec.push_back(i);
  if (op->getNumOperands() == 5) {
    mlir::Value steps = normaliseOptional(op->getOperand(4));
    if (steps) {
      if (mlir::failed(extractIntVector(steps, stepsVec)))
        return false;
    }
  }
  if (stepsVec.empty())
    stepsVec.assign(axesVec.size(), 1);
  if (axesVec.size() != startsVec.size() ||
      axesVec.size() != endsVec.size() || axesVec.size() != stepsVec.size())
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
      info[axis].staticSize = clampSlice(
          dim, SliceParams{startsVec[k], endsVec[k], step});
    }
  }
  return true;
}

/// Same as `resolveSliceExtents` but accepts runtime starts/ends (only axes
/// and steps must fold). Used when the slice op has compile-time-known axes
/// but runtime indices (e.g. the per-token attention slice inside Qwen
/// vision's outlined loop body). For each touched axis sets
/// `runtimeFromOperand = true` with the `paramIndex` so the caller can emit
/// the `tensor.extract` + clamp arith at IR-build time.
static bool resolveSliceExtentsRuntimeIndices(
    mlir::Operation *op, mlir::RankedTensorType dataType,
    llvm::SmallVectorImpl<SliceAxisInfo> &info) {
  if (op->getNumOperands() < 3)
    return false;
  int64_t rank = dataType.getRank();

  // starts/ends operand types must be ranked 1-D.
  auto sTy = mlir::dyn_cast<mlir::RankedTensorType>(op->getOperand(1).getType());
  auto eTy = mlir::dyn_cast<mlir::RankedTensorType>(op->getOperand(2).getType());
  if (!sTy || !eTy || sTy.getRank() != 1 || eTy.getRank() != 1)
    return false;
  // Their element count must agree with the axes count.
  if (sTy.isDynamicDim(0) || eTy.isDynamicDim(0))
    return false;
  int64_t nIdx = sTy.getDimSize(0);
  if (eTy.getDimSize(0) != nIdx)
    return false;

  llvm::SmallVector<int64_t> axesVec, stepsVec;
  if (op->getNumOperands() >= 4) {
    mlir::Value axes = normaliseOptional(op->getOperand(3));
    if (axes) {
      if (mlir::failed(extractIntVector(axes, axesVec)))
        return false;
    }
  }
  if (axesVec.empty())
    for (int64_t i : llvm::seq<int64_t>(rank))
      axesVec.push_back(i);
  if (static_cast<int64_t>(axesVec.size()) != nIdx)
    return false;
  if (op->getNumOperands() == 5) {
    mlir::Value steps = normaliseOptional(op->getOperand(4));
    if (steps) {
      if (mlir::failed(extractIntVector(steps, stepsVec)))
        return false;
    }
  }
  if (stepsVec.empty())
    stepsVec.assign(axesVec.size(), 1);
  if (axesVec.size() != stepsVec.size())
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
    if (step <= 0)
      return false;
    info[axis].useUpperBound = false;
    info[axis].step = step;
    info[axis].runtimeFromOperand = true;
    info[axis].paramIndex = static_cast<int64_t>(k);
  }
  return true;
}

/// Emit host-side `index` arithmetic to compute the logical slice extent on
/// a dynamic-input-dim axis given constant start/end/step. `dim` is the
/// `tensor.dim` (or memref.dim) Value. Implements ONNX clamping semantics.
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

// ---------------------------------------------------------------------------
// Patterns
// ---------------------------------------------------------------------------

// NOTE on the omitted SliceDecompose pattern: an earlier prototype rewrote
// fully-const, all-static-dim slices directly to `tensor.extract_slice`
// (bufferizes to a zero-copy `memref.subview`). It was intentionally dropped
// from the registered pattern set because the resulting subview carries a
// strided layout while many downstream HIP custom kernels assume contiguous
// packed memref descriptors and read wrong elements from strided parents —
// losing the cosine improvement that exact-extent sizing in SliceToHip
// already delivers via the kernel-copy path. Re-add only after auditing
// every downstream consumer (matmul / norm / elementwise / gqa kernels)
// for strided-input support.

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

    // Compute exact per-axis sizes from compile-time-foldable params.
    // Three cases per dynamic output dim:
    //   * Touched axis, static input dim → IntegerAttr (compile-time extent).
    //   * Touched axis, dynamic input dim, constant indices → host-side
    //     `index` arith on `tensor.dim` (descriptor-only, no GPU read).
    //   * Untouched axis OR runtime indices → upper bound = data.dim[i].
    // Sizing the buffer to the LOGICAL extent (rather than the data-dim
    // upper bound) means downstream consumers see the correctly-sized
    // output and don't fold the kernel's zero-padded tail into reductions
    // or matmuls — the root cause of cosine collapse in dynamic-shape
    // vision encoders. Zero-size extents are NOT shrunk (MIOpen broadcast
    // tensor descriptors reject dim 0); those keep the upper-bound contract.
    llvm::SmallVector<SliceAxisInfo> sliceInfo;
    bool haveExtents = resolveSliceExtents(op, dataType, sliceInfo);
    if (!haveExtents)
      // Runtime starts/ends path (e.g. per-token attention slices inside
      // Qwen vision's outlined loop body). axes/steps must still fold.
      haveExtents =
          resolveSliceExtentsRuntimeIndices(op, dataType, sliceInfo);

    // Capture starts/ends/steps operands for the runtime-from-operand path.
    mlir::Value startsTensor = op->getOperand(1);
    mlir::Value endsTensor = op->getOperand(2);

    auto guardWithUpper = [&](mlir::Value extent, mlir::Value upper) {
      // Empty-slice (extent == 0) gets sized to the data-dim upper bound.
      // For the `Slice(x, k, k, axis) -> Loop` pattern (an empty placeholder
      // used as a Loop accumulator's v_init), the v_init buffer MUST be
      // large enough to hold the loop body's accumulated writes — the
      // `hip-fix-loop-accumulator-offset` pass replaces the body's
      // frozen-dim offset with `iter * chunk_size`, which only writes safely
      // when the underlying buffer covers `max_trip * chunk_size` rows.
      // `upper` = `data.dim(axis)` is exactly that bound for the canonical
      // Qwen-style windowed-attention pattern.
      mlir::Value zero =
          mlir::arith::ConstantIndexOp::create(rewriter, loc, 0);
      mlir::Value isZero = mlir::arith::CmpIOp::create(
          rewriter, loc, mlir::arith::CmpIPredicate::eq, extent, zero);
      return mlir::arith::SelectOp::create(rewriter, loc, isZero, upper,
                                           extent)
          .getResult();
    };

    auto extractI64ToIndex = [&](mlir::Value tensor1D, int64_t k) {
      mlir::Value kIdx =
          mlir::arith::ConstantIndexOp::create(rewriter, loc, k);
      mlir::Value elem =
          mlir::tensor::ExtractOp::create(rewriter, loc, tensor1D, kIdx);
      auto intTy = mlir::cast<mlir::IntegerType>(elem.getType());
      if (intTy.getWidth() != 64)
        elem = mlir::arith::ExtSIOp::create(rewriter, loc,
                                            rewriter.getI64Type(), elem);
      // Cast i64 -> index.
      return mlir::arith::IndexCastOp::create(
                 rewriter, loc, rewriter.getIndexType(), elem)
          .getResult();
    };

    auto emitRuntimeOperandExtent = [&](mlir::Value dim, int64_t k,
                                         int64_t step) {
      using mlir::arith::AddIOp;
      using mlir::arith::ConstantIndexOp;
      using mlir::arith::DivSIOp;
      using mlir::arith::MaxSIOp;
      using mlir::arith::MinSIOp;
      using mlir::arith::SubIOp;
      mlir::Value zeroI = ConstantIndexOp::create(rewriter, loc, 0);
      mlir::Value start = extractI64ToIndex(startsTensor, k);
      mlir::Value end = extractI64ToIndex(endsTensor, k);
      // ONNX neg-index: if start < 0 then start += dim. Same for end.
      mlir::Value sLt0 = mlir::arith::CmpIOp::create(
          rewriter, loc, mlir::arith::CmpIPredicate::slt, start, zeroI);
      mlir::Value sPlusDim = AddIOp::create(rewriter, loc, start, dim);
      mlir::Value sNorm = mlir::arith::SelectOp::create(rewriter, loc, sLt0,
                                                        sPlusDim, start);
      mlir::Value eLt0 = mlir::arith::CmpIOp::create(
          rewriter, loc, mlir::arith::CmpIPredicate::slt, end, zeroI);
      mlir::Value ePlusDim = AddIOp::create(rewriter, loc, end, dim);
      mlir::Value eNorm = mlir::arith::SelectOp::create(rewriter, loc, eLt0,
                                                        ePlusDim, end);
      // Clamp to [0, dim].
      mlir::Value sLo = MaxSIOp::create(rewriter, loc, sNorm, zeroI);
      mlir::Value sClamped = MinSIOp::create(rewriter, loc, sLo, dim);
      mlir::Value eLo = MaxSIOp::create(rewriter, loc, eNorm, zeroI);
      mlir::Value eClamped = MinSIOp::create(rewriter, loc, eLo, dim);
      mlir::Value eGeS = MaxSIOp::create(rewriter, loc, eClamped, sClamped);
      mlir::Value diff = SubIOp::create(rewriter, loc, eGeS, sClamped);
      mlir::Value stepM1 = ConstantIndexOp::create(rewriter, loc, step - 1);
      mlir::Value stepC = ConstantIndexOp::create(rewriter, loc, step);
      mlir::Value diffPlusM = AddIOp::create(rewriter, loc, diff, stepM1);
      return DivSIOp::create(rewriter, loc, diffPlusM, stepC).getResult();
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
              : mlir::arith::ConstantIndexOp::create(
                    rewriter, loc, dataType.getDimSize(i))
                    .getResult();

      if (!haveExtents || sliceInfo[i].useUpperBound) {
        dynSizes.push_back(upper);
        continue;
      }
      if (sliceInfo[i].staticSize >= 0) {
        if (sliceInfo[i].staticSize == 0)
          dynSizes.push_back(upper);
        else
          dynSizes.push_back(mlir::arith::ConstantIndexOp::create(
              rewriter, loc, sliceInfo[i].staticSize));
        continue;
      }
      if (sliceInfo[i].runtimeFromDim) {
        mlir::Value dim =
            mlir::tensor::DimOp::create(rewriter, loc, data, i);
        mlir::Value extent =
            emitRuntimeExtent(rewriter, loc, dim, sliceInfo[i].runtimeStart,
                              sliceInfo[i].runtimeEnd, sliceInfo[i].step);
        dynSizes.push_back(guardWithUpper(extent, upper));
        continue;
      }
      if (sliceInfo[i].runtimeFromOperand) {
        mlir::Value dim =
            dataType.isDynamicDim(i)
                ? mlir::tensor::DimOp::create(rewriter, loc, data, i)
                      .getResult()
                : mlir::arith::ConstantIndexOp::create(
                      rewriter, loc, dataType.getDimSize(i))
                      .getResult();
        mlir::Value extent =
            emitRuntimeOperandExtent(dim, sliceInfo[i].paramIndex,
                                     sliceInfo[i].step);
        dynSizes.push_back(guardWithUpper(extent, upper));
        continue;
      }
      dynSizes.push_back(upper);
    }
    mlir::Value init =
        mlir::tensor::EmptyOp::create(rewriter, loc, resultType.getShape(),
                                      resultType.getElementType(), dynSizes);

    auto hipOp =
        mlir::hip::SliceOp::create(rewriter, loc, resultType, context, data,
                                   starts, ends, axes, steps, init);
    rewriter.replaceOp(op, hipOp->getResult(0));
    return mlir::success();
  }
};

} // namespace

void populateSliceConversionPatterns(RewritePatternSet &patterns,
                                     MLIRContext *ctx) {
  patterns.add<SliceToHip>(ctx);
}

} // namespace hip
} // namespace mlir
