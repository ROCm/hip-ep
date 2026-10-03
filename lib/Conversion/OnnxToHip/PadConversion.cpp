/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"
#include "ReadbackScalar.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallBitVector.h"

namespace mlir {
namespace hip {
namespace {

static bool isRankOneIntTensor(mlir::Value value) {
  auto type = mlir::dyn_cast<mlir::RankedTensorType>(value.getType());
  return type && type.getRank() == 1 &&
         (type.getElementType().isInteger(32) ||
          type.getElementType().isInteger(64));
}

static bool areValidAxes(int64_t rank, llvm::ArrayRef<int64_t> axes) {
  llvm::SmallBitVector seen(rank);
  for (int64_t axis : axes) {
    if (axis < 0)
      axis += rank;
    if (axis < 0 || axis >= rank || seen.test(axis))
      return false;
    seen.set(axis);
  }
  return true;
}

/// Read entry `[idx]` from a 1-D int tensor on the host and return an
/// `index`-typed value.
///
/// A non-inline `pads` source may become device-resident when the standalone
/// externalizer runs after conversion, so a bare `tensor.extract` would
/// bufferize to an UNSYNCHRONIZED host `memref.load` -- a SEGV on targets where
/// the source is true device memory. `readbackShapeEntryToHost` folds an
/// inspectable carrier value and otherwise emits a synchronized
/// `hip.readback_scalar` (D2H + stream sync). See ReadbackScalar.h.
static mlir::Value extractAsIndex(mlir::PatternRewriter &rewriter,
                                  mlir::Location loc, mlir::Value ctx,
                                  mlir::Value tensor1d, int64_t idx) {
  mlir::Value extracted =
      readbackShapeEntryToHost(rewriter, loc, ctx, tensor1d, idx);
  return mlir::arith::IndexCastOp::create(rewriter, loc,
                                          rewriter.getIndexType(), extracted);
}

/// Build a `tensor.empty` for the Pad output, computing dynamic dims as
///   out_dim[i] = data_dim[i] + pads[i] + pads[i + N]
/// where `N` is the number of padded axes. Two cases:
///
///   * pads is a compile-time constant: use the literal pad amounts.
///   * pads is dynamic: emit `tensor.extract` + `arith.addi`.
///
/// When `axes` is supplied and not the default identity, only those axes
/// participate; dims outside `axes` keep their input size. We require
/// `axes` to be either absent or a compile-time constant -- a dynamic
/// `axes` would make the per-dim pad lookup data-dependent, which we
/// can't express with `tensor.empty` dynsizes.
///
/// `staticPads` / `staticAxes` are read directly from dense constant carriers
/// during compute conversion. When pads are genuinely runtime-dynamic, the
/// synchronized readback path remains the destination-sizing fallback.
static mlir::FailureOr<mlir::Value>
buildPadOutputInit(mlir::PatternRewriter &rewriter, mlir::Location loc,
                   mlir::Operation *op, mlir::Value ctx,
                   mlir::RankedTensorType resultType, mlir::Value data,
                   mlir::Value pads, mlir::Value axes,
                   std::optional<llvm::ArrayRef<int64_t>> staticPads,
                   std::optional<llvm::ArrayRef<int64_t>> staticAxes) {
  // Fully static result: the empty tensor needs no dynamic sizes, and we do
  // not have to look at `pads` / `axes` at all. This is important when
  // either operand is a function argument (dynamic) but the output shape is
  // still fixed -- the per-axis math below would otherwise bail out and the
  // whole onnx.Pad op would stay unconverted.
  if (resultType.hasStaticShape()) {
    return mlir::tensor::EmptyOp::create(rewriter, loc, resultType.getShape(),
                                         resultType.getElementType(),
                                         mlir::ValueRange{})
        .getResult();
  }

  auto dataType = mlir::cast<mlir::RankedTensorType>(data.getType());
  int64_t rank = dataType.getRank();

  // Build the axis -> pad-index map. By default every axis is padded in
  // order, so axis i lives at slot i in the pads vector. If `axes` is a
  // compile-time constant we honour it.
  llvm::SmallVector<int64_t> axesVec;
  if (staticAxes)
    axesVec.assign(staticAxes->begin(), staticAxes->end());
  if (!axes)
    for (int64_t i = 0; i < rank; ++i)
      axesVec.push_back(i);
  llvm::DenseMap<int64_t, int64_t> axisToSlot;
  for (auto [slot, a] : llvm::enumerate(axesVec)) {
    if (a < 0)
      a += rank;
    if (a < 0 || a >= rank || axisToSlot.contains(a))
      return rewriter.notifyMatchFailure(
          op, "Pad axes must be unique and within the data rank");
    axisToSlot[a] = static_cast<int64_t>(slot);
  }
  int64_t nPadded = static_cast<int64_t>(axesVec.size());

  if (staticPads) {
    llvm::SmallVector<mlir::OpFoldResult> resultShape;
    if (mlir::failed(mlir::hip::reifyPadShape(rewriter, loc, data, pads, axes,
                                              staticPads, staticAxes,
                                              resultShape)))
      return rewriter.notifyMatchFailure(
          op, "constant Pad parameters are invalid for the data shape");
    mlir::FailureOr<mlir::Value> init = createEmptyTensorFromReifiedShape(
        rewriter, loc, resultType, resultShape);
    if (mlir::failed(init))
      return rewriter.notifyMatchFailure(
          op, "Pad result type is incompatible with inferred extents");
    return *init;
  }

  llvm::SmallVector<mlir::Value> dynSizes;
  for (int64_t i = 0; i < resultType.getRank(); ++i) {
    if (!resultType.isDynamicDim(i))
      continue;

    // data dim contribution (constant when known, runtime otherwise).
    mlir::Value dataDim;
    if (i < rank && !dataType.isDynamicDim(i))
      dataDim = mlir::arith::ConstantIndexOp::create(rewriter, loc,
                                                     dataType.getDimSize(i));
    else
      dataDim = mlir::tensor::DimOp::create(rewriter, loc, data, i);

    auto it = axisToSlot.find(i);
    if (it == axisToSlot.end()) {
      // Not in `axes` -- this axis is not padded.
      dynSizes.push_back(dataDim);
      continue;
    }
    int64_t slot = it->second;

    mlir::Value begin = extractAsIndex(rewriter, loc, ctx, pads, slot);
    mlir::Value end = extractAsIndex(rewriter, loc, ctx, pads, slot + nPadded);
    mlir::Value sum =
        mlir::arith::AddIOp::create(rewriter, loc, dataDim, begin);
    sum = mlir::arith::AddIOp::create(rewriter, loc, sum, end);
    dynSizes.push_back(sum);
  }

  return mlir::tensor::EmptyOp::create(rewriter, loc, resultType.getShape(),
                                       resultType.getElementType(), dynSizes)
      .getResult();
}

/// onnx.Pad -> hip.pad
///
/// ONNX layout: Pad(data, pads, [constant_value], [axes]) {mode}.
/// Optional inputs may be present and typed `none` (onnx.NoValue) when omitted
/// by the producer.
struct PadToHip : public mlir::RewritePattern {
  PadToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Pad", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    if (op->getNumOperands() < 2 || op->getNumOperands() > 4 ||
        op->getNumResults() != 1)
      return rewriter.notifyMatchFailure(op, "expected 2-4 inputs, 1 output");

    auto ctxOrFailure = getContextArg(op, rewriter);
    if (mlir::failed(ctxOrFailure))
      return mlir::failure();
    mlir::Value context = *ctxOrFailure;

    mlir::Location loc = op->getLoc();
    mlir::Value data = op->getOperand(0);
    mlir::Value pads = op->getOperand(1);

    auto isNone = [](mlir::Value v) -> bool {
      return v && mlir::isa<mlir::NoneType>(v.getType());
    };

    mlir::Value constantValue = nullptr;
    if (op->getNumOperands() > 2 && !isNone(op->getOperand(2)))
      constantValue = op->getOperand(2);
    mlir::Value axes = nullptr;
    if (op->getNumOperands() > 3 && !isNone(op->getOperand(3)))
      axes = op->getOperand(3);

    auto dataType = mlir::dyn_cast<mlir::RankedTensorType>(data.getType());
    auto resultType =
        mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
    if (!dataType || !resultType)
      return rewriter.notifyMatchFailure(
          op, "Pad data and result must be ranked tensors");
    if (!isRankOneIntTensor(pads))
      return rewriter.notifyMatchFailure(
          op, "Pad pads must be a rank-1 i32 or i64 tensor");
    if (axes && !isRankOneIntTensor(axes))
      return rewriter.notifyMatchFailure(
          op, "Pad axes must be a rank-1 i32 or i64 tensor");

    llvm::SmallVector<int64_t> padsValues;
    std::optional<llvm::ArrayRef<int64_t>> staticPads;
    if (extractConstantIntVector(pads, padsValues))
      staticPads = padsValues;
    llvm::SmallVector<int64_t> axesValues;
    std::optional<llvm::ArrayRef<int64_t>> staticAxes;
    if (axes) {
      if (extractConstantIntVector(axes, axesValues))
        staticAxes = axesValues;
      else if (!resultType.hasStaticShape())
        return rewriter.notifyMatchFailure(
            op, "dynamic `axes` operand is not supported by Pad conversion");
    }
    if (staticAxes && !areValidAxes(dataType.getRank(), *staticAxes))
      return rewriter.notifyMatchFailure(
          op, "Pad axes must be unique and within the data rank");

    if (staticPads && (!axes || staticAxes)) {
      auto inferredShape = mlir::hip::inferPadShape(dataType.getShape(),
                                                    *staticPads, staticAxes);
      if (mlir::failed(inferredShape))
        return rewriter.notifyMatchFailure(
            op, "constant Pad parameters are invalid for the data shape");
      if (!isResultTypeCompatibleWithPayloadShape(resultType, *inferredShape))
        return rewriter.notifyMatchFailure(
            op, "Pad result type contradicts constant parameters");
    }

    // Build the output buffer. When the result is fully static, the helper
    // collapses to a `tensor.empty` with no dynsizes (identical to the old
    // `createEmptyTensor` behaviour). When at least one dim is dynamic, the
    // dynamic dims are computed as data_dim + pads_begin + pads_end at IR
    // build time from a dense carrier when available, otherwise through a
    // synchronized readback of genuinely runtime pads.
    auto initOrFailure =
        buildPadOutputInit(rewriter, loc, op, context, resultType, data, pads,
                           axes, staticPads, staticAxes);
    if (mlir::failed(initOrFailure))
      return mlir::failure();
    mlir::Value init = *initOrFailure;

    mlir::StringAttr modeAttr;
    if (auto attr = op->getAttrOfType<mlir::StringAttr>("mode"))
      modeAttr = attr;
    else
      modeAttr = rewriter.getStringAttr("constant");

    // Build operands [ctx, data, pads, cval?, axes?, output] and segment
    // sizes for AttrSizedOperandSegments.
    mlir::SmallVector<mlir::Value> operands;
    operands.push_back(context);
    operands.push_back(data);
    operands.push_back(pads);
    if (constantValue)
      operands.push_back(constantValue);
    if (axes)
      operands.push_back(axes);
    operands.push_back(init);

    llvm::SmallVector<int32_t, 6> segmentSizes = {
        /*ctx=*/1,
        /*data=*/1,
        /*pads=*/1,
        /*constant_value=*/constantValue ? 1 : 0,
        /*axes=*/axes ? 1 : 0,
        /*output=*/1};

    mlir::SmallVector<mlir::NamedAttribute> attrs;
    attrs.push_back(rewriter.getNamedAttr("mode", modeAttr));
    if (staticPads)
      attrs.push_back(rewriter.getNamedAttr(
          "static_pads", rewriter.getDenseI64ArrayAttr(*staticPads)));
    if (staticAxes)
      attrs.push_back(rewriter.getNamedAttr(
          "static_axes", rewriter.getDenseI64ArrayAttr(*staticAxes)));

    mlir::OperationState state(loc, "hip.pad");
    state.addOperands(operands);
    state.addAttributes(attrs);
    state.addTypes({resultType});
    state.addAttribute("operand_segment_sizes",
                       rewriter.getDenseI32ArrayAttr(segmentSizes));

    mlir::Operation *hipOp = rewriter.create(state);
    rewriter.replaceOp(op, hipOp->getResults());
    return mlir::success();
  }
};

} // namespace

void populatePadConversionPatterns(RewritePatternSet &patterns,
                                   MLIRContext *ctx) {
  patterns.add<PadToHip>(ctx);
}

} // namespace hip
} // namespace mlir
