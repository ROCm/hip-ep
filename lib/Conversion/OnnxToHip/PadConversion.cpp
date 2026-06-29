/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"
#include "ReadbackScalar.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"

namespace mlir {
namespace hip {
namespace {

/// Try to recognise \p v as a compile-time 1-D integer constant tensor
/// (matching the forms produced by `lowerOnnxConstants`). Returns null
/// otherwise. Mirrors the helper in SliceConversion.cpp.
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

/// Read entry `[idx]` from a 1-D int tensor on the host and return an
/// `index`-typed value.
///
/// The `pads` operand is frequently an externalized constant (its inline
/// `dense` value is stripped by constant externalization), so a bare
/// `tensor.extract` would bufferize to an UNSYNCHRONIZED host `memref.load` of
/// the device-resident constants blob -- a SEGV on targets where that blob is
/// true device memory. `readbackShapeEntryToHost` folds the value when the
/// constant is still inline and otherwise emits a synchronized
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
/// `padsAttr` / `axesAttr` carry the compile-time `pads` / `axes` values when
/// the pre-lowering `PadShapeFold` stamped them onto the op (the common case:
/// the operand was an inline constant before externalization). When present we
/// use them directly -- no operand read, no `hip.readback_scalar`. When absent
/// (genuinely runtime-dynamic `pads`) we fall back to reading the operand via
/// the synchronized readback path.
static mlir::FailureOr<mlir::Value> buildPadOutputInit(
    mlir::PatternRewriter &rewriter, mlir::Location loc, mlir::Operation *op,
    mlir::Value ctx, mlir::RankedTensorType resultType, mlir::Value data,
    mlir::Value pads, mlir::Value axes, llvm::ArrayRef<int64_t> padsAttr,
    bool hasPadsAttr, llvm::ArrayRef<int64_t> axesAttr, bool hasAxesAttr) {
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
  if (hasAxesAttr) {
    // Stamped by PadShapeFold before externalization.
    axesVec.assign(axesAttr.begin(), axesAttr.end());
  } else if (axes) {
    if (mlir::failed(extractIntVector(axes, axesVec)))
      return rewriter.notifyMatchFailure(
          op, "dynamic `axes` operand is not supported by Pad conversion");
  }
  if (axesVec.empty())
    for (int64_t i = 0; i < rank; ++i)
      axesVec.push_back(i);
  for (int64_t &a : axesVec)
    if (a < 0)
      a += rank;

  llvm::DenseMap<int64_t, int64_t> axisToSlot;
  for (auto [slot, axis] : llvm::enumerate(axesVec))
    axisToSlot[axis] = static_cast<int64_t>(slot);
  int64_t nPadded = static_cast<int64_t>(axesVec.size());

  // Decide whether we can use compile-time pad values. Prefer the attribute
  // stamped by PadShapeFold (captured before externalization); otherwise try
  // to read an inline operand (only succeeds when `pads` was small enough to
  // stay inline). A miss leaves `padsAreConst == false` -> readback fallback.
  llvm::SmallVector<int64_t> padsConst;
  if (hasPadsAttr)
    padsConst.assign(padsAttr.begin(), padsAttr.end());
  else
    (void)extractIntVector(pads, padsConst);
  bool padsAreConst = static_cast<int64_t>(padsConst.size()) == 2 * nPadded;

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

    mlir::Value begin, end;
    if (padsAreConst) {
      begin =
          mlir::arith::ConstantIndexOp::create(rewriter, loc, padsConst[slot]);
      end = mlir::arith::ConstantIndexOp::create(rewriter, loc,
                                                 padsConst[slot + nPadded]);
    } else {
      begin = extractAsIndex(rewriter, loc, ctx, pads, slot);
      end = extractAsIndex(rewriter, loc, ctx, pads, slot + nPadded);
    }
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

    // ONNX `Pad.constant_value` is a scalar; hip.pad takes it BY VALUE as a
    // plain float/integer SSA value (no buffer, no memory space). Resolve the
    // 0-D constant_value tensor to a host scalar: `readbackScalarToHost` folds
    // a compile-time constant (the overwhelmingly common pad value, e.g. 0.0)
    // to an `arith.constant` with zero device traffic, and otherwise emits a
    // synchronized `hip.readback_scalar`. First normalize a producer's
    // redundant single-element 1-D tensor to rank-0 (a zero-cost reshape) so
    // the readback helper sees the expected 0-D operand.
    //
    // Before: %cv : tensor<1xf32>
    // After:  %cv0 = tensor.collapse_shape %cv [] : tensor<1xf32> into tensor<f32>
    //         %s   = arith.constant <val> : f32   // constant fold, or
    //         %s   = hip.readback_scalar(%ctx, %cv0 : tensor<f32>) -> f32
    if (constantValue) {
      if (auto cvTy =
              mlir::dyn_cast<mlir::RankedTensorType>(constantValue.getType())) {
        if (cvTy.getRank() != 0 && cvTy.hasStaticShape() &&
            cvTy.getNumElements() == 1) {
          auto scalarTy =
              mlir::RankedTensorType::get({}, cvTy.getElementType());
          llvm::SmallVector<mlir::ReassociationIndices> noReassoc;
          constantValue = mlir::tensor::CollapseShapeOp::create(
              rewriter, loc, scalarTy, constantValue, noReassoc);
        }
      }
      constantValue =
          readbackScalarToHost(rewriter, loc, context, constantValue);
    }

    auto resultType =
        mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());

    // Compile-time pads/axes stamped by the pre-lowering PadShapeFold pattern
    // (before externalization stripped the inline constant). Their presence
    // lets buildPadOutputInit fold the output shape with zero device traffic;
    // absence falls back to the synchronized readback path.
    llvm::ArrayRef<int64_t> padsAttr;
    bool hasPadsAttr = false;
    if (auto a =
            op->getAttrOfType<mlir::DenseI64ArrayAttr>("hipdnn.pad_amounts")) {
      padsAttr = a.asArrayRef();
      hasPadsAttr = true;
    }
    llvm::ArrayRef<int64_t> axesAttr;
    bool hasAxesAttr = false;
    if (auto a =
            op->getAttrOfType<mlir::DenseI64ArrayAttr>("hipdnn.pad_axes")) {
      axesAttr = a.asArrayRef();
      hasAxesAttr = true;
    }

    // Build the output buffer. When the result is fully static, the helper
    // collapses to a `tensor.empty` with no dynsizes (identical to the old
    // `createEmptyTensor` behaviour). When at least one dim is dynamic, the
    // dynamic dims are computed as data_dim + pads_begin + pads_end at IR
    // build time (using the stamped constant `pads` if available, an inline
    // operand if still present, otherwise a synchronized readback).
    auto initOrFailure =
        buildPadOutputInit(rewriter, loc, op, context, resultType, data, pads,
                           axes, padsAttr, hasPadsAttr, axesAttr, hasAxesAttr);
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
