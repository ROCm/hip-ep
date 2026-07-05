/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"
#include "ReadbackScalar.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Sequence.h"

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

/// Read entry `[idx]` from `padsHost` (the host copy of `pads`) as an `index`.
///
/// `padsHost` comes from `hip.transfer_to_host(%pads)`, so this is a synced
/// host read. A bare `tensor.extract` of the device `pads` would instead be an
/// unsynced host load of the device constants blob and SEGV where that blob is
/// real device memory. Only used on the runtime path (the compile-time path
/// uses literal pad amounts and never extracts).
///
/// Before:  %e = tensor.extract %pads_device[%idx]   // unsynced device load
/// After:   %e = tensor.extract %padsHost[%idx]      // synced host load
static mlir::Value extractAsIndex(mlir::PatternRewriter &rewriter,
                                  mlir::Location loc, mlir::Value padsHost,
                                  int64_t idx) {
  mlir::Value index = mlir::arith::ConstantIndexOp::create(rewriter, loc, idx);
  mlir::Value extracted = mlir::tensor::ExtractOp::create(
      rewriter, loc, padsHost, mlir::ValueRange{index});
  return mlir::arith::IndexCastOp::create(rewriter, loc,
                                          rewriter.getIndexType(), extracted);
}

/// Build the device-space init (see `createDeviceAllocTensor` in
/// OnnxToHipUtils.h) for the Pad output, sizing dynamic dims as
///   out_dim[i] = data_dim[i] + pads[i] + pads[i + N]
/// where `N` is the number of padded axes. Two cases:
///
///   * pads is a compile-time constant: use the literal amounts.
///   * pads is dynamic: emit `tensor.extract` + `arith.addi`.
///
/// With a non-default `axes`, only those axes are padded; other dims keep their
/// input size. `axes` must be absent or a compile-time constant -- a dynamic
/// `axes` would make the pad lookup data-dependent, which we can't express in
/// `alloc_tensor` dynsizes.
///
/// `padsAttr` / `axesAttr` are the compile-time values stamped by PadShapeFold
/// (the common case); if present we use them, else an inline-constant `pads`
/// operand, else a synced read of `padsHost` (see extractAsIndex). So `pads` is
/// the device operand (only for the inline fold) and `padsHost` its host copy
/// (only on the runtime path).
static mlir::FailureOr<mlir::Value> buildPadOutputInit(
    mlir::PatternRewriter &rewriter, mlir::Location loc, mlir::Operation *op,
    mlir::RankedTensorType resultType, mlir::Value data, mlir::Value pads,
    mlir::Value padsHost, mlir::Value axes, llvm::ArrayRef<int64_t> padsAttr,
    bool hasPadsAttr, llvm::ArrayRef<int64_t> axesAttr, bool hasAxesAttr) {
  // Fully static result: no dynamic sizes, so we don't touch `pads` / `axes`.
  // Important when an operand is dynamic (a function arg) but the output shape
  // is still fixed -- the per-axis math below would otherwise bail and leave
  // the onnx.Pad unconverted.
  if (resultType.hasStaticShape()) {
    return createDeviceAllocTensor(rewriter, loc, resultType,
                                   mlir::ValueRange{});
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
    for (int64_t i : llvm::seq<int64_t>(0, rank))
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
  for (int64_t i : llvm::seq<int64_t>(0, resultType.getRank())) {
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
      begin = extractAsIndex(rewriter, loc, padsHost, slot);
      end = extractAsIndex(rewriter, loc, padsHost, slot + nPadded);
    }
    mlir::Value sum =
        mlir::arith::AddIOp::create(rewriter, loc, dataDim, begin);
    sum = mlir::arith::AddIOp::create(rewriter, loc, sum, end);
    dynSizes.push_back(sum);
  }

  return createDeviceAllocTensor(rewriter, loc, resultType, dynSizes);
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

    // ONNX `Pad.constant_value` is a scalar; hip.pad takes it BY VALUE (no
    // buffer). A compile-time constant folds to an `arith.constant`; a runtime
    // value is brought to host via the same `hip.transfer_to_host` as
    // pads/axes, then read with `tensor.extract` (safe: the operand is the
    // post-sync host copy, not a device buffer).
    //
    // Before: %cv : tensor<f32>   (runtime, non-constant)
    // After:  %cvH = hip.transfer_to_host(%ctx, %cv : tensor<f32>) ->
    // tensor<f32>
    //         %s   = tensor.extract %cvH[] : tensor<f32>
    // (constant case folds instead to:  %s = arith.constant <val> : f32)
    if (constantValue) {
      auto cvTy = mlir::cast<mlir::RankedTensorType>(constantValue.getType());
      mlir::Type cvElemTy = cvTy.getElementType();
      if (mlir::DenseElementsAttr dense = getConstantDense(constantValue);
          dense && dense.getNumElements() == 1) {
        constantValue =
            materializeConstScalar(rewriter, loc, dense, cvElemTy, 0);
      } else {
        // Collapse a 1-element 1-D tensor to rank-0 (zero-cost) so the
        // transfer/extract work on a 0-D scalar.
        if (cvTy.getRank() != 0 && cvTy.hasStaticShape() &&
            cvTy.getNumElements() == 1) {
          auto scalarTy = mlir::RankedTensorType::get({}, cvElemTy);
          llvm::SmallVector<mlir::ReassociationIndices> noReassoc;
          constantValue = mlir::tensor::CollapseShapeOp::create(
              rewriter, loc, scalarTy, constantValue, noReassoc);
        }
        mlir::Value cvHost =
            TransferToHostOp::create(rewriter, loc, constantValue.getType(),
                                     context, constantValue)
                .getResult();
        constantValue = mlir::tensor::ExtractOp::create(rewriter, loc, cvHost,
                                                        mlir::ValueRange{})
                            .getResult();
      }
    }

    auto resultType =
        mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());

    // Compile-time pads/axes stamped by the pre-lowering PadShapeFold pattern
    // (before externalization stripped the inline constant). If present,
    // buildPadOutputInit folds the output shape with no device traffic; if not,
    // it reads the amounts from the host transfer copy below.
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

    // Bring `pads` (and `axes`) to host ONCE via hip.transfer_to_host: wrap_pad
    // reads them CPU-side, and the dynamic-output path reads the same host copy
    // via tensor.extract (see extractAsIndex). One shared transfer instead of a
    // per-dim hip.readback_scalar; bufferizes to a host buffer + async D2H +
    // sync.
    //
    // Before: %pads : tensor<4xi64>   (device operand)
    // After:  %ph = hip.transfer_to_host(%ctx, %pads : tensor<4xi64>) ->
    // tensor<4xi64>
    mlir::Value padsHost =
        TransferToHostOp::create(rewriter, loc, pads.getType(), context, pads)
            .getResult();
    mlir::Value axesHost;
    if (axes)
      axesHost =
          TransferToHostOp::create(rewriter, loc, axes.getType(), context, axes)
              .getResult();

    // Build the output buffer as a DEVICE-space bufferization.alloc_tensor (see
    // createDeviceAllocTensor). Static result -> no dynsizes; dynamic dims are
    // computed as data_dim + pads_begin + pads_end at build time (from the
    // stamped constant `pads`, an inline operand, or a synced read of
    // padsHost).
    auto initOrFailure =
        buildPadOutputInit(rewriter, loc, op, resultType, data, pads, padsHost,
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
    operands.push_back(padsHost);
    if (constantValue)
      operands.push_back(constantValue);
    if (axes)
      operands.push_back(axesHost);
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
