/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- PadShapeFold.cpp - Pre-lowering Pad normalization ----------------===//
//
// Two pre-lowering patterns for `onnx.Pad`: one rewrites the opset<11
// attribute-carrying form into the operand form (PadLegacyAttrsToOperands,
// documented at its definition), the other stamps compile-time pad amounts as
// attributes, described below.
//
// Sibling pre-lowering fold to GatherShapeFold / ReshapeShapeFold. Captures
// the compile-time value of `onnx.Pad`'s `pads` (and optional `axes`) operand
// onto the op while its producer is still a generic ONNX constant. The later
// `lowerOnnxConstants` sweep creates an inspectable `hip.constant`, and
// PadConversion runs before the standalone externalizer. Stamping still gives
// shape construction stable provenance independent of the rewritten producer.
//
// Why this matters
// ----------------
// `onnx.Pad` with a dynamic output shape needs the per-axis pad amounts on the
// HOST to size the output buffer (out_dim[i] = data_dim[i] + begin + end).
// `pads` is almost always a compile-time ONNX constant. This ONNX-rooted
// pre-rewrite runs before the producer changes dialect and stamps the values
// onto the Pad op. PadConversion can then size dynamic outputs without relying
// on a particular constant producer form or emitting synchronized D2H
// readbacks. Genuinely runtime-dynamic `pads` carry no attribute and still use
// the readback path (correctness preserved).
//
//   Before (this pattern):
//     %pads = onnx.Constant {value = dense<[0,1,0,1]> : tensor<4xi64>}
//     %out  = onnx.Pad(%data, %pads) {mode = "constant"}
//
//   After (this pattern):
//     %pads = onnx.Constant {value = dense<[0,1,0,1]> : tensor<4xi64>}
//     %out  = onnx.Pad(%data, %pads)
//               {mode = "constant", hipdnn.pad_amounts = array<i64: 0,1,0,1>}
//
//   (the `pads` operand is left untouched -- the hip.pad kernel still reads it
//    on the GPU; only the host-side output-shape math now uses the attribute.)
//
// Implementation notes
// --------------------
//   * Roots on `onnx.Pad`. Idempotent: bails if `hipdnn.pad_amounts` is already
//     set, so the greedy `ExistingOps`-strictness pre-lowering loop quiesces.
//   * Only fires when the result has at least one dynamic dim (the static-shape
//     case never reads `pads` in PadConversion, so stamping would be useless
//     churn).
//   * Reads the inline value from `onnx.Constant`'s `value` attr (or
//     `arith.constant`). If `pads` is not such an inline constant the op is
//     left unchanged -- it is a genuine runtime `pads` and PadConversion's
//     readback fallback handles it.
//   * Leaves the `onnx.Constant` ops in place; carrier lowering, DCE, and the
//     later standalone externalizer handle them as usual.
//
//===----------------------------------------------------------------------===//

#include "OnnxToHipUtils.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Debug.h"

#include <optional>

#define DEBUG_TYPE "pad-shape-fold"

STATISTIC(NumPadConstStamps,
          "Number of onnx.Pad ops whose constant pads/axes were stamped as "
          "attributes before carrier lowering");

STATISTIC(NumPadLegacyRewrites,
          "Number of attribute-carrying (opset<11) onnx.Pad ops rewritten to "
          "the operand form");

namespace mlir {
namespace hip {

namespace {

/// Return the values of `v` as an int64 vector when `v` is an inline 1-D
/// integer constant (`arith.constant` or `onnx.Constant {value = dense<...>}`),
/// or std::nullopt otherwise. Mirrors the inline-constant recognition used by
/// the sibling shape folds. It deliberately matches generic ONNX/arith
/// producers because this ONNX-rooted pre-rewrite runs before carrier lowering.
static std::optional<llvm::SmallVector<int64_t>>
getInlineIntVector(mlir::Value v) {
  if (!v)
    return std::nullopt;
  mlir::Operation *defOp = v.getDefiningOp();
  if (!defOp)
    return std::nullopt;

  mlir::DenseElementsAttr dense;
  if (auto cst = mlir::dyn_cast<mlir::arith::ConstantOp>(defOp))
    dense = mlir::dyn_cast<mlir::DenseElementsAttr>(cst.getValue());
  else if (defOp->getName().getStringRef() == "onnx.Constant")
    dense = defOp->getAttrOfType<mlir::DenseElementsAttr>("value");

  if (!dense)
    return std::nullopt;
  auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(dense.getType());
  if (!tensorType || tensorType.getRank() != 1)
    return std::nullopt;
  auto elemTy = tensorType.getElementType();
  if (!elemTy.isInteger(64) && !elemTy.isInteger(32))
    return std::nullopt;

  llvm::SmallVector<int64_t> out;
  for (mlir::APInt entry : dense.getValues<mlir::APInt>())
    out.push_back(entry.getSExtValue());
  return out;
}

/// Rewrite the opset<11 `Pad` form -- a lone `data` operand, with `pads` (and
/// the fill `value`) carried as attributes -- into the operand form every later
/// Pad pattern expects. Nothing in this pipeline upgrades opsets, so a tf2onnx
/// opset-9 export arrives at PadConversion with a single operand.
///
/// Done here rather than by teaching PadConversion a second input form, so the
/// whole Pad path downstream keeps a single shape: the next round of the
/// pre-lowering loop stamps the materialized constant through
/// PadStampConstShape, and conversion reads the same provenance it already uses
/// for the modern form. Mirrors SliceLegacyAttrsToOperands.
///
///   Before:
///     %out = onnx.Pad(%data) {mode = "reflect", pads = [0, 1, 0, 1]}
///   After:
///     %p   = onnx.Constant {value = dense<[0, 1, 0, 1]> : tensor<4xi64>}
///     %out = onnx.Pad(%data, %p) {mode = "reflect"}
struct PadLegacyAttrsToOperands : public mlir::RewritePattern {
  PadLegacyAttrsToOperands(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Pad", /*benefit=*/2, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    // Only the legacy schema has a lone `data`; the form this pattern emits
    // carries at least two operands, so it cannot match its own output.
    if (op->getNumOperands() != 1 || op->getNumResults() != 1)
      return rewriter.notifyMatchFailure(op, "pad.not_legacy_arity");

    llvm::SmallVector<int64_t> padsVec;
    if (auto dense = op->getAttrOfType<mlir::DenseI64ArrayAttr>("pads")) {
      padsVec.assign(dense.asArrayRef().begin(), dense.asArrayRef().end());
    } else if (auto array = op->getAttrOfType<mlir::ArrayAttr>("pads")) {
      for (mlir::Attribute entry : array) {
        auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(entry);
        if (!intAttr)
          return rewriter.notifyMatchFailure(op, "pad.legacy_pads_not_ints");
        padsVec.push_back(intAttr.getValue().getSExtValue());
      }
    } else {
      return rewriter.notifyMatchFailure(op, "pad.legacy_pads_missing");
    }

    // The legacy layout is [begin...; end...] over every axis, so the entry
    // count is twice the input rank. Checking it here keeps a malformed export
    // from reaching the output-shape math as a plausible-looking vector.
    mlir::Value data = op->getOperand(0);
    auto dataType = mlir::dyn_cast<mlir::RankedTensorType>(data.getType());
    if (!dataType)
      return rewriter.notifyMatchFailure(op, "pad.legacy_unranked_data");
    if (static_cast<int64_t>(padsVec.size()) != 2 * dataType.getRank())
      return rewriter.notifyMatchFailure(op, "pad.legacy_pads_arity");

    // `value` is the legacy spelling of the `constant_value` operand, and only
    // `constant` mode reads it. Positive zero is the default on both sides, so
    // a +0.0 or absent attribute needs no operand at all. Anything else,
    // including -0.0, has to be materialized in the data's element type to
    // keep the fill bit-exact.
    std::optional<llvm::APFloat> fill;
    mlir::FloatType fillType;
    if (auto valueAttr = op->getAttrOfType<mlir::FloatAttr>("value")) {
      if (!valueAttr.getValue().isPosZero()) {
        fillType = mlir::dyn_cast<mlir::FloatType>(dataType.getElementType());
        if (!fillType)
          return rewriter.notifyMatchFailure(op, "pad.legacy_value_not_float");
        fill = valueAttr.getValue();
        bool losesInfo = false;
        fill->convert(fillType.getFloatSemantics(),
                      llvm::APFloat::rmNearestTiesToEven, &losesInfo);
      }
    }

    // Every check that can decline is above this line: the pre-lowering loop
    // retries ops whose patterns fail, so a decline after inserting constants
    // would leave dead ones behind on each round.
    mlir::Location loc = op->getLoc();
    llvm::SmallVector<mlir::Value> operands{data};

    auto i64Type = rewriter.getI64Type();
    auto padsTensorType = mlir::RankedTensorType::get(
        {static_cast<int64_t>(padsVec.size())}, i64Type);
    {
      mlir::OperationState cst(loc, "onnx.Constant");
      cst.addTypes(padsTensorType);
      cst.addAttribute("value",
                       mlir::DenseElementsAttr::get(
                           padsTensorType, llvm::ArrayRef<int64_t>(padsVec)));
      operands.push_back(rewriter.create(cst)->getResult(0));
    }

    if (fill) {
      auto scalarType = mlir::RankedTensorType::get({}, fillType);
      mlir::OperationState cst(loc, "onnx.Constant");
      cst.addTypes(scalarType);
      cst.addAttribute("value",
                       mlir::DenseElementsAttr::get(
                           scalarType, llvm::ArrayRef<llvm::APFloat>{*fill}));
      operands.push_back(rewriter.create(cst)->getResult(0));
    }

    // Carry everything except the two attributes that just became operands, so
    // `mode` and the exporter's node naming survive the rewrite.
    llvm::SmallVector<mlir::NamedAttribute> attrs;
    for (mlir::NamedAttribute attr : op->getAttrs())
      if (attr.getName() != "pads" && attr.getName() != "value")
        attrs.push_back(attr);

    mlir::OperationState state(loc, "onnx.Pad");
    state.addOperands(operands);
    state.addAttributes(attrs);
    state.addTypes(op->getResult(0).getType());
    rewriter.replaceOp(op, rewriter.create(state)->getResult(0));

    LLVM_DEBUG(llvm::dbgs()
               << "[" DEBUG_TYPE "] rewrote attribute-form Pad ("
               << padsVec.size() << " entries) to the operand form\n");
    ++NumPadLegacyRewrites;
    return mlir::success();
  }
};

struct PadStampConstShape : public mlir::RewritePattern {
  PadStampConstShape(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Pad", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    // Idempotent: already stamped.
    if (op->hasAttr("hipdnn.pad_amounts"))
      return rewriter.notifyMatchFailure(op, "pad.already_stamped");

    if (op->getNumOperands() < 2)
      return rewriter.notifyMatchFailure(op, "pad.arity");

    // Only the dynamic-output case reads `pads` on the host in PadConversion;
    // a fully static result never needs the values, so stamping is pointless.
    auto resultType =
        mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
    if (!resultType || resultType.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "pad.static_result");

    auto padsVec = getInlineIntVector(op->getOperand(1));
    if (!padsVec)
      return rewriter.notifyMatchFailure(op, "pad.pads_not_inline_const");

    // `axes` is operand 3 when present and not an onnx.NoValue. Stamp it too so
    // PadConversion's axis->slot mapping does not depend on the rewritten
    // `axes` producer form. Absent/None axes => default identity, handled by
    // PadConversion without an attribute.
    std::optional<llvm::SmallVector<int64_t>> axesVec;
    if (op->getNumOperands() > 3) {
      mlir::Value axes = op->getOperand(3);
      bool axesIsNone = axes && mlir::isa<mlir::NoneType>(axes.getType());
      if (!axesIsNone) {
        axesVec = getInlineIntVector(axes);
        if (!axesVec)
          return rewriter.notifyMatchFailure(op, "pad.axes_not_inline_const");
      }
    }

    rewriter.modifyOpInPlace(op, [&] {
      op->setAttr("hipdnn.pad_amounts",
                  rewriter.getDenseI64ArrayAttr(*padsVec));
      if (axesVec)
        op->setAttr("hipdnn.pad_axes", rewriter.getDenseI64ArrayAttr(*axesVec));
    });

    LLVM_DEBUG(llvm::dbgs()
               << "[" DEBUG_TYPE "] stamped pad_amounts (" << padsVec->size()
               << " entries)" << (axesVec ? " + pad_axes" : "") << "\n");
    ++NumPadConstStamps;
    return mlir::success();
  }
};

} // namespace

void populatePadShapeFoldPatterns(mlir::RewritePatternSet &patterns,
                                  mlir::MLIRContext *ctx) {
  patterns.add<PadLegacyAttrsToOperands, PadStampConstShape>(ctx);
}

} // namespace hip
} // namespace mlir
