/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- SliceShapeFold.cpp - Pre-lowering Slice param stamping ------------===//
//
// Sibling pre-lowering fold to PadShapeFold. Captures the compile-time value
// of `onnx.Slice`'s starts/ends/axes/steps operands onto the op as attributes
// while their producers are still generic ONNX constants. The first carrier
// sweep then creates inspectable `hip.constant` ops, SliceDecompose runs during
// compute conversion, and only afterward does the standalone externalizer run.
//
// SwinV2 window/partition slices use small 1-element i64 constants for every
// parameter. Stamping them before their producer changes dialect gives
// SliceDecompose stable ONNX provenance and avoids falling through to the stub
// `hip.slice` runtime op.
//
//   Before:
//     %s = onnx.Constant {value = dense<0> : tensor<1xi64>}
//     %e = onnx.Constant {value = dense<8> : tensor<1xi64>}
//     %a = onnx.Constant {value = dense<1> : tensor<1xi64>}
//     %out = onnx.Slice(%data, %s, %e, %a)
//
//   After:
//     %out = onnx.Slice(%data, %s, %e, %a)
//              {hipdnn.slice_starts = array<i64: 0>,
//               hipdnn.slice_ends   = array<i64: 8>,
//               hipdnn.slice_axes   = array<i64: 1>}
//
//===----------------------------------------------------------------------===//

#include "OnnxToHipUtils.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Debug.h"

#include <optional>

#define DEBUG_TYPE "slice-shape-fold"

STATISTIC(NumSliceConstStamps,
          "Number of onnx.Slice ops whose constant params were stamped as "
          "attributes before carrier lowering");

STATISTIC(NumSliceLegacyRewrites,
          "Number of attribute-carrying (opset<10) onnx.Slice ops rewritten to "
          "the operand form");

namespace mlir {
namespace hip {

namespace {

static mlir::Value normaliseOptional(mlir::Value v) {
  if (!v)
    return v;
  auto defOp = v.getDefiningOp();
  if (defOp && defOp->getName().getStringRef() == "onnx.NoValue")
    return mlir::Value();
  return v;
}

/// Inline 1-D integer constant (`arith.constant` or `onnx.Constant`), or
/// std::nullopt. This ONNX-rooted pre-rewrite runs before carrier lowering.
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

/// Rewrite the opset<10 `Slice-1` form -- a lone `data` operand, with
/// starts/ends/axes carried as attributes -- into the operand form every later
/// Slice pattern expects. Nothing in this pipeline upgrades opsets, so an
/// opset-9 export arrives at SliceDecompose with one operand and is turned away
/// by its 3-5 operand arity check, then survives to bufferization.
///
/// Done here rather than by teaching SliceConversion a second input form so the
/// whole Slice path downstream keeps a single shape: the next round of the
/// pre-lowering loop stamps these constants through SliceStampConstParams, and
/// the decomposition then reads the same `hipdnn.slice_*` provenance it already
/// uses for the modern form.
///
/// `ends` keeps whatever sentinel the exporter wrote for "to the end", which
/// needs no special casing: SliceDecompose clamps every bound against the data
/// dim, mapping any sentinel onto the extent.
///
///   Before:
///     %out = onnx.Slice(%data) {starts = [0, 0, 0, 3],
///                               ends = [..], axes = [0, 1, 2, 3]}
///   After:
///     %s = onnx.Constant {value = dense<[0, 0, 0, 3]> : tensor<4xi64>}
///     %e = onnx.Constant ...
///     %a = onnx.Constant ...
///     %out = onnx.Slice(%data, %s, %e, %a)
struct SliceLegacyAttrsToOperands : public mlir::RewritePattern {
  SliceLegacyAttrsToOperands(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Slice", /*benefit=*/2, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    // Only the legacy schema has a lone `data`; the form this pattern emits
    // carries four operands, so it cannot match its own output.
    if (op->getNumOperands() != 1 || op->getNumResults() != 1)
      return rewriter.notifyMatchFailure(op, "slice.not_legacy_arity");

    auto readInts = [&](llvm::StringRef name,
                        llvm::SmallVectorImpl<int64_t> &out) -> bool {
      auto attr = op->getAttrOfType<mlir::ArrayAttr>(name);
      if (!attr)
        return false;
      for (mlir::Attribute entry : attr) {
        auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(entry);
        if (!intAttr)
          return false;
        out.push_back(intAttr.getValue().getSExtValue());
      }
      return true;
    };

    llvm::SmallVector<int64_t> startsVec, endsVec, axesVec;
    if (!readInts("starts", startsVec) || !readInts("ends", endsVec))
      return rewriter.notifyMatchFailure(op, "slice.legacy_attrs_missing");
    if (startsVec.empty() || startsVec.size() != endsVec.size())
      return rewriter.notifyMatchFailure(op,
                                         "slice.legacy_starts_ends_mismatch");

    // `axes` is optional, defaulting to the leading dims. Spell the default out
    // instead of leaving the operand off: SliceDecompose derives an absent
    // `axes` from the data RANK, and a legacy op may name fewer axes than the
    // data has dims, which would then mismatch starts/ends and fall through.
    if (!readInts("axes", axesVec)) {
      axesVec.clear();
      for (int64_t i :
           llvm::seq<int64_t>(static_cast<int64_t>(startsVec.size())))
        axesVec.push_back(i);
    }
    if (axesVec.size() != startsVec.size())
      return rewriter.notifyMatchFailure(op, "slice.legacy_axes_mismatch");

    mlir::Location loc = op->getLoc();
    auto i64Type = rewriter.getI64Type();
    auto materialize = [&](llvm::ArrayRef<int64_t> vals) -> mlir::Value {
      auto type = mlir::RankedTensorType::get(
          {static_cast<int64_t>(vals.size())}, i64Type);
      mlir::OperationState state(loc, "onnx.Constant");
      state.addTypes(type);
      state.addAttribute("value", mlir::DenseElementsAttr::get(type, vals));
      return rewriter.create(state)->getResult(0);
    };

    mlir::OperationState state(loc, "onnx.Slice");
    state.addOperands({op->getOperand(0), materialize(startsVec),
                       materialize(endsVec), materialize(axesVec)});
    state.addTypes(op->getResult(0).getType());
    rewriter.replaceOp(op, rewriter.create(state)->getResult(0));

    LLVM_DEBUG(llvm::dbgs()
               << "[" DEBUG_TYPE << "] rewrote attribute-form Slice ("
               << startsVec.size() << " axes) to the operand form\n");
    ++NumSliceLegacyRewrites;
    return mlir::success();
  }
};

struct SliceStampConstParams : public mlir::RewritePattern {
  SliceStampConstParams(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Slice", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    if (op->hasAttr("hipdnn.slice_starts"))
      return rewriter.notifyMatchFailure(op, "slice.already_stamped");

    if (op->getNumOperands() < 3 || op->getNumOperands() > 5)
      return rewriter.notifyMatchFailure(op, "slice.arity");

    auto startsVec = getInlineIntVector(op->getOperand(1));
    if (!startsVec)
      return rewriter.notifyMatchFailure(op, "slice.starts_not_inline_const");
    auto endsVec = getInlineIntVector(op->getOperand(2));
    if (!endsVec)
      return rewriter.notifyMatchFailure(op, "slice.ends_not_inline_const");

    std::optional<llvm::SmallVector<int64_t>> axesVec;
    if (op->getNumOperands() >= 4) {
      mlir::Value axes = normaliseOptional(op->getOperand(3));
      if (axes) {
        axesVec = getInlineIntVector(axes);
        if (!axesVec)
          return rewriter.notifyMatchFailure(op, "slice.axes_not_inline_const");
      }
    }

    std::optional<llvm::SmallVector<int64_t>> stepsVec;
    if (op->getNumOperands() == 5) {
      mlir::Value steps = normaliseOptional(op->getOperand(4));
      if (steps) {
        stepsVec = getInlineIntVector(steps);
        if (!stepsVec)
          return rewriter.notifyMatchFailure(op,
                                             "slice.steps_not_inline_const");
      }
    }

    rewriter.modifyOpInPlace(op, [&] {
      op->setAttr("hipdnn.slice_starts",
                  rewriter.getDenseI64ArrayAttr(*startsVec));
      op->setAttr("hipdnn.slice_ends", rewriter.getDenseI64ArrayAttr(*endsVec));
      if (axesVec)
        op->setAttr("hipdnn.slice_axes",
                    rewriter.getDenseI64ArrayAttr(*axesVec));
      if (stepsVec)
        op->setAttr("hipdnn.slice_steps",
                    rewriter.getDenseI64ArrayAttr(*stepsVec));
    });

    LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE << "] stamped slice params ("
                            << startsVec->size() << " entries)\n");
    ++NumSliceConstStamps;
    return mlir::success();
  }
};

} // namespace

void populateSliceShapeFoldPatterns(mlir::RewritePatternSet &patterns,
                                    MLIRContext *ctx) {
  patterns.add<SliceLegacyAttrsToOperands, SliceStampConstParams>(ctx);
}

} // namespace hip
} // namespace mlir
