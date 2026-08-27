/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- PadShapeFold.cpp - Pre-lowering Pad const-pads stamping -----------===//
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
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Debug.h"

#include <optional>

#define DEBUG_TYPE "pad-shape-fold"

STATISTIC(NumPadConstStamps,
          "Number of onnx.Pad ops whose constant pads/axes were stamped as "
          "attributes before carrier lowering");

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
  patterns.add<PadStampConstShape>(ctx);
}

} // namespace hip
} // namespace mlir
