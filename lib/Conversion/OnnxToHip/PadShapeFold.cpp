/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- PadShapeFold.cpp - Pre-lowering Pad const-pads stamping -----------===//
//
// Sibling pre-lowering fold to GatherShapeFold / ReshapeShapeFold. Captures
// the compile-time value of `onnx.Pad`'s `pads` (and optional `axes`) operand
// onto the op as attributes BEFORE `lowerOnnxConstants` externalizes the
// constant -- so the later PadConversion (which runs in `convertComputeOps`,
// AFTER externalization) can compute the dynamic output shape from those
// attributes without reading the operand.
//
// Why this matters
// ----------------
// `onnx.Pad` with a dynamic output shape needs the per-axis pad amounts on the
// HOST to size the output buffer (out_dim[i] = data_dim[i] + begin + end).
// `pads` is almost always a compile-time constant in the ONNX model, but it is
// frequently large enough (or a model-wide initializer) that
// `lowerOnnxConstants` externalizes it: the `onnx.Constant` becomes a
// `memref.global` with a NULL initial_value (bytes live in `constants.bin`).
// By the time PadConversion runs, the inline value is gone, so a fold attempt
// fails and the fallback emits a synchronized `hip.readback_scalar` (D2H) per
// pad entry -- runtime device traffic for a value that is known at compile
// time. This pre-lowering fold runs while the value is still inline and stamps
// it onto the op, eliminating the readback for the (overwhelmingly common)
// constant-`pads` case. Genuinely runtime-dynamic `pads` carry no attribute
// and still use the readback path in PadConversion (correctness preserved).
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
//   * Leaves the `onnx.Constant` ops in place; DCE / externalization handle
//     them as usual.
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
          "attributes before externalization");

namespace mlir {
namespace hip {

namespace {

/// Return the values of `v` as an int64 vector when `v` is an inline 1-D
/// integer constant (`arith.constant` or `onnx.Constant {value = dense<...>}`),
/// or std::nullopt otherwise. Mirrors the inline-constant recognition used by
/// the sibling shape folds; deliberately does NOT peek through
/// `bufferization.to_tensor(memref.get_global)` because this pattern runs
/// BEFORE externalization, where that form does not yet exist.
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
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PadStampConstShape)
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
    // PadConversion's axis->slot mapping does not have to read an (also
    // possibly externalized) `axes` constant. Absent/None axes => default
    // identity, handled by PadConversion without an attribute.
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
