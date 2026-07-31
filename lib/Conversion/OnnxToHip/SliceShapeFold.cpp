/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- SliceShapeFold.cpp - Pre-lowering Slice param stamping ------------===//
//
// Sibling pre-lowering fold to PadShapeFold. Captures the compile-time value
// of `onnx.Slice`'s starts/ends/axes/steps operands onto the op as attributes
// BEFORE `lowerOnnxConstants` externalizes the constants -- so SliceDecompose
// (which runs in `convertComputeOps`, after externalization) can rewrite to
// `tensor.extract_slice` without reading the operand.
//
// SwinV2 window/partition slices use small 1-element i64 constants for every
// param; production builds externalize even those into `memref.global` entries
// with null `initial_value` (bytes live in constants.bin / ORT mem). Without
// this stamp SliceDecompose fails `extractIntVector` and every slice falls
// through to the stub `hip.slice` runtime op.
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
          "attributes before externalization");

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
/// std::nullopt. Runs before externalization so `bufferization.to_tensor` is
/// intentionally not handled here.
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
          return rewriter.notifyMatchFailure(op, "slice.steps_not_inline_const");
      }
    }

    rewriter.modifyOpInPlace(op, [&] {
      op->setAttr("hipdnn.slice_starts",
                  rewriter.getDenseI64ArrayAttr(*startsVec));
      op->setAttr("hipdnn.slice_ends",
                  rewriter.getDenseI64ArrayAttr(*endsVec));
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
  patterns.add<SliceStampConstParams>(ctx);
}

} // namespace hip
} // namespace mlir
