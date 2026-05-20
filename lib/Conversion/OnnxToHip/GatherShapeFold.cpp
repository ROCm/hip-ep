/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- GatherShapeFold.cpp - Pre-lowering Gather(Shape) folding ----------===//
//
// Pre-lowering walk that recognizes the common transformer-shape-arithmetic
// idiom
//
//   %shape = onnx.Shape(%x)                  : tensor<Nxi64>
//   %idx   = onnx.Constant{value=dense<[k]>} : tensor<...xi64>
//   %res   = onnx.Gather(%shape, %idx)       : tensor<...xi64>
//
// and folds it to
//
//   %dim   = tensor.dim %x, %k               : index
//   %dim64 = arith.index_cast %dim           : index to i64
//   %res   = tensor.from_elements %dim64     : tensor<...xi64>
//
// Why this matters (gfx1151 dynseqlen regression):
//   Pre-dynseqlen, all shapes were static and this whole chain folded to a
//   compile-time constant that GenerateInterface placed in constants.bin
//   (already on GPU at init -- no per-step host store).  Dynseqlen broke the
//   constant fold, so the same arithmetic now happens at runtime.  The
//   bufferized form of the full Shape result is `memref<Nxi64>` written by N
//   host stores, then absorbed by PoolAllocs into the GPU pool.  On gfx1151
//   the pool is real device memory and the host stores SEGV.
//
// Folding here shrinks `memref<Nxi64>` to `memref<i64>` (one host store)
// AND localizes the boundary, so the late hip.scalar_to_gpu_buf rewrite can
// match the residual `memref.alloc + single store + GPU consumer` pattern
// and replace it with an explicit H2D copy.
//
// Runs BEFORE lowerOnnxConstants so the index value is still inline in the
// onnx.Constant `value` attribute (after externalization it would be hidden
// behind memref.get_global pointing into the constants blob).
//
// Non-goals
// ---------
//   - Folding Gather over arbitrary tensors: this fold matches ONLY when
//     the Gather source is `onnx.Shape`.  Generic Gather lowerings live
//     elsewhere and run later in the pipeline.
//   - Folding Gather with non-constant indices: `getInlineScalarIndex`
//     requires an `onnx.Constant` operand whose `value` attribute is a
//     single i32/i64 element.  Anything else survives unchanged so the
//     general Gather conversion can handle it.
//   - Multi-element Gather results: the result type must be 0-D (scalar
//     Gather) or 1-D with a single element (1xi64).  Anything wider is
//     left to the general Gather lowering -- folding it here would
//     require materializing a multi-dim `tensor.from_elements`, which is
//     exactly what `ShapeConversion.cpp` is for.
//   - Constant-folding the gathered Shape itself: that responsibility
//     lives in `ShapeConversion.cpp`.  The two passes are intentionally
//     orthogonal -- this fold collapses Shape -> Gather chains;
//     ShapeConversion handles the standalone Shape op when the chain
//     doesn't end in Gather.
//
//===----------------------------------------------------------------------===//

#include "OnnxToHipUtils.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir {
namespace hip {

namespace {

/// Return the single integer value of a 0-D or single-element ONNX
/// constant, or std::nullopt if the op is not eligible.
static std::optional<int64_t> getInlineScalarIndex(mlir::Operation *constOp) {
  if (!constOp || constOp->getName().getStringRef() != "onnx.Constant")
    return std::nullopt;
  auto valueAttr = constOp->getAttrOfType<mlir::DenseElementsAttr>("value");
  if (!valueAttr)
    return std::nullopt;
  if (valueAttr.getNumElements() != 1)
    return std::nullopt;
  auto elemType = valueAttr.getElementType();
  if (!elemType.isInteger(32) && !elemType.isInteger(64))
    return std::nullopt;
  return (*valueAttr.getValues<mlir::APInt>().begin()).getSExtValue();
}

} // namespace

mlir::LogicalResult foldGatherShapeBeforeLowering(mlir::func::FuncOp funcOp) {
  // Collect Gather ops first; rewriting in place during walk is unsafe.
  llvm::SmallVector<mlir::Operation *> gathers;
  funcOp.walk([&](mlir::Operation *op) {
    if (op->getName().getStringRef() == "onnx.Gather")
      gathers.push_back(op);
  });

  for (mlir::Operation *gatherOp : gathers) {
    if (gatherOp->getNumOperands() != 2)
      continue;
    mlir::Operation *shapeOp = gatherOp->getOperand(0).getDefiningOp();
    mlir::Operation *constOp = gatherOp->getOperand(1).getDefiningOp();
    if (!shapeOp || shapeOp->getName().getStringRef() != "onnx.Shape")
      continue;

    auto idxOpt = getInlineScalarIndex(constOp);
    if (!idxOpt)
      continue;

    // ONNX Gather has `axis` attribute; for Shape (1-D) only axis 0 is
    // meaningful. Older exporters omit the attribute (default 0).
    int64_t axis = 0;
    if (auto axisAttr = gatherOp->getAttrOfType<mlir::IntegerAttr>("axis"))
      axis = axisAttr.getSInt();
    if (axis != 0)
      continue;

    mlir::Value shapeInput = shapeOp->getOperand(0);
    auto inputType =
        mlir::dyn_cast<mlir::RankedTensorType>(shapeInput.getType());
    if (!inputType)
      continue;
    int64_t rank = inputType.getRank();

    // Normalize Shape's start/end per ONNX Shape-15 spec.  The Gather index
    // k is an index INTO the 1-D Shape result of length (end - start), not
    // into the input's full rank — so the negative-index normalization on k
    // must use the SUB-RANGE length, not the full rank, per ONNX Gather
    // semantics on a 1-D tensor of length L (k += L if k < 0).
    int64_t start = 0;
    int64_t end = rank;
    if (auto startAttr = shapeOp->getAttrOfType<mlir::IntegerAttr>("start"))
      start = startAttr.getSInt();
    if (auto endAttr = shapeOp->getAttrOfType<mlir::IntegerAttr>("end"))
      end = endAttr.getSInt();
    if (start < 0)
      start += rank;
    if (end < 0)
      end += rank;
    start = std::max(start, int64_t(0));
    end = std::min(end, rank);
    int64_t rangeLen = std::max(end - start, int64_t(0));
    if (rangeLen == 0)
      continue;

    int64_t k = *idxOpt;
    if (k < 0)
      k += rangeLen; // ONNX Gather: normalize against TARGET tensor length.
    if (k < 0 || k >= rangeLen)
      continue;
    int64_t absDim = start + k;

    auto resultType = mlir::dyn_cast<mlir::RankedTensorType>(
        gatherOp->getResult(0).getType());
    if (!resultType)
      continue;
    if (!resultType.getElementType().isInteger(64))
      continue;
    // Result must be 0-D (scalar Gather) or 1-D with one element
    // (1-element-tensor Gather). Anything else is not the pattern we fold.
    if (resultType.getRank() == 0) {
      // OK, scalar.
    } else if (resultType.getRank() == 1 && resultType.getDimSize(0) == 1) {
      // OK, 1xi64.
    } else {
      continue;
    }

    mlir::Location loc = gatherOp->getLoc();
    mlir::OpBuilder builder(gatherOp);
    auto i64Type = builder.getI64Type();

    mlir::Value dimI64;
    if (inputType.isDynamicDim(absDim)) {
      mlir::Value dimVal =
          mlir::tensor::DimOp::create(builder, loc, shapeInput, absDim);
      dimI64 = mlir::arith::IndexCastOp::create(builder, loc, i64Type, dimVal);
    } else {
      dimI64 = mlir::arith::ConstantOp::create(
          builder, loc,
          builder.getI64IntegerAttr(inputType.getDimSize(absDim)));
    }

    mlir::Value newResult = mlir::tensor::FromElementsOp::create(
        builder, loc, resultType, mlir::ValueRange{dimI64});
    gatherOp->getResult(0).replaceAllUsesWith(newResult);
    gatherOp->erase();
    // Leave shapeOp/constOp in place; they may have other uses, and a later
    // canonicalization pass / DCE will remove them if not.
  }
  return mlir::success();
}

} // namespace hip
} // namespace mlir
