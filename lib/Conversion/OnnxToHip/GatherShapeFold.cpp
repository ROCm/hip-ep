/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- GatherShapeFold.cpp - Pre-lowering Gather(Shape) folding ----------===//
//
// Pre-lowering pattern that recognizes the common transformer-shape-arithmetic
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
// Why this matters (dynamic-sequence-length regression):
//   With fully static shapes, this whole chain folds to a compile-time
//   constant that GenerateInterface places in constants.bin (already on GPU
//   at init -- no per-step host store).  Dynamic sequence length breaks the
//   constant fold, so the same arithmetic now happens at runtime.  The
//   bufferized form of the full Shape result is `memref<Nxi64>` written by N
//   host stores, then absorbed by PoolAllocs into the GPU pool.  On targets
//   where the pool is real device memory, the host stores SEGV.
//
// Folding here shrinks `memref<Nxi64>` to `memref<i64>` (one host store)
// AND localizes the boundary, so the late `--hip-materialize-host-scalars`
// pass (lib/Dialect/Transforms/MaterializeHostScalars.cpp) can match the
// residual `memref.alloc + host store + GPU consumer` pattern and divert
// the alloc out of the GPU pool into a `hipHostMalloc(hipHostMallocMapped)`
// scratch buffer (`hip.get_host_scratch` view) that is host-writable AND
// GPU-readable on UMA — avoiding the host-store-into-device-memory SEGV.
//
// Implemented as a RewritePattern rooted on `onnx.Gather` and run BEFORE
// `lowerOnnxConstants` so the index value is still inline in the
// `onnx.Constant` `value` attribute (after externalization it would be
// hidden behind `memref.get_global` pointing into the constants blob).
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
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "gather-shape-fold"

STATISTIC(NumGatherShapeFolds,
          "Number of Gather(Shape(x), const_idx) idioms folded to "
          "tensor.from_elements(tensor.dim(x, k))");

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

struct GatherOfShapeToDim : public mlir::RewritePattern {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GatherOfShapeToDim)
  GatherOfShapeToDim(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Gather", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *gatherOp,
                  mlir::PatternRewriter &rewriter) const override {
    if (gatherOp->getNumOperands() != 2)
      return rewriter.notifyMatchFailure(gatherOp, "gather.arity");

    mlir::Operation *shapeOp = gatherOp->getOperand(0).getDefiningOp();
    mlir::Operation *constOp = gatherOp->getOperand(1).getDefiningOp();
    if (!shapeOp || shapeOp->getName().getStringRef() != "onnx.Shape")
      return rewriter.notifyMatchFailure(gatherOp, "data.not_shape");

    auto idxOpt = getInlineScalarIndex(constOp);
    if (!idxOpt)
      return rewriter.notifyMatchFailure(gatherOp, "index.not_inline_scalar");

    // ONNX Gather has `axis` attribute; for Shape (1-D) only axis 0 is
    // meaningful. Older exporters omit the attribute (default 0).
    int64_t axis = 0;
    if (auto axisAttr = gatherOp->getAttrOfType<mlir::IntegerAttr>("axis"))
      axis = axisAttr.getSInt();
    if (axis != 0)
      return rewriter.notifyMatchFailure(gatherOp, "gather.axis_nonzero");

    mlir::Value shapeInput = shapeOp->getOperand(0);
    auto inputType =
        mlir::dyn_cast<mlir::RankedTensorType>(shapeInput.getType());
    if (!inputType)
      return rewriter.notifyMatchFailure(gatherOp, "shape.input_unranked");
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
      return rewriter.notifyMatchFailure(gatherOp, "shape.empty_range");

    int64_t k = *idxOpt;
    if (k < 0)
      k += rangeLen; // ONNX Gather: normalize against TARGET tensor length.
    if (k < 0 || k >= rangeLen)
      return rewriter.notifyMatchFailure(gatherOp, "index.out_of_range");
    int64_t absDim = start + k;

    auto resultType = mlir::dyn_cast<mlir::RankedTensorType>(
        gatherOp->getResult(0).getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(gatherOp, "result.not_ranked");
    if (!resultType.getElementType().isInteger(64))
      return rewriter.notifyMatchFailure(gatherOp, "result.not_i64");
    // Result must be 0-D (scalar Gather) or 1-D with one element (1xi64).
    // Anything else is not the pattern we fold.
    bool resultShapeOK =
        (resultType.getRank() == 0) ||
        (resultType.getRank() == 1 && resultType.getDimSize(0) == 1);
    if (!resultShapeOK)
      return rewriter.notifyMatchFailure(gatherOp, "result.shape_unsupported");

    mlir::Location loc = gatherOp->getLoc();
    auto i64Type = rewriter.getI64Type();

    mlir::Value dimI64;
    if (inputType.isDynamicDim(absDim)) {
      mlir::Value dimVal =
          mlir::tensor::DimOp::create(rewriter, loc, shapeInput, absDim);
      dimI64 = mlir::arith::IndexCastOp::create(rewriter, loc, i64Type, dimVal);
    } else {
      dimI64 = mlir::arith::ConstantOp::create(
          rewriter, loc,
          rewriter.getI64IntegerAttr(inputType.getDimSize(absDim)));
    }

    mlir::Value newResult = mlir::tensor::FromElementsOp::create(
        rewriter, loc, resultType, mlir::ValueRange{dimI64});

    LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] folded " << gatherOp->getName()
                            << " on dim " << absDim << "\n");
    ++NumGatherShapeFolds;
    rewriter.replaceOp(gatherOp, newResult);
    // Leave shapeOp/constOp in place; they may have other uses, and a
    // later canonicalization pass / DCE will remove them if not.
    return mlir::success();
  }
};

} // namespace

void populateGatherShapeFoldPatterns(mlir::RewritePatternSet &patterns,
                                     mlir::MLIRContext *ctx) {
  patterns.add<GatherOfShapeToDim>(ctx);
}

} // namespace hip
} // namespace mlir
