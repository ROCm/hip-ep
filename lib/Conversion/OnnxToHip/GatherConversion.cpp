//===- GatherConversion.cpp - ONNX-to-HIP Gather conversion --- *- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.  All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
//
// Why this conversion exists
// --------------------------
// `onnx.Gather` selects rows along an arbitrary axis using an index
// tensor.  In LLM workloads the dominant pattern is the embedding
// table lookup at the start of each step (axis = 0, indices are token
// IDs); the same op also shows up during decode when slicing past-key /
// past-value caches.  Both feed `hip.gather` so the runtime can pick the
// right kernel based on operand shape -- a single rewrite covers the
// whole spectrum.
//
// Non-obvious choices
// -------------------
// * `axis` is canonicalized to a non-negative value here; the runtime
//   wrapper expects it that way and we don't want every kernel to repeat
//   the wrap-around math.
// * Negative indices follow the ONNX convention (`-1` means last row).
//   We do not normalize them at conversion time because doing so would
//   require materializing an `arith.addi` per element -- the runtime
//   handles negatives directly via a single mod operation on the GPU.
// * Output shape is computed as `data.shape[:axis] + indices.shape +
//   data.shape[axis+1:]`, matching the ONNX semantics exactly.
//
//===----------------------------------------------------------------------===//

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

//===----------------------------------------------------------------------===//
// ONNX Gather -> HIP Gather
//===----------------------------------------------------------------------===//

struct GatherToHip : public RewritePattern {
  GatherToHip(MLIRContext* ctx)
      : RewritePattern("onnx.Gather", /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation* op,
                                PatternRewriter& rewriter) const override {
    auto ctxOrFailure = getContextArg(op, rewriter);
    if (failed(ctxOrFailure))
      return rewriter.notifyMatchFailure(op, "missing context argument");
    Value context = *ctxOrFailure;

    Location loc = op->getLoc();
    Value data = op->getOperand(0);
    Value indices = op->getOperand(1);

    // Get axis attribute from ONNX Gather operation
    int64_t axis = op->getAttrOfType<IntegerAttr>("axis").getSInt();
    auto axisAttr = rewriter.getI64IntegerAttr(axis);

    // Get result type
    auto resultType = cast<RankedTensorType>(op->getResult(0).getType());
    auto dataType = cast<RankedTensorType>(data.getType());
    auto indicesType = cast<RankedTensorType>(indices.getType());

    // Normalize negative axis for dimension calculations only
    int64_t normalizedAxis = axis < 0 ? axis + dataType.getRank() : axis;

    // Create output tensor with dynamic shape support
    // Output shape: [data[0:axis], indices.shape, data[axis+1:]]
    llvm::SmallVector<Value> dynSizes;
    int64_t outDimIdx = 0;

    // Copy dimensions before axis from data
    for (auto i : llvm::seq<int64_t>(0, normalizedAxis)) {
      if (outDimIdx < resultType.getRank() &&
          resultType.isDynamicDim(outDimIdx))
        dynSizes.push_back(tensor::DimOp::create(rewriter, loc, data, i));
      outDimIdx++;
    }
    // Copy all dimensions from indices
    for (auto i : llvm::seq<int64_t>(0, indicesType.getRank())) {
      if (outDimIdx < resultType.getRank() &&
          resultType.isDynamicDim(outDimIdx))
        dynSizes.push_back(tensor::DimOp::create(rewriter, loc, indices, i));
      outDimIdx++;
    }
    // Copy dimensions after axis from data
    for (auto i : llvm::seq<int64_t>(normalizedAxis + 1, dataType.getRank())) {
      if (outDimIdx < resultType.getRank() &&
          resultType.isDynamicDim(outDimIdx))
        dynSizes.push_back(tensor::DimOp::create(rewriter, loc, data, i));
      outDimIdx++;
    }

    Value init = tensor::EmptyOp::create(rewriter, loc, resultType.getShape(),
                                         resultType.getElementType(), dynSizes);

    // Create hip.gather operation
    auto gatherOp = mlir::hip::GatherOp::create(
        rewriter, loc, resultType, context, data, indices, init, axisAttr);

    rewriter.replaceOp(op, gatherOp->getResult(0));
    return success();
  }
};

} // namespace

void mlir::hip::populateGatherConversionPatterns(RewritePatternSet& patterns,
                                                 MLIRContext* ctx) {
  patterns.add<GatherToHip>(ctx);
}

} // namespace hip
} // namespace mlir
