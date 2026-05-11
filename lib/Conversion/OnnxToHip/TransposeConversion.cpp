//===- TransposeConversion.cpp - ONNX-to-HIP Transpose conversion - *- C++
//-*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.  All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

/// onnx.Transpose -> hip.transpose
struct TransposeToHip : public RewritePattern {
  TransposeToHip(MLIRContext* ctx)
      : RewritePattern("onnx.Transpose", /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation* op,
                                PatternRewriter& rewriter) const override;
};

LogicalResult TransposeToHip::matchAndRewrite(Operation* op,
                                              PatternRewriter& rewriter) const {
  auto ctxOrFailure = getContextArg(op, rewriter);
  if (failed(ctxOrFailure))
    return failure();
  Value context = *ctxOrFailure;

  Location loc = op->getLoc();
  Value data = op->getOperand(0);

  auto permAttr = op->getAttrOfType<ArrayAttr>("perm");
  if (!permAttr)
    return op->emitOpError("hip.transpose requires explicit perm attribute");

  int64_t dim0 = -1, dim1 = -1;
  int64_t mismatchCount = 0;
  for (auto [permIdx, attr] : llvm::enumerate(permAttr)) {
    int64_t p = cast<IntegerAttr>(attr).getValue().getSExtValue();
    if (p != static_cast<int64_t>(permIdx)) {
      ++mismatchCount;
      if (dim0 < 0)
        dim0 = static_cast<int64_t>(permIdx);
      else if (dim1 < 0)
        dim1 = static_cast<int64_t>(permIdx);
    }
  }
  if (mismatchCount != 2 || dim0 < 0 || dim1 < 0)
    return op->emitOpError("perm must swap exactly two dimensions");
  int64_t p0 = cast<IntegerAttr>(permAttr[dim0]).getValue().getSExtValue();
  int64_t p1 = cast<IntegerAttr>(permAttr[dim1]).getValue().getSExtValue();
  if (p0 != dim1 || p1 != dim0)
    return op->emitOpError("perm must swap exactly two dimensions");

  auto resultType = cast<RankedTensorType>(op->getResult(0).getType());

  // Transpose: output dim i corresponds to input dim perm[i].
  llvm::SmallVector<Value> dynSizes;
  for (auto [outDimIdx, attr] : llvm::enumerate(permAttr)) {
    if (resultType.isDynamicDim(outDimIdx)) {
      const int64_t srcDim = cast<IntegerAttr>(attr).getValue().getSExtValue();
      dynSizes.push_back(tensor::DimOp::create(rewriter, loc, data, srcDim));
    }
  }

  Value init = tensor::EmptyOp::create(rewriter, loc, resultType.getShape(),
                                       resultType.getElementType(), dynSizes);

  Value d0 = arith::ConstantIndexOp::create(rewriter, loc, dim0);
  Value d1 = arith::ConstantIndexOp::create(rewriter, loc, dim1);

  auto hipOp = mlir::hip::TransposeOp::create(rewriter, loc, resultType,
                                              context, d0, d1, data, init);
  rewriter.replaceOp(op, hipOp->getResult(0));
  return success();
}

} // namespace

void mlir::hip::populateTransposeConversionPatterns(RewritePatternSet& patterns,
                                                    MLIRContext* ctx) {
  patterns.add<TransposeToHip>(ctx);
}

} // namespace hip
} // namespace mlir
