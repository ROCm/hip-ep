/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- TriluConversion.cpp - onnx.Trilu -> hip.trilu ---------------------===//
//
// Converts onnx.Trilu to hip.trilu. The diagonal offset k must be a constant
// scalar (or omitted, which means 0). upper defaults to 1. A splat input is
// passed as a rank-0 tensor that the kernel broadcasts.
//
//===----------------------------------------------------------------------===//

#include "OnnxToHipUtils.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

namespace mlir {
namespace hip {
namespace {

mlir::DenseElementsAttr constantTensor(mlir::Value value) {
  mlir::Operation *defOp = value.getDefiningOp();
  if (!defOp)
    return nullptr;
  if (auto cst = mlir::dyn_cast<mlir::arith::ConstantOp>(defOp))
    return mlir::dyn_cast<mlir::DenseElementsAttr>(cst.getValue());
  if (auto attr = defOp->getAttr("value"))
    if (auto dense = mlir::dyn_cast<mlir::DenseElementsAttr>(attr))
      return dense;
  return nullptr;
}

mlir::FailureOr<int64_t> diagonalOffset(mlir::Value k) {
  if (!k || mlir::isa<mlir::NoneType>(k.getType()))
    return 0;
  mlir::DenseElementsAttr dense = constantTensor(k);
  if (!dense || dense.getNumElements() != 1)
    return mlir::failure();
  auto elemType = dense.getElementType();
  if (!mlir::isa<mlir::IntegerType>(elemType))
    return mlir::failure();
  return (*dense.value_begin<llvm::APInt>()).getSExtValue();
}

struct TriluToHip : public mlir::RewritePattern {
  TriluToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Trilu", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    if (op->getNumResults() != 1 || op->getNumOperands() < 1 ||
        op->getNumOperands() > 2)
      return rewriter.notifyMatchFailure(
          op, "onnx.Trilu expects one result and an optional k");

    mlir::Value input = op->getOperand(0);
    auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(input.getType());
    if (!inputType || inputType.getRank() < 2)
      return rewriter.notifyMatchFailure(op, "ranked tensor of rank >= 2");
    if (!mlir::isa<mlir::FloatType>(inputType.getElementType()))
      return rewriter.notifyMatchFailure(op, "float element type required");

    mlir::Value kValue =
        op->getNumOperands() == 2 ? op->getOperand(1) : nullptr;
    auto diagonal = diagonalOffset(kValue);
    if (mlir::failed(diagonal))
      return rewriter.notifyMatchFailure(op, "k must be a constant scalar");

    int64_t upper = 1;
    if (auto attr = op->getAttrOfType<mlir::IntegerAttr>("upper"))
      upper = attr.getValue().getSExtValue() != 0 ? 1 : 0;

    auto ctxOrFailure = getContextArg(op, rewriter);
    if (mlir::failed(ctxOrFailure))
      return mlir::failure();

    auto resultType =
        mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
    if (!resultType ||
        resultType.getElementType() != inputType.getElementType())
      return rewriter.notifyMatchFailure(op, "result type must match input");

    // ConstantOfShape lowers to tensor.splat; take the splat path below once
    // it has.
    if (mlir::Operation *defOp = input.getDefiningOp())
      if (defOp->getName().getStringRef() == "onnx.ConstantOfShape")
        return rewriter.notifyMatchFailure(
            op, "waiting for ConstantOfShape lowering");

    mlir::Location loc = op->getLoc();
    mlir::Value init;
    // A dynamic tensor.splat bufferizes to host loops that store into device
    // memory. Pass its scalar as a rank-0 tensor.empty + linalg.fill (a rank-0
    // splat would fold to a host memref.global) and let the kernel broadcast.
    auto splat = input.getDefiningOp<mlir::tensor::SplatOp>();
    if (splat && splat.getType() == resultType) {
      init = mlir::tensor::EmptyOp::create(rewriter, loc, resultType,
                                           splat.getDynamicSizes());
      auto scalarType =
          mlir::RankedTensorType::get({}, inputType.getElementType());
      mlir::Value empty = mlir::tensor::EmptyOp::create(
          rewriter, loc, scalarType, mlir::ValueRange{});
      input = mlir::linalg::FillOp::create(rewriter, loc,
                                           mlir::ValueRange{splat.getInput()},
                                           mlir::ValueRange{empty})
                  .getResult(0);
    } else {
      init = createEmptyTensor(rewriter, loc, resultType, input);
    }
    // SI64Attr is a signed 64-bit integer. getI64IntegerAttr is signless and
    // fails the hip.trilu attribute constraint.
    auto si64 = rewriter.getIntegerType(64, /*isSigned=*/true);
    auto hipOp = mlir::hip::TriluOp::create(
        rewriter, loc, resultType, *ctxOrFailure, input, init,
        rewriter.getIntegerAttr(si64, *diagonal),
        rewriter.getIntegerAttr(si64, upper));
    rewriter.replaceOp(op, hipOp->getResult(0));
    return mlir::success();
  }
};

} // namespace

void populateTriluConversionPatterns(RewritePatternSet &patterns,
                                     MLIRContext *ctx) {
  patterns.add<TriluToHip>(ctx);
}

} // namespace hip
} // namespace mlir
