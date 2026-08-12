/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- GatherConversion.cpp - Convert onnx.Gather to hipsr.gather --------===//
//
// onnx.Gather selects entries along one axis, which hipsr.gather models
// directly. ONNX lets `axis` count back from the end; the hipsr operation
// takes it normalized, so the conversion resolves it. The placeholder's shape
// region is left empty for hipsr-populate-shape-region, as for every DPS
// operation.
//
// Every gather converts, including the `Gather(Shape(x))` idiom that reads one
// extent. Its operands are plain tensors at this point, so nothing yet forces
// that chain onto the device; where such shape-only values live is decided by
// memory space during bufferization, as it is for the `onnx.Shape` result that
// feeds `hipsr.expand`.
//
//===----------------------------------------------------------------------===//

#include "OnnxToHipsrUtils.h"

#include "hip/Conversion/OnnxToHipsr/OnnxToHipsr.h"
#include "hip/Dialect/Hipsr/IR/HipsrOps.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace hipsr {
namespace {

struct GatherToHipsr : public ::mlir::RewritePattern {
  GatherToHipsr(::mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Gather", /*benefit=*/1, ctx) {}

  ::mlir::LogicalResult
  matchAndRewrite(::mlir::Operation *op,
                  ::mlir::PatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 2 || op->getNumResults() != 1) {
      return rewriter.notifyMatchFailure(
          op, "expected two operands and a single result");
    }

    ::mlir::Value data = op->getOperand(0);
    ::mlir::Value indices = op->getOperand(1);
    auto dataType = ::mlir::dyn_cast<::mlir::RankedTensorType>(data.getType());
    auto indicesType =
        ::mlir::dyn_cast<::mlir::RankedTensorType>(indices.getType());
    auto resultType =
        ::mlir::dyn_cast<::mlir::RankedTensorType>(op->getResult(0).getType());
    if (!dataType || !indicesType || !resultType) {
      return rewriter.notifyMatchFailure(op, "expected ranked tensor types");
    }
    if (dataType.getElementType() != resultType.getElementType()) {
      return rewriter.notifyMatchFailure(
          op, "expected matching data and result element types");
    }
    if (!indicesType.getElementType().isIntOrIndex()) {
      return rewriter.notifyMatchFailure(op, "expected integer indices");
    }

    int64_t rank = dataType.getRank();
    int64_t axis = 0;
    // getSInt asserts on a signedness this attribute may not carry, so read
    // the raw APInt.
    if (auto axisAttr = op->getAttrOfType<::mlir::IntegerAttr>("axis")) {
      axis = axisAttr.getValue().getSExtValue();
    }
    if (axis < 0) {
      axis += rank;
    }
    if (axis < 0 || axis >= rank) {
      return rewriter.notifyMatchFailure(op, "expected axis in [-rank, rank)");
    }
    if (resultType.getRank() != rank - 1 + indicesType.getRank()) {
      return rewriter.notifyMatchFailure(
          op, "result rank must be data rank - 1 plus indices rank");
    }

    ::mlir::FailureOr<::mlir::Value> ctx = getHipsrContextArg(op, rewriter);
    if (::mlir::failed(ctx)) {
      return ::mlir::failure();
    }

    ::mlir::Location loc = op->getLoc();
    ::mlir::Value init =
        PlaceholderOp::create(rewriter, loc, ::mlir::TypeRange{resultType},
                              *ctx, ::mlir::ValueRange{data, indices},
                              PlaceholderType::Normal)
            .getResult(0);
    auto gatherOp =
        GatherOp::create(rewriter, loc, ::mlir::TypeRange{resultType}, *ctx,
                         data, indices, init, rewriter.getI64IntegerAttr(axis));
    rewriter.replaceOp(op, gatherOp.getResult(0));
    return ::mlir::success();
  }
};

} // namespace

void populateGatherConversionPatterns(::mlir::RewritePatternSet &patterns,
                                      ::mlir::MLIRContext *ctx) {
  patterns.add<GatherToHipsr>(ctx);
}

} // namespace hipsr
} // namespace mlir
