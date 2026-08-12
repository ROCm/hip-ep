/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- ScatterNDConversion.cpp - onnx.ScatterND -> hipsr.scatter_nd ------===//
//
// onnx.ScatterND overwrites the slices its indices address, which
// hipsr.scatter_nd models directly. ONNX also reduces a duplicate index into
// its destination through `reduction`; that is a read-modify-write rather than
// an overwrite, so this rejects the four reducing modes. The placeholder's
// shape region is left empty for hipsr-populate-shape-region, as for every DPS
// operation.
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

struct ScatterNDToHipsr : public ::mlir::RewritePattern {
  ScatterNDToHipsr(::mlir::MLIRContext *ctx)
      : RewritePattern("onnx.ScatterND", /*benefit=*/1, ctx) {}

  ::mlir::LogicalResult
  matchAndRewrite(::mlir::Operation *op,
                  ::mlir::PatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 3 || op->getNumResults() != 1) {
      return rewriter.notifyMatchFailure(
          op, "expected three operands and a single result");
    }
    if (auto reduction = op->getAttrOfType<::mlir::StringAttr>("reduction")) {
      if (reduction.getValue() != "none") {
        return rewriter.notifyMatchFailure(op, "expected reduction = none");
      }
    }

    ::mlir::Value data = op->getOperand(0);
    ::mlir::Value indices = op->getOperand(1);
    ::mlir::Value updates = op->getOperand(2);
    auto dataType = ::mlir::dyn_cast<::mlir::RankedTensorType>(data.getType());
    auto indicesType =
        ::mlir::dyn_cast<::mlir::RankedTensorType>(indices.getType());
    auto updatesType =
        ::mlir::dyn_cast<::mlir::RankedTensorType>(updates.getType());
    auto resultType =
        ::mlir::dyn_cast<::mlir::RankedTensorType>(op->getResult(0).getType());
    if (!dataType || !indicesType || !updatesType || !resultType) {
      return rewriter.notifyMatchFailure(op, "expected ranked tensor types");
    }
    if (!indicesType.getElementType().isIntOrIndex()) {
      return rewriter.notifyMatchFailure(op, "expected integer indices");
    }
    if (dataType.getElementType() != updatesType.getElementType() ||
        dataType.getElementType() != resultType.getElementType()) {
      return rewriter.notifyMatchFailure(
          op, "expected matching data, updates and result element types");
    }

    ::mlir::FailureOr<::mlir::Value> ctx = getHipsrContextArg(op, rewriter);
    if (::mlir::failed(ctx)) {
      return ::mlir::failure();
    }

    // The output takes the data's shape, so the data alone drives the
    // placeholder.
    ::mlir::Location loc = op->getLoc();
    ::mlir::Value init = PlaceholderOp::create(
                             rewriter, loc, ::mlir::TypeRange{resultType}, *ctx,
                             ::mlir::ValueRange{data}, PlaceholderType::Normal)
                             .getResult(0);
    auto scatterOp =
        ScatterNDOp::create(rewriter, loc, ::mlir::TypeRange{resultType}, *ctx,
                            data, indices, updates, init);
    rewriter.replaceOp(op, scatterOp.getResult(0));
    return ::mlir::success();
  }
};

} // namespace

void populateScatterNDConversionPatterns(::mlir::RewritePatternSet &patterns,
                                         ::mlir::MLIRContext *ctx) {
  patterns.add<ScatterNDToHipsr>(ctx);
}

} // namespace hipsr
} // namespace mlir
