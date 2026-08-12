/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- TransposeConversion.cpp - Convert onnx.Transpose to hipsr.transpose =//
//
// onnx.Transpose permutes axes, which hipsr.transpose models directly. ONNX
// leaves `perm` optional and defaults it to the reverse permutation; the hipsr
// operation always carries it, so the conversion materializes the default. The
// placeholder's shape region is left empty for hipsr-populate-shape-region, as
// for every DPS operation.
//
//===----------------------------------------------------------------------===//

#include "OnnxToHipsrUtils.h"

#include "hip/Conversion/OnnxToHipsr/OnnxToHipsr.h"
#include "hip/Dialect/Hipsr/IR/HipsrOps.h"

#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>

namespace mlir {
namespace hipsr {
namespace {

// Reads ONNX Transpose's optional `perm`, defaulting to the reverse
// permutation. Returns nullopt when the attribute is present but is not a list
// of integers of the input's rank.
std::optional<::llvm::SmallVector<int64_t>> resolvePerm(::mlir::Operation *op,
                                                        int64_t rank) {
  auto permAttr = op->getAttrOfType<::mlir::ArrayAttr>("perm");
  if (!permAttr) {
    return ::llvm::to_vector(::llvm::reverse(::llvm::seq<int64_t>(0, rank)));
  }
  if (static_cast<int64_t>(permAttr.size()) != rank) {
    return std::nullopt;
  }
  ::llvm::SmallVector<int64_t> perm;
  perm.reserve(rank);
  for (::mlir::Attribute entry : permAttr) {
    auto intAttr = ::mlir::dyn_cast<::mlir::IntegerAttr>(entry);
    if (!intAttr) {
      return std::nullopt;
    }
    perm.push_back(intAttr.getValue().getSExtValue());
  }
  return perm;
}

struct TransposeToHipsr : public ::mlir::RewritePattern {
  TransposeToHipsr(::mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Transpose", /*benefit=*/1, ctx) {}

  ::mlir::LogicalResult
  matchAndRewrite(::mlir::Operation *op,
                  ::mlir::PatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 1 || op->getNumResults() != 1) {
      return rewriter.notifyMatchFailure(
          op, "expected one operand and a single result");
    }

    ::mlir::Value input = op->getOperand(0);
    auto inputType =
        ::mlir::dyn_cast<::mlir::RankedTensorType>(input.getType());
    auto resultType =
        ::mlir::dyn_cast<::mlir::RankedTensorType>(op->getResult(0).getType());
    if (!inputType || !resultType) {
      return rewriter.notifyMatchFailure(op, "expected ranked tensor types");
    }
    if (inputType.getElementType() != resultType.getElementType()) {
      return rewriter.notifyMatchFailure(
          op, "expected matching input and result element types");
    }
    if (inputType.getRank() != resultType.getRank()) {
      return rewriter.notifyMatchFailure(
          op, "expected the result rank to equal the input rank");
    }

    std::optional<::llvm::SmallVector<int64_t>> perm =
        resolvePerm(op, inputType.getRank());
    if (!perm || !::mlir::isPermutationVector(*perm)) {
      return rewriter.notifyMatchFailure(
          op, "expected perm to be a permutation of [0, rank)");
    }

    ::mlir::FailureOr<::mlir::Value> ctx = getHipsrContextArg(op, rewriter);
    if (::mlir::failed(ctx)) {
      return ::mlir::failure();
    }

    ::mlir::Location loc = op->getLoc();
    ::mlir::Value init = PlaceholderOp::create(
                             rewriter, loc, ::mlir::TypeRange{resultType}, *ctx,
                             ::mlir::ValueRange{input}, PlaceholderType::Normal)
                             .getResult(0);
    auto transposeOp =
        TransposeOp::create(rewriter, loc, ::mlir::TypeRange{resultType}, *ctx,
                            input, init, rewriter.getDenseI64ArrayAttr(*perm));
    rewriter.replaceOp(op, transposeOp.getResult(0));
    return ::mlir::success();
  }
};

} // namespace

void populateTransposeConversionPatterns(::mlir::RewritePatternSet &patterns,
                                         ::mlir::MLIRContext *ctx) {
  patterns.add<TransposeToHipsr>(ctx);
}

} // namespace hipsr
} // namespace mlir
