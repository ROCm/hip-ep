/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- TransposeMatMulFold.cpp - Fold Transpose into MatMul --------------===//
//
// Pre-lowering pattern that recognizes the common attention idiom
//
//   %kt = onnx.Transpose(%k) {perm = [0, 1, ..., r-1, r, r-2]}
//   %scores = onnx.MatMul(%q, %kt)
//
// and folds it to
//
//   %scores = onnx.MatMul(%q, %k) {hipdnn.transB = 1}
//
// Only the last-two-dimension swap is matched (Q @ K^T in transformers).
// MatMulConversion passes the stamped attributes through to hip.matmul, which
// applies the transpose inside hipBLASLt instead of a separate transpose
// kernel.
//
//===----------------------------------------------------------------------===//

#include "OnnxToHipUtils.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/Statistic.h"

#define DEBUG_TYPE "transpose-matmul-fold"

STATISTIC(NumTransposeMatMulFolds,
          "Number of Transpose(last-two-swap) -> MatMul folds");

namespace mlir {
namespace hip {

namespace {

static bool isLastTwoDimSwap(ArrayRef<int64_t> perm) {
  int64_t r = perm.size();
  if (r < 2)
    return false;
  for (int64_t i = 0; i < r - 2; ++i) {
    if (perm[i] != i)
      return false;
  }
  return perm[r - 2] == r - 1 && perm[r - 1] == r - 2;
}

static std::optional<SmallVector<int64_t>>
getExplicitTransposePerm(mlir::Operation *transposeOp) {
  if (!transposeOp || transposeOp->getName().getStringRef() != "onnx.Transpose")
    return std::nullopt;
  SmallVector<int64_t> perm;
  if (auto permAttr =
          transposeOp->getAttrOfType<mlir::DenseIntElementsAttr>("perm")) {
    perm.reserve(permAttr.size());
    for (mlir::APInt v : permAttr.getValues<mlir::APInt>())
      perm.push_back(v.getSExtValue());
    return perm;
  }
  if (auto permAttr = transposeOp->getAttrOfType<mlir::ArrayAttr>("perm")) {
    perm.reserve(permAttr.size());
    for (mlir::Attribute elem : permAttr) {
      auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(elem);
      if (!intAttr)
        return std::nullopt;
      // Not getSInt(): only *scalar* ONNX INT attributes import as si64. The
      // elements of an INTS attribute are signless, as is a perm this pipeline
      // builds itself through getI64ArrayAttr(), and getSInt() asserts on both.
      perm.push_back(intAttr.getValue().getSExtValue());
    }
    return perm;
  }
  return std::nullopt;
}

static bool isFoldableLastTwoSwap(mlir::Operation *transposeOp) {
  auto permOpt = getExplicitTransposePerm(transposeOp);
  if (!permOpt)
    return false;
  if (!isLastTwoDimSwap(*permOpt))
    return false;
  if (!transposeOp->getResult(0).hasOneUse())
    return false;
  return true;
}

struct FoldTransposeIntoMatMul : public mlir::RewritePattern {
  FoldTransposeIntoMatMul(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.MatMul", /*benefit=*/2, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *matmulOp,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::Value a = matmulOp->getOperand(0);
    mlir::Value b = matmulOp->getOperand(1);
    int64_t transA = 0;
    int64_t transB = 0;

    if (mlir::Operation *trA = a.getDefiningOp()) {
      if (isFoldableLastTwoSwap(trA)) {
        a = trA->getOperand(0);
        transA = 1;
      }
    }
    if (mlir::Operation *trB = b.getDefiningOp()) {
      if (isFoldableLastTwoSwap(trB)) {
        b = trB->getOperand(0);
        transB = 1;
      }
    }
    if (transA == 0 && transB == 0)
      return rewriter.notifyMatchFailure(matmulOp, "no foldable transpose");

    mlir::Location loc = matmulOp->getLoc();
    mlir::OperationState state(loc, "onnx.MatMul");
    state.addOperands({a, b});
    state.addTypes(matmulOp->getResultTypes());
    for (mlir::NamedAttribute attr : matmulOp->getAttrs()) {
      if (attr.getName() == "hipdnn.transA" ||
          attr.getName() == "hipdnn.transB")
        continue;
      state.addAttribute(attr.getName(), attr.getValue());
    }
    if (transA)
      state.addAttribute("hipdnn.transA", rewriter.getI64IntegerAttr(transA));
    if (transB)
      state.addAttribute("hipdnn.transB", rewriter.getI64IntegerAttr(transB));

    mlir::Operation *newMatmul = rewriter.create(state);
    rewriter.replaceOp(matmulOp, newMatmul->getResults());
    ++NumTransposeMatMulFolds;
    return mlir::success();
  }
};

} // namespace

void populateTransposeMatMulFoldPatterns(RewritePatternSet &patterns,
                                         MLIRContext *ctx) {
  patterns.add<FoldTransposeIntoMatMul>(ctx);
}

} // namespace hip
} // namespace mlir
