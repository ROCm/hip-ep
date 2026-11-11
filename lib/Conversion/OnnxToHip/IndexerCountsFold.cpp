/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- IndexerCountsFold.cpp - Per-row counts from the QSA indexer --------===//
//
// DynamicSparseAttention takes the number of valid selected_indices entries
// of every query row, which exporters compute from a qsa
// SparseAttentionIndexer's output with
//
//   counts = ReduceSum(Where(GreaterOrEqual(selected_indices, 0), 1, 0),
//                      axes = [-1], keepdims = 0)                 [B, S]
//
// - The glue lowers to four kernels over the whole [B, S, K * r + r - 1]
//   selection.
// - The indexer writes each row's valid entries, all non-negative token
//   positions, ahead of the -1 fill, so it already knows the exact count.
// - The fold gives the onnx.Custom a third i32 [B, S] result for it, marked
//   with kSelectedCountsAttr, and replaces the glue.
// - SparseAttentionIndexerConversion is the only reader of the marked form.
//   Other uses of selected_indices still read the indices.
//
//===----------------------------------------------------------------------===//

#include "OnnxToHipUtils.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"

#include <optional>

namespace mlir {
namespace hip {
namespace {

/// Returns the value of a single-element integer onnx.Constant, or
/// std::nullopt when `v` has another producer.
std::optional<int64_t> scalarConstant(Value v) {
  Operation *op = onnxProducer(v, "onnx.Constant");
  auto attr = op ? op->getAttrOfType<DenseIntElementsAttr>("value") : nullptr;
  if (!attr || attr.getNumElements() != 1) {
    return std::nullopt;
  }
  return (*attr.getValues<APInt>().begin()).getSExtValue();
}

struct IndexerCountsFold : public RewritePattern {
  IndexerCountsFold(MLIRContext *ctx)
      : RewritePattern("onnx.ReduceSum", /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *reduce,
                                PatternRewriter &rewriter) const override {
    auto keepdims = reduce->getAttrOfType<IntegerAttr>("keepdims");
    auto countsType =
        dyn_cast<RankedTensorType>(reduce->getResult(0).getType());
    if (reduce->getNumOperands() != 2 || !keepdims || keepdims.getInt() != 0 ||
        !countsType || countsType.getRank() != 2 ||
        !countsType.getElementType().isInteger(32)) {
      return rewriter.notifyMatchFailure(
          reduce, "not an i32 rank 2 ReduceSum without keepdims");
    }
    Operation *where = onnxProducer(reduce->getOperand(0), "onnx.Where");
    Operation *compare =
        where ? onnxProducer(where->getOperand(0), "onnx.GreaterOrEqual")
              : nullptr;
    if (!compare || scalarConstant(where->getOperand(1)) != 1 ||
        scalarConstant(where->getOperand(2)) != 0 ||
        scalarConstant(compare->getOperand(1)) != 0) {
      return rewriter.notifyMatchFailure(
          reduce, "not ReduceSum(Where(GreaterOrEqual(x, 0), 1, 0))");
    }

    auto selected = dyn_cast<OpResult>(compare->getOperand(0));
    Operation *indexer = selected ? selected.getOwner() : nullptr;
    auto selectedType =
        selected ? dyn_cast<RankedTensorType>(selected.getType()) : nullptr;
    if (!indexer || selected.getResultNumber() != 0 ||
        !isQsaSparseAttentionIndexer(indexer) || !selectedType ||
        selectedType.getRank() != 3) {
      return rewriter.notifyMatchFailure(
          reduce, "does not count a qsa SparseAttentionIndexer's selection");
    }
    std::optional<int64_t> axis = scalarConstant(reduce->getOperand(1));
    if (axis != -1 && axis != 2) {
      return rewriter.notifyMatchFailure(reduce,
                                         "must reduce the selection axis");
    }
    if (failed(verifyCompatibleShape(selectedType.getShape().take_front(2),
                                     countsType.getShape()))) {
      return rewriter.notifyMatchFailure(reduce, "counts must be [B, S] of the "
                                                 "selection");
    }

    Value counts;
    if (indexer->hasAttr(kSelectedCountsAttr)) {
      if (indexer->getNumResults() != 3 ||
          indexer->getResult(2).getType() != countsType) {
        return rewriter.notifyMatchFailure(
            reduce, "indexer counts result has another type");
      }
      counts = indexer->getResult(2);
    } else {
      if (indexer->getNumResults() != 2) {
        return rewriter.notifyMatchFailure(reduce,
                                           "indexer must have 2 results");
      }
      rewriter.setInsertionPoint(indexer);
      OperationState state(indexer->getLoc(), indexer->getName());
      state.addOperands(indexer->getOperands());
      state.addTypes(indexer->getResultTypes());
      state.addTypes(countsType);
      state.addAttributes(indexer->getAttrs());
      state.addAttribute(kSelectedCountsAttr, rewriter.getUnitAttr());
      Operation *folded = rewriter.create(state);
      rewriter.replaceOp(indexer, folded->getResults().take_front(2));
      counts = folded->getResult(2);
    }
    rewriter.replaceOp(reduce, counts);
    if (where->use_empty()) {
      rewriter.eraseOp(where);
    }
    if (compare->use_empty()) {
      rewriter.eraseOp(compare);
    }
    return success();
  }
};

} // namespace

void populateIndexerCountsFoldPatterns(RewritePatternSet &patterns,
                                       MLIRContext *ctx) {
  patterns.add<IndexerCountsFold>(ctx);
}

} // namespace hip
} // namespace mlir
