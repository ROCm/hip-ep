/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- QsaPairFold.cpp - Pair the QSA indexer with its attention ----------===//
//
// A qsa SparseAttentionIndexer without a mask feeds DynamicSparseAttention as
//
//   sel, present, counts = SparseAttentionIndexer(..., past_sequence_length)
//   out = DynamicSparseAttention(..., Reshape(sel, [-1, K]),
//                                Reshape(counts, [-1]), ...)
//
// with K = token_budget + compress_ratio - 1 (counts comes from
// IndexerCountsFold). Row (b, s) sees the P + s + 1 positions up to its own
// (P = past_sequence_length), and while that is at most K the indexer selects
// all of them: the first min(K - P, S) rows of every batch are dense. Scoring
// and sorting them is wasted, and the attention would read K indices just to
// visit positions 0 .. P + s.
//
// When every reader of the selection is such an attention, this fold stamps
// kQsaPairAttr on the indexer and appends its past_sequence_length to each
// attention at kDsaPastSequenceLengthIdx. The converted indexer then leaves the
// dense rows unwritten (their counts stay exact) and lists the blocks of the
// other rows in ascending order instead of by score; the converted attention
// takes the positions of a dense row in order. Attention does not depend on
// candidate order, so the result is the unpaired one up to fp32 summation
// order. A selection that is a graph output, or read by anything else, keeps
// the unpaired form.
//
//===----------------------------------------------------------------------===//

#include "OnnxToHipUtils.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"

namespace mlir {
namespace hip {
namespace {

constexpr unsigned kIndexerMaskIdx = 6;
constexpr unsigned kIndexerPastSequenceLengthIdx = 12;
constexpr unsigned kDsaAuxKeyIdx = 5;
constexpr unsigned kDsaSelectedIndicesIdx = 7;
constexpr unsigned kDsaSelectedCountsIdx = 8;

Value optionalOperand(Operation *op, unsigned idx) {
  if (idx >= op->getNumOperands()) {
    return nullptr;
  }
  Value v = op->getOperand(idx);
  return isa<NoneType>(v.getType()) ? nullptr : v;
}

bool isCustom(Operation *op, StringRef function) {
  auto fn = op->getAttrOfType<StringAttr>("function_name");
  auto domain = op->getAttrOfType<StringAttr>("domain_name");
  return op->getName().getStringRef() == "onnx.Custom" && fn &&
         fn.getValue() == function && domain &&
         domain.getValue() == "com.microsoft";
}

StringRef strAttr(Operation *op, StringRef name, StringRef defaultVal) {
  auto attr = op->getAttrOfType<StringAttr>(name);
  return attr ? attr.getValue() : defaultVal;
}

/// `v`, or the input of the onnx.Reshape producing it.
Value throughReshape(Value v) {
  Operation *op = v.getDefiningOp();
  return op && op->getName().getStringRef() == "onnx.Reshape"
             ? op->getOperand(0)
             : v;
}

bool compatibleDim(int64_t a, int64_t b) {
  return ShapedType::isDynamic(a) || ShapedType::isDynamic(b) || a == b;
}

struct QsaPairFold : public RewritePattern {
  QsaPairFold(MLIRContext *ctx)
      : RewritePattern("onnx.Custom", /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *indexer,
                                PatternRewriter &rewriter) const override {
    if (!isCustom(indexer, "SparseAttentionIndexer") ||
        strAttr(indexer, "policy_mode", "") != "qsa" ||
        !indexer->hasAttr(kSelectedCountsAttr) ||
        indexer->hasAttr(kQsaPairAttr) || indexer->getNumResults() != 3) {
      return rewriter.notifyMatchFailure(
          indexer, "not an unpaired qsa indexer with a counts result");
    }
    Value pastSeqLen = optionalOperand(indexer, kIndexerPastSequenceLengthIdx);
    if (optionalOperand(indexer, kIndexerMaskIdx) || !pastSeqLen) {
      return rewriter.notifyMatchFailure(
          indexer, "needs past_sequence_length and no mask");
    }
    auto ratio = indexer->getAttrOfType<IntegerAttr>("compress_ratio");
    auto budget = indexer->getAttrOfType<IntegerAttr>("token_budget");
    Value selected = indexer->getResult(0);
    Value counts = indexer->getResult(2);
    auto selectedType = dyn_cast<RankedTensorType>(selected.getType());
    auto queryType =
        dyn_cast<RankedTensorType>(indexer->getOperand(0).getType());
    if (!ratio || !budget || !selectedType || selectedType.getRank() != 3 ||
        !queryType || queryType.getRank() != 3 ||
        selectedType.getDimSize(2) != budget.getValue().getSExtValue() +
                                          ratio.getValue().getSExtValue() - 1) {
      return rewriter.notifyMatchFailure(
          indexer, "selection must be [B, S, token_budget + r - 1]");
    }
    const int64_t width = selectedType.getDimSize(2);

    // Every reader of the selection must be a Reshape to [B * S, K] feeding
    // only the selected_indices of an attention that also takes this
    // indexer's counts.
    llvm::SetVector<Operation *> attentions;
    for (Operation *reshape : selected.getUsers()) {
      auto rowsType =
          reshape->getName().getStringRef() == "onnx.Reshape" &&
                  reshape->getNumResults() == 1
              ? dyn_cast<RankedTensorType>(reshape->getResult(0).getType())
              : RankedTensorType();
      if (!rowsType || reshape->getOperand(0) != selected ||
          rowsType.getRank() != 2 || rowsType.getDimSize(1) != width) {
        return rewriter.notifyMatchFailure(
            indexer, "selection has a reader other than a [B * S, K] Reshape");
      }
      for (OpOperand &use : reshape->getResult(0).getUses()) {
        Operation *dsa = use.getOwner();
        if (use.getOperandNumber() != kDsaSelectedIndicesIdx ||
            !isCustom(dsa, "DynamicSparseAttention") ||
            strAttr(dsa, "attention_mode", "selected_only") !=
                "selected_only" ||
            strAttr(dsa, "selected_kv_source", "main") != "main" ||
            dsa->getNumOperands() <= kDsaSelectedCountsIdx ||
            dsa->getNumOperands() > kDsaMaxInputs ||
            throughReshape(dsa->getOperand(kDsaSelectedCountsIdx)) != counts ||
            !isa<NoneType>(dsa->getOperand(kDsaAuxKeyIdx).getType()) ||
            optionalOperand(dsa, kDsaPastSequenceLengthIdx)) {
          return rewriter.notifyMatchFailure(
              indexer, "selection is read by something other than the "
                       "selected_indices of an attention over its counts");
        }
        auto dsaQueryType =
            dyn_cast<RankedTensorType>(dsa->getOperand(0).getType());
        if (!dsaQueryType || dsaQueryType.getRank() != 3 ||
            !compatibleDim(dsaQueryType.getDimSize(0),
                           queryType.getDimSize(0)) ||
            !compatibleDim(dsaQueryType.getDimSize(1),
                           queryType.getDimSize(1))) {
          return rewriter.notifyMatchFailure(
              indexer, "attention rows are not the indexer's [B, S] rows");
        }
        attentions.insert(dsa);
      }
    }
    if (attentions.empty()) {
      return rewriter.notifyMatchFailure(indexer, "selection is not read");
    }

    rewriter.modifyOpInPlace(indexer, [&] {
      indexer->setAttr(kQsaPairAttr, rewriter.getUnitAttr());
    });
    for (Operation *dsa : attentions) {
      SmallVector<Value> operands = llvm::to_vector(dsa->getOperands());
      operands.resize(kDsaMaxInputs, dsa->getOperand(kDsaAuxKeyIdx));
      operands[kDsaPastSequenceLengthIdx] = pastSeqLen;
      rewriter.setInsertionPoint(dsa);
      OperationState state(dsa->getLoc(), dsa->getName());
      state.addOperands(operands);
      state.addTypes(dsa->getResultTypes());
      state.addAttributes(dsa->getAttrs());
      Operation *paired = rewriter.create(state);
      rewriter.replaceOp(dsa, paired->getResults());
    }
    return success();
  }
};

} // namespace

void populateQsaPairFoldPatterns(RewritePatternSet &patterns,
                                 MLIRContext *ctx) {
  patterns.add<QsaPairFold>(ctx);
}

} // namespace hip
} // namespace mlir
