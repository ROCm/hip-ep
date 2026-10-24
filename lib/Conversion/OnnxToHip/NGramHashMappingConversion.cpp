/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

#include <limits>

namespace mlir {
namespace hip {
namespace {

/// com.microsoft.NGramHashMapping -> hip.ngram_hash_mapping
struct NGramHashMappingToHip : public mlir::RewritePattern {
  NGramHashMappingToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Custom", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override;
};

mlir::LogicalResult
NGramHashMappingToHip::matchAndRewrite(mlir::Operation *op,
                                       mlir::PatternRewriter &rewriter) const {
  auto funcNameAttr = op->getAttrOfType<mlir::StringAttr>("function_name");
  if (!funcNameAttr || funcNameAttr.getValue() != "NGramHashMapping") {
    return rewriter.notifyMatchFailure(op, "not a NGramHashMapping operation");
  }
  auto domainAttr = op->getAttrOfType<mlir::StringAttr>("domain_name");
  if (!domainAttr || domainAttr.getValue() != "com.microsoft") {
    return rewriter.notifyMatchFailure(
        op, "domain must be com.microsoft for NGramHashMapping");
  }

  size_t numOps = op->getNumOperands();
  if (numOps < 3 || numOps > 7) {
    return rewriter.notifyMatchFailure(op,
                                       "NGramHashMapping expects 3-7 operands");
  }
  size_t numResults = op->getNumResults();
  if (numResults < 1 || numResults > 2) {
    return rewriter.notifyMatchFailure(op,
                                       "NGramHashMapping expects 1-2 results");
  }

  auto readInt = [&](llvm::StringRef name) -> std::optional<int64_t> {
    auto attr = op->getAttrOfType<mlir::IntegerAttr>(name);
    if (!attr) {
      return std::nullopt;
    }
    return attr.getValue().getSExtValue();
  };
  std::optional<int64_t> maxNgramSize = readInt("max_ngram_size");
  std::optional<int64_t> nHeadPerNgram = readInt("n_head_per_ngram");
  std::optional<int64_t> padId = readInt("pad_id");
  if (!maxNgramSize || !nHeadPerNgram || !padId) {
    return rewriter.notifyMatchFailure(
        op, "missing max_ngram_size / n_head_per_ngram / pad_id");
  }
  int64_t resetOnEos = readInt("reset_on_eos").value_or(0);

  auto getOptionalOperand = [&](size_t idx) -> mlir::Value {
    if (idx >= numOps) {
      return nullptr;
    }
    mlir::Value val = op->getOperand(idx);
    if (mlir::isa<mlir::NoneType>(val.getType())) {
      return nullptr;
    }
    return val;
  };

  mlir::Value inputIds = op->getOperand(0);
  mlir::Value multipliers = op->getOperand(1);
  mlir::Value vocabSizes = op->getOperand(2);
  mlir::Value pastIds = getOptionalOperand(3);
  mlir::Value headOffsets = getOptionalOperand(4);
  mlir::Value eosTokenId = getOptionalOperand(5);
  mlir::Value segmentIds = getOptionalOperand(6);

  auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(inputIds.getType());
  if (!inputType || inputType.getRank() != 2) {
    return rewriter.notifyMatchFailure(op, "input_ids must be ranked rank-2");
  }
  if (inputType.getElementType().isSignlessInteger(32) &&
      (*padId < std::numeric_limits<int32_t>::min() ||
       *padId > std::numeric_limits<int32_t>::max())) {
    return rewriter.notifyMatchFailure(op, "pad_id does not fit in i32");
  }
  if (eosTokenId) {
    auto eosType = mlir::dyn_cast<mlir::RankedTensorType>(eosTokenId.getType());
    if (!eosType || eosType.getRank() != 0) {
      return rewriter.notifyMatchFailure(op, "eos_token_id must be rank 0");
    }
  }

  auto hashType =
      mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
  if (!hashType || hashType.getRank() != 3 || hashType.isDynamicDim(2)) {
    return rewriter.notifyMatchFailure(op,
                                       "hash_ids must be rank 3 with static H");
  }
  mlir::RankedTensorType presentType;
  if (numResults == 2 &&
      !mlir::isa<mlir::NoneType>(op->getResult(1).getType())) {
    presentType =
        mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(1).getType());
    if (!presentType || presentType.getRank() != 2 ||
        presentType.isDynamicDim(1)) {
      return rewriter.notifyMatchFailure(
          op, "present_ids must be rank 2 with static L");
    }
  }

  auto ctxOrFailure = getContextArg(op, rewriter);
  if (mlir::failed(ctxOrFailure)) {
    return rewriter.notifyMatchFailure(op, "missing context argument");
  }

  mlir::Location loc = op->getLoc();
  mlir::SmallVector<mlir::Value> operands = {*ctxOrFailure, inputIds,
                                             multipliers, vocabSizes};
  for (mlir::Value v : {pastIds, headOffsets, eosTokenId, segmentIds}) {
    if (v) {
      operands.push_back(v);
    }
  }
  mlir::SmallVector<mlir::Type> resultTypes = {hashType};
  operands.push_back(createEmptyTensor(rewriter, loc, hashType, inputIds));
  if (presentType) {
    resultTypes.push_back(presentType);
    operands.push_back(createEmptyTensor(rewriter, loc, presentType, inputIds));
  }

  llvm::SmallVector<int32_t> segmentSizes = {
      1,
      1,
      1,
      1,
      pastIds ? 1 : 0,
      headOffsets ? 1 : 0,
      eosTokenId ? 1 : 0,
      segmentIds ? 1 : 0,
      static_cast<int32_t>(resultTypes.size())};

  auto state = mlir::OperationState(loc, "hip.ngram_hash_mapping");
  state.addOperands(operands);
  state.addTypes(resultTypes);
  state.addAttribute("max_ngram_size",
                     rewriter.getI64IntegerAttr(*maxNgramSize));
  state.addAttribute("n_head_per_ngram",
                     rewriter.getI64IntegerAttr(*nHeadPerNgram));
  state.addAttribute("pad_id", rewriter.getI64IntegerAttr(*padId));
  state.addAttribute("reset_on_eos", rewriter.getI64IntegerAttr(resetOnEos));
  state.addAttribute("operand_segment_sizes",
                     rewriter.getDenseI32ArrayAttr(segmentSizes));
  mlir::Operation *hipOp = rewriter.create(state);

  llvm::SmallVector<mlir::Value> replacements = {hipOp->getResult(0)};
  if (numResults == 2) {
    replacements.push_back(presentType ? hipOp->getResult(1) : mlir::Value{});
  }
  rewriter.replaceOp(op, replacements);
  return mlir::success();
}

} // namespace

void populateNGramHashMappingConversionPatterns(RewritePatternSet &patterns,
                                                MLIRContext *ctx) {
  patterns.add<NGramHashMappingToHip>(ctx);
}

} // namespace hip
} // namespace mlir
