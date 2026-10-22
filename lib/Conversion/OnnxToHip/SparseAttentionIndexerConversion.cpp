/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

#include <cmath>

namespace mlir {
namespace hip {
namespace {

constexpr int64_t kHeadSize = 128;
constexpr int64_t kMaxHeads = 8;
constexpr int64_t kMaxBlockTopK = 512;

/// com.microsoft.SparseAttentionIndexer (policy_mode = qsa)
///   -> hip.sparse_attention_indexer
struct SparseAttentionIndexerToHip : public mlir::RewritePattern {
  SparseAttentionIndexerToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Custom", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override;
};

mlir::LogicalResult SparseAttentionIndexerToHip::matchAndRewrite(
    mlir::Operation *op, mlir::PatternRewriter &rewriter) const {
  auto funcNameAttr = op->getAttrOfType<mlir::StringAttr>("function_name");
  if (!funcNameAttr || funcNameAttr.getValue() != "SparseAttentionIndexer") {
    return rewriter.notifyMatchFailure(
        op, "not a SparseAttentionIndexer operation");
  }
  auto domainAttr = op->getAttrOfType<mlir::StringAttr>("domain_name");
  if (!domainAttr || domainAttr.getValue() != "com.microsoft") {
    return rewriter.notifyMatchFailure(
        op, "domain must be com.microsoft for SparseAttentionIndexer");
  }

  auto policy = op->getAttrOfType<mlir::StringAttr>("policy_mode");
  if (!policy || policy.getValue() != "qsa") {
    return rewriter.notifyMatchFailure(op, "policy_mode must be qsa");
  }
  if (op->hasAttr("index_topk") || op->hasAttr("head_weight_scale")) {
    return rewriter.notifyMatchFailure(
        op, "index_topk / head_weight_scale are csa-only attributes");
  }
  auto ratioAttr = op->getAttrOfType<mlir::IntegerAttr>("compress_ratio");
  auto budgetAttr = op->getAttrOfType<mlir::IntegerAttr>("token_budget");
  if (!ratioAttr || !budgetAttr) {
    return rewriter.notifyMatchFailure(
        op, "compress_ratio and token_budget are required");
  }
  const int64_t ratio = ratioAttr.getValue().getSExtValue();
  const int64_t budget = budgetAttr.getValue().getSExtValue();
  if (ratio <= 0 || budget <= 0 || budget % ratio != 0 ||
      budget / ratio > kMaxBlockTopK) {
    return rewriter.notifyMatchFailure(
        op, "token_budget must be a positive multiple of compress_ratio with "
            "at most 512 blocks");
  }
  float epsilon = 1e-6f;
  if (auto attr = op->getAttrOfType<mlir::FloatAttr>("epsilon")) {
    epsilon = static_cast<float>(attr.getValueAsDouble());
  }
  if (epsilon < 0.0f) {
    return rewriter.notifyMatchFailure(op, "epsilon must be non-negative");
  }

  const size_t numOps = op->getNumOperands();
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
  if (numOps < 8 || numOps > 14) {
    return rewriter.notifyMatchFailure(
        op, "SparseAttentionIndexer expects 8-14 operands");
  }
  if (getOptionalOperand(1)) {
    return rewriter.notifyMatchFailure(op, "only packed QK is supported");
  }
  for (size_t idx = 8; idx < numOps; ++idx) {
    if (getOptionalOperand(idx)) {
      return rewriter.notifyMatchFailure(
          op, "csa inputs and past_sequence_length are not supported");
    }
  }
  mlir::Value query = op->getOperand(0);
  mlir::Value queryNormWeight = getOptionalOperand(2);
  mlir::Value keyNormWeight = getOptionalOperand(3);
  mlir::Value cosCache = getOptionalOperand(4);
  mlir::Value sinCache = getOptionalOperand(5);
  mlir::Value mask = getOptionalOperand(6);
  mlir::Value pastKey = getOptionalOperand(7);
  if (!queryNormWeight || !keyNormWeight || !cosCache || !sinCache ||
      !pastKey) {
    return rewriter.notifyMatchFailure(
        op, "norm weights, cos/sin caches and past_key are required");
  }

  auto queryType = mlir::dyn_cast<mlir::RankedTensorType>(query.getType());
  if (!queryType || queryType.getRank() != 3 || queryType.isDynamicDim(2)) {
    return rewriter.notifyMatchFailure(
        op, "query must be rank 3 with a static width");
  }
  mlir::Type elem = queryType.getElementType();
  if (!elem.isF16() && !elem.isF32()) {
    return rewriter.notifyMatchFailure(op, "element type must be f16 or f32");
  }
  const int64_t width = queryType.getDimSize(2);
  const int64_t numHeads = width / kHeadSize - 1;
  if (width % kHeadSize != 0 || numHeads < 1 || numHeads > kMaxHeads) {
    return rewriter.notifyMatchFailure(
        op, "query must be packed [B, S, (N + 1) * 128] with 1 <= N <= 8");
  }

  auto matches = [](mlir::Value v, mlir::Type expectedElem,
                    llvm::ArrayRef<int64_t> expected) {
    auto type = mlir::dyn_cast<mlir::RankedTensorType>(v.getType());
    if (!type || type.getElementType() != expectedElem ||
        type.getRank() != static_cast<int64_t>(expected.size())) {
      return false;
    }
    for (auto [i, want] : llvm::enumerate(expected)) {
      int64_t got = type.getDimSize(i);
      if (want != mlir::ShapedType::kDynamic &&
          got != mlir::ShapedType::kDynamic && got != want) {
        return false;
      }
    }
    return true;
  };
  constexpr int64_t kAny = mlir::ShapedType::kDynamic;
  const int64_t batch = queryType.getDimSize(0);
  const int64_t seq = queryType.getDimSize(1);

  if (!matches(queryNormWeight, elem, {kHeadSize}) ||
      !matches(keyNormWeight, elem, {kHeadSize})) {
    return rewriter.notifyMatchFailure(
        op, "query/key norm weights must be [128] of the query type");
  }
  auto cosType = mlir::dyn_cast<mlir::RankedTensorType>(cosCache.getType());
  if (!cosType || (cosType.getRank() != 2 && cosType.getRank() != 3) ||
      cosType.isDynamicDim(cosType.getRank() - 1)) {
    return rewriter.notifyMatchFailure(
        op, "cos_cache must be rank 2 or 3 with a static rotary width");
  }
  const int64_t cosRank = cosType.getRank();
  const int64_t rotary = cosType.getDimSize(cosRank - 1);
  if (rotary <= 0 || rotary % 2 != 0 || rotary > kHeadSize) {
    return rewriter.notifyMatchFailure(
        op, "rotary width must be even and in (0, 128]");
  }
  llvm::SmallVector<int64_t> cacheShape = {kAny, rotary};
  if (cosRank == 3) {
    cacheShape.insert(cacheShape.begin(), batch);
  }
  if (!matches(cosCache, elem, cacheShape) ||
      !matches(sinCache, elem, cacheShape)) {
    return rewriter.notifyMatchFailure(
        op, "cos/sin caches must be [X, R] or [B, X, R] of the query type");
  }
  if (!matches(pastKey, elem, {batch, kAny, kHeadSize})) {
    return rewriter.notifyMatchFailure(op,
                                       "past_key must be [B, P, 128] of the "
                                       "query type");
  }
  auto pastType = mlir::cast<mlir::RankedTensorType>(pastKey.getType());
  const int64_t past = pastType.getDimSize(1);
  const int64_t total =
      mlir::ShapedType::isDynamic(past) || mlir::ShapedType::isDynamic(seq)
          ? kAny
          : past + seq;
  if (mask && !matches(mask, rewriter.getI64Type(), {batch, total})) {
    return rewriter.notifyMatchFailure(op, "mask must be i64 [B, P + S]");
  }

  if (op->getNumResults() != 2) {
    return rewriter.notifyMatchFailure(
        op, "SparseAttentionIndexer qsa expects 2 results");
  }
  const int64_t capacity = budget + ratio - 1;
  if (!matches(op->getResult(0), rewriter.getI32Type(),
               {batch, seq, capacity})) {
    return rewriter.notifyMatchFailure(
        op, "selected_indices must be i32 [B, S, token_budget + r - 1]");
  }
  if (!matches(op->getResult(1), elem, {batch, total, kHeadSize})) {
    return rewriter.notifyMatchFailure(
        op, "present_key must be [B, P + S, 128] of the query type");
  }
  auto selectedType =
      mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());
  auto presentType =
      mlir::cast<mlir::RankedTensorType>(op->getResult(1).getType());

  float scale = 1.0f / std::sqrt(static_cast<float>(kHeadSize));
  if (auto attr = op->getAttrOfType<mlir::FloatAttr>("scale")) {
    scale = static_cast<float>(attr.getValueAsDouble());
  }

  auto ctxOrFailure = getContextArg(op, rewriter);
  if (mlir::failed(ctxOrFailure)) {
    return rewriter.notifyMatchFailure(op, "missing context argument");
  }

  mlir::Location loc = op->getLoc();
  mlir::Value selectedInit =
      createEmptyTensor(rewriter, loc, selectedType, query);

  llvm::SmallVector<mlir::Value> presentDims;
  if (presentType.isDynamicDim(0)) {
    presentDims.push_back(
        mlir::tensor::DimOp::create(rewriter, loc, query, int64_t{0}));
  }
  if (presentType.isDynamicDim(1)) {
    mlir::Value pastLen =
        mlir::tensor::DimOp::create(rewriter, loc, pastKey, int64_t{1});
    mlir::Value seqLen =
        mlir::tensor::DimOp::create(rewriter, loc, query, int64_t{1});
    presentDims.push_back(
        mlir::arith::AddIOp::create(rewriter, loc, pastLen, seqLen));
  }
  mlir::Value presentInit =
      mlir::tensor::EmptyOp::create(rewriter, loc, presentType.getShape(),
                                    presentType.getElementType(), presentDims);

  llvm::SmallVector<mlir::Value> operands = {
      *ctxOrFailure, query,    queryNormWeight, keyNormWeight,
      cosCache,      sinCache, pastKey};
  if (mask) {
    operands.push_back(mask);
  }
  operands.push_back(selectedInit);
  operands.push_back(presentInit);

  auto state = mlir::OperationState(loc, "hip.sparse_attention_indexer");
  state.addOperands(operands);
  state.addTypes({selectedType, presentType});
  state.addAttribute("compress_ratio", rewriter.getI64IntegerAttr(ratio));
  state.addAttribute("token_budget", rewriter.getI64IntegerAttr(budget));
  state.addAttribute("epsilon", rewriter.getF32FloatAttr(epsilon));
  state.addAttribute("scale", rewriter.getF32FloatAttr(scale));
  mlir::Operation *hipOp = rewriter.create(state);
  rewriter.replaceOp(op, hipOp->getResults());
  return mlir::success();
}

} // namespace

void populateSparseAttentionIndexerConversionPatterns(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.add<SparseAttentionIndexerToHip>(ctx);
}

} // namespace hip
} // namespace mlir
