/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

#include <cmath>

namespace mlir {
namespace hip {
namespace {

/// com.microsoft.DynamicSparseAttention -> hip.dynamic_sparse_attention
struct DynamicSparseAttentionToHip : public mlir::RewritePattern {
  DynamicSparseAttentionToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Custom", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override;
};

mlir::LogicalResult DynamicSparseAttentionToHip::matchAndRewrite(
    mlir::Operation *op, mlir::PatternRewriter &rewriter) const {
  auto funcNameAttr = op->getAttrOfType<mlir::StringAttr>("function_name");
  if (!funcNameAttr || funcNameAttr.getValue() != "DynamicSparseAttention") {
    return rewriter.notifyMatchFailure(
        op, "not a DynamicSparseAttention operation");
  }
  auto domainAttr = op->getAttrOfType<mlir::StringAttr>("domain_name");
  if (!domainAttr || domainAttr.getValue() != "com.microsoft") {
    return rewriter.notifyMatchFailure(
        op, "domain must be com.microsoft for DynamicSparseAttention");
  }

  auto readStr = [&](llvm::StringRef name, llvm::StringRef defaultVal) {
    auto attr = op->getAttrOfType<mlir::StringAttr>(name);
    return attr ? attr.getValue() : defaultVal;
  };
  auto readInt = [&](llvm::StringRef name, int64_t defaultVal) {
    auto attr = op->getAttrOfType<mlir::IntegerAttr>(name);
    return attr ? attr.getValue().getSExtValue() : defaultVal;
  };
  if (readStr("attention_mode", "selected_only") != "selected_only") {
    return rewriter.notifyMatchFailure(op,
                                       "attention_mode must be selected_only");
  }
  if (readStr("selected_kv_source", "main") != "main") {
    return rewriter.notifyMatchFailure(op, "selected_kv_source must be main");
  }
  if (readInt("is_causal", 1) != 1) {
    return rewriter.notifyMatchFailure(op, "is_causal must be 1");
  }
  if (readInt("do_rotary", 0) != 0) {
    return rewriter.notifyMatchFailure(op, "do_rotary is not supported");
  }
  if (readInt("smooth_softmax", 0) != 0) {
    return rewriter.notifyMatchFailure(op, "smooth_softmax is not supported");
  }
  if (readInt("local_window_size", -1) != -1) {
    return rewriter.notifyMatchFailure(op,
                                       "local_window_size is not supported");
  }
  if (readInt("auxiliary_kv_shared", 0) != 0) {
    return rewriter.notifyMatchFailure(op,
                                       "auxiliary_kv_shared is not supported");
  }
  const int64_t numHeads = readInt("num_heads", 0);
  const int64_t kvNumHeads = readInt("kv_num_heads", 0);
  if (numHeads <= 0 || kvNumHeads <= 0 || numHeads % kvNumHeads != 0) {
    return rewriter.notifyMatchFailure(
        op, "num_heads must be a positive multiple of kv_num_heads");
  }
  if (numHeads / kvNumHeads > 16) {
    return rewriter.notifyMatchFailure(
        op, "num_heads / kv_num_heads must be at most 16");
  }

  size_t numOps = op->getNumOperands();
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

  if (numOps < 11 || numOps > 17) {
    return rewriter.notifyMatchFailure(
        op, "DynamicSparseAttention expects 11-17 operands");
  }
  mlir::Value query = op->getOperand(0);
  mlir::Value key = getOptionalOperand(1);
  mlir::Value value = getOptionalOperand(2);
  mlir::Value pastKey = getOptionalOperand(3);
  mlir::Value pastValue = getOptionalOperand(4);
  mlir::Value selectedIndices = getOptionalOperand(7);
  mlir::Value selectedCounts = getOptionalOperand(8);
  mlir::Value seqlensK = getOptionalOperand(9);
  if (!key || !value) {
    return rewriter.notifyMatchFailure(op, "packed QKV is not supported");
  }
  if (!pastKey || !pastValue) {
    return rewriter.notifyMatchFailure(op, "past_key/past_value are required");
  }
  if (getOptionalOperand(5) || getOptionalOperand(6)) {
    return rewriter.notifyMatchFailure(op, "auxiliary KV is not supported");
  }
  if (!selectedIndices || !selectedCounts || !seqlensK) {
    return rewriter.notifyMatchFailure(
        op, "selected_indices, selected_counts and seqlens_k are required");
  }
  for (size_t idx = 11; idx < 17; ++idx) {
    if (getOptionalOperand(idx)) {
      return rewriter.notifyMatchFailure(
          op, "rotary caches, position_ids, q/k norm weights and head_sink "
              "are not supported");
    }
  }

  auto queryType = mlir::dyn_cast<mlir::RankedTensorType>(query.getType());
  if (!queryType || queryType.getRank() != 3 || queryType.isDynamicDim(2) ||
      queryType.getDimSize(2) % numHeads != 0) {
    return rewriter.notifyMatchFailure(
        op, "query must be [B, S, num_heads * D] with a static last dim");
  }
  const int64_t headSize = queryType.getDimSize(2) / numHeads;
  if (headSize != 128 && headSize != 256) {
    return rewriter.notifyMatchFailure(op, "head size must be 128 or 256");
  }
  auto pastType = mlir::dyn_cast<mlir::RankedTensorType>(pastKey.getType());
  if (!pastType || pastType.getRank() != 4 ||
      !pastType.getElementType().isF16()) {
    return rewriter.notifyMatchFailure(op, "past_key must be f16 rank 4");
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
  mlir::Type f16 = rewriter.getF16Type();
  mlir::Type i32 = rewriter.getI32Type();
  const int64_t batch = queryType.getDimSize(0);
  const int64_t seq = queryType.getDimSize(1);
  const int64_t rows =
      mlir::ShapedType::isDynamic(batch) || mlir::ShapedType::isDynamic(seq)
          ? kAny
          : batch * seq;
  const llvm::SmallVector<int64_t> cacheShape = {
      batch, kvNumHeads, pastType.getDimSize(2), headSize};
  if (!matches(query, f16, {batch, seq, numHeads * headSize})) {
    return rewriter.notifyMatchFailure(op, "query must be f16");
  }
  if (!matches(key, f16, {batch, seq, kvNumHeads * headSize}) ||
      !matches(value, f16, {batch, seq, kvNumHeads * headSize})) {
    return rewriter.notifyMatchFailure(
        op, "key and value must be f16 [B, S, kv_num_heads * D]");
  }
  if (!matches(pastKey, f16, cacheShape) ||
      !matches(pastValue, f16, cacheShape)) {
    return rewriter.notifyMatchFailure(
        op, "past caches must be f16 [B, kv_num_heads, capacity, D]");
  }
  if (!matches(selectedIndices, i32, {rows, kAny})) {
    return rewriter.notifyMatchFailure(
        op, "selected_indices must be i32 [B * S, max_selected]");
  }
  if (!matches(selectedCounts, i32, {rows})) {
    return rewriter.notifyMatchFailure(op,
                                       "selected_counts must be i32 [B * S]");
  }
  if (!matches(seqlensK, i32, {batch})) {
    return rewriter.notifyMatchFailure(op, "seqlens_k must be i32 [B]");
  }

  if (op->getNumResults() != 3) {
    return rewriter.notifyMatchFailure(
        op, "DynamicSparseAttention expects 3 results");
  }
  auto outputType =
      mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
  if (!outputType ||
      !matches(op->getResult(0), f16, {batch, seq, numHeads * headSize})) {
    return rewriter.notifyMatchFailure(op, "output must be f16 like query");
  }
  auto withStaticHeadSize = [&](mlir::RankedTensorType type) {
    llvm::SmallVector<int64_t> shape(type.getShape());
    shape[3] = headSize;
    return mlir::RankedTensorType::get(shape, type.getElementType());
  };
  auto getPresentType = [&](unsigned idx,
                            mlir::Value past) -> mlir::RankedTensorType {
    mlir::Type type = op->getResult(idx).getType();
    if (mlir::isa<mlir::NoneType>(type)) {
      return withStaticHeadSize(
          mlir::cast<mlir::RankedTensorType>(past.getType()));
    }
    auto ranked = mlir::dyn_cast<mlir::RankedTensorType>(type);
    if (!ranked || !matches(op->getResult(idx), f16, cacheShape)) {
      return nullptr;
    }
    return withStaticHeadSize(ranked);
  };
  auto presentKeyType = getPresentType(1, pastKey);
  auto presentValueType = getPresentType(2, pastValue);
  if (!presentKeyType || !presentValueType) {
    return rewriter.notifyMatchFailure(
        op, "present caches must match the past cache shape");
  }

  float scale = 1.0f / std::sqrt(static_cast<float>(headSize));
  if (auto attr = op->getAttrOfType<mlir::FloatAttr>("scale")) {
    scale = static_cast<float>(attr.getValueAsDouble());
  }

  auto ctxOrFailure = getContextArg(op, rewriter);
  if (mlir::failed(ctxOrFailure)) {
    return rewriter.notifyMatchFailure(op, "missing context argument");
  }

  mlir::Location loc = op->getLoc();
  auto refine = [&](mlir::Value v) -> mlir::Value {
    auto type = mlir::cast<mlir::RankedTensorType>(v.getType());
    if (!type.isDynamicDim(3)) {
      return v;
    }
    return mlir::tensor::CastOp::create(rewriter, loc, withStaticHeadSize(type),
                                        v);
  };
  pastKey = refine(pastKey);
  pastValue = refine(pastValue);
  mlir::Value outputInit = createEmptyTensor(rewriter, loc, outputType, query);
  mlir::Value presentKeyInit =
      createEmptyTensor(rewriter, loc, presentKeyType, pastKey);
  mlir::Value presentValueInit =
      createEmptyTensor(rewriter, loc, presentValueType, pastValue);

  auto state = mlir::OperationState(loc, "hip.dynamic_sparse_attention");
  state.addOperands({*ctxOrFailure, query, key, value, pastKey, pastValue,
                     selectedIndices, selectedCounts, seqlensK, outputInit,
                     presentKeyInit, presentValueInit});
  state.addTypes({outputType, presentKeyType, presentValueType});
  state.addAttribute("num_heads", rewriter.getI64IntegerAttr(numHeads));
  state.addAttribute("kv_num_heads", rewriter.getI64IntegerAttr(kvNumHeads));
  state.addAttribute("scale", rewriter.getF32FloatAttr(scale));
  mlir::Operation *hipOp = rewriter.create(state);
  llvm::SmallVector<mlir::Value, 3> results(hipOp->getResults());
  for (unsigned idx = 1; idx < 3; ++idx) {
    mlir::Type origType = op->getResult(idx).getType();
    if (mlir::isa<mlir::RankedTensorType>(origType) &&
        origType != results[idx].getType()) {
      results[idx] =
          mlir::tensor::CastOp::create(rewriter, loc, origType, results[idx]);
    }
  }
  rewriter.replaceOp(op, results);
  return mlir::success();
}

} // namespace

void populateDynamicSparseAttentionConversionPatterns(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.add<DynamicSparseAttentionToHip>(ctx);
}

} // namespace hip
} // namespace mlir
