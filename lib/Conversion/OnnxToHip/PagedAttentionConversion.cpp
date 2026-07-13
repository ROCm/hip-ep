/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===- PagedAttentionConversion.cpp - PagedAttention ONNX→HIP lowering ---===//
//
// Matches onnx.Custom("com.microsoft", "PagedAttention") and emits
// hip.paged_attention.
//
// ORT PagedAttention input order (com.microsoft.PagedAttention):
//   [0] query        — [num_tokens, hidden_size] or packed QKV
//   [1] key_cache    — [num_blocks, block_size, kv_num_heads, head_dim]  NHD
//   [2] value_cache  — same as key_cache
//   [3] block_table  — [batch_size, max_blocks_per_seq]  int32
//   [4] slot_mapping — [num_tokens]  int32
//   [5] sequence_lengths — [batch_size]  int32
//   [6] key          — [num_tokens, G*D]  optional (null if packed QKV)
//   [7] value        — [num_tokens, G*D]  optional (null if packed QKV)
//   [8] cos_cache    — [max_pos, D/2]  optional (RoPE)
//   [9] sin_cache    — [max_pos, D/2]  optional (RoPE)
//
// Attributes:
//   num_heads, kv_num_heads, scale, do_rotary (int, default 0)
//
//===----------------------------------------------------------------------===//

#include "OnnxToHipUtils.h"

#include <cmath>

namespace mlir {
namespace hip {
namespace {

struct PagedAttentionToHip : public mlir::RewritePattern {
  PagedAttentionToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Custom", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override;
};

mlir::LogicalResult PagedAttentionToHip::matchAndRewrite(
    mlir::Operation *op, mlir::PatternRewriter &rewriter) const {
  auto funcNameAttr = op->getAttrOfType<mlir::StringAttr>("function_name");
  if (!funcNameAttr || funcNameAttr.getValue() != "PagedAttention")
    return rewriter.notifyMatchFailure(op, "not a PagedAttention operation");

  auto domainAttr = op->getAttrOfType<mlir::StringAttr>("domain_name");
  if (!domainAttr || domainAttr.getValue() != "com.microsoft")
    return rewriter.notifyMatchFailure(op, "domain must be com.microsoft");

  auto ctxOrFailure = getContextArg(op, rewriter);
  if (mlir::failed(ctxOrFailure))
    return rewriter.notifyMatchFailure(op, "missing context argument");
  mlir::Value context = *ctxOrFailure;

  mlir::Location loc = op->getLoc();
  const size_t numOps = op->getNumOperands();

  if (numOps < 6)
    return rewriter.notifyMatchFailure(
        op, "PagedAttention requires at least 6 operands");

  // Helper: get optional operand.
  auto getOptionalOperand = [&](size_t idx) -> mlir::Value {
    if (idx >= numOps)
      return nullptr;
    mlir::Value val = op->getOperand(idx);
    if (mlir::isa<mlir::NoneType>(val.getType()))
      return nullptr;
    return val;
  };

  // Required inputs (indices 0-5).
  mlir::Value query = op->getOperand(0);
  mlir::Value keyCache = op->getOperand(1);
  mlir::Value valueCache = op->getOperand(2);
  mlir::Value blockTable = op->getOperand(3);
  mlir::Value slotMapping = op->getOperand(4);
  mlir::Value seqLens = op->getOperand(5);

  // Optional inputs.
  mlir::Value key = getOptionalOperand(6);
  mlir::Value value = getOptionalOperand(7);
  mlir::Value cosCache = getOptionalOperand(8);
  mlir::Value sinCache = getOptionalOperand(9);

  // Attributes.
  auto numHeadsAttrOnnx = op->getAttrOfType<mlir::IntegerAttr>("num_heads");
  if (!numHeadsAttrOnnx)
    return rewriter.notifyMatchFailure(op, "missing num_heads attribute");
  const int64_t numHeads = numHeadsAttrOnnx.getValue().getSExtValue();

  auto kvNumHeadsAttrOnnx =
      op->getAttrOfType<mlir::IntegerAttr>("kv_num_heads");
  if (!kvNumHeadsAttrOnnx)
    return rewriter.notifyMatchFailure(op, "missing kv_num_heads attribute");

  auto getI64Attr = [&](const char *name, int64_t def) -> mlir::IntegerAttr {
    auto a = op->getAttrOfType<mlir::IntegerAttr>(name);
    return a ? rewriter.getI64IntegerAttr(a.getValue().getSExtValue())
             : rewriter.getI64IntegerAttr(def);
  };
  auto getF32Attr = [&](const char *name, float def) -> mlir::FloatAttr {
    auto a = op->getAttrOfType<mlir::FloatAttr>(name);
    return a ? a : rewriter.getF32FloatAttr(def);
  };

  // Derive default scale from query shape.
  float defaultScale = 0.0f;
  if (auto qt = mlir::dyn_cast<mlir::RankedTensorType>(query.getType())) {
    if (qt.hasRank() && qt.getRank() >= 2) {
      int64_t hidden = qt.getDimSize(1);
      if (hidden != mlir::ShapedType::kDynamic && numHeads > 0) {
        int64_t headDim = hidden / numHeads;
        defaultScale = 1.0f / std::sqrt(static_cast<float>(headDim));
      }
    }
  }

  auto numHeadsAttr =
      rewriter.getI64IntegerAttr(numHeads);
  auto kvNumHeadsAttr = rewriter.getI64IntegerAttr(
      kvNumHeadsAttrOnnx.getValue().getSExtValue());
  auto scaleAttr = getF32Attr("scale", defaultScale);
  auto doRotaryAttr = getI64Attr("do_rotary", 0);

  // Shape attributes derived from tensors.
  // key_cache shape: [num_blocks, block_size, G, D]
  auto keyCacheType =
      mlir::dyn_cast<mlir::RankedTensorType>(keyCache.getType());
  if (!keyCacheType || keyCacheType.getRank() != 4)
    return rewriter.notifyMatchFailure(
        op, "key_cache must be a rank-4 tensor [num_blocks, bs, G, D]");

  const int64_t blockSize = keyCacheType.getDimSize(1);
  const int64_t headDim = keyCacheType.getDimSize(3);

  // num_tokens from query shape.
  int64_t numTokens = mlir::ShapedType::kDynamic;
  if (auto qt = mlir::dyn_cast<mlir::RankedTensorType>(query.getType()))
    if (qt.hasRank() && qt.getRank() >= 1)
      numTokens = qt.getDimSize(0);

  // batch_size from seqLens shape.
  int64_t batchSize = mlir::ShapedType::kDynamic;
  if (auto st = mlir::dyn_cast<mlir::RankedTensorType>(seqLens.getType()))
    if (st.hasRank() && st.getRank() >= 1)
      batchSize = st.getDimSize(0);

  // max_blocks_per_seq from block_table shape.
  int64_t maxBlocksPerSeq = mlir::ShapedType::kDynamic;
  if (auto bt = mlir::dyn_cast<mlir::RankedTensorType>(blockTable.getType()))
    if (bt.hasRank() && bt.getRank() >= 2)
      maxBlocksPerSeq = bt.getDimSize(1);

  // element_size_bytes from query dtype.
  int64_t elemSize = 2; // default fp16
  if (auto qt = mlir::dyn_cast<mlir::RankedTensorType>(query.getType())) {
    if (qt.getElementType().isF32())
      elemSize = 4;
  }

  auto numTokensAttr = rewriter.getI64IntegerAttr(numTokens);
  auto batchSizeAttr = rewriter.getI64IntegerAttr(batchSize);
  auto headDimAttr = rewriter.getI64IntegerAttr(headDim);
  auto elemSizeAttr = rewriter.getI64IntegerAttr(elemSize);
  auto blockSizeAttr = rewriter.getI64IntegerAttr(blockSize);
  auto maxBlocksAttr = rewriter.getI64IntegerAttr(maxBlocksPerSeq);

  // Output type: [num_tokens, H*D].
  if (op->getNumResults() != 1)
    return rewriter.notifyMatchFailure(op,
                                       "PagedAttention expects exactly 1 output");
  auto outputType =
      mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());
  mlir::Value outputInit = createEmptyTensor(rewriter, loc, outputType, query);

  // Build operands: context + required inputs + optional inputs + output.
  mlir::SmallVector<mlir::Value> operands;
  operands.push_back(context);
  operands.push_back(query);
  if (key)
    operands.push_back(key);
  if (value)
    operands.push_back(value);
  operands.push_back(keyCache);
  operands.push_back(valueCache);
  operands.push_back(blockTable);
  operands.push_back(slotMapping);
  operands.push_back(seqLens);
  operands.push_back(outputInit);
  if (cosCache)
    operands.push_back(cosCache);
  if (sinCache)
    operands.push_back(sinCache);

  // Attributes.
  mlir::SmallVector<mlir::NamedAttribute> attrs;
  attrs.push_back(rewriter.getNamedAttr("num_heads", numHeadsAttr));
  attrs.push_back(rewriter.getNamedAttr("kv_num_heads", kvNumHeadsAttr));
  attrs.push_back(rewriter.getNamedAttr("scale", scaleAttr));
  attrs.push_back(rewriter.getNamedAttr("do_rotary", doRotaryAttr));
  attrs.push_back(rewriter.getNamedAttr("num_tokens", numTokensAttr));
  attrs.push_back(rewriter.getNamedAttr("batch_size", batchSizeAttr));
  attrs.push_back(rewriter.getNamedAttr("head_dim", headDimAttr));
  attrs.push_back(rewriter.getNamedAttr("element_size_bytes", elemSizeAttr));
  attrs.push_back(rewriter.getNamedAttr("block_size", blockSizeAttr));
  attrs.push_back(rewriter.getNamedAttr("max_blocks_per_seq", maxBlocksAttr));

  // operand_segment_sizes for AttrSizedOperandSegments.
  // Order matches Hip_PagedAttentionOp argument order:
  // ctx(1), query(1), key(0|1), value(0|1), key_cache(1), value_cache(1),
  // block_table(1), slot_mapping(1), sequence_lengths(1), output(1),
  // cos_cache(0|1), sin_cache(0|1)
  llvm::SmallVector<int32_t> segSizes = {
      1,             // ctx
      1,             // query
      key ? 1 : 0,   // key
      value ? 1 : 0, // value
      1,             // key_cache
      1,             // value_cache
      1,             // block_table
      1,             // slot_mapping
      1,             // sequence_lengths
      1,             // output
      cosCache ? 1 : 0, // cos_cache
      sinCache ? 1 : 0, // sin_cache
  };
  attrs.push_back(rewriter.getNamedAttr(
      "operand_segment_sizes", rewriter.getDenseI32ArrayAttr(segSizes)));

  auto state = mlir::OperationState(loc, "hip.paged_attention");
  state.addOperands(operands);
  state.addAttributes(attrs);
  // hip.paged_attention produces no results (DPS out-args carry the output).

  rewriter.create(state);

  // Replace original op results with outputInit (the DPS output buffer).
  rewriter.replaceOp(op, {outputInit});
  return mlir::success();
}

} // namespace

void populatePagedAttentionConversionPatterns(mlir::RewritePatternSet &patterns,
                                              mlir::MLIRContext *ctx) {
  patterns.add<PagedAttentionToHip>(ctx);
}

} // namespace hip
} // namespace mlir
