/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- HipGqaBuilder.cpp - Shared hip.gqa op builder ---------------------===//
//
// Builds a hip.gqa op with the AttrSizedOperandSegments layout used by the
// Whisper attention lowering paths.  Used by both MultiHeadAttentionConversion
// (Whisper decoder self-attn + cross-attn) and AttentionConversion (Whisper
// encoder fused-QKV self-attn) so the 19-slot operand-segments table and
// OperationState boilerplate live in exactly one place.
//
// The two callers map onto the same builder by passing nullptr for the
// past_key / past_value operands when no KV cache is involved (encoder /
// cross-attn).  See callers for the dispatch logic that decides which
// no_causal / seqlens_k / total_seq_len values to feed in.
//
//===----------------------------------------------------------------------===//

#include "OnnxToHipUtils.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir {
namespace hip {

mlir::LogicalResult buildHipGqaCall(
    mlir::Operation *op, mlir::PatternRewriter &rewriter, mlir::Value context,
    mlir::Value query, mlir::Value key, mlir::Value value, mlir::Value pastKey,
    mlir::Value pastValue, mlir::Value seqlensK, mlir::Value totalSeqLen,
    int64_t numHeads, float scale, bool noCausal,
    mlir::RankedTensorType outputType, mlir::RankedTensorType presentKeyType,
    mlir::RankedTensorType presentValueType, mlir::Value attentionBias) {
  mlir::Location loc = op->getLoc();

  // DPS init buffers. present_* follows past_* when a cache is concatenated
  // in place. With no past, a rank-4 BNSH buffer is filled from a rank-3 BSH
  // query: batch is dim 0 and sequence is dim 2. Same-index tensor.dim would
  // read the query's hidden size as the sequence extent.
  auto makePresentInit = [&](mlir::RankedTensorType presentType,
                             mlir::Value past) -> mlir::FailureOr<mlir::Value> {
    mlir::Value source = past ? past : query;
    auto sourceType = mlir::dyn_cast<mlir::RankedTensorType>(source.getType());
    const bool bshQuery = !past && sourceType && sourceType.getRank() == 3 &&
                          presentType.getRank() == 4;
    llvm::SmallVector<mlir::Value> dynSizes;
    for (int64_t dimIdx : llvm::seq<int64_t>(presentType.getRank())) {
      if (!presentType.isDynamicDim(dimIdx))
        continue;
      int64_t srcDim = dimIdx;
      if (bshQuery) {
        if (dimIdx == 0)
          srcDim = 0;
        else if (dimIdx == 2)
          srcDim = 1;
        else
          return mlir::failure();
      }
      dynSizes.push_back(
          mlir::tensor::DimOp::create(rewriter, loc, source, srcDim));
    }
    return mlir::Value(
        mlir::tensor::EmptyOp::create(rewriter, loc, presentType.getShape(),
                                      presentType.getElementType(), dynSizes));
  };

  mlir::Value outputInit = createEmptyTensor(rewriter, loc, outputType, query);
  auto presentKeyInitOr = makePresentInit(presentKeyType, pastKey);
  auto presentValueInitOr = makePresentInit(presentValueType, pastValue);
  if (mlir::failed(presentKeyInitOr) || mlir::failed(presentValueInitOr))
    return mlir::failure();
  mlir::Value presentKeyInit = *presentKeyInitOr;
  mlir::Value presentValueInit = *presentValueInitOr;

  llvm::SmallVector<mlir::Type> resultTypes = {outputType, presentKeyType,
                                               presentValueType};

  // Operand order matches Hip_GqaOp's AttrSizedOperandSegments layout.
  llvm::SmallVector<mlir::Value> operands;
  operands.push_back(context);
  operands.push_back(query);
  operands.push_back(key);
  operands.push_back(value);
  if (pastKey)
    operands.push_back(pastKey);
  if (pastValue)
    operands.push_back(pastValue);
  operands.push_back(seqlensK);
  operands.push_back(totalSeqLen);
  if (attentionBias)
    operands.push_back(attentionBias);
  operands.push_back(outputInit);
  operands.push_back(presentKeyInit);
  operands.push_back(presentValueInit);

  // segmentSizes order MUST match HipOps.td Hip_GqaOp argument order.
  llvm::SmallVector<int32_t> segmentSizes;
  segmentSizes.push_back(1);                     // ctx
  segmentSizes.push_back(1);                     // query
  segmentSizes.push_back(1);                     // key
  segmentSizes.push_back(1);                     // value
  segmentSizes.push_back(pastKey ? 1 : 0);       // past_key
  segmentSizes.push_back(pastValue ? 1 : 0);     // past_value
  segmentSizes.push_back(1);                     // seqlens_k
  segmentSizes.push_back(1);                     // total_seq_len
  segmentSizes.push_back(0);                     // cos_cache
  segmentSizes.push_back(0);                     // sin_cache
  segmentSizes.push_back(0);                     // position_ids
  segmentSizes.push_back(attentionBias ? 1 : 0); // attention_bias
  segmentSizes.push_back(0);                     // head_sink
  segmentSizes.push_back(0);                     // k_scale
  segmentSizes.push_back(0);                     // v_scale
  segmentSizes.push_back(1);                     // output
  segmentSizes.push_back(1);                     // present_key
  segmentSizes.push_back(1);                     // present_value
  segmentSizes.push_back(0);                     // output_qk

  llvm::SmallVector<mlir::NamedAttribute> attrs;
  attrs.push_back(
      rewriter.getNamedAttr("num_heads", rewriter.getI64IntegerAttr(numHeads)));
  attrs.push_back(rewriter.getNamedAttr("kv_num_heads",
                                        rewriter.getI64IntegerAttr(numHeads)));
  attrs.push_back(
      rewriter.getNamedAttr("scale", rewriter.getF32FloatAttr(scale)));
  attrs.push_back(
      rewriter.getNamedAttr("no_causal", rewriter.getBoolAttr(noCausal)));

  mlir::OperationState gqaState(loc, "hip.gqa");
  gqaState.addOperands(operands);
  gqaState.addTypes(resultTypes);
  gqaState.addAttributes(attrs);
  gqaState.addAttribute("operand_segment_sizes",
                        rewriter.getDenseI32ArrayAttr(segmentSizes));
  mlir::Operation *gqaOp = rewriter.create(gqaState);

  // Replace by positional mapping: the original op's results [0..N) become
  // the hip.gqa's results [0..N).  Callers that don't expose present_* (e.g.
  // encoder Attention with a single output, or cross-attn with one output)
  // simply have fewer original results and the trailing hip.gqa results are
  // dropped on the floor.
  llvm::SmallVector<mlir::Value> replacements;
  for (size_t i = 0; i < op->getNumResults(); ++i)
    replacements.push_back(gqaOp->getResult(i));
  rewriter.replaceOp(op, replacements);
  return mlir::success();
}

} // namespace hip
} // namespace mlir
