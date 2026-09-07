/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/APInt.h"

namespace mlir {
namespace hip {
namespace {

/// onnx.Attention (ONNX opset 23/24) -> hip.gqa
///
/// General lowering of the standard ONNX scaled-dot-product attention op onto
/// the existing hip.gqa runtime.  RoPE is expected to be applied upstream
/// (separate onnx.RotaryEmbedding nodes).  This pattern is intentionally NOT
/// model-specific — it covers every combination of the ONNX Attention operator
/// that the hip.gqa runtime can service:
///
///   * Q/K/V rank-3 [B, S, hidden] (num_heads split by the q/kv_num_heads
///     attributes) AND rank-4 BNSH [B, N, S, head] (num_heads inferred from
///     the shape when the attributes are absent).  Rank-4 inputs are
///     transposed+flattened to the rank-3 BSHD layout the runtime consumes,
///     and the rank-3 output is expanded+transposed back to rank-4.
///   * 1 output (Y only) OR 3 outputs (Y, present_key, present_value).  When
///     the ONNX op does not request present KV, internal DPS present buffers
///     are synthesized (hip.gqa always writes present_key/present_value); their
///     results are dropped on the floor.
///   * is_causal = 1 (built-in causal mask) with OR without an external
///   additive
///     mask; is_causal = 0 with an external additive mask (threaded through the
///     hip.gqa attention_bias operand); is_causal = 0 with no mask
///     (bidirectional / encoder attention).
///   * With OR without a past KV cache (BNSH past_key/past_value).
///
/// Runtime causal/mask contract (see gqa.cpp), mirroring the ONNX Attention
/// reference (scores += attn_mask; then, if is_causal, mask the upper
/// triangle):
///   - no_causal = (is_causal == 0). The built-in causal mask is applied
///     whenever `!no_causal` (is_causal == 1), INDEPENDENTLY of whether an
///     additive mask is present -- so is_causal=1 + mask applies BOTH (the mask
///     is added, then the causal triangle is masked). This is idempotent when
///     the mask already encodes causal (the common HF export, where the mask is
///     GreaterOrEqual(q_pos,k_pos) & padding & sliding-window baked to 0/-inf)
///     and correct when it does not (e.g. a padding-only mask + is_causal).
///   - An external float mask is threaded through the `attention_bias` operand
///     and is always added to the scores.
///   - `no_causal && !attention_bias` selects the bidirectional no-past path.
/// Combinations the runtime cannot express are rejected loudly rather than
/// silently miscompiled (see the guards below).
///
/// Not yet supported (rejected with a clear message): the 4th qk_matmul_output
/// result, the opset-24 nonpad_kv_seqlen external-cache input, and boolean
/// attn_mask (the runtime's attention_bias is additive-float only).
///
/// Mask-extent assumption: the additive mask is passed to the runtime verbatim;
/// its batch/head dims broadcast (extent 1 -> all), but its q/kv extents must
/// equal the query seq length and total (past+current) KV length respectively.
/// The runtime bias kernel does NOT broadcast the q dim (a mask with q==1 while
/// sq>1 would be read out of range), so exports that emit a full mask (q==S,
/// the common case) are correct; a q-broadcast mask is not modelled.
///
/// Before (Gemma-style decoder layer, is_causal=0, external fp16 mask, 3 out):
///   %y, %pk, %pv = onnx.Attention(%q, %k, %v, %mask, %past_k, %past_v)
///       {q_num_heads = 16, kv_num_heads = 8, is_causal = 0, scale = 1.0}
///
/// After:
///   %cur       = tensor.dim %q, %c1        // current KV tokens (== query seq)
///   %past      = tensor.dim %past_k, %c2   // valid past KV length (0 prefill)
///   %tot       = arith.addi %past, %cur    // present seq = past + current
///   %seqlens_k = tensor.from_elements (%tot - 1)  : tensor<1xi32>
///   %total_seq = tensor.from_elements %tot        : tensor<i32>
///   %pk_init   = tensor.empty(%B, %tot)    // present sized to %tot, NOT %past
///   %y, %pk, %pv = hip.gqa(%ctx)
///       ins(%q, %k, %v, %past_k, %past_v, %seqlens_k, %total_seq, %mask)
///       outs(%y_init, %pk_init, %pv_init)
///       {num_heads = 16, kv_num_heads = 8, scale = 1.0, no_causal = true}
///
/// present KV is concat(past, current) along the seq axis, so its seq extent is
/// past_seq + current_seq.  A fresh prefill arrives with an EMPTY past
/// (past_seq == 0): sizing present from dim(past_key, 2) alone would collapse
/// it to a zero-length buffer, the output allocator would then hand the runtime
/// a null present_key/present_value, and wrap_group_query_attention would
/// reject the call and leave the attention output zero-filled.
struct OnnxAttentionToHip : public mlir::RewritePattern {
  OnnxAttentionToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Attention", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override;
};

mlir::LogicalResult
OnnxAttentionToHip::matchAndRewrite(mlir::Operation *op,
                                    mlir::PatternRewriter &rewriter) const {
  mlir::Location loc = op->getLoc();

  size_t numOps = op->getNumOperands();
  if (numOps < 3 || numOps > 7)
    return rewriter.notifyMatchFailure(
        op, "onnx.Attention expects 3-7 operands (Q,K,V,[mask],[past_k],"
            "[past_v],[nonpad_kv_seqlen])");

  auto isLive = [](mlir::Value v) {
    return v && !mlir::isa<mlir::NoneType>(v.getType());
  };

  if (numOps == 7 && isLive(op->getOperand(6)))
    return rewriter.notifyMatchFailure(
        op, "nonpad_kv_seqlen (external-cache mode) not supported yet");

  mlir::Value query = op->getOperand(0);
  mlir::Value key = op->getOperand(1);
  mlir::Value value = op->getOperand(2);
  mlir::Value attnMask = numOps > 3 ? op->getOperand(3) : mlir::Value();
  mlir::Value pastKey = numOps > 4 ? op->getOperand(4) : mlir::Value();
  mlir::Value pastValue = numOps > 5 ? op->getOperand(5) : mlir::Value();

  if (attnMask && !isLive(attnMask))
    attnMask = nullptr;
  if (pastKey && !isLive(pastKey))
    pastKey = nullptr;
  if (pastValue && !isLive(pastValue))
    pastValue = nullptr;

  if ((pastKey == nullptr) != (pastValue == nullptr))
    return rewriter.notifyMatchFailure(
        op, "past_key and past_value must both be present or both absent");

  // === Input types / rank ===

  auto queryType = mlir::dyn_cast<mlir::RankedTensorType>(query.getType());
  auto keyType = mlir::dyn_cast<mlir::RankedTensorType>(key.getType());
  auto valueType = mlir::dyn_cast<mlir::RankedTensorType>(value.getType());
  if (!queryType || !keyType || !valueType)
    return rewriter.notifyMatchFailure(op, "Q/K/V must be ranked tensors");

  const int64_t rank = queryType.getRank();
  if (rank != 3 && rank != 4)
    return rewriter.notifyMatchFailure(
        op, "Q must be rank-3 [B,S,hidden] or rank-4 BNSH [B,N,S,head]");
  if (keyType.getRank() != rank || valueType.getRank() != rank)
    return rewriter.notifyMatchFailure(op, "Q/K/V must share the same rank");
  const bool rank4 = (rank == 4);

  // === Head counts ===
  //
  // ONNX requires q_num_heads/kv_num_heads for rank-3 inputs; for rank-4 inputs
  // they are optional and derivable from the (static) head dimension.
  auto qNumHeadsAttr = op->getAttrOfType<mlir::IntegerAttr>("q_num_heads");
  auto kvNumHeadsAttr = op->getAttrOfType<mlir::IntegerAttr>("kv_num_heads");
  int64_t qNumHeads = 0;
  int64_t kvNumHeads = 0;
  if (qNumHeadsAttr && kvNumHeadsAttr) {
    qNumHeads = qNumHeadsAttr.getValue().getSExtValue();
    kvNumHeads = kvNumHeadsAttr.getValue().getSExtValue();
  } else if (rank4 && !queryType.isDynamicDim(1) && !keyType.isDynamicDim(1)) {
    qNumHeads = queryType.getDimSize(1);
    kvNumHeads = keyType.getDimSize(1);
  } else {
    return rewriter.notifyMatchFailure(
        op, "missing q_num_heads/kv_num_heads (required for rank-3 inputs, and "
            "for rank-4 inputs with a dynamic head dimension)");
  }
  if (qNumHeads <= 0 || kvNumHeads <= 0)
    return rewriter.notifyMatchFailure(op, "head counts must be > 0");
  if (qNumHeads % kvNumHeads != 0)
    return rewriter.notifyMatchFailure(
        op, "q_num_heads must be divisible by kv_num_heads");

  auto getI64 = [&](const char *name, int64_t defaultVal) {
    auto a = op->getAttrOfType<mlir::IntegerAttr>(name);
    return a ? a.getValue().getSExtValue() : defaultVal;
  };
  auto getFloat = [&](const char *name, float defaultVal) {
    auto a = op->getAttrOfType<mlir::FloatAttr>(name);
    return a ? a.getValue().convertToFloat() : defaultVal;
  };

  int64_t isCausal = getI64("is_causal", 0);
  float scale = getFloat("scale", 0.0f);
  float softcap = getFloat("softcap", 0.0f);

  // === Results ===

  size_t numResults = op->getNumResults();
  if (numResults < 1 || numResults > 4)
    return rewriter.notifyMatchFailure(
        op, "onnx.Attention expects 1-4 results (Y,[present_k],[present_v],"
            "[qk])");
  if (numResults > 3)
    return rewriter.notifyMatchFailure(op,
                                       "qk_matmul_output not supported yet");

  // === Causal / mask contract ===
  //
  // Runtime (gqa.cpp) mirrors the ONNX Attention reference: the external float
  // mask (attention_bias) is ALWAYS added to the scores, and the built-in
  // causal mask is applied whenever !no_causal (== is_causal==1), regardless of
  // whether a mask is present. So is_causal=1 + mask applies BOTH; is_causal=0
  // + mask applies only the mask; no_causal && no mask selects the
  // bidirectional no-past path.
  if (attnMask) {
    auto maskType = mlir::dyn_cast<mlir::RankedTensorType>(attnMask.getType());
    if (!maskType)
      return rewriter.notifyMatchFailure(op, "attn_mask must be a ranked "
                                             "tensor");
    // The runtime's attention_bias is additive float; a boolean keep/drop mask
    // would need conversion to additive 0/-inf, which is not modelled here.
    if (maskType.getElementType().isInteger(1))
      return rewriter.notifyMatchFailure(
          op, "boolean attn_mask not supported (runtime attention_bias is "
              "additive-float only)");
  }
  // is_causal=0 with no mask is bidirectional attention, which the runtime
  // services only as a no-past path (bidirectional_no_past ignores past KV).
  if (isCausal == 0 && !attnMask && pastKey)
    return rewriter.notifyMatchFailure(
        op, "is_causal=0 without an attn_mask but WITH a past KV cache is "
            "ambiguous (bidirectional attention has no causal cache)");

  // no_causal = (is_causal == 0). is_causal=1 => built-in causal is applied by
  // the runtime IN ADDITION to any additive mask; is_causal=0 => causal is
  // skipped and the mask (if any) carries all masking.
  bool noCausal = (isCausal == 0);

  auto ctxOrFailure = getContextArg(op, rewriter);
  if (mlir::failed(ctxOrFailure))
    return rewriter.notifyMatchFailure(op, "missing context argument");
  mlir::Value context = *ctxOrFailure;

  auto i32Ty = rewriter.getIntegerType(32);
  const int64_t kDyn = mlir::ShapedType::kDynamic;

  // === Rank-4 BNSH -> rank-3 BSHD helper ===============================
  //
  //   %t = hip.transpose %v {perm=[0,2,1,3]}   // [B,N,S,D] -> [B,S,N,D]
  //   %r = tensor.collapse_shape %t [[0],[1],[2,3]]  // -> [B,S,N*D]
  //
  // N and D are architecture constants (static); B and S may be dynamic.  The
  // transpose materialises a dense [B,S,N,D] buffer so collapsing the trailing
  // (N,D) dims is a pure metadata op.
  auto bnsdToBshd = [&](mlir::Value v) -> mlir::FailureOr<mlir::Value> {
    auto t = mlir::cast<mlir::RankedTensorType>(v.getType());
    int64_t B = t.getDimSize(0), N = t.getDimSize(1), S = t.getDimSize(2),
            D = t.getDimSize(3);
    if (N == kDyn || D == kDyn)
      return rewriter.notifyMatchFailure(
          op, "rank-4 attention requires static num_heads and head_dim");
    mlir::Type e = t.getElementType();
    auto transTy = mlir::RankedTensorType::get({B, S, N, D}, e);
    llvm::SmallVector<mlir::Value> dyn;
    if (B == kDyn)
      dyn.push_back(mlir::tensor::DimOp::create(rewriter, loc, v, 0));
    if (S == kDyn)
      dyn.push_back(mlir::tensor::DimOp::create(rewriter, loc, v, 2));
    mlir::Value tinit = mlir::tensor::EmptyOp::create(
        rewriter, loc, transTy.getShape(), e, dyn);
    auto transOp =
        mlir::hip::TransposeOp::create(rewriter, loc, context, v, tinit,
                                       rewriter.getI64ArrayAttr({0, 2, 1, 3}));
    mlir::Value transposed = transOp->getResult(0);
    auto collTy = mlir::RankedTensorType::get({B, S, N * D}, e);
    llvm::SmallVector<mlir::ReassociationIndices> re = {{0}, {1}, {2, 3}};
    auto collapseOp = mlir::tensor::CollapseShapeOp::create(
        rewriter, loc, collTy, transposed, re);
    return mlir::Value(collapseOp.getResult());
  };

  // === rank-3 BSHD [B,S,N*D] -> rank-4 BNSH [B,N,S,D] helper ===========
  //
  //   %e = tensor.expand_shape %v [[0],[1],[2,3]]   // [B,S,H] -> [B,S,N,D]
  //   %r = hip.transpose %e {perm=[0,2,1,3]}        // -> [B,N,S,D]
  auto bshdToBnsd = [&](mlir::Value v,
                        mlir::RankedTensorType tt) -> mlir::Value {
    int64_t B = tt.getDimSize(0), N = tt.getDimSize(1), S = tt.getDimSize(2),
            D = tt.getDimSize(3);
    mlir::Type e = tt.getElementType();
    auto expTy = mlir::RankedTensorType::get({B, S, N, D}, e);
    llvm::SmallVector<mlir::OpFoldResult> outShape;
    outShape.push_back(
        B == kDyn
            ? mlir::OpFoldResult(
                  mlir::tensor::DimOp::create(rewriter, loc, v, 0).getResult())
            : mlir::OpFoldResult(rewriter.getIndexAttr(B)));
    outShape.push_back(
        S == kDyn
            ? mlir::OpFoldResult(
                  mlir::tensor::DimOp::create(rewriter, loc, v, 1).getResult())
            : mlir::OpFoldResult(rewriter.getIndexAttr(S)));
    outShape.push_back(mlir::OpFoldResult(rewriter.getIndexAttr(N)));
    outShape.push_back(mlir::OpFoldResult(rewriter.getIndexAttr(D)));
    llvm::SmallVector<mlir::ReassociationIndices> re = {{0}, {1}, {2, 3}};
    auto expandOp = mlir::tensor::ExpandShapeOp::create(rewriter, loc, expTy, v,
                                                        re, outShape);
    mlir::Value expanded = expandOp.getResult();
    llvm::SmallVector<mlir::Value> dyn;
    if (B == kDyn)
      dyn.push_back(mlir::tensor::DimOp::create(rewriter, loc, expanded, 0));
    if (S == kDyn)
      dyn.push_back(mlir::tensor::DimOp::create(rewriter, loc, expanded, 1));
    mlir::Value tinit =
        mlir::tensor::EmptyOp::create(rewriter, loc, tt.getShape(), e, dyn);
    auto transOp =
        mlir::hip::TransposeOp::create(rewriter, loc, context, expanded, tinit,
                                       rewriter.getI64ArrayAttr({0, 2, 1, 3}));
    return transOp->getResult(0);
  };

  // Flatten rank-4 Q/K/V to the rank-3 BSHD layout the runtime consumes.
  mlir::Value q3 = query, k3 = key, v3 = value;
  if (rank4) {
    auto q3Or = bnsdToBshd(query);
    auto k3Or = bnsdToBshd(key);
    auto v3Or = bnsdToBshd(value);
    if (mlir::failed(q3Or) || mlir::failed(k3Or) || mlir::failed(v3Or))
      return mlir::failure();
    q3 = *q3Or;
    k3 = *k3Or;
    v3 = *v3Or;
  }

  // === seqlens_k / total_seq_len ======================================
  //
  // present KV = concat(past, current) along the seq axis, so the present seq
  // extent is past_seq + current_seq.
  mlir::Value curKvSeqIdx =
      mlir::tensor::DimOp::create(rewriter, loc, k3, 1).getResult();
  mlir::Value pastSeqIdx =
      pastKey
          ? mlir::tensor::DimOp::create(rewriter, loc, pastKey, 2).getResult()
          : mlir::arith::ConstantIndexOp::create(rewriter, loc, 0).getResult();
  mlir::Value totalKvIdx =
      mlir::arith::AddIOp::create(rewriter, loc, pastSeqIdx, curKvSeqIdx)
          .getResult();

  // seqlens_k[b] = total_seq - 1 (ORT GQA convention total_seq = seqlens_k + 1;
  // the runtime derives past_len = total_seq - sq). Ignored by the runtime on
  // the bidirectional no-past path (no_causal with no past operand, where
  // total_seq is just the KV extent), but computed uniformly here.
  mlir::Value oneIdx =
      mlir::arith::ConstantIndexOp::create(rewriter, loc, 1).getResult();
  mlir::Value totalKvM1Idx =
      mlir::arith::SubIOp::create(rewriter, loc, totalKvIdx, oneIdx)
          .getResult();
  mlir::Value seqlensKI32 =
      mlir::arith::IndexCastOp::create(rewriter, loc, i32Ty, totalKvM1Idx);
  auto seqlensKType = mlir::RankedTensorType::get({1}, i32Ty);
  mlir::Value seqlensK = mlir::tensor::FromElementsOp::create(
      rewriter, loc, seqlensKType, mlir::ValueRange{seqlensKI32});

  // === hip.gqa output type + init =====================================
  //
  // hip.gqa writes a rank-3 [B, S, num_heads*v_head_dim] output. For rank-3
  // inputs this is exactly the op's result-0 type; for rank-4 inputs the op's
  // result-0 is [B, N, S, v_head_dim], so build the flattened rank-3 type here
  // and expand+transpose the runtime output back afterwards.  The output init
  // is emitted before the present inits so the DPS operands materialise in the
  // canonical output -> present_key -> present_value order.
  auto res0Type =
      mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());
  mlir::RankedTensorType gqaOutType;
  if (rank4) {
    int64_t B = res0Type.getDimSize(0), N = res0Type.getDimSize(1),
            S = res0Type.getDimSize(2), Dv = res0Type.getDimSize(3);
    if (N == kDyn || Dv == kDyn)
      return rewriter.notifyMatchFailure(
          op, "rank-4 attention requires static num_heads and head_dim");
    gqaOutType =
        mlir::RankedTensorType::get({B, S, N * Dv}, res0Type.getElementType());
  } else {
    gqaOutType = res0Type;
  }
  mlir::Value outputInit = createEmptyTensor(rewriter, loc, gqaOutType, q3);

  // === present_key / present_value types ==============================
  //
  // present is always rank-4 BNSH [B, kv_heads, total_seq, head_dim] (ONNX
  // Attention emits present in BNSH regardless of the Q/K/V rank). When the op
  // requests the present outputs, use their result types; otherwise synthesize
  // internal DPS buffers (hip.gqa always writes present_key/present_value).
  //
  // Synthesis needs the (static) per-head dims: for rank-3 K/V the head_dim is
  // hidden/kv_num_heads; for rank-4 K/V it is the last dim directly.
  auto perHeadDim = [&](mlir::RankedTensorType t) -> int64_t {
    if (rank4)
      return t.getDimSize(3);
    int64_t hidden = t.getDimSize(2);
    if (hidden == kDyn)
      return kDyn;
    return hidden / kvNumHeads;
  };
  int64_t kHeadDim = perHeadDim(keyType);
  int64_t vHeadDim = perHeadDim(valueType);
  int64_t batchDim = queryType.getDimSize(0);

  auto synthPresent = [&](int64_t headDim,
                          mlir::Type elem) -> mlir::RankedTensorType {
    return mlir::RankedTensorType::get({batchDim, kvNumHeads, kDyn, headDim},
                                       elem);
  };

  mlir::RankedTensorType presentKeyType =
      numResults >= 2
          ? mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(1).getType())
          : synthPresent(kHeadDim, keyType.getElementType());
  mlir::RankedTensorType presentValueType =
      numResults >= 3
          ? mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(2).getType())
          : synthPresent(vHeadDim, valueType.getElementType());

  if (!presentKeyType || !presentValueType || presentKeyType.getRank() != 4 ||
      presentValueType.getRank() != 4)
    return rewriter.notifyMatchFailure(
        op, "present_key/value must be rank-4 BNSH tensors");
  if (presentKeyType.getDimSize(3) == kDyn ||
      presentValueType.getDimSize(3) == kDyn)
    return rewriter.notifyMatchFailure(
        op,
        "present_key/value head_dim must be static (architecture constant)");

  // total_seq_len scalar = present KV buffer capacity. When the present seq dim
  // is static (provided result type), emit a constant; otherwise emit the
  // runtime past+current total.
  mlir::Value totalSeqLen;
  auto totalSeqLenType = mlir::RankedTensorType::get({}, i32Ty);
  if (!presentKeyType.isDynamicDim(2)) {
    auto attr = mlir::DenseElementsAttr::get(
        totalSeqLenType,
        llvm::APInt(32, presentKeyType.getDimSize(2), /*isSigned=*/true));
    totalSeqLen = mlir::arith::ConstantOp::create(rewriter, loc, attr);
  } else {
    mlir::Value totalKvI32 =
        mlir::arith::IndexCastOp::create(rewriter, loc, i32Ty, totalKvIdx);
    totalSeqLen = mlir::tensor::FromElementsOp::create(
        rewriter, loc, totalSeqLenType, mlir::ValueRange{totalKvI32});
  }

  // Build the present_key/present_value DPS init buffers with seq dim = total
  // (past + current). present is rank-4 BNSH [batch, kv_heads, seq, head_dim];
  // batch (dim 0) and seq (dim 2) are the only dynamic dims in practice.
  auto buildPresentInit =
      [&](mlir::RankedTensorType t) -> mlir::FailureOr<mlir::Value> {
    llvm::SmallVector<mlir::Value> dynSizes;
    for (int64_t dimIdx : llvm::seq<int64_t>(t.getRank())) {
      if (!t.isDynamicDim(dimIdx))
        continue;
      if (dimIdx == 2)
        dynSizes.push_back(totalKvIdx);
      else if (dimIdx == 0)
        dynSizes.push_back(
            mlir::tensor::DimOp::create(rewriter, loc, query, 0).getResult());
      else if (pastKey)
        dynSizes.push_back(
            mlir::tensor::DimOp::create(rewriter, loc, pastKey, dimIdx)
                .getResult());
      else
        return mlir::failure();
    }
    return mlir::Value(mlir::tensor::EmptyOp::create(
        rewriter, loc, t.getShape(), t.getElementType(), dynSizes));
  };

  mlir::FailureOr<mlir::Value> presentKeyInitOr =
      buildPresentInit(presentKeyType);
  mlir::FailureOr<mlir::Value> presentValueInitOr =
      buildPresentInit(presentValueType);
  if (mlir::failed(presentKeyInitOr) || mlir::failed(presentValueInitOr))
    return rewriter.notifyMatchFailure(
        op, "present_key/present_value has an unsupported dynamic dim "
            "(only batch and seq may be dynamic)");
  mlir::Value presentKeyInit = *presentKeyInitOr;
  mlir::Value presentValueInit = *presentValueInitOr;

  // === Build the hip.gqa op ===========================================

  llvm::SmallVector<mlir::Type> resultTypes = {gqaOutType, presentKeyType,
                                               presentValueType};

  llvm::SmallVector<mlir::Value> operands;
  operands.push_back(context);
  operands.push_back(q3);
  operands.push_back(k3);
  operands.push_back(v3);
  if (pastKey)
    operands.push_back(pastKey);
  if (pastValue)
    operands.push_back(pastValue);
  operands.push_back(seqlensK);
  operands.push_back(totalSeqLen);
  if (attnMask)
    operands.push_back(attnMask);
  operands.push_back(outputInit);
  operands.push_back(presentKeyInit);
  operands.push_back(presentValueInit);

  llvm::SmallVector<int32_t> segmentSizes;
  segmentSizes.push_back(1);                 // ctx
  segmentSizes.push_back(1);                 // query
  segmentSizes.push_back(1);                 // key
  segmentSizes.push_back(1);                 // value
  segmentSizes.push_back(pastKey ? 1 : 0);   // past_key
  segmentSizes.push_back(pastValue ? 1 : 0); // past_value
  segmentSizes.push_back(1);                 // seqlens_k
  segmentSizes.push_back(1);                 // total_seq_len
  segmentSizes.push_back(0);                 // cos_cache
  segmentSizes.push_back(0);                 // sin_cache
  segmentSizes.push_back(0);                 // position_ids
  segmentSizes.push_back(attnMask ? 1 : 0);  // attention_bias
  segmentSizes.push_back(0);                 // head_sink
  segmentSizes.push_back(0);                 // k_scale
  segmentSizes.push_back(0);                 // v_scale
  segmentSizes.push_back(1);                 // output
  segmentSizes.push_back(1);                 // present_key
  segmentSizes.push_back(1);                 // present_value
  segmentSizes.push_back(0);                 // output_qk

  llvm::SmallVector<mlir::NamedAttribute> attrs;
  attrs.push_back(rewriter.getNamedAttr("num_heads",
                                        rewriter.getI64IntegerAttr(qNumHeads)));
  attrs.push_back(rewriter.getNamedAttr(
      "kv_num_heads", rewriter.getI64IntegerAttr(kvNumHeads)));
  attrs.push_back(
      rewriter.getNamedAttr("scale", rewriter.getF32FloatAttr(scale)));
  attrs.push_back(
      rewriter.getNamedAttr("softcap", rewriter.getF32FloatAttr(softcap)));
  attrs.push_back(
      rewriter.getNamedAttr("no_causal", rewriter.getBoolAttr(noCausal)));
  // AttentionWindowFold recovers the window during pre-lowering from the
  // additive mask, which is the only place a pre-opset-25 export can put it,
  // and stamps the result here. Absent the stamp the op keeps hip.gqa's -1
  // default and the runtime scores the full key range, as it did before. Only a
  // positive window is forwarded, so a recovered 0 or a hand-written 0 cannot
  // be mistaken for "windowed".
  if (auto stampedWindow =
          op->getAttrOfType<mlir::IntegerAttr>("hipdnn.local_window_size")) {
    int64_t window = stampedWindow.getValue().getSExtValue();
    if (window > 0)
      attrs.push_back(rewriter.getNamedAttr(
          "local_window_size", rewriter.getI64IntegerAttr(window)));
  }

  mlir::OperationState gqaState(loc, "hip.gqa");
  gqaState.addOperands(operands);
  gqaState.addTypes(resultTypes);
  gqaState.addAttributes(attrs);
  gqaState.addAttribute("operand_segment_sizes",
                        rewriter.getDenseI32ArrayAttr(segmentSizes));
  mlir::Operation *gqaOp = rewriter.create(gqaState);

  // === Map hip.gqa results back to the onnx.Attention results ==========
  mlir::Value finalOut = gqaOp->getResult(0);
  if (rank4)
    finalOut = bshdToBnsd(finalOut, res0Type);

  llvm::SmallVector<mlir::Value> newResults;
  newResults.push_back(finalOut);
  if (numResults >= 2)
    newResults.push_back(gqaOp->getResult(1)); // present_key (BNSH)
  if (numResults >= 3)
    newResults.push_back(gqaOp->getResult(2)); // present_value (BNSH)

  rewriter.replaceOp(op, newResults);
  return mlir::success();
}

} // namespace

void populateOnnxAttentionConversionPatterns(RewritePatternSet &patterns,
                                             MLIRContext *ctx) {
  patterns.add<OnnxAttentionToHip>(ctx);
}

} // namespace hip
} // namespace mlir
