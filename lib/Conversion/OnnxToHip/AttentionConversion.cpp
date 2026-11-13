/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir {
namespace hip {
namespace {

/// onnx.Custom(com.microsoft.Attention) -> hip.gqa.
///
/// The legacy fused-QKV Attention node (Whisper encoder, CLIP text encoder,
/// BERT, diffusion mid-blocks) is rewritten to:
///
///   1. One QKV projection over the fused weight (no transpose):
///        qkv_proj   = hip.matmul(hidden, qkv_w)   : [B,S,H] @ [H,3H]
///        qkv_biased = hip.add(qkv_proj, qkv_b)
///      ORT stores the fused weight as [H, q+k+v]. When qkv_hidden_sizes is
///      absent and that weight is static [H, 3H], the three sizes are inferred.
///
///   2. Three tensor.extract_slice ops splitting the last axis into Q / K / V.
///
///   3. One hip.gqa. unidirectional=0 is bidirectional (no_causal).
///      unidirectional=1 is causal. Batch and sequence may be dynamic. A
///      rank-1 length or rank-2 padding mask_index is expanded into
///      attention_bias. Past KV, an already-built attention_bias operand, and
///      past_sequence_length stay unsupported.
///
/// Before:
///   %out = onnx.Custom(%hidden, %qkv_w, %qkv_b) {function_name = "Attention",
///            domain_name = "com.microsoft", num_heads = 1}
///   // %qkv_w : tensor<512x1536xf16>, hidden batch/sequence dynamic
/// After:
///   %qkv = hip.matmul %ctx, %hidden, %qkv_w
///   %biased = hip.add %ctx, %qkv, %qkv_b
///   %q = tensor.extract_slice %biased[0, 0, 0][batch, seq, 512][1, 1, 1]
///   ...
///   %out = hip.gqa(%ctx) ins(%q, %k, %v, %seqlens, %total_seq, ...)
///            {no_causal = true, num_heads = 1, kv_num_heads = 1}
struct AttentionToHip : public mlir::RewritePattern {
  AttentionToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Custom", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override;
};

/// Rank-0 onnx.Constant holding \p value.
mlir::Value buildRank0Constant(mlir::PatternRewriter &rewriter,
                               mlir::Location loc, mlir::TypedAttr value) {
  auto type = mlir::RankedTensorType::get({}, value.getType());
  mlir::OperationState state(loc, "onnx.Constant");
  state.addTypes(type);
  state.addAttribute("value", mlir::DenseElementsAttr::get(type, value));
  return rewriter.create(state)->getResult(0);
}

/// Size of dimension \p dim of \p value, folded when static.
mlir::OpFoldResult dimOf(mlir::PatternRewriter &rewriter, mlir::Location loc,
                         mlir::Value value, int64_t dim) {
  auto type = mlir::cast<mlir::RankedTensorType>(value.getType());
  if (!type.isDynamicDim(dim))
    return rewriter.getIndexAttr(type.getDimSize(dim));
  return mlir::tensor::DimOp::create(rewriter, loc, value, dim).getResult();
}

/// The 1-D i64 target shape hip.expand takes for \p dims.
mlir::Value buildShapeTensor(mlir::PatternRewriter &rewriter,
                             mlir::Location loc,
                             llvm::ArrayRef<mlir::OpFoldResult> dims) {
  auto i64Ty = rewriter.getI64Type();
  llvm::SmallVector<mlir::Value> elems;
  for (mlir::OpFoldResult dim : dims) {
    if (auto value = mlir::dyn_cast<mlir::Value>(dim)) {
      elems.push_back(
          mlir::arith::IndexCastOp::create(rewriter, loc, i64Ty, value));
      continue;
    }
    elems.push_back(mlir::arith::ConstantOp::create(
        rewriter, loc,
        rewriter.getI64IntegerAttr(*mlir::getConstantIntValue(dim))));
  }
  auto shapeType =
      mlir::RankedTensorType::get({static_cast<int64_t>(elems.size())}, i64Ty);
  return mlir::tensor::FromElementsOp::create(rewriter, loc, shapeType, elems);
}

/// Rank-1 mask_index is a per-batch count of valid leading tokens. Rank-2 is a
/// 0/1 token mask (nonzero = keep). hip.gqa indexes attention_bias as
/// [batch, heads, query, key] and does not broadcast a query extent of 1, so
/// the key mask is repeated on every query row. heads is 1 and does broadcast.
///
/// mask_index is usually computed on the device, so the bias is built from HIP
/// ops. Host loops would read it from, and write the bias into, device memory.
///
///   keep [B, S] = rank 1: hip.less(range(S)[1, S], mask[B, 1])
///                 rank 2: hip.equal(mask, 0) selects the filter instead
///   row  [B, S] = hip.where(keep, 0, filter)
///   bias         = hip.expand(row[B, 1, 1, S]) : [B, 1, S, S]
mlir::FailureOr<mlir::Value> buildPaddingBias(
    mlir::PatternRewriter &rewriter, mlir::Location loc, mlir::Value context,
    mlir::Value hidden, mlir::RankedTensorType hiddenType, mlir::Value mask,
    mlir::RankedTensorType maskType, mlir::Type biasElem, double filter) {
  const int64_t batch = hiddenType.getDimSize(0);
  const int64_t seqLen = hiddenType.getDimSize(1);
  mlir::OpFoldResult batchSize = dimOf(rewriter, loc, hidden, 0);
  mlir::OpFoldResult seqSize = dimOf(rewriter, loc, hidden, 1);
  auto i1Ty = rewriter.getI1Type();

  mlir::Value accepted =
      buildRank0Constant(rewriter, loc, rewriter.getFloatAttr(biasElem, 0.0));
  mlir::Value rejected = buildRank0Constant(
      rewriter, loc, rewriter.getFloatAttr(biasElem, filter));

  mlir::Value cond, onTrue, onFalse;
  if (maskType.getRank() == 1) {
    auto lenType = mlir::cast<mlir::IntegerType>(maskType.getElementType());
    auto keysType = mlir::RankedTensorType::get({seqLen}, lenType);
    mlir::Value keys;
    if (!hiddenType.isDynamicDim(1)) {
      llvm::SmallVector<llvm::APInt> indices;
      for (int64_t i : llvm::seq<int64_t>(seqLen))
        indices.push_back(
            llvm::APInt(lenType.getWidth(), i, /*isSigned=*/true));
      mlir::OperationState state(loc, "onnx.Constant");
      state.addTypes(keysType);
      state.addAttribute("value",
                         mlir::DenseElementsAttr::get(keysType, indices));
      keys = rewriter.create(state)->getResult(0);
    } else {
      mlir::Value seq = mlir::cast<mlir::Value>(seqSize);
      mlir::Value start = buildRank0Constant(
          rewriter, loc, rewriter.getIntegerAttr(lenType, 0));
      mlir::Value delta = buildRank0Constant(
          rewriter, loc, rewriter.getIntegerAttr(lenType, 1));
      mlir::Value limit = mlir::tensor::FromElementsOp::create(
          rewriter, loc, mlir::RankedTensorType::get({}, lenType),
          mlir::ValueRange{
              mlir::arith::IndexCastOp::create(rewriter, loc, lenType, seq)});
      mlir::Value init = mlir::tensor::EmptyOp::create(rewriter, loc, keysType,
                                                       mlir::ValueRange{seq});
      keys = mlir::hip::RangeOp::create(rewriter, loc, context, start, limit,
                                        delta, init)
                 ->getResult(0);
    }

    llvm::SmallVector<mlir::ReassociationIndices> toPair = {{0, 1}};
    auto keyRowType = mlir::RankedTensorType::get({1, seqLen}, lenType);
    mlir::Value keyRow = mlir::tensor::ExpandShapeOp::create(
        rewriter, loc, keyRowType, keys, toPair,
        llvm::SmallVector<mlir::OpFoldResult>{rewriter.getIndexAttr(1),
                                              seqSize});
    const int64_t maskBatch = maskType.getDimSize(0);
    auto lenColType = mlir::RankedTensorType::get({maskBatch, 1}, lenType);
    mlir::Value lenCol = mlir::tensor::ExpandShapeOp::create(
        rewriter, loc, lenColType, mask, toPair,
        llvm::SmallVector<mlir::OpFoldResult>{dimOf(rewriter, loc, mask, 0),
                                              rewriter.getIndexAttr(1)});

    auto keepType = mlir::RankedTensorType::get({maskBatch, seqLen}, i1Ty);
    llvm::SmallVector<mlir::Value> keepDyn;
    if (keepType.isDynamicDim(0))
      keepDyn.push_back(mlir::tensor::DimOp::create(rewriter, loc, mask, 0));
    if (keepType.isDynamicDim(1))
      keepDyn.push_back(mlir::cast<mlir::Value>(seqSize));
    mlir::Value keepInit =
        mlir::tensor::EmptyOp::create(rewriter, loc, keepType, keepDyn);
    cond = mlir::hip::LessOp::create(rewriter, loc, context, keyRow, lenCol,
                                     keepInit)
               ->getResult(0);
    onTrue = accepted;
    onFalse = rejected;
  } else if (maskType.getElementType().isInteger(1)) {
    cond = mask;
    onTrue = accepted;
    onFalse = rejected;
  } else {
    mlir::Value zero = buildRank0Constant(
        rewriter, loc, rewriter.getIntegerAttr(maskType.getElementType(), 0));
    auto padType = mlir::RankedTensorType::get(maskType.getShape(), i1Ty);
    mlir::Value padInit = createEmptyTensor(rewriter, loc, padType, mask);
    cond =
        mlir::hip::EqualOp::create(rewriter, loc, context, mask, zero, padInit)
            ->getResult(0);
    onTrue = rejected;
    onFalse = accepted;
  }

  auto condType = mlir::cast<mlir::RankedTensorType>(cond.getType());
  auto rowType = mlir::RankedTensorType::get(condType.getShape(), biasElem);
  mlir::Value rowInit = createEmptyTensor(rewriter, loc, rowType, cond);
  mlir::Value row = mlir::hip::WhereOp::create(rewriter, loc, context, cond,
                                               onTrue, onFalse, rowInit)
                        ->getResult(0);

  auto row4Type = mlir::RankedTensorType::get(
      {rowType.getDimSize(0), 1, 1, rowType.getDimSize(1)}, biasElem);
  mlir::Value row4 = mlir::tensor::ExpandShapeOp::create(
      rewriter, loc, row4Type, row,
      llvm::SmallVector<mlir::ReassociationIndices>{{0, 1, 2}, {3}},
      llvm::SmallVector<mlir::OpFoldResult>{
          dimOf(rewriter, loc, row, 0), rewriter.getIndexAttr(1),
          rewriter.getIndexAttr(1), dimOf(rewriter, loc, row, 1)});

  auto biasType =
      mlir::RankedTensorType::get({batch, 1, seqLen, seqLen}, biasElem);
  llvm::SmallVector<mlir::OpFoldResult> biasDims = {
      batchSize, rewriter.getIndexAttr(1), seqSize, seqSize};
  llvm::SmallVector<mlir::Value> biasDyn;
  for (mlir::OpFoldResult dim : biasDims)
    if (auto value = mlir::dyn_cast<mlir::Value>(dim))
      biasDyn.push_back(value);
  mlir::Value biasInit =
      mlir::tensor::EmptyOp::create(rewriter, loc, biasType, biasDyn);
  return mlir::hip::ExpandOp::create(rewriter, loc, context, row4,
                                     buildShapeTensor(rewriter, loc, biasDims),
                                     biasInit)
      ->getResult(0);
}

mlir::LogicalResult
AttentionToHip::matchAndRewrite(mlir::Operation *op,
                                mlir::PatternRewriter &rewriter) const {
  auto funcNameAttr = op->getAttrOfType<mlir::StringAttr>("function_name");
  if (!funcNameAttr || funcNameAttr.getValue() != "Attention")
    return rewriter.notifyMatchFailure(op, "not an Attention operation");

  auto domainAttr = op->getAttrOfType<mlir::StringAttr>("domain_name");
  if (!domainAttr || domainAttr.getValue() != "com.microsoft")
    return rewriter.notifyMatchFailure(
        op, "domain must be com.microsoft for Attention");

  if (op->getNumResults() != 1)
    return rewriter.notifyMatchFailure(
        op, "Attention with present outputs is not lowered");
  if (op->getNumOperands() < 3)
    return rewriter.notifyMatchFailure(
        op, "Attention expects at least 3 operands (input, weights, bias)");

  auto operandOrNull = [&](size_t index) -> mlir::Value {
    if (index >= op->getNumOperands())
      return {};
    mlir::Value value = op->getOperand(index);
    if (!value || mlir::isa<mlir::NoneType>(value.getType()))
      return {};
    return value;
  };
  // Operands: hidden, weight, bias, mask_index, past, attention_bias,
  // past_sequence_length. Only mask_index is lowered.
  mlir::Value mask = operandOrNull(3);
  for (size_t i = 4; i < op->getNumOperands(); ++i) {
    if (operandOrNull(i))
      return rewriter.notifyMatchFailure(
          op, "unsupported optional Attention operand (past / attention_bias "
              "/ past_seqlen)");
  }

  mlir::Value hidden = op->getOperand(0);
  mlir::Value qkvW = op->getOperand(1);
  mlir::Value qkvB = op->getOperand(2);

  auto numHeadsAttrOnnx = op->getAttrOfType<mlir::IntegerAttr>("num_heads");
  if (!numHeadsAttrOnnx)
    return rewriter.notifyMatchFailure(op, "missing num_heads attribute");
  const int64_t numHeads = numHeadsAttrOnnx.getValue().getSExtValue();
  if (numHeads <= 0)
    return rewriter.notifyMatchFailure(op, "num_heads must be > 0");

  auto getI64 = [&](const char *name, int64_t defaultVal) {
    auto a = op->getAttrOfType<mlir::IntegerAttr>(name);
    return a ? a.getValue().getSExtValue() : defaultVal;
  };
  if (getI64("do_rotary", 0) != 0)
    return rewriter.notifyMatchFailure(op, "do_rotary != 0 not supported");
  const int64_t unidirectional = getI64("unidirectional", 0);
  if (unidirectional != 0 && unidirectional != 1)
    return rewriter.notifyMatchFailure(
        op, "unidirectional must be 0 (bidirectional) or 1 (causal)");
  if (getI64("past_present_share_buffer", 0) != 0)
    return rewriter.notifyMatchFailure(
        op, "past_present_share_buffer != 0 not supported");
  // unidirectional=0 skips the runtime causal triangle.
  const bool noCausal = unidirectional == 0;

  // Scale: ONNX attr if present, else 0.0 so hip.gqa computes
  // 1/sqrt(head_size).
  auto scaleAttrOnnx = op->getAttrOfType<mlir::FloatAttr>("scale");
  const float scale =
      scaleAttrOnnx ? scaleAttrOnnx.getValue().convertToFloat() : 0.0f;

  auto hiddenType = mlir::dyn_cast<mlir::RankedTensorType>(hidden.getType());
  if (!hiddenType || hiddenType.getRank() != 3)
    return rewriter.notifyMatchFailure(
        op, "hidden must be a rank-3 tensor [B, S, H]");
  auto qkvWType = mlir::dyn_cast<mlir::RankedTensorType>(qkvW.getType());
  if (!qkvWType || qkvWType.getRank() != 2 || !qkvWType.hasStaticShape())
    return rewriter.notifyMatchFailure(
        op, "qkv_w must be a static rank-2 tensor [H, 3H]");
  auto outputType =
      mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
  if (!outputType || outputType.getRank() != 3)
    return rewriter.notifyMatchFailure(
        op, "Attention output must be a rank-3 tensor");

  const int64_t weightIn = qkvWType.getDimSize(0);
  const int64_t weightOut = qkvWType.getDimSize(1);
  int64_t qSize = 0;
  auto qkvSizesAttr = op->getAttrOfType<mlir::ArrayAttr>("qkv_hidden_sizes");
  if (qkvSizesAttr) {
    if (qkvSizesAttr.size() != 3)
      return rewriter.notifyMatchFailure(
          op, "qkv_hidden_sizes must be a 3-element array");
    llvm::SmallVector<int64_t, 3> qkvSizes;
    for (mlir::Attribute a : qkvSizesAttr) {
      auto ia = mlir::dyn_cast<mlir::IntegerAttr>(a);
      if (!ia)
        return rewriter.notifyMatchFailure(
            op, "qkv_hidden_sizes entries must be integers");
      qkvSizes.push_back(ia.getValue().getSExtValue());
    }
    if (qkvSizes[0] != qkvSizes[1] || qkvSizes[1] != qkvSizes[2])
      return rewriter.notifyMatchFailure(
          op, "qkv_hidden_sizes entries must be equal (multi-head, HPG=1)");
    qSize = qkvSizes[0];
  } else if (weightOut % 3 == 0 && weightOut / 3 == weightIn) {
    qSize = weightOut / 3;
  } else {
    return rewriter.notifyMatchFailure(
        op, "qkv_hidden_sizes is absent and qkv_w is not static [H, 3H]");
  }

  const int64_t qkvHidden = qSize * 3;
  if (!hiddenType.isDynamicDim(2) && hiddenType.getDimSize(2) != weightIn)
    return rewriter.notifyMatchFailure(op,
                                       "hidden size must match qkv_w.shape[0]");
  if (weightOut != qkvHidden)
    return rewriter.notifyMatchFailure(
        op, "qkv_w shape must be [hidden, sum(qkv_hidden_sizes)]");
  if (qSize % numHeads != 0)
    return rewriter.notifyMatchFailure(
        op, "qkv_hidden_sizes[0] must be divisible by num_heads");

  auto dimsMatch = [](int64_t lhs, int64_t rhs) {
    return lhs == rhs || mlir::ShapedType::isDynamic(lhs) ||
           mlir::ShapedType::isDynamic(rhs);
  };
  if (!dimsMatch(outputType.getDimSize(0), hiddenType.getDimSize(0)) ||
      !dimsMatch(outputType.getDimSize(1), hiddenType.getDimSize(1)) ||
      !dimsMatch(outputType.getDimSize(2), hiddenType.getDimSize(2)))
    return rewriter.notifyMatchFailure(
        op, "Attention output shape must match input hidden shape");

  auto ctxOrFailure = getContextArg(op, rewriter);
  if (mlir::failed(ctxOrFailure))
    return rewriter.notifyMatchFailure(op, "missing context argument");
  mlir::Value context = *ctxOrFailure;

  mlir::Type elemType = hiddenType.getElementType();
  mlir::Value attentionBias;
  if (mask) {
    auto maskType = mlir::dyn_cast<mlir::RankedTensorType>(mask.getType());
    if (!maskType || !maskType.getElementType().isInteger())
      return rewriter.notifyMatchFailure(
          op, "mask_index must be a ranked integer tensor");
    if (maskType.getRank() != 1 && maskType.getRank() != 2)
      return rewriter.notifyMatchFailure(
          op, "mask_index must be rank 1 (lengths) or rank 2 (token mask)");
    if (maskType.getRank() == 1 && maskType.getElementType().isInteger(1))
      return rewriter.notifyMatchFailure(
          op, "rank-1 mask_index must be a token count, not i1");
    if (!dimsMatch(maskType.getDimSize(0), hiddenType.getDimSize(0)) ||
        (maskType.getRank() == 2 &&
         !dimsMatch(maskType.getDimSize(1), hiddenType.getDimSize(1))))
      return rewriter.notifyMatchFailure(
          op, "mask_index shape does not match hidden batch/sequence");
    if (!elemType.isF16() && !elemType.isF32())
      return rewriter.notifyMatchFailure(
          op, "padding mask requires f16 or f32 hidden");
    double filter = -10000.0;
    if (auto filterAttr =
            op->getAttrOfType<mlir::FloatAttr>("mask_filter_value"))
      filter = filterAttr.getValue().convertToDouble();
    auto biasOr =
        buildPaddingBias(rewriter, op->getLoc(), context, hidden, hiddenType,
                         mask, maskType, elemType, filter);
    if (mlir::failed(biasOr))
      return mlir::failure();
    attentionBias = *biasOr;
  }

  mlir::Location loc = op->getLoc();
  const int64_t batch = hiddenType.getDimSize(0);
  const int64_t seqLen = hiddenType.getDimSize(1);

  mlir::Value batchDyn, seqDyn;
  if (hiddenType.isDynamicDim(0))
    batchDyn = mlir::tensor::DimOp::create(rewriter, loc, hidden, 0);
  if (hiddenType.isDynamicDim(1))
    seqDyn = mlir::tensor::DimOp::create(rewriter, loc, hidden, 1);
  llvm::SmallVector<mlir::Value> prefixDyn;
  if (batchDyn)
    prefixDyn.push_back(batchDyn);
  if (seqDyn)
    prefixDyn.push_back(seqDyn);

  // The fused weight is already [H, 3H], so hip.matmul consumes it directly.
  auto projType =
      mlir::RankedTensorType::get({batch, seqLen, qkvHidden}, elemType);
  mlir::Value projInit =
      mlir::tensor::EmptyOp::create(rewriter, loc, projType, prefixDyn);
  mlir::Value qkvProj =
      mlir::hip::MatmulOp::create(rewriter, loc, projType, context, hidden,
                                  qkvW, projInit)
          .getResult(0);

  mlir::Value addInit =
      mlir::tensor::EmptyOp::create(rewriter, loc, projType, prefixDyn);
  mlir::Value qkvBiased =
      mlir::hip::AddOp::create(rewriter, loc, projType, context, qkvProj, qkvB,
                               addInit)
          .getResult(0);

  auto sliceType =
      mlir::RankedTensorType::get({batch, seqLen, qSize}, elemType);
  auto foldDim = [&](mlir::Value dyn,
                     int64_t staticSize) -> mlir::OpFoldResult {
    if (dyn)
      return dyn;
    return rewriter.getIndexAttr(staticSize);
  };
  auto buildSlice = [&](int64_t offsetOnLastAxis) -> mlir::Value {
    llvm::SmallVector<mlir::OpFoldResult, 3> offsets = {
        rewriter.getIndexAttr(0), rewriter.getIndexAttr(0),
        rewriter.getIndexAttr(offsetOnLastAxis)};
    llvm::SmallVector<mlir::OpFoldResult, 3> sizes = {
        foldDim(batchDyn, batch), foldDim(seqDyn, seqLen),
        rewriter.getIndexAttr(qSize)};
    llvm::SmallVector<mlir::OpFoldResult, 3> strides(3,
                                                     rewriter.getIndexAttr(1));
    mlir::OperationState sliceState(
        loc, mlir::tensor::ExtractSliceOp::getOperationName());
    mlir::tensor::ExtractSliceOp::build(rewriter, sliceState, sliceType,
                                        qkvBiased, offsets, sizes, strides);
    return rewriter.create(sliceState)->getResult(0);
  };
  mlir::Value query = buildSlice(0);
  mlir::Value key = buildSlice(qSize);
  mlir::Value value = buildSlice(2 * qSize);

  // seqlens_k is a [B] i32 tensor. Bidirectional no-past attention ignores it
  // and treats every key as valid, so store the full sequence length. Causal
  // no-past attention uses ORT's prefill sentinel (-1): the runtime then sets
  // total_seq = S and past_len = 0. Storing S itself would be read as S+1 and
  // rejected because the present buffer only holds S keys.
  auto i32Ty = rewriter.getIntegerType(32);
  auto seqlensKType = mlir::RankedTensorType::get(
      {hiddenType.isDynamicDim(0) ? mlir::ShapedType::kDynamic : batch}, i32Ty);
  // A dynamic batch broadcasts one rank-0 value on the device; a host fill
  // loop would store into a pooled device buffer.
  auto broadcastToBatch = [&](mlir::Value scalar) {
    mlir::Value shape =
        buildShapeTensor(rewriter, loc, mlir::OpFoldResult(batchDyn));
    mlir::Value init = mlir::tensor::EmptyOp::create(
        rewriter, loc, seqlensKType, mlir::ValueRange{batchDyn});
    return mlir::hip::ExpandOp::create(rewriter, loc, context, scalar, shape,
                                       init)
        ->getResult(0);
  };
  mlir::Value seqlensK;
  if (!noCausal) {
    if (batchDyn) {
      seqlensK = broadcastToBatch(buildRank0Constant(
          rewriter, loc, rewriter.getIntegerAttr(i32Ty, -1)));
    } else {
      auto attr = mlir::DenseElementsAttr::get(
          seqlensKType, rewriter.getIntegerAttr(i32Ty, -1));
      seqlensK = mlir::arith::ConstantOp::create(rewriter, loc, attr);
    }
  } else if (!batchDyn && !seqDyn) {
    llvm::SmallVector<llvm::APInt, 1> seqlensVals(
        static_cast<size_t>(batch), llvm::APInt(32, seqLen, /*isSigned=*/true));
    auto seqlensKAttr = mlir::DenseElementsAttr::get(
        seqlensKType, llvm::ArrayRef<llvm::APInt>(seqlensVals));
    seqlensK = mlir::arith::ConstantOp::create(rewriter, loc, seqlensKAttr);
  } else if (!batchDyn) {
    mlir::Value seqI32 =
        mlir::arith::IndexCastOp::create(rewriter, loc, i32Ty, seqDyn);
    llvm::SmallVector<mlir::Value> elems(static_cast<size_t>(batch), seqI32);
    seqlensK = mlir::tensor::FromElementsOp::create(rewriter, loc, seqlensKType,
                                                    elems);
  } else {
    mlir::Value seqScalar;
    if (seqDyn)
      seqScalar = mlir::tensor::FromElementsOp::create(
          rewriter, loc, mlir::RankedTensorType::get({}, i32Ty),
          mlir::ValueRange{
              mlir::arith::IndexCastOp::create(rewriter, loc, i32Ty, seqDyn)});
    else
      seqScalar = buildRank0Constant(rewriter, loc,
                                     rewriter.getIntegerAttr(i32Ty, seqLen));
    seqlensK = broadcastToBatch(seqScalar);
  }

  // The runtime derives the KV span from seqlens_k. total_seq_len is still a
  // required scalar and records the present-buffer capacity S.
  auto totalSeqLenType = mlir::RankedTensorType::get({}, i32Ty);
  mlir::Value totalSeqLen;
  if (!seqDyn) {
    auto totalSeqLenAttr = mlir::DenseElementsAttr::get(
        totalSeqLenType, llvm::APInt(32, seqLen, /*isSigned=*/true));
    totalSeqLen =
        mlir::arith::ConstantOp::create(rewriter, loc, totalSeqLenAttr);
  } else {
    mlir::Value seqI32 =
        mlir::arith::IndexCastOp::create(rewriter, loc, i32Ty, seqDyn);
    totalSeqLen = mlir::tensor::FromElementsOp::create(
        rewriter, loc, totalSeqLenType, mlir::ValueRange{seqI32});
  }

  const int64_t headSize = qSize / numHeads;
  auto presentType = mlir::RankedTensorType::get(
      {batch, numHeads, seqLen, headSize}, elemType);

  return buildHipGqaCall(op, rewriter, context, query, key, value,
                         /*pastKey=*/nullptr, /*pastValue=*/nullptr, seqlensK,
                         totalSeqLen, numHeads, scale, noCausal, outputType,
                         presentType, presentType, attentionBias);
}

} // namespace

void populateAttentionConversionPatterns(RewritePatternSet &patterns,
                                         MLIRContext *ctx) {
  patterns.add<AttentionToHip>(ctx);
}

} // namespace hip
} // namespace mlir
