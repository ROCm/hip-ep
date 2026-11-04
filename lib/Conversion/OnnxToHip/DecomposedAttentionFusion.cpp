/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- DecomposedAttentionFusion.cpp - Fold primitive SDPA into MHA -------===//
//
// Exporters that do not emit a fused attention op (e.g. the diffusers UNet
// export) write unmasked, non-causal scaled dot-product attention as
// primitive ops, one chain per head group:
//
//   %q4 = onnx.Reshape(%q, [B, S, N, H])
//   %qt = onnx.Transpose(%q4) {perm = [0, 2, 1, 3]}     // [B, N, Sq, H]
//   %qs = onnx.Mul(%qt, %cq)                            // optional
//   %k4 = onnx.Reshape(%k, [B, S, N, H])
//   %kt = onnx.Transpose(%k4) {perm = [0, 2, 3, 1]}     // [B, N, H, Skv]
//   %ks = onnx.Mul(%kt, %ck)                            // optional
//   %s  = onnx.MatMul(%qs, %ks)                         // [B, N, Sq, Skv]
//   %ss = onnx.Mul(%s, %c) | onnx.Div(%s, %c)           // optional
//   %p  = onnx.Softmax(%ss) {axis = -1}
//   %v4 = onnx.Reshape(%v, [B, S, N, H])
//   %vt = onnx.Transpose(%v4) {perm = [0, 2, 1, 3]}     // [B, N, Skv, H]
//   %o  = onnx.MatMul(%p, %vt)                          // [B, N, Sq, H]
//   %ot = onnx.Transpose(%o) {perm = [0, 2, 1, 3]}      // [B, Sq, N, H]
//
// Lowered op by op, the score matrix [B, N, Sq, Skv] is written to and read
// back from DRAM by the score GEMM, the softmax and the value GEMM; at long
// sequences the three ops are bound by that traffic. The chain is replaced by
//
//   %a  = onnx.Custom(%q, %k, %v) {com.microsoft.MultiHeadAttention,
//                                  num_heads = N, scale = cq * ck * c}
//   %ot = tensor.expand_shape %a [B, Sq, N*H] -> [B, Sq, N, H]
//
// which MultiHeadAttentionConversion lowers to hip.multi_head_attention, whose
// runtime runs non-causal prefill as one flash-attention kernel that never
// materializes the scores.
//
// Rooted on onnx.Softmax and run with the other pre-lowering fusions, BEFORE
// lowerOnnxConstants, so the scale factors are still onnx.Constant values.
//
//===----------------------------------------------------------------------===//

#include "OnnxToHipUtils.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Debug.h"

#undef DEBUG_TYPE
#define DEBUG_TYPE "decomposed-attention-fusion"

STATISTIC(NumAttentionFused,
          "Number of primitive scaled dot-product attention chains folded "
          "into com.microsoft.MultiHeadAttention");

namespace mlir {
namespace hip {

namespace {

bool isOnnxOp(mlir::Operation *op, llvm::StringRef name) {
  return op && op->getName().getStringRef() == name;
}

/// The value of a one-element floating-point onnx.Constant, looking through
/// Cast / CastLike.
std::optional<double> getScalarFloatConstant(mlir::Value v) {
  mlir::Operation *def = v.getDefiningOp();
  if (!def)
    return std::nullopt;
  if ((isOnnxOp(def, "onnx.Cast") || isOnnxOp(def, "onnx.CastLike")) &&
      def->getNumOperands() >= 1)
    return getScalarFloatConstant(def->getOperand(0));
  if (!isOnnxOp(def, "onnx.Constant"))
    return std::nullopt;
  auto dense =
      mlir::dyn_cast_or_null<mlir::DenseElementsAttr>(def->getAttr("value"));
  if (!dense || dense.getNumElements() != 1 ||
      !mlir::isa<mlir::FloatType>(dense.getElementType()))
    return std::nullopt;
  return (*dense.getValues<llvm::APFloat>().begin()).convertToDouble();
}

std::optional<llvm::SmallVector<int64_t>> getPerm(mlir::Operation *op) {
  llvm::SmallVector<int64_t> perm;
  if (auto dense = op->getAttrOfType<mlir::DenseIntElementsAttr>("perm")) {
    for (const llvm::APInt &v : dense.getValues<llvm::APInt>())
      perm.push_back(v.getSExtValue());
    return perm;
  }
  if (auto array = op->getAttrOfType<mlir::ArrayAttr>("perm")) {
    for (mlir::Attribute elem : array) {
      auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(elem);
      if (!intAttr)
        return std::nullopt;
      perm.push_back(intAttr.getValue().getSExtValue());
    }
    return perm;
  }
  return std::nullopt;
}

/// The single-use producer of `v` when it is the ONNX op `name`.
mlir::Operation *singleUseProducer(mlir::Value v, llvm::StringRef name) {
  mlir::Operation *def = v.getDefiningOp();
  if (!isOnnxOp(def, name) || !v.hasOneUse())
    return nullptr;
  return def;
}

/// When `v` is Mul(x, c) or Mul(c, x) with a scalar constant c and one use,
/// returns x and multiplies `scale` by c; otherwise returns `v` unchanged.
mlir::Value peelScalarMul(mlir::Value v, double &scale,
                          llvm::SmallVectorImpl<mlir::Operation *> &chain) {
  mlir::Operation *mul = singleUseProducer(v, "onnx.Mul");
  if (!mul || mul->getNumOperands() != 2)
    return v;
  for (unsigned i = 0; i < 2; ++i) {
    if (auto c = getScalarFloatConstant(mul->getOperand(1 - i))) {
      scale *= *c;
      chain.push_back(mul);
      return mul->getOperand(i);
    }
  }
  return v;
}

/// One head-split operand: `v` = Transpose(Reshape(x3, [B, S, N, H]), perm)
/// with x3 rank-3 [B, S, N * H]. Returns x3 and the [B, S, N, H] shape.
struct HeadSplit {
  mlir::Value input;
  mlir::RankedTensorType split;
};

std::optional<HeadSplit>
matchHeadSplit(mlir::Value v, llvm::ArrayRef<int64_t> expectedPerm,
               llvm::SmallVectorImpl<mlir::Operation *> &chain) {
  mlir::Operation *transpose = singleUseProducer(v, "onnx.Transpose");
  if (!transpose)
    return std::nullopt;
  auto perm = getPerm(transpose);
  if (!perm || llvm::ArrayRef<int64_t>(*perm) != expectedPerm)
    return std::nullopt;
  mlir::Operation *reshape =
      singleUseProducer(transpose->getOperand(0), "onnx.Reshape");
  if (!reshape)
    return std::nullopt;
  mlir::Value input = reshape->getOperand(0);
  auto inType = mlir::dyn_cast<mlir::RankedTensorType>(input.getType());
  auto split =
      mlir::dyn_cast<mlir::RankedTensorType>(reshape->getResult(0).getType());
  if (!inType || !split || inType.getRank() != 3 || split.getRank() != 4 ||
      !inType.getElementType().isF16() || split.isDynamicDim(2) ||
      split.isDynamicDim(3) || inType.isDynamicDim(2) ||
      inType.getDimSize(2) != split.getDimSize(2) * split.getDimSize(3))
    return std::nullopt;
  // The reshape only splits the hidden dim: batch and sequence carry over.
  for (int64_t d : {0, 1})
    if (!inType.isDynamicDim(d) && !split.isDynamicDim(d) &&
        inType.getDimSize(d) != split.getDimSize(d))
      return std::nullopt;
  chain.push_back(transpose);
  chain.push_back(reshape);
  return HeadSplit{input, split};
}

struct DecomposedAttentionToMha : public mlir::RewritePattern {
  DecomposedAttentionToMha(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Softmax", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *softmax,
                  mlir::PatternRewriter &rewriter) const override {
    auto scoresType =
        mlir::dyn_cast<mlir::RankedTensorType>(softmax->getResult(0).getType());
    if (!scoresType || scoresType.getRank() != 4)
      return rewriter.notifyMatchFailure(softmax, "scores not rank 4");
    auto axisAttr = softmax->getAttrOfType<mlir::IntegerAttr>("axis");
    const int64_t axis = axisAttr ? axisAttr.getValue().getSExtValue() : 0;
    if (!axisAttr || (axis != -1 && axis != 3))
      return rewriter.notifyMatchFailure(softmax, "softmax not over keys");

    llvm::SmallVector<mlir::Operation *> chain;
    double scale = 1.0;

    // Scores, with an optional scalar Mul or Div applied after the GEMM.
    mlir::Value scores = softmax->getOperand(0);
    if (mlir::Operation *div = singleUseProducer(scores, "onnx.Div")) {
      auto c = getScalarFloatConstant(div->getOperand(1));
      if (!c || *c == 0.0)
        return rewriter.notifyMatchFailure(softmax, "scores div not scalar");
      scale /= *c;
      chain.push_back(div);
      scores = div->getOperand(0);
    } else {
      scores = peelScalarMul(scores, scale, chain);
    }
    mlir::Operation *qk = singleUseProducer(scores, "onnx.MatMul");
    if (!qk || qk->getNumOperands() != 2 || qk->hasAttr("hipdnn.transB"))
      return rewriter.notifyMatchFailure(softmax, "scores not a plain MatMul");
    chain.push_back(qk);

    auto q = matchHeadSplit(peelScalarMul(qk->getOperand(0), scale, chain),
                            {0, 2, 1, 3}, chain);
    auto k = matchHeadSplit(peelScalarMul(qk->getOperand(1), scale, chain),
                            {0, 2, 3, 1}, chain);
    if (!q || !k)
      return rewriter.notifyMatchFailure(softmax, "Q or K not head-split");

    // Probabilities feed exactly one value GEMM, whose output is merged back
    // to [B, Sq, N, H].
    mlir::Value probs = softmax->getResult(0);
    if (!probs.hasOneUse())
      return rewriter.notifyMatchFailure(softmax, "probs not one use");
    mlir::Operation *pv = *probs.getUsers().begin();
    if (!isOnnxOp(pv, "onnx.MatMul") || pv->getNumOperands() != 2 ||
        pv->getOperand(0) != probs || pv->hasAttr("hipdnn.transB"))
      return rewriter.notifyMatchFailure(softmax, "probs not P @ V");
    auto v = matchHeadSplit(pv->getOperand(1), {0, 2, 1, 3}, chain);
    if (!v)
      return rewriter.notifyMatchFailure(softmax, "V not head-split");
    mlir::Value attn = pv->getResult(0);
    if (!attn.hasOneUse())
      return rewriter.notifyMatchFailure(softmax, "P @ V not one use");
    mlir::Operation *merge = *attn.getUsers().begin();
    auto mergePerm =
        isOnnxOp(merge, "onnx.Transpose") ? getPerm(merge) : std::nullopt;
    if (!mergePerm || llvm::ArrayRef<int64_t>(*mergePerm) !=
                          llvm::ArrayRef<int64_t>{0, 2, 1, 3})
      return rewriter.notifyMatchFailure(softmax, "output not head-merged");
    auto mergedType =
        mlir::dyn_cast<mlir::RankedTensorType>(merge->getResult(0).getType());

    // Q, K and V share the head count and head size; K and V share the key
    // length. MultiHeadAttention takes them as [B, S, N * H].
    const int64_t heads = q->split.getDimSize(2);
    const int64_t headSize = q->split.getDimSize(3);
    for (const HeadSplit *hs : {&*k, &*v})
      if (hs->split.getDimSize(2) != heads ||
          hs->split.getDimSize(3) != headSize)
        return rewriter.notifyMatchFailure(softmax, "Q/K/V heads differ");
    if (!mergedType || mergedType.getRank() != 4 ||
        !mergedType.getElementType().isF16())
      return rewriter.notifyMatchFailure(softmax, "output type");

    auto queryType = mlir::cast<mlir::RankedTensorType>(q->input.getType());
    auto outType = mlir::RankedTensorType::get(
        {queryType.getDimSize(0), queryType.getDimSize(1), heads * headSize},
        rewriter.getF16Type());

    mlir::Location loc = merge->getLoc();
    rewriter.setInsertionPoint(merge);
    mlir::OperationState state(loc, "onnx.Custom");
    state.addOperands({q->input, k->input, v->input});
    state.addTypes(outType);
    state.addAttribute("function_name",
                       rewriter.getStringAttr("MultiHeadAttention"));
    state.addAttribute("domain_name", rewriter.getStringAttr("com.microsoft"));
    state.addAttribute(
        "num_heads",
        rewriter.getIntegerAttr(rewriter.getIntegerType(64, /*isSigned=*/true),
                                heads));
    // An explicit scale: MultiHeadAttention reads 0 (its default) as
    // 1/sqrt(H), which is wrong when the chain applied no scale.
    state.addAttribute("scale", rewriter.getF32FloatAttr(scale));
    if (auto name = softmax->getAttrOfType<mlir::StringAttr>("onnx_node_name"))
      state.addAttribute("onnx_node_name", name);
    mlir::Operation *mha = rewriter.create(state);

    // [B, Sq, N * H] -> [B, Sq, N, H]: the layout the merge transpose made.
    llvm::SmallVector<mlir::OpFoldResult> mergedShape;
    for (int64_t d : {0, 1})
      mergedShape.push_back(
          queryType.isDynamicDim(d)
              ? mlir::OpFoldResult(
                    mlir::tensor::DimOp::create(rewriter, loc, q->input, d)
                        .getResult())
              : mlir::OpFoldResult(
                    rewriter.getIndexAttr(queryType.getDimSize(d))));
    mergedShape.push_back(rewriter.getIndexAttr(heads));
    mergedShape.push_back(rewriter.getIndexAttr(headSize));
    auto expanded = mlir::tensor::ExpandShapeOp::create(
        rewriter, loc, mergedType, mha->getResult(0),
        llvm::ArrayRef<mlir::ReassociationIndices>{{0}, {1}, {2, 3}},
        mergedShape);
    rewriter.replaceOp(merge, expanded.getResult());

    // Erase the chain consumer-first; scale constants are shared and left to
    // the onnx.* use_empty DCE at the end of ConvertOnnxToHipPass.
    rewriter.eraseOp(pv);
    rewriter.eraseOp(softmax);
    for (mlir::Operation *op : chain)
      if (op->use_empty())
        rewriter.eraseOp(op);

    LLVM_DEBUG(llvm::dbgs()
               << "[" DEBUG_TYPE "] fused N=" << heads << " H=" << headSize
               << " scale=" << scale << " at " << loc << "\n");
    ++NumAttentionFused;
    return mlir::success();
  }
};

} // namespace

void populateDecomposedAttentionFusionPatterns(
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *ctx) {
  patterns.add<DecomposedAttentionToMha>(ctx);
}

} // namespace hip
} // namespace mlir
