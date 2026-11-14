/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- AttentionSubgraphFold.cpp - Primitive-op attention to MHA ----------===//
//
// Pre-lowering pattern that recognizes self-attention written out with
// primitive ops, the form SigLIP / ViT vision encoders export:
//
//   q = Transpose(Reshape(Q, [B,S,N,H])) {perm = [0,2,1,3]}   [* sq]
//   k = Transpose(Reshape(K, [B,S,N,H])) {perm = [0,2,3,1]}   [* sk]
//   v = Transpose(Reshape(V, [B,S,N,H])) {perm = [0,2,1,3]}
//   o = Reshape(Transpose(MatMul(Softmax(MatMul(q, k) [* s]), v)
//                         {perm = [0,2,1,3]}), [B,S,N*H])
//
// and replaces `o` with com.microsoft.MultiHeadAttention(Q, K, V) with
// num_heads = N and scale = sq * sk * s. That lowers to the flash prefill
// kernel, which never materializes the B x N x S x S scores the matmul ->
// softmax -> matmul chain writes and reads back.
//
// Only the unmasked fp16 form is matched: the runtime MHA path implements
// neither an additive attention bias nor a key padding mask yet.
//
//===----------------------------------------------------------------------===//

#include "OnnxToHipUtils.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/Statistic.h"

#define DEBUG_TYPE "attention-subgraph-fold"

STATISTIC(NumAttentionSubgraphFolds,
          "Number of primitive-op attention subgraphs folded to MHA");

namespace mlir {
namespace hip {

namespace {

static mlir::Operation *defOf(mlir::Value v, llvm::StringRef name) {
  mlir::Operation *def = v ? v.getDefiningOp() : nullptr;
  return def && def->getName().getStringRef() == name ? def : nullptr;
}

static bool hasPerm(mlir::Operation *transpose, ArrayRef<int64_t> want) {
  SmallVector<int64_t> perm;
  if (auto dense = transpose->getAttrOfType<mlir::DenseIntElementsAttr>("perm")) {
    for (mlir::APInt v : dense.getValues<mlir::APInt>())
      perm.push_back(v.getSExtValue());
  } else if (auto arr = transpose->getAttrOfType<mlir::ArrayAttr>("perm")) {
    for (mlir::Attribute elem : arr) {
      auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(elem);
      if (!intAttr)
        return false;
      perm.push_back(intAttr.getValue().getSExtValue());
    }
  } else {
    return false;
  }
  return llvm::ArrayRef<int64_t>(perm) == want;
}

static std::optional<double> scalarFloat(mlir::Value v) {
  mlir::Operation *def = v ? v.getDefiningOp() : nullptr;
  if (!def)
    return std::nullopt;
  if (def->getName().getStringRef() == "onnx.Constant")
    if (auto f = def->getAttrOfType<mlir::FloatAttr>("value_float"))
      return f.getValueAsDouble();
  mlir::DenseElementsAttr dense;
  if (auto cst = mlir::dyn_cast<mlir::arith::ConstantOp>(def))
    dense = mlir::dyn_cast<mlir::DenseElementsAttr>(cst.getValue());
  else if (def->getName().getStringRef() == "onnx.Constant")
    dense = def->getAttrOfType<mlir::DenseElementsAttr>("value");
  if (!dense || dense.getNumElements() != 1 ||
      !mlir::isa<mlir::FloatType>(dense.getElementType()))
    return std::nullopt;
  return (*dense.getValues<mlir::APFloat>().begin()).convertToDouble();
}

// Steps over a single-use Mul by a scalar constant, folding it into `scale`.
static mlir::Value peelScale(mlir::Value v, double &scale) {
  mlir::Operation *mul = defOf(v, "onnx.Mul");
  if (!mul || !mul->getResult(0).hasOneUse())
    return v;
  for (unsigned i : {0u, 1u}) {
    if (auto s = scalarFloat(mul->getOperand(i))) {
      scale *= *s;
      return mul->getOperand(1 - i);
    }
  }
  return v;
}

struct HeadSplit {
  mlir::Value source; // [B, S, N*H]
  int64_t numHeads;
  int64_t headSize;
};

// Transpose(Reshape(x [B,S,N*H] -> [B,S,N,H])) with the given perm.
static std::optional<HeadSplit> matchHeadSplit(mlir::Value v,
                                               ArrayRef<int64_t> perm) {
  mlir::Operation *transpose = defOf(v, "onnx.Transpose");
  if (!transpose || !transpose->getResult(0).hasOneUse() ||
      !hasPerm(transpose, perm))
    return std::nullopt;
  mlir::Operation *reshape = defOf(transpose->getOperand(0), "onnx.Reshape");
  if (!reshape || !reshape->getResult(0).hasOneUse())
    return std::nullopt;
  auto in = mlir::dyn_cast<mlir::RankedTensorType>(reshape->getOperand(0).getType());
  auto out = mlir::dyn_cast<mlir::RankedTensorType>(reshape->getResult(0).getType());
  if (!in || !out || in.getRank() != 3 || out.getRank() != 4 ||
      !in.getElementType().isF16())
    return std::nullopt;
  int64_t n = out.getDimSize(2), h = out.getDimSize(3);
  if (mlir::ShapedType::isDynamic(n) || mlir::ShapedType::isDynamic(h) ||
      in.isDynamicDim(2) || in.getDimSize(2) != n * h)
    return std::nullopt;
  for (int i : {0, 1})
    if (!in.isDynamicDim(i) && !out.isDynamicDim(i) &&
        in.getDimSize(i) != out.getDimSize(i))
      return std::nullopt;
  return HeadSplit{reshape->getOperand(0), n, h};
}

// Erases `op` when unused, then its ONNX / constant producers that become
// unused, e.g. the Shape -> Gather -> Concat chain feeding a Reshape.
static std::optional<int64_t> constantInt(mlir::Value v) {
  mlir::Operation *def = v ? v.getDefiningOp() : nullptr;
  if (!def)
    return std::nullopt;
  mlir::DenseElementsAttr dense;
  if (auto cst = mlir::dyn_cast<mlir::arith::ConstantOp>(def))
    dense = mlir::dyn_cast<mlir::DenseElementsAttr>(cst.getValue());
  else if (def->getName().getStringRef() == "onnx.Constant")
    dense = def->getAttrOfType<mlir::DenseElementsAttr>("value");
  if (!dense || dense.getNumElements() != 1 ||
      !dense.getElementType().isIntOrIndex())
    return std::nullopt;
  return (*dense.getValues<mlir::APInt>().begin()).getSExtValue();
}

// One element of a Concat-built shape: a constant, or [Unsqueeze]
// Gather(Shape(t), i) where dim i of t is static.
static std::optional<int64_t> shapeElement(mlir::Value v) {
  if (mlir::Operation *unsqueeze = defOf(v, "onnx.Unsqueeze"))
    v = unsqueeze->getOperand(0);
  if (auto c = constantInt(v))
    return c;
  mlir::Operation *gather = defOf(v, "onnx.Gather");
  mlir::Operation *shape = gather ? defOf(gather->getOperand(0), "onnx.Shape") : nullptr;
  if (!shape)
    return std::nullopt;
  if (auto start = shape->getAttrOfType<mlir::IntegerAttr>("start");
      start && start.getValue().getSExtValue() != 0)
    return std::nullopt;
  auto index = constantInt(gather->getOperand(1));
  auto type = mlir::dyn_cast<mlir::RankedTensorType>(shape->getOperand(0).getType());
  if (!index || !type)
    return std::nullopt;
  int64_t i = *index < 0 ? *index + type.getRank() : *index;
  if (i < 0 || i >= type.getRank() || type.isDynamicDim(i))
    return std::nullopt;
  return type.getDimSize(i);
}

// The last extent of a rank-3 Reshape target: a static result dim, or the
// last element of a Concat-built shape.
static std::optional<int64_t> lastTargetExtent(mlir::Operation *reshape,
                                               mlir::RankedTensorType outType) {
  if (!outType.isDynamicDim(2))
    return outType.getDimSize(2);
  mlir::Operation *concat = defOf(reshape->getOperand(1), "onnx.Concat");
  if (!concat || concat->getNumOperands() != 3)
    return std::nullopt;
  return shapeElement(concat->getOperand(2));
}

static void eraseIfDead(mlir::PatternRewriter &rewriter, mlir::Operation *op) {
  if (!op || !op->use_empty())
    return;
  llvm::StringRef name = op->getName().getStringRef();
  if (!name.starts_with("onnx.") && !mlir::isa<mlir::arith::ConstantOp>(op))
    return;
  llvm::SmallVector<mlir::Operation *> producers;
  for (mlir::Value operand : op->getOperands())
    if (mlir::Operation *def = operand.getDefiningOp())
      producers.push_back(def);
  rewriter.eraseOp(op);
  for (mlir::Operation *def : producers)
    eraseIfDead(rewriter, def);
}

struct FoldAttentionSubgraph : public mlir::RewritePattern {
  FoldAttentionSubgraph(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Reshape", /*benefit=*/3, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *root,
                  mlir::PatternRewriter &rewriter) const override {
    auto outType = mlir::dyn_cast<mlir::RankedTensorType>(root->getResult(0).getType());
    if (!outType || outType.getRank() != 3 || !outType.getElementType().isF16())
      return rewriter.notifyMatchFailure(root, "not a rank-3 f16 reshape");

    mlir::Operation *merge = defOf(root->getOperand(0), "onnx.Transpose");
    if (!merge || !merge->getResult(0).hasOneUse() ||
        !hasPerm(merge, {0, 2, 1, 3}))
      return rewriter.notifyMatchFailure(root, "no head-merge transpose");
    mlir::Operation *pv = defOf(merge->getOperand(0), "onnx.MatMul");
    if (!pv || !pv->getResult(0).hasOneUse() || pv->hasAttr("hipdnn.transA") ||
        pv->hasAttr("hipdnn.transB"))
      return rewriter.notifyMatchFailure(root, "no probs x V matmul");
    mlir::Operation *softmax = defOf(pv->getOperand(0), "onnx.Softmax");
    if (!softmax || !softmax->getResult(0).hasOneUse())
      return rewriter.notifyMatchFailure(root, "no softmax");
    if (auto axis = softmax->getAttrOfType<mlir::IntegerAttr>("axis")) {
      int64_t a = axis.getValue().getSExtValue();
      if (a != -1 && a != 3)
        return rewriter.notifyMatchFailure(root, "softmax not over keys");
    }

    double scale = 1.0;
    mlir::Value scores = peelScale(softmax->getOperand(0), scale);
    mlir::Operation *qk = defOf(scores, "onnx.MatMul");
    if (!qk || !qk->getResult(0).hasOneUse() || qk->hasAttr("hipdnn.transA") ||
        qk->hasAttr("hipdnn.transB"))
      return rewriter.notifyMatchFailure(root, "no Q x K^T matmul");

    mlir::Value qSide = peelScale(qk->getOperand(0), scale);
    mlir::Value kSide = peelScale(qk->getOperand(1), scale);
    auto q = matchHeadSplit(qSide, {0, 2, 1, 3});
    auto k = matchHeadSplit(kSide, {0, 2, 3, 1});
    auto v = matchHeadSplit(pv->getOperand(1), {0, 2, 1, 3});
    if (!q || !k || !v)
      return rewriter.notifyMatchFailure(root, "Q/K/V head split not matched");
    std::optional<int64_t> last = lastTargetExtent(root, outType);
    if (q->numHeads != k->numHeads || q->numHeads != v->numHeads ||
        q->headSize != k->headSize || q->headSize != v->headSize || !last ||
        (*last != -1 && *last != q->numHeads * q->headSize))
      return rewriter.notifyMatchFailure(root, "inconsistent head shapes");

    mlir::OperationState state(root->getLoc(), "onnx.Custom");
    state.addOperands({q->source, k->source, v->source});
    state.addTypes(root->getResultTypes());
    state.addAttribute("function_name", rewriter.getStringAttr("MultiHeadAttention"));
    state.addAttribute("domain_name", rewriter.getStringAttr("com.microsoft"));
    state.addAttribute("num_heads", rewriter.getI64IntegerAttr(q->numHeads));
    state.addAttribute("scale", rewriter.getF32FloatAttr(static_cast<float>(scale)));
    mlir::Operation *mha = rewriter.create(state);

    // ONNX ops are not known to be side-effect free, so the replaced chain is
    // erased here rather than left to dead-code elimination.
    mlir::Operation *outShape = root->getNumOperands() > 1
                                    ? root->getOperand(1).getDefiningOp()
                                    : nullptr;
    rewriter.replaceOp(root, mha->getResults());
    eraseIfDead(rewriter, merge);
    eraseIfDead(rewriter, outShape);
    ++NumAttentionSubgraphFolds;
    return mlir::success();
  }
};

} // namespace

void populateAttentionSubgraphFoldPatterns(RewritePatternSet &patterns,
                                           MLIRContext *ctx) {
  patterns.add<FoldAttentionSubgraph>(ctx);
}

} // namespace hip
} // namespace mlir
