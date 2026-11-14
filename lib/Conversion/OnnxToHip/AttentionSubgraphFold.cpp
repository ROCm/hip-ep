/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- AttentionSubgraphFold.cpp - Primitive-op attention to MHA ----------===//
//
// Pre-lowering pattern that recognizes self-attention written out with
// primitive ops, the form vision encoders export, and replaces it with
// com.microsoft.MultiHeadAttention, which lowers to the flash prefill kernel
// instead of materializing the scores through a matmul, a softmax and a second
// matmul.
//
// Batched form (SigLIP / Gemma vision; X are [B,S,N,H]):
//
//   q = Transpose(Xq) {perm = [0,2,1,3]}                      [* sq]
//   k = Transpose(Xk) {perm = [0,2,3,1]}                      [* sk]
//       (or Transpose{0,1,3,2}(Transpose{0,2,1,3}(Xk)), or
//        Transpose{0,2,1,3}(Xk) under hipdnn.transB)
//   v = Transpose(Xv) {perm = [0,2,1,3]}
//   s = MatMul(q, k) [* sc]  then optionally  + bias
//                            or  Where(cond, s, -big)
//   o = Reshape(Transpose(MatMul(Softmax(s), v)) {perm = [0,2,1,3]}, [B,S,D])
//
// Unbatched form (Qwen-style ViT; X are [S,N,H]): the same with the rank-3
// perms [1,0,2] / [1,2,0] / [0,2,1] and an output Reshape to [S,D].
//
// The replacement is MultiHeadAttention(Q, K, V) on [B,S,N*H] tensors with
// num_heads = N and scale = sq * sk * sc. An X produced by a Reshape of such a
// tensor is used directly; otherwise a Reshape is inserted. An additive bias
// becomes attention_bias; Where(cond, s, -big) becomes the attention_bias
// Where(cond, 0, -big), or, when cond compares integer segment ids
// (Equal(Unsqueeze(q, [1]), Unsqueeze(k, [0]))), an int32 [2, S] attention_bias
// holding those ids. Only fp16 is matched, which is what the runtime MHA
// implements.
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

static mlir::RankedTensorType f16Type(mlir::Value v) {
  auto type = v ? mlir::dyn_cast<mlir::RankedTensorType>(v.getType()) : nullptr;
  return type && type.getElementType().isF16() ? type : nullptr;
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

static mlir::DenseElementsAttr constantDense(mlir::Value v) {
  mlir::Operation *def = v ? v.getDefiningOp() : nullptr;
  if (!def)
    return nullptr;
  if (auto cst = mlir::dyn_cast<mlir::arith::ConstantOp>(def))
    return mlir::dyn_cast<mlir::DenseElementsAttr>(cst.getValue());
  if (def->getName().getStringRef() == "onnx.Constant")
    return def->getAttrOfType<mlir::DenseElementsAttr>("value");
  return nullptr;
}

static std::optional<double> scalarFloat(mlir::Value v) {
  if (mlir::Operation *def = v ? v.getDefiningOp() : nullptr;
      def && def->getName().getStringRef() == "onnx.Constant")
    if (auto f = def->getAttrOfType<mlir::FloatAttr>("value_float"))
      return f.getValueAsDouble();
  mlir::DenseElementsAttr dense = constantDense(v);
  if (!dense || dense.getNumElements() != 1 ||
      !mlir::isa<mlir::FloatType>(dense.getElementType()))
    return std::nullopt;
  return (*dense.getValues<mlir::APFloat>().begin()).convertToDouble();
}

static std::optional<int64_t> constantInt(mlir::Value v) {
  mlir::DenseElementsAttr dense = constantDense(v);
  if (!dense || dense.getNumElements() != 1 ||
      !dense.getElementType().isIntOrIndex())
    return std::nullopt;
  return (*dense.getValues<mlir::APInt>().begin()).getSExtValue();
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

// A per-head source X: [B,S,N,H] (batched) or [S,N,H].
struct HeadSource {
  mlir::Value x;
  int64_t numHeads;
  int64_t headSize;
};

static std::optional<HeadSource> headSource(mlir::Value x, bool batched) {
  auto type = f16Type(x);
  if (!type || type.getRank() != (batched ? 4 : 3))
    return std::nullopt;
  int64_t n = type.getDimSize(type.getRank() - 2);
  int64_t h = type.getDimSize(type.getRank() - 1);
  if (mlir::ShapedType::isDynamic(n) || mlir::ShapedType::isDynamic(h))
    return std::nullopt;
  return HeadSource{x, n, h};
}

// Q or V: Transpose(X) moving the head axis in front of the sequence axis.
static std::optional<HeadSource> matchQV(mlir::Value v, bool batched) {
  mlir::Operation *t = defOf(v, "onnx.Transpose");
  if (!t || !t->getResult(0).hasOneUse())
    return std::nullopt;
  if (!hasPerm(t, batched ? ArrayRef<int64_t>{0, 2, 1, 3}
                          : ArrayRef<int64_t>{1, 0, 2}))
    return std::nullopt;
  return headSource(t->getOperand(0), batched);
}

// K as the right-hand side of Q x K^T.
static std::optional<HeadSource> matchK(mlir::Value v, bool batched,
                                        bool transB) {
  if (transB)
    return matchQV(v, batched);
  mlir::Operation *t = defOf(v, "onnx.Transpose");
  if (!t || !t->getResult(0).hasOneUse())
    return std::nullopt;
  if (hasPerm(t, batched ? ArrayRef<int64_t>{0, 2, 3, 1}
                         : ArrayRef<int64_t>{1, 2, 0}))
    return headSource(t->getOperand(0), batched);
  if (hasPerm(t, batched ? ArrayRef<int64_t>{0, 1, 3, 2}
                         : ArrayRef<int64_t>{0, 2, 1}))
    return matchQV(t->getOperand(0), batched);
  return std::nullopt;
}

// One element of a shape tensor: a constant, or [Unsqueeze] Gather(Shape(t),
// i) where dim i of t is static.
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

// The last extent of a Reshape target: a static result dim, the last element
// of a constant shape, or the last element of a Concat-built shape.
static std::optional<int64_t> lastTargetExtent(mlir::Operation *reshape,
                                               mlir::RankedTensorType outType) {
  int64_t last = outType.getRank() - 1;
  if (!outType.isDynamicDim(last))
    return outType.getDimSize(last);
  if (mlir::DenseElementsAttr dense = constantDense(reshape->getOperand(1));
      dense && dense.getElementType().isIntOrIndex() && dense.getNumElements() > 0)
    return (*(dense.getValues<mlir::APInt>().begin() + (dense.getNumElements() - 1)))
        .getSExtValue();
  mlir::Operation *concat = defOf(reshape->getOperand(1), "onnx.Concat");
  if (!concat || concat->getNumOperands() != static_cast<unsigned>(outType.getRank()))
    return std::nullopt;
  return shapeElement(concat->getOperand(concat->getNumOperands() - 1));
}

static mlir::Value createI64Constant(mlir::PatternRewriter &rewriter,
                                     mlir::Location loc, ArrayRef<int64_t> values) {
  auto type = mlir::RankedTensorType::get({static_cast<int64_t>(values.size())},
                                          rewriter.getI64Type());
  mlir::OperationState state(loc, "onnx.Constant");
  state.addTypes(type);
  state.addAttribute("value", mlir::DenseIntElementsAttr::get(type, values));
  return rewriter.create(state)->getResult(0);
}

static mlir::Value createF16Scalar(mlir::PatternRewriter &rewriter,
                                   mlir::Location loc, double value) {
  auto type = mlir::RankedTensorType::get({}, rewriter.getF16Type());
  llvm::APFloat half(value);
  bool losesInfo = false;
  half.convert(llvm::APFloat::IEEEhalf(), llvm::APFloat::rmNearestTiesToEven,
               &losesInfo);
  mlir::OperationState state(loc, "onnx.Constant");
  state.addTypes(type);
  state.addAttribute("value", mlir::DenseElementsAttr::get(type, half));
  return rewriter.create(state)->getResult(0);
}

// [B,S,N*H] for the MHA: the 3-D input of a Reshape that produced X, or a new
// Reshape of X ([B,S,N,H] -> [B,S,N*H], [S,N,H] -> [1,S,N*H]).
static mlir::Value toMhaInput(mlir::PatternRewriter &rewriter, mlir::Location loc,
                              const HeadSource &src, bool batched) {
  int64_t hidden = src.numHeads * src.headSize;
  if (mlir::Operation *reshape = defOf(src.x, "onnx.Reshape"); reshape && batched) {
    auto in = f16Type(reshape->getOperand(0));
    if (in && in.getRank() == 3 && !in.isDynamicDim(2) && in.getDimSize(2) == hidden)
      return reshape->getOperand(0);
  }
  auto xType = mlir::cast<mlir::RankedTensorType>(src.x.getType());
  SmallVector<int64_t> dims =
      batched ? SmallVector<int64_t>{xType.getDimSize(0), xType.getDimSize(1), hidden}
              : SmallVector<int64_t>{1, xType.getDimSize(0), hidden};
  mlir::Value shape = createI64Constant(
      rewriter, loc,
      batched ? ArrayRef<int64_t>{0, 0, hidden} : ArrayRef<int64_t>{1, -1, hidden});
  mlir::OperationState state(loc, "onnx.Reshape");
  state.addOperands({src.x, shape});
  state.addTypes(mlir::RankedTensorType::get(dims, xType.getElementType()));
  return rewriter.create(state)->getResult(0);
}

// Where(cond, 0, masked) as an fp16 attention bias. Every layer usually masks
// with the same cond, so one bias is built right after cond and shared.
static mlir::Value maskBias(mlir::PatternRewriter &rewriter, mlir::Value cond,
                            double masked) {
  for (mlir::Operation *user : cond.getUsers()) {
    if (user->getName().getStringRef() != "onnx.Where" ||
        user->getOperand(0) != cond || !f16Type(user->getResult(0)))
      continue;
    auto zero = scalarFloat(user->getOperand(1));
    auto fill = scalarFloat(user->getOperand(2));
    if (zero && *zero == 0.0 && fill && *fill == masked)
      return user->getResult(0);
  }
  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointAfterValue(cond);
  mlir::Location loc = cond.getLoc();
  auto condType = mlir::cast<mlir::RankedTensorType>(cond.getType());
  mlir::OperationState where(loc, "onnx.Where");
  where.addOperands({cond, createF16Scalar(rewriter, loc, 0.0),
                     createF16Scalar(rewriter, loc, masked)});
  where.addTypes(
      mlir::RankedTensorType::get(condType.getShape(), rewriter.getF16Type()));
  return rewriter.create(where)->getResult(0);
}

// cond = Equal(Unsqueeze(q, [1]), Unsqueeze(k, [0])) over integer segment ids
// q, k of shape [S]: a block-diagonal mask keeping the keys in the query's
// segment. Returns (q, k).
static std::optional<std::pair<mlir::Value, mlir::Value>>
matchSegments(mlir::Value cond) {
  mlir::Operation *equal = defOf(cond, "onnx.Equal");
  if (!equal)
    return std::nullopt;
  auto unsqueezed = [](mlir::Value v, int64_t axis) -> mlir::Value {
    mlir::Operation *u = defOf(v, "onnx.Unsqueeze");
    if (!u || u->getNumOperands() != 2)
      return {};
    mlir::DenseElementsAttr axes = constantDense(u->getOperand(1));
    auto type = mlir::dyn_cast<mlir::RankedTensorType>(u->getOperand(0).getType());
    if (!axes || axes.getNumElements() != 1 || !type || type.getRank() != 1 ||
        !mlir::isa<mlir::IntegerType>(type.getElementType()))
      return {};
    int64_t a = (*axes.getValues<mlir::APInt>().begin()).getSExtValue();
    return (a < 0 ? a + 2 : a) == axis ? u->getOperand(0) : mlir::Value();
  };
  for (unsigned swap : {0u, 1u}) {
    mlir::Value q = unsqueezed(equal->getOperand(swap), 1);
    mlir::Value k = unsqueezed(equal->getOperand(1 - swap), 0);
    if (q && k)
      return std::make_pair(q, k);
  }
  return std::nullopt;
}

// [2, S] int32: the query segment ids, then the key segment ids.
static mlir::Value packSegments(mlir::PatternRewriter &rewriter,
                                mlir::Location loc, mlir::Value q,
                                mlir::Value k) {
  mlir::Value axis0 = createI64Constant(rewriter, loc, {0});
  SmallVector<mlir::Value, 2> rows;
  for (mlir::Value ids : {q, k}) {
    auto type = mlir::cast<mlir::RankedTensorType>(ids.getType());
    mlir::OperationState cast(loc, "onnx.Cast");
    cast.addOperands(ids);
    cast.addTypes(mlir::RankedTensorType::get(type.getShape(), rewriter.getI32Type()));
    mlir::Value ids32 = rewriter.create(cast)->getResult(0);
    mlir::OperationState unsqueeze(loc, "onnx.Unsqueeze");
    unsqueeze.addOperands({ids32, axis0});
    unsqueeze.addTypes(mlir::RankedTensorType::get({1, type.getDimSize(0)},
                                                   rewriter.getI32Type()));
    rows.push_back(rewriter.create(unsqueeze)->getResult(0));
  }
  auto qType = mlir::cast<mlir::RankedTensorType>(q.getType());
  mlir::OperationState concat(loc, "onnx.Concat");
  concat.addOperands(rows);
  concat.addAttribute("axis", rewriter.getIntegerAttr(
                                  rewriter.getIntegerType(64, /*isSigned=*/true), 0));
  concat.addTypes(mlir::RankedTensorType::get({2, qType.getDimSize(0)},
                                              rewriter.getI32Type()));
  return rewriter.create(concat)->getResult(0);
}

// Erases `op` when unused, then its ONNX / constant producers that become
// unused, e.g. the Shape -> Gather -> Concat chain feeding a Reshape.
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
    auto outType = f16Type(root->getResult(0));
    if (!outType || (outType.getRank() != 3 && outType.getRank() != 2))
      return rewriter.notifyMatchFailure(root, "not a rank-2/3 f16 reshape");
    const bool batched = outType.getRank() == 3;

    mlir::Operation *merge = defOf(root->getOperand(0), "onnx.Transpose");
    if (!merge || !merge->getResult(0).hasOneUse() ||
        !hasPerm(merge, batched ? ArrayRef<int64_t>{0, 2, 1, 3}
                                : ArrayRef<int64_t>{1, 0, 2}))
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
      if (a != -1 && a != (batched ? 3 : 2))
        return rewriter.notifyMatchFailure(root, "softmax not over keys");
    }

    // Scores, with an optional additive bias or Where mask around them.
    double scale = 1.0;
    mlir::Value scores = softmax->getOperand(0);
    mlir::Value bias, cond;
    double maskValue = 0.0;
    auto isQk = [&](mlir::Value v) {
      double s = 1.0;
      return defOf(peelScale(v, s), "onnx.MatMul") != nullptr;
    };
    if (mlir::Operation *add = defOf(scores, "onnx.Add");
        add && add->getResult(0).hasOneUse()) {
      unsigned qkSide = isQk(add->getOperand(0)) ? 0 : isQk(add->getOperand(1)) ? 1 : 2;
      if (qkSide == 2)
        return rewriter.notifyMatchFailure(root, "add without Q x K^T");
      bias = add->getOperand(1 - qkSide);
      scores = add->getOperand(qkSide);
    } else if (mlir::Operation *where = defOf(scores, "onnx.Where");
               where && where->getResult(0).hasOneUse()) {
      auto masked = scalarFloat(where->getOperand(2));
      if (!masked || *masked > -1.0e4 || !isQk(where->getOperand(1)))
        return rewriter.notifyMatchFailure(root, "where is not a key mask");
      cond = where->getOperand(0);
      maskValue = *masked;
      scores = where->getOperand(1);
    }
    scores = peelScale(scores, scale);
    mlir::Operation *qk = defOf(scores, "onnx.MatMul");
    if (!qk || !qk->getResult(0).hasOneUse() || qk->hasAttr("hipdnn.transA"))
      return rewriter.notifyMatchFailure(root, "no Q x K^T matmul");
    const bool transB = qk->hasAttr("hipdnn.transB");

    mlir::Value qSide = peelScale(qk->getOperand(0), scale);
    mlir::Value kSide = peelScale(qk->getOperand(1), scale);
    auto q = matchQV(qSide, batched);
    auto k = matchK(kSide, batched, transB);
    auto v = matchQV(pv->getOperand(1), batched);
    if (!q || !k || !v)
      return rewriter.notifyMatchFailure(root, "Q/K/V head split not matched");
    std::optional<int64_t> last = lastTargetExtent(root, outType);
    if (q->numHeads != k->numHeads || q->numHeads != v->numHeads ||
        q->headSize != k->headSize || q->headSize != v->headSize || !last ||
        (*last != -1 && *last != q->numHeads * q->headSize))
      return rewriter.notifyMatchFailure(root, "inconsistent head shapes");
    if (bias) {
      auto biasType = mlir::dyn_cast<mlir::RankedTensorType>(bias.getType());
      if (!biasType || biasType.getRank() > (batched ? 4 : 3) ||
          !biasType.getElementType().isF16())
        return rewriter.notifyMatchFailure(root, "unsupported bias");
    }

    mlir::Location loc = root->getLoc();
    mlir::Value qIn = toMhaInput(rewriter, loc, *q, batched);
    mlir::Value kIn = toMhaInput(rewriter, loc, *k, batched);
    mlir::Value vIn = toMhaInput(rewriter, loc, *v, batched);
    if (cond) {
      // A segment-id mask travels as [2, S] int32 ids, which the flash kernel
      // compares per element, instead of an S x S fp16 bias it has to read.
      if (auto segments = matchSegments(cond))
        bias = packSegments(rewriter, loc, segments->first, segments->second);
      else
        bias = maskBias(rewriter, cond, maskValue);
    }

    mlir::OperationState state(loc, "onnx.Custom");
    state.addOperands({qIn, kIn, vIn});
    if (bias) {
      for (int i = 0; i < 2; ++i) {
        mlir::OperationState none(loc, "onnx.NoValue");
        none.addTypes(rewriter.getNoneType());
        none.addAttribute("value", rewriter.getUnitAttr());
        state.addOperands(rewriter.create(none)->getResult(0));
      }
      state.addOperands(bias);
    }
    auto qInType = mlir::cast<mlir::RankedTensorType>(qIn.getType());
    state.addTypes(batched ? root->getResult(0).getType()
                           : mlir::RankedTensorType::get(
                                 {1, qInType.getDimSize(1), qInType.getDimSize(2)},
                                 outType.getElementType()));
    state.addAttribute("function_name", rewriter.getStringAttr("MultiHeadAttention"));
    state.addAttribute("domain_name", rewriter.getStringAttr("com.microsoft"));
    state.addAttribute("num_heads", rewriter.getI64IntegerAttr(q->numHeads));
    state.addAttribute("scale", rewriter.getF32FloatAttr(static_cast<float>(scale)));
    mlir::Value result = rewriter.create(state)->getResult(0);
    if (!batched) {
      mlir::OperationState reshape(loc, "onnx.Reshape");
      reshape.addOperands({result, root->getOperand(1)});
      reshape.addTypes(root->getResult(0).getType());
      result = rewriter.create(reshape)->getResult(0);
    }

    // ONNX ops are not known to be side-effect free, so the replaced chain is
    // erased here rather than left to dead-code elimination.
    mlir::Operation *outShape = batched ? root->getOperand(1).getDefiningOp() : nullptr;
    rewriter.replaceOp(root, result);
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
