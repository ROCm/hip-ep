/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- LayerNormFusion.cpp - Fold decomposed LayerNorm into one op --------===//
//
// Exports of SigLIP / CLIP style encoders (e.g. SmolVLM's vision tower) spell
// LayerNorm over the last axis as primitives:
//
//   %mean = onnx.ReduceMean(%x) {axes = [-1], keepdims = 1}
//   %d    = onnx.Sub(%x, %mean)
//   %sq   = onnx.Pow(%d, 2.0)  // or onnx.Mul(%d, %d)
//   %var  = onnx.ReduceMean(%sq) {axes = [-1], keepdims = 1}
//   %s    = onnx.Sqrt(onnx.Add(%var, %eps))
//   %n    = onnx.Div(%d, %s)
//   %y    = onnx.Add(onnx.Mul(%n, %gamma), %beta)
//
//   ->  %y = onnx.LayerNormalization(%x, %gamma, %beta) {axis = -1, epsilon}
//
// Evaluated op by op in fp16 the squared deviations overflow to inf once
// |x - mean| exceeds ~256 (SmolVLM's post_layernorm input reaches ~460), and
// those rows normalize to 0. hip.layer_norm accumulates in fp32, matching the
// fused LayerNormalization that ORT's CPU provider runs for the same graph.
//
// Rooted on onnx.Div; runs with the other pre-lowering fusions, before
// `lowerOnnxConstants`, so epsilon / exponent / axes are still onnx.Constant.
// Benefit 2 so it wins over the broadcasting Div -> Mul(Reciprocal) rewrite
// rooted on the same op.
//
//===----------------------------------------------------------------------===//

#include "OnnxToHipUtils.h"

#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "layer-norm-fusion"

STATISTIC(NumLayerNormFused, "Number of decomposed LayerNorm chains folded to "
                             "onnx.LayerNormalization");

namespace mlir {
namespace hip {

namespace {

static mlir::Operation *defOpNamed(mlir::Value v, llvm::StringRef name) {
  mlir::Operation *def = v ? v.getDefiningOp() : nullptr;
  return def && def->getName().getStringRef() == name ? def : nullptr;
}

static std::optional<double> getScalarFloatConstant(mlir::Value v) {
  mlir::Operation *def = v.getDefiningOp();
  if (!def)
    return std::nullopt;
  llvm::StringRef opName = def->getName().getStringRef();
  if ((opName == "onnx.Cast" || opName == "onnx.CastLike") &&
      def->getNumOperands() >= 1)
    return getScalarFloatConstant(def->getOperand(0));
  if (opName != "onnx.Constant")
    return std::nullopt;
  auto dense =
      mlir::dyn_cast_or_null<mlir::DenseElementsAttr>(def->getAttr("value"));
  if (!dense || dense.getNumElements() != 1)
    return std::nullopt;
  if (mlir::isa<mlir::FloatType>(dense.getElementType()))
    return (*dense.getValues<llvm::APFloat>().begin()).convertToDouble();
  if (mlir::isa<mlir::IntegerType>(dense.getElementType()))
    return static_cast<double>(
        (*dense.getValues<llvm::APInt>().begin()).getSExtValue());
  return std::nullopt;
}

/// True when \p op is ReduceMean(keepdims=1) over exactly the last axis of a
/// rank-\p rank input.
static bool isLastAxisMean(mlir::Operation *op, int64_t rank) {
  if (!op || op->getName().getStringRef() != "onnx.ReduceMean")
    return false;
  if (auto k = op->getAttrOfType<mlir::IntegerAttr>("keepdims"))
    if (k.getValue().getSExtValue() != 1)
      return false;
  llvm::SmallVector<int64_t> axes;
  if (op->getNumOperands() > 1) {
    if (!matchConstantIntTensor(op->getOperand(1), axes)) {
      auto def = op->getOperand(1).getDefiningOp();
      auto dense = def ? def->getAttrOfType<mlir::DenseIntElementsAttr>("value")
                       : nullptr;
      if (!dense)
        return false;
      for (const llvm::APInt &v : dense.getValues<llvm::APInt>())
        axes.push_back(v.getSExtValue());
    }
  } else if (auto attr = op->getAttrOfType<mlir::ArrayAttr>("axes")) {
    for (mlir::Attribute a : attr)
      axes.push_back(
          mlir::cast<mlir::IntegerAttr>(a).getValue().getSExtValue());
  }
  return axes.size() == 1 && (axes[0] == -1 || axes[0] == rank - 1);
}

/// For a binary op, the operand that is not \p known (null if neither or
/// both are).
static mlir::Value otherOperand(mlir::Operation *op, mlir::Value known) {
  if (op->getNumOperands() != 2)
    return nullptr;
  mlir::Value a = op->getOperand(0), b = op->getOperand(1);
  if (a == known && b != known)
    return b;
  if (b == known && a != known)
    return a;
  return nullptr;
}

/// \p v is a rank-1 tensor whose length is the normalized dim \p dim.
static bool isPerChannel(mlir::Value v, int64_t dim) {
  auto t = mlir::dyn_cast<mlir::RankedTensorType>(v.getType());
  return t && t.getRank() == 1 && !t.isDynamicDim(0) && t.getDimSize(0) == dim;
}

struct DecomposedLayerNormToLayerNorm : public mlir::RewritePattern {
  DecomposedLayerNormToLayerNorm(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Div", /*benefit=*/2, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *divOp,
                  mlir::PatternRewriter &rewriter) const override {
    if (divOp->getNumOperands() != 2)
      return rewriter.notifyMatchFailure(divOp, "div.arity");
    mlir::Value d = divOp->getOperand(0);

    mlir::Operation *sqrtOp = defOpNamed(divOp->getOperand(1), "onnx.Sqrt");
    if (!sqrtOp)
      return rewriter.notifyMatchFailure(divOp, "denominator not Sqrt");
    mlir::Operation *epsAdd = defOpNamed(sqrtOp->getOperand(0), "onnx.Add");
    if (!epsAdd || epsAdd->getNumOperands() != 2)
      return rewriter.notifyMatchFailure(divOp, "sqrt input not Add");
    mlir::Value var;
    std::optional<double> eps;
    for (int i = 0; i < 2 && !eps; ++i) {
      eps = getScalarFloatConstant(epsAdd->getOperand(i));
      var = epsAdd->getOperand(1 - i);
    }
    if (!eps)
      return rewriter.notifyMatchFailure(divOp, "epsilon not a scalar const");

    mlir::Operation *subOp = defOpNamed(d, "onnx.Sub");
    if (!subOp)
      return rewriter.notifyMatchFailure(divOp, "numerator not Sub");
    mlir::Value x = subOp->getOperand(0);
    auto xType = mlir::dyn_cast<mlir::RankedTensorType>(x.getType());
    if (!xType || xType.getRank() < 1 ||
        !mlir::isa<mlir::FloatType>(xType.getElementType()) ||
        xType.isDynamicDim(xType.getRank() - 1) ||
        xType != divOp->getResult(0).getType())
      return rewriter.notifyMatchFailure(divOp, "x type");
    int64_t rank = xType.getRank();

    mlir::Operation *meanOp = subOp->getOperand(1).getDefiningOp();
    if (!isLastAxisMean(meanOp, rank) || meanOp->getOperand(0) != x)
      return rewriter.notifyMatchFailure(divOp, "mean not ReduceMean(x, -1)");

    mlir::Operation *varOp = var.getDefiningOp();
    if (!isLastAxisMean(varOp, rank))
      return rewriter.notifyMatchFailure(divOp, "var not ReduceMean(.., -1)");
    mlir::Operation *sqOp = varOp->getOperand(0).getDefiningOp();
    bool isSquare = false;
    if (sqOp && sqOp->getName().getStringRef() == "onnx.Pow")
      isSquare = sqOp->getOperand(0) == d &&
                 getScalarFloatConstant(sqOp->getOperand(1)) == 2.0;
    else if (sqOp && sqOp->getName().getStringRef() == "onnx.Mul")
      isSquare = sqOp->getOperand(0) == d && sqOp->getOperand(1) == d;
    if (!isSquare)
      return rewriter.notifyMatchFailure(divOp, "var input not (x-mean)^2");

    // Affine tail: Mul(n, gamma) then optional Add(., beta).
    int64_t dim = xType.getDimSize(rank - 1);
    mlir::Value n = divOp->getResult(0);
    if (!n.hasOneUse())
      return rewriter.notifyMatchFailure(divOp, "normalized has many uses");
    mlir::Operation *mulOp = *n.getUsers().begin();
    if (mulOp->getName().getStringRef() != "onnx.Mul")
      return rewriter.notifyMatchFailure(divOp, "no scale Mul");
    mlir::Value gamma = otherOperand(mulOp, n);
    if (!gamma || !isPerChannel(gamma, dim) ||
        mulOp->getResult(0).getType() != xType)
      return rewriter.notifyMatchFailure(divOp, "scale not per-channel");
    mlir::Operation *last = mulOp;
    mlir::Value beta;
    if (mulOp->getResult(0).hasOneUse()) {
      mlir::Operation *addOp = *mulOp->getResult(0).getUsers().begin();
      if (addOp->getName().getStringRef() == "onnx.Add") {
        mlir::Value b = otherOperand(addOp, mulOp->getResult(0));
        if (b && isPerChannel(b, dim) &&
            addOp->getResult(0).getType() == xType) {
          beta = b;
          last = addOp;
        }
      }
    }

    mlir::OperationState state(last->getLoc(), "onnx.LayerNormalization");
    state.addOperands({x, gamma});
    if (beta)
      state.addOperands(beta);
    state.addTypes(xType);
    auto si64 = rewriter.getIntegerType(64, /*isSigned=*/true);
    state.addAttribute("axis", rewriter.getIntegerAttr(si64, -1));
    state.addAttribute("epsilon",
                       rewriter.getF32FloatAttr(static_cast<float>(*eps)));
    state.addAttribute("stash_type", rewriter.getIntegerAttr(si64, 1));
    for (llvm::StringRef name : {"node.outputs", "onnx_node_name"})
      if (mlir::Attribute a = last->getAttr(name))
        state.addAttribute(name, a);
    rewriter.setInsertionPoint(last);
    mlir::Operation *ln = rewriter.create(state);
    rewriter.replaceOp(last, ln->getResult(0));

    for (mlir::Operation *op :
         {mulOp, divOp, sqrtOp, epsAdd, varOp, sqOp, subOp, meanOp})
      if (op != last && op->use_empty())
        rewriter.eraseOp(op);

    LLVM_DEBUG(llvm::dbgs()
               << "[" DEBUG_TYPE "] fused at " << ln->getLoc() << "\n");
    ++NumLayerNormFused;
    return mlir::success();
  }
};

} // namespace

void populateLayerNormFusionPatterns(mlir::RewritePatternSet &patterns,
                                     mlir::MLIRContext *ctx) {
  patterns.add<DecomposedLayerNormToLayerNorm>(ctx);
}

} // namespace hip
} // namespace mlir
