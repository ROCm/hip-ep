/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- GroupNormFusion.cpp - InstanceNorm-spelled GroupNorm -> GroupNorm --===//
//
// PyTorch exports nn.GroupNorm over an NCHW activation as an instance norm of
// a [N, G, C/G * spatial] view plus the per-channel affine:
//
//   r = Reshape(x, [N, G, -1])
//   n = InstanceNormalization(r, ones[G], zeros[G], epsilon)
//   y = Add(Mul(Reshape(n, shape(x)), gamma[C, 1, 1]), beta[C, 1, 1])
//
// and diffusers' ResNet blocks follow it with SiLU, spelled Mul(y, Sigmoid(y)).
// Lowered as written that is the instance norm plus two elementwise passes over
// the activation, four with the SiLU. com.microsoft.GroupNorm with
// channels_last = 0 computes the same in one statistics read and one
// read-write, so the chain is rewritten into it, taking the SiLU along when y
// feeds nothing else.
//
// Only the plain spelling is taken: the instance norm's own scale and bias
// must be ones and zeros (a per-group affine would have to be folded into
// gamma and beta), and gamma / beta must broadcast along the channel axis
// only. Rooted on onnx.InstanceNormalization and run in the pre-lowering set,
// while the constants are still onnx.Constant.
//
//===----------------------------------------------------------------------===//

#include "OnnxToHipUtils.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"

#include <iterator>

#undef DEBUG_TYPE
#define DEBUG_TYPE "group-norm-fusion"

STATISTIC(NumGroupNormFused,
          "Number of InstanceNorm-spelled GroupNorm chains rewritten to "
          "com.microsoft.GroupNorm");

namespace mlir {
namespace hip {

namespace {

bool isOnnxOp(mlir::Operation *op, llvm::StringRef name) {
  return op && op->getName().getStringRef() == name;
}

/// Whether `v` is an onnx.Constant whose every element equals `expected`.
bool isSplatConstant(mlir::Value v, double expected) {
  mlir::Operation *def = v.getDefiningOp();
  if (!isOnnxOp(def, "onnx.Constant"))
    return false;
  auto attr =
      mlir::dyn_cast_or_null<mlir::DenseElementsAttr>(def->getAttr("value"));
  if (!attr || !mlir::isa<mlir::FloatType>(attr.getElementType()))
    return false;
  for (const llvm::APFloat &x : attr.getValues<llvm::APFloat>())
    if (x.convertToDouble() != expected)
      return false;
  return true;
}

/// For a binary op with operand `x`, the other operand when it broadcasts
/// along `x`'s channel axis (1) only, i.e. has shape [C, 1, ...] or
/// [1, C, 1, ...] against a static NCHW `x`.
mlir::Value channelOperand(mlir::Operation *op, mlir::Value x) {
  if (!op || op->getNumOperands() != 2 || op->getNumResults() != 1)
    return nullptr;
  mlir::Value other;
  if (op->getOperand(0) == x)
    other = op->getOperand(1);
  else if (op->getOperand(1) == x)
    other = op->getOperand(0);
  else
    return nullptr;
  auto xType = mlir::cast<mlir::RankedTensorType>(x.getType());
  auto oType = mlir::dyn_cast<mlir::RankedTensorType>(other.getType());
  if (!oType || !oType.hasStaticShape() ||
      oType.getElementType() != xType.getElementType() ||
      op->getResult(0).getType() != xType)
    return nullptr;
  const int64_t rank = xType.getRank();
  const int64_t oRank = oType.getRank();
  if (oRank < rank - 1 || oRank > rank)
    return nullptr;
  for (int64_t d : llvm::seq<int64_t>(0, oRank)) {
    const int64_t xd = d + (rank - oRank); // right-aligned broadcast
    const int64_t want = xd == 1 ? xType.getDimSize(1) : 1;
    if (oType.getDimSize(d) != want)
      return nullptr;
  }
  return other;
}

mlir::IntegerAttr si64Attr(mlir::MLIRContext *ctx, int64_t v) {
  return mlir::IntegerAttr::get(
      mlir::IntegerType::get(ctx, 64, mlir::IntegerType::Signed), v);
}

mlir::Value reshapeTo1D(mlir::PatternRewriter &rewriter, mlir::Location loc,
                        mlir::Value v, int64_t length) {
  auto type = mlir::cast<mlir::RankedTensorType>(v.getType());
  auto target = mlir::RankedTensorType::get({length}, type.getElementType());
  auto shapeType = mlir::RankedTensorType::get({1}, rewriter.getI64Type());
  mlir::OperationState constState(loc, "onnx.Constant");
  constState.addTypes(shapeType);
  constState.addAttribute(
      "value",
      mlir::DenseElementsAttr::get(shapeType, llvm::ArrayRef<int64_t>{length}));
  mlir::Value shape = rewriter.create(constState)->getResult(0);
  mlir::OperationState state(loc, "onnx.Reshape");
  state.addOperands({v, shape});
  state.addTypes(target);
  state.addAttribute("allowzero", si64Attr(rewriter.getContext(), 0));
  return rewriter.create(state)->getResult(0);
}

struct InstanceNormGroupNormToGroupNorm : public mlir::RewritePattern {
  InstanceNormGroupNormToGroupNorm(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.InstanceNormalization", /*benefit=*/2, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *inorm,
                  mlir::PatternRewriter &rewriter) const override {
    if (inorm->getNumOperands() != 3 || inorm->getNumResults() != 1)
      return rewriter.notifyMatchFailure(inorm, "inorm.arity");
    if (!isSplatConstant(inorm->getOperand(1), 1.0) ||
        !isSplatConstant(inorm->getOperand(2), 0.0))
      return rewriter.notifyMatchFailure(inorm, "inorm.affine_not_identity");

    // r = Reshape(x, [N, G, L]) over a static NCHW x.
    mlir::Operation *view = inorm->getOperand(0).getDefiningOp();
    if (!isOnnxOp(view, "onnx.Reshape") || !inorm->getOperand(0).hasOneUse())
      return rewriter.notifyMatchFailure(inorm, "input.not_reshape");
    mlir::Value x = view->getOperand(0);
    auto xType = mlir::dyn_cast<mlir::RankedTensorType>(x.getType());
    auto rType =
        mlir::dyn_cast<mlir::RankedTensorType>(inorm->getOperand(0).getType());
    if (!xType || !xType.hasStaticShape() || xType.getRank() < 3 || !rType ||
        !rType.hasStaticShape() || rType.getRank() != 3)
      return rewriter.notifyMatchFailure(inorm, "shapes.not_static");
    const int64_t n = xType.getDimSize(0), c = xType.getDimSize(1);
    const int64_t groups = rType.getDimSize(1);
    if (rType.getDimSize(0) != n || groups <= 0 || c % groups != 0 ||
        rType.getDimSize(2) * groups != xType.getNumElements() / n)
      return rewriter.notifyMatchFailure(inorm, "view.not_groups");

    // y = Add(Mul(Reshape(n, shape(x)), gamma), beta).
    mlir::Value normed = inorm->getResult(0);
    if (!normed.hasOneUse())
      return rewriter.notifyMatchFailure(inorm, "inorm.not_one_use");
    mlir::Operation *unview = *normed.getUsers().begin();
    if (!isOnnxOp(unview, "onnx.Reshape") ||
        unview->getResult(0).getType() != xType ||
        !unview->getResult(0).hasOneUse())
      return rewriter.notifyMatchFailure(inorm, "output.not_reshape_back");
    mlir::Operation *mul = *unview->getResult(0).getUsers().begin();
    if (!isOnnxOp(mul, "onnx.Mul"))
      return rewriter.notifyMatchFailure(inorm, "affine.no_mul");
    mlir::Value gamma = channelOperand(mul, unview->getResult(0));
    if (!gamma || !mul->getResult(0).hasOneUse())
      return rewriter.notifyMatchFailure(inorm, "affine.gamma");
    mlir::Operation *add = *mul->getResult(0).getUsers().begin();
    if (!isOnnxOp(add, "onnx.Add"))
      return rewriter.notifyMatchFailure(inorm, "affine.no_add");
    mlir::Value beta = channelOperand(add, mul->getResult(0));
    if (!beta)
      return rewriter.notifyMatchFailure(inorm, "affine.beta");

    // SiLU when y feeds exactly Sigmoid(y) and Mul(y, Sigmoid(y)).
    mlir::Value y = add->getResult(0);
    mlir::Operation *sigmoid = nullptr, *gate = nullptr;
    if (!y.hasOneUse() && std::distance(y.use_begin(), y.use_end()) == 2) {
      for (mlir::Operation *user : y.getUsers())
        if (isOnnxOp(user, "onnx.Sigmoid"))
          sigmoid = user;
      if (sigmoid && sigmoid->getResult(0).hasOneUse()) {
        mlir::Operation *m = *sigmoid->getResult(0).getUsers().begin();
        if (isOnnxOp(m, "onnx.Mul") && m->getNumOperands() == 2 &&
            llvm::is_contained(m->getOperands(), y) &&
            llvm::is_contained(m->getOperands(), sigmoid->getResult(0)) &&
            llvm::is_contained(y.getUsers(), m))
          gate = m;
      }
      if (!gate)
        sigmoid = nullptr;
    }
    mlir::Operation *last = gate ? gate : add;

    mlir::Location loc = inorm->getLoc();
    rewriter.setInsertionPoint(last);
    mlir::Value gamma1 = reshapeTo1D(rewriter, loc, gamma, c);
    mlir::Value beta1 = reshapeTo1D(rewriter, loc, beta, c);
    float epsilon = 1e-5f;
    if (auto eps = inorm->getAttrOfType<mlir::FloatAttr>("epsilon"))
      epsilon = static_cast<float>(eps.getValueAsDouble());

    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::OperationState state(loc, "onnx.Custom");
    state.addOperands({x, gamma1, beta1});
    state.addTypes(xType);
    state.addAttribute("function_name", rewriter.getStringAttr("GroupNorm"));
    state.addAttribute("domain_name", rewriter.getStringAttr("com.microsoft"));
    state.addAttribute("groups", si64Attr(ctx, groups));
    state.addAttribute("activation", si64Attr(ctx, gate ? 1 : 0));
    state.addAttribute("channels_last", si64Attr(ctx, 0));
    state.addAttribute("epsilon", rewriter.getF32FloatAttr(epsilon));
    mlir::Value groupNorm = rewriter.create(state)->getResult(0);
    rewriter.replaceOp(last, groupNorm);

    // Users before producers; `last` went with the replacement.
    for (mlir::Operation *op :
         {sigmoid, gate ? add : nullptr, mul, unview, inorm, view})
      if (op && op->use_empty())
        rewriter.eraseOp(op);
    ++NumGroupNormFused;
    return mlir::success();
  }
};

} // namespace

void populateGroupNormFusionPatterns(RewritePatternSet &patterns,
                                     MLIRContext *ctx) {
  patterns.add<InstanceNormGroupNormToGroupNorm>(ctx);
}

} // namespace hip
} // namespace mlir
