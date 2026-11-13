/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- gelu_fusion.cpp - elementwise GELU chains to hip.gelu --------------===//
//
// A frontend that cannot keep GELU as one op emits it as primitive HIP ops.
// The match lives here, not in a frontend conversion, so every frontend (ONNX,
// Torch, ...) gets the fused kernel and none repeats the match.
//
// Tanh form:
//   Before:
//     %h  = hip.mul(%ctx) ins(%x, %c0_5)          // 0.5 * x
//     %x2 = hip.mul(%ctx) ins(%x, %x)             // x^3, or hip.pow {3.0}
//     %x3 = hip.mul(%ctx) ins(%x2, %x)
//     %c  = hip.mul(%ctx) ins(%x3, %c0_044715)
//     %s  = hip.add(%ctx) ins(%x, %c)
//     %t  = hip.mul(%ctx) ins(%s, %c0_7979)       // sqrt(2 / pi)
//     %th = hip.tanh(%ctx) ins(%t)
//     %p  = hip.add(%ctx) ins(%th, %c1)
//     %y  = hip.mul(%ctx) ins(%h, %p)
//   After:
//     %y = hip.gelu(%ctx) ins(%x) outs(%init) {approximate = "tanh"}
//
// Erf form:
//   Before:
//     %h  = hip.mul(%ctx) ins(%x, %c0_5)
//     %d  = hip.div(%ctx) ins(%x, %c1_4142)       // or hip.mul by 0.7071
//     %e  = hip.erf(%ctx) ins(%d)
//     %p  = hip.add(%ctx) ins(%e, %c1)
//     %y  = hip.mul(%ctx) ins(%h, %p)
//   After:
//     %y = hip.gelu(%ctx) ins(%x) outs(%init) {approximate = "none"}
//
// Every multiply and add commutes, so each operand order is matched. A scalar
// constant is a rank-0 or one-element tensor that is a hip.constant, an
// arith.constant, a tensor.splat of one, or a hip.cast, arith.truncf or
// arith.extf of any of those.
//
// Safety: each intermediate must have a single value use (a tensor.dim query
// is not a value use; it is retargeted to x). Every intermediate must have the
// type of x, so hip.gelu, a flat elementwise kernel, computes what the chain
// computed; the only other operands are the scalar constants, so the dynamic
// extents of the intermediates are those of x.
//
//===----------------------------------------------------------------------===//

#include "gelu_fusion.hpp"

#include "hip/Dialect/IR/HipDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

#include <cmath>
#include <optional>
#include <utility>

namespace hip {
namespace fusion_transform {
namespace {

// Same tolerance rationale as the ONNX GELU matchers: an f16 round trip moves
// sqrt(2/pi) and 0.044715 in the fourth digit.
constexpr double kConstantTolerance = 1e-3;
constexpr double kHalf = 0.5;
constexpr double kOne = 1.0;
constexpr double kCubic = 0.044715;
constexpr double kSqrt2OverPi = 0.7978845608028654;
constexpr double kSqrt2 = 1.4142135623730951;

bool nearlyEqual(double a, double b) {
  return std::abs(a - b) <= kConstantTolerance * std::abs(b);
}

std::optional<double> floatValue(mlir::DenseElementsAttr attr) {
  if (!attr || attr.getNumElements() != 1 ||
      !mlir::isa<mlir::FloatType>(attr.getElementType()))
    return std::nullopt;
  llvm::APFloat value = *attr.getValues<llvm::APFloat>().begin();
  bool lostInfo = false;
  value.convert(llvm::APFloat::IEEEdouble(), llvm::APFloat::rmNearestTiesToEven,
                &lostInfo);
  return value.convertToDouble();
}

/// The value of a scalar constant operand, looking through the wrappers the
/// frontends add. std::nullopt when `value` is not a single float constant.
std::optional<double> scalarConstant(mlir::Value value) {
  mlir::Operation *def = value.getDefiningOp();
  if (!def)
    return std::nullopt;
  if (auto constant = mlir::dyn_cast<mlir::hip::ConstantOp>(def))
    return floatValue(mlir::dyn_cast_or_null<mlir::DenseElementsAttr>(
        constant.getValueAttr()));
  if (auto constant = mlir::dyn_cast<mlir::arith::ConstantOp>(def)) {
    if (auto scalar = mlir::dyn_cast<mlir::FloatAttr>(constant.getValue())) {
      llvm::APFloat v = scalar.getValue();
      bool lostInfo = false;
      v.convert(llvm::APFloat::IEEEdouble(), llvm::APFloat::rmNearestTiesToEven,
                &lostInfo);
      return v.convertToDouble();
    }
    return floatValue(
        mlir::dyn_cast<mlir::DenseElementsAttr>(constant.getValue()));
  }
  if (auto splat = mlir::dyn_cast<mlir::tensor::SplatOp>(def))
    return scalarConstant(splat.getInput());
  if (auto cast = mlir::dyn_cast<mlir::hip::CastOp>(def))
    return scalarConstant(cast.getInput());
  // The Torch frontend splats arith.truncf of an f64 constant.
  if (auto trunc = mlir::dyn_cast<mlir::arith::TruncFOp>(def))
    return scalarConstant(trunc.getIn());
  if (auto ext = mlir::dyn_cast<mlir::arith::ExtFOp>(def))
    return scalarConstant(ext.getIn());
  return std::nullopt;
}

bool isScalar(mlir::Value value, double expected) {
  std::optional<double> v = scalarConstant(value);
  return v && nearlyEqual(*v, expected);
}

/// For a commutative binary op with one scalar constant operand equal to
/// `expected`, the other operand; null when that does not hold.
template <typename BinaryOp>
mlir::Value otherThanConstant(BinaryOp op, double expected) {
  bool lhsConst = isScalar(op.getLhs(), expected);
  bool rhsConst = isScalar(op.getRhs(), expected);
  if (lhsConst == rhsConst)
    return {};
  return lhsConst ? op.getRhs() : op.getLhs();
}

/// True when `value` has exactly one value use, by `consumer`. A tensor.dim
/// reads only the type and does not count.
bool onlyUsedBy(mlir::Value value, mlir::Operation *consumer) {
  mlir::Operation *seen = nullptr;
  for (mlir::OpOperand &use : value.getUses()) {
    if (mlir::isa<mlir::tensor::DimOp>(use.getOwner()))
      continue;
    if (seen && seen != use.getOwner())
      return false;
    if (seen == use.getOwner())
      return false; // the consumer reads it twice
    seen = use.getOwner();
  }
  return seen == consumer;
}

/// The op that defines `value` when it is an `OpTy` whose only value use is
/// `consumer`.
template <typename OpTy>
OpTy singleUseDef(mlir::Value value, mlir::Operation *consumer) {
  auto op = value.getDefiningOp<OpTy>();
  return op && onlyUsedBy(value, consumer) ? op : OpTy();
}

struct Chain {
  mlir::Value x;
  mlir::StringRef approximate;
  // Consumers first, so each op is erased after everything that reads it.
  llvm::SmallVector<mlir::Operation *> ops;
};

/// x^3 as hip.pow {3.0}(x), or as x*x*x in either grouping.
bool matchCube(mlir::Value cube, mlir::Operation *consumer, mlir::Value x,
               Chain &chain) {
  if (auto pow = singleUseDef<mlir::hip::PowOp>(cube, consumer)) {
    if (pow.getInput() != x ||
        !nearlyEqual(pow.getExponent().convertToDouble(), 3.0))
      return false;
    chain.ops.push_back(pow);
    return true;
  }
  auto outer = singleUseDef<mlir::hip::MulOp>(cube, consumer);
  if (!outer)
    return false;
  for (auto [square, rest] : {std::pair(outer.getLhs(), outer.getRhs()),
                              std::pair(outer.getRhs(), outer.getLhs())}) {
    if (rest != x)
      continue;
    auto inner = singleUseDef<mlir::hip::MulOp>(square, outer);
    if (!inner || inner.getLhs() != x || inner.getRhs() != x)
      continue;
    chain.ops.push_back(outer);
    chain.ops.push_back(inner);
    return true;
  }
  return false;
}

/// tanh(sqrt(2/pi) * (x + 0.044715 * x^3)): `argument` is tanh's operand.
bool matchTanhArgument(mlir::Value argument, mlir::Operation *consumer,
                       mlir::Value x, Chain &chain) {
  auto scale = singleUseDef<mlir::hip::MulOp>(argument, consumer);
  if (!scale)
    return false;
  mlir::Value inner = otherThanConstant(scale, kSqrt2OverPi);
  if (!inner)
    return false;
  auto sum = singleUseDef<mlir::hip::AddOp>(inner, scale);
  if (!sum)
    return false;
  mlir::Value cubicTerm;
  if (sum.getLhs() == x)
    cubicTerm = sum.getRhs();
  else if (sum.getRhs() == x)
    cubicTerm = sum.getLhs();
  else
    return false;
  auto cubicScale = singleUseDef<mlir::hip::MulOp>(cubicTerm, sum);
  if (!cubicScale)
    return false;
  mlir::Value cube = otherThanConstant(cubicScale, kCubic);
  if (!cube)
    return false;
  chain.ops.push_back(scale);
  chain.ops.push_back(sum);
  chain.ops.push_back(cubicScale);
  return matchCube(cube, cubicScale, x, chain);
}

/// erf(x / sqrt(2)) or erf(x * (1 / sqrt(2))): `argument` is erf's operand.
bool matchErfArgument(mlir::Value argument, mlir::Operation *consumer,
                      mlir::Value x, Chain &chain) {
  if (auto div = singleUseDef<mlir::hip::DivOp>(argument, consumer)) {
    if (div.getLhs() != x || !isScalar(div.getRhs(), kSqrt2))
      return false;
    chain.ops.push_back(div);
    return true;
  }
  if (auto mul = singleUseDef<mlir::hip::MulOp>(argument, consumer)) {
    if (otherThanConstant(mul, 1.0 / kSqrt2) != x)
      return false;
    chain.ops.push_back(mul);
    return true;
  }
  return false;
}

struct GeluFusion : public mlir::OpRewritePattern<mlir::hip::MulOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::hip::MulOp root,
                  mlir::PatternRewriter &rewriter) const override {
    Chain chain;
    bool matched = false;
    // root = (0.5 * x) * (1 + tanh(...) | 1 + erf(...)), either operand order.
    for (auto [halfValue, restValue] :
         {std::pair(root.getLhs(), root.getRhs()),
          std::pair(root.getRhs(), root.getLhs())}) {
      auto half = singleUseDef<mlir::hip::MulOp>(halfValue, root);
      auto plusOne = singleUseDef<mlir::hip::AddOp>(restValue, root);
      if (!half || !plusOne)
        continue;
      mlir::Value x = otherThanConstant(half, kHalf);
      mlir::Value activation = otherThanConstant(plusOne, kOne);
      if (!x || !activation)
        continue;

      Chain attempt;
      attempt.x = x;
      attempt.ops.push_back(half);
      attempt.ops.push_back(plusOne);
      if (auto tanh = singleUseDef<mlir::hip::TanhOp>(activation, plusOne)) {
        attempt.ops.push_back(tanh);
        attempt.approximate = "tanh";
        if (!matchTanhArgument(tanh.getX(), tanh, x, attempt))
          continue;
      } else if (auto erf =
                     singleUseDef<mlir::hip::ErfOp>(activation, plusOne)) {
        attempt.ops.push_back(erf);
        attempt.approximate = "none";
        if (!matchErfArgument(erf.getX(), erf, x, attempt))
          continue;
      } else {
        continue;
      }
      chain = std::move(attempt);
      matched = true;
      break;
    }
    if (!matched)
      return rewriter.notifyMatchFailure(root, "not a GELU chain");

    mlir::Value x = chain.x;
    auto xType = mlir::dyn_cast<mlir::RankedTensorType>(x.getType());
    auto resultType =
        mlir::dyn_cast<mlir::RankedTensorType>(root->getResult(0).getType());
    if (!xType || xType != resultType)
      return rewriter.notifyMatchFailure(root, "GELU does not broadcast");
    mlir::Type elemType = xType.getElementType();
    if (!elemType.isF16() && !elemType.isF32() && !elemType.isBF16())
      return rewriter.notifyMatchFailure(root,
                                         "GELU supports only f16, f32, bf16");
    for (mlir::Operation *op : chain.ops) {
      // The HIP context is operand 0 of every HIP op.
      if (op->getOperand(0) != root.getCtx() ||
          op->getResult(0).getType() != xType)
        return rewriter.notifyMatchFailure(
            root, "GELU ops must share a context and the type of x");
    }

    // Initial values that exist only to feed the erased ops.
    llvm::SmallVector<mlir::Operation *> initDefs;
    for (mlir::Operation *op : chain.ops)
      for (mlir::Value init :
           mlir::cast<mlir::DestinationStyleOpInterface>(op).getDpsInits())
        if (mlir::Operation *def = init.getDefiningOp())
          initDefs.push_back(def);

    for (mlir::Operation *op : chain.ops)
      rewriter.replaceUsesWithIf(
          op->getResult(0), x, [](mlir::OpOperand &operand) {
            return mlir::isa<mlir::tensor::DimOp>(operand.getOwner());
          });
    mlir::hip::GeluOp gelu = mlir::hip::GeluOp::create(
        rewriter, root.getLoc(), resultType, root.getCtx(), x, root.getOutput(),
        rewriter.getStringAttr(chain.approximate));
    rewriter.replaceOp(root, gelu.getResult(0));
    for (mlir::Operation *op : chain.ops)
      rewriter.eraseOp(op);
    llvm::SmallPtrSet<mlir::Operation *, 8> seen;
    for (mlir::Operation *def : initDefs)
      if (seen.insert(def).second && mlir::isOpTriviallyDead(def))
        rewriter.eraseOp(def);
    return mlir::success();
  }
};

} // namespace

void populateGeluFusionPattern(mlir::RewritePatternSet &patterns,
                               mlir::PatternBenefit benefit) {
  patterns.add<GeluFusion>(patterns.getContext(), benefit);
}

} // namespace fusion_transform
} // namespace hip
