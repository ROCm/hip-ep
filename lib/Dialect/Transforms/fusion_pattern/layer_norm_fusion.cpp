/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- layer_norm_fusion.cpp - primitive LayerNorm chains to one hip op ---===//
//
// clang-format off
// Before:
//   %mean = hip.reduce_mean(%ctx) ins(%x, %axes) {keepdims = 1, reduced_axes = [-1]}
//   %d    = hip.sub(%ctx) ins(%x, %mean)
//   %sq   = hip.mul(%ctx) ins(%d, %d)          // or hip.pow {exponent = 2}
//   %var  = hip.reduce_mean(%ctx) ins(%sq, %axes)
//   %a    = hip.add(%ctx) ins(%var, %eps)
//   %n    = hip.mul(%ctx) ins(%d, hip.reciprocal(hip.sqrt(%a)))
//   %y    = hip.add(%ctx) ins(hip.mul(%ctx) ins(%n, %gamma), %beta)
// After:
//   %y = hip.layer_norm(%ctx) ins(%x, %gamma, %beta) {axis = -1, epsilon}
// clang-format on
//
// The normalization step may also be hip.div by sqrt, hip.pow {0.5} or
// {-0.5}, or hip.div(1, sqrt). A multiply by 1 in front of the reciprocal is
// the broadcasting Div(1, sqrt) rewrite. The affine tail is optional; with
// no per-channel scale the norm gets a constant scale of ones.
//
// A rank>4 elementwise op is packed to rank 4 as collapse -> hip op ->
// expand before this pass. One collapse/expand pair is skipped so the match
// still sees the original tensors, provided its unpacked side has x's shape.
//
// A same-shape hip.add that produces the norm input becomes
// hip.skip_layer_norm. Later users of that add, in the same block and after
// the norm, read the sum the kernel writes. A broadcasting add, or one whose
// dynamic extents are not proven equal, is left alone.
//
//===----------------------------------------------------------------------===//

#include "layer_norm_fusion.hpp"

#include "fusion_utils.hpp"
#include "runtime_shape.hpp"

#include "hip/Dialect/IR/HipDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"

#include <optional>

namespace hip {
namespace fusion_transform {
namespace {

using mlir::Operation;
using mlir::Value;

void keep(llvm::SmallVectorImpl<Operation *> &ops,
          llvm::SmallPtrSetImpl<Operation *> &seen, Operation *op) {
  if (op && seen.insert(op).second)
    ops.push_back(op);
}

/// Erases the ops of \p ops that became dead, until none does. An op with
/// side effects (a call producing a destination tensor) stays.
void eraseDeadOps(mlir::PatternRewriter &rewriter,
                  llvm::SmallVectorImpl<Operation *> &ops) {
  bool progress = true;
  while (progress) {
    progress = false;
    for (Operation *&op : ops) {
      if (!op || !mlir::isOpTriviallyDead(op))
        continue;
      rewriter.eraseOp(op);
      op = nullptr;
      progress = true;
    }
  }
}

/// Skip one rank>4 pack step: collapse into the op, expand out of it.
/// fuseLayerNorm checks afterwards that each skipped reshape is a pack.
Value definingCompute(Value v, llvm::SmallVectorImpl<Operation *> &pack,
                      llvm::SmallPtrSetImpl<Operation *> &seen) {
  if (auto collapse = v.getDefiningOp<mlir::tensor::CollapseShapeOp>()) {
    keep(pack, seen, collapse);
    v = collapse.getSrc();
  }
  if (auto expand = v.getDefiningOp<mlir::tensor::ExpandShapeOp>()) {
    keep(pack, seen, expand);
    v = expand.getSrc();
  }
  return v;
}

/// True when every reshape in \p ops is a pack of an x-shaped value: its
/// unpacked side has x's extents, with 1 allowed on the reduced last axis.
/// Reshapes keep row-major order, so a packed elementwise op then computes
/// what the unpacked op would. A reshape to any other shape could move
/// elements across rows ([1, 2, 2] -> [1, 4] -> [1, 1, 4] reduces both rows
/// together).
bool reshapesArePacks(llvm::ArrayRef<Operation *> ops,
                      mlir::RankedTensorType xType) {
  auto unpacked = [&](mlir::Type type) {
    auto t = mlir::dyn_cast<mlir::RankedTensorType>(type);
    if (!t || t.getRank() != xType.getRank())
      return false;
    int64_t last = t.getRank() - 1;
    for (int64_t i = 0; i < last; ++i)
      if (t.getDimSize(i) != xType.getDimSize(i))
        return false;
    return t.getDimSize(last) == xType.getDimSize(last) ||
           t.getDimSize(last) == 1;
  };
  for (Operation *op : ops) {
    if (auto collapse =
            mlir::dyn_cast_or_null<mlir::tensor::CollapseShapeOp>(op)) {
      if (!unpacked(collapse.getSrcType()))
        return false;
    } else if (auto expand =
                   mlir::dyn_cast_or_null<mlir::tensor::ExpandShapeOp>(op)) {
      if (!unpacked(expand.getResultType()))
        return false;
    }
  }
  return true;
}

Operation *onlyUser(Value v) {
  Operation *seen = nullptr;
  for (mlir::OpOperand &use : v.getUses()) {
    if (mlir::isa<mlir::tensor::DimOp>(use.getOwner()))
      continue;
    if (seen)
      return nullptr;
    seen = use.getOwner();
  }
  return seen;
}

/// The value consumers see. The pack restore is the expand of the hip op.
std::pair<Operation *, Value> producedOf(Operation *op) {
  Value result = op->getResult(0);
  Operation *user = onlyUser(result);
  if (auto expand = mlir::dyn_cast_or_null<mlir::tensor::ExpandShapeOp>(user))
    if (expand.getSrc() == result)
      return {expand, expand.getResult()};
  return {op, result};
}

bool sameContext(Operation *op, Value ctx) {
  return op && op->getNumOperands() > 0 && op->getOperand(0) == ctx;
}

std::optional<double> integerScalar(Value v) {
  auto constant =
      mlir::dyn_cast_or_null<mlir::hip::ConstantOp>(v.getDefiningOp());
  if (!constant)
    return std::nullopt;
  auto dense =
      mlir::dyn_cast_or_null<mlir::DenseElementsAttr>(constant.getValueAttr());
  if (!dense || dense.getNumElements() != 1)
    return std::nullopt;
  auto intType = mlir::dyn_cast<mlir::IntegerType>(dense.getElementType());
  if (!intType)
    return std::nullopt;
  // Unsigned and i1 zero-extend; signed integers sign-extend.
  return (*dense.getValues<llvm::APInt>().begin())
      .roundToDouble(/*isSigned=*/!intType.isUnsigned() &&
                     intType.getWidth() != 1);
}

/// Scalar float, including a cast rounded to its result type. An integer
/// constant cast to float uses the integer's own extension, then the float
/// rounding (an f32 1e-8 is 0 in f16; ui8 255 cast to f32 is 255).
std::optional<double> epsilonOf(Value v) {
  if (std::optional<double> value = getScalarFloatConstant(v))
    return value;
  auto cast = mlir::dyn_cast_or_null<mlir::hip::CastOp>(v.getDefiningOp());
  if (!cast)
    return std::nullopt;
  auto shaped = mlir::dyn_cast<mlir::ShapedType>(cast->getResult(0).getType());
  auto floatType =
      shaped ? mlir::dyn_cast<mlir::FloatType>(shaped.getElementType())
             : nullptr;
  if (!floatType)
    return std::nullopt;
  std::optional<double> source = integerScalar(cast.getInput());
  if (!source)
    return std::nullopt;
  llvm::APFloat rounded(*source);
  bool lostInfo = false;
  rounded.convert(floatType.getFloatSemantics(),
                  llvm::APFloat::rmNearestTiesToEven, &lostInfo);
  return rounded.convertToDouble();
}

bool isLastAxisMean(Operation *op, Value data, int64_t rank) {
  auto mean = mlir::dyn_cast_or_null<mlir::hip::ReduceMeanOp>(op);
  if (!mean || mean.getData() != data || mean.getKeepdims() != 1)
    return false;
  auto axes = mean.getReducedAxes();
  if (!axes || axes->size() != 1)
    return false;
  int64_t axis = (*axes)[0];
  if (axis < 0)
    axis += rank;
  return axis == rank - 1;
}

bool isPerChannel(Value v, int64_t dim) {
  auto type = mlir::dyn_cast<mlir::RankedTensorType>(v.getType());
  return type && type.getRank() == 1 && !type.isDynamicDim(0) &&
         type.getDimSize(0) == dim;
}

Value matchStd(Value v, llvm::SmallVectorImpl<Operation *> &chain,
               llvm::SmallPtrSetImpl<Operation *> &seen) {
  Value compute = definingCompute(v, chain, seen);
  Operation *op = compute.getDefiningOp();
  if (auto sqrt = mlir::dyn_cast_or_null<mlir::hip::SqrtOp>(op)) {
    keep(chain, seen, sqrt);
    return sqrt.getX();
  }
  if (auto pow = mlir::dyn_cast_or_null<mlir::hip::PowOp>(op)) {
    if (pow.getExponent().convertToDouble() == 0.5) {
      keep(chain, seen, pow);
      return pow.getInput();
    }
  }
  return nullptr;
}

Value matchReciprocalSqrt(Value compute,
                          llvm::SmallVectorImpl<Operation *> &chain,
                          llvm::SmallPtrSetImpl<Operation *> &seen) {
  Operation *op = compute.getDefiningOp();
  Value std;
  if (auto reciprocal = mlir::dyn_cast_or_null<mlir::hip::ReciprocalOp>(op)) {
    std = reciprocal.getX();
  } else if (auto div = mlir::dyn_cast_or_null<mlir::hip::DivOp>(op)) {
    if (getScalarFloatConstant(div.getLhs()) != 1.0)
      return nullptr;
    std = div.getRhs();
  } else {
    return nullptr;
  }
  Value a = matchStd(std, chain, seen);
  if (a)
    keep(chain, seen, op);
  return a;
}

/// 1/sqrt(a): reciprocal, div by one, pow -0.5, or a multiply by 1 in front
/// of the reciprocal (the broadcasting Div(1, sqrt) rewrite).
Value matchInvStd(Value v, llvm::SmallVectorImpl<Operation *> &chain,
                  llvm::SmallPtrSetImpl<Operation *> &seen) {
  Value compute = definingCompute(v, chain, seen);
  if (auto pow =
          mlir::dyn_cast_or_null<mlir::hip::PowOp>(compute.getDefiningOp())) {
    if (pow.getExponent().convertToDouble() == -0.5) {
      keep(chain, seen, pow);
      return pow.getInput();
    }
    return nullptr;
  }
  if (auto mul =
          mlir::dyn_cast_or_null<mlir::hip::MulOp>(compute.getDefiningOp())) {
    Value rest;
    if (getScalarFloatConstant(mul.getLhs()) == 1.0)
      rest = mul.getRhs();
    else if (getScalarFloatConstant(mul.getRhs()) == 1.0)
      rest = mul.getLhs();
    else
      return nullptr;
    Value a =
        matchReciprocalSqrt(definingCompute(rest, chain, seen), chain, seen);
    if (!a)
      return nullptr;
    keep(chain, seen, mul);
    return a;
  }
  return matchReciprocalSqrt(compute, chain, seen);
}

/// Points the tensor.dim queries of a chain value at x. A chain value of x's
/// rank has x's extents except on the reduced axis, which its type states as
/// a static 1. A packed value of another rank keeps its queries, and so the
/// ops that compute it.
void retargetChainShapeQueries(mlir::PatternRewriter &rewriter, Value from,
                               Value x) {
  if (from.getType() == x.getType()) {
    retargetShapeQueries(rewriter, from, x);
    return;
  }
  auto fromType = mlir::dyn_cast<mlir::RankedTensorType>(from.getType());
  auto xType = mlir::cast<mlir::RankedTensorType>(x.getType());
  if (!fromType || fromType.getRank() != xType.getRank())
    return;
  mlir::OpBuilder::InsertionGuard guard(rewriter);
  for (mlir::OpOperand &use : llvm::make_early_inc_range(from.getUses())) {
    auto dim = mlir::dyn_cast<mlir::tensor::DimOp>(use.getOwner());
    if (!dim)
      continue;
    std::optional<int64_t> index = dim.getConstantIndex();
    if (!index || *index < 0 || *index >= fromType.getRank())
      continue;
    if (fromType.isDynamicDim(*index)) {
      rewriter.modifyOpInPlace(dim, [&] { use.set(x); });
    } else {
      rewriter.setInsertionPoint(dim);
      rewriter.replaceOpWithNewOp<mlir::arith::ConstantIndexOp>(
          dim, fromType.getDimSize(*index));
    }
  }
}

bool operandIs(Value operand, Value visible, Value compute);

/// The operand of a binary HIP op that is not the norm result. The norm side
/// may be the value itself or the collapse that packs it.
Value affineOther(Operation *op, Value visible, Value compute) {
  Value lhs, rhs;
  if (auto mul = mlir::dyn_cast<mlir::hip::MulOp>(op)) {
    lhs = mul.getLhs();
    rhs = mul.getRhs();
  } else if (auto add = mlir::dyn_cast<mlir::hip::AddOp>(op)) {
    lhs = add.getLhs();
    rhs = add.getRhs();
  } else {
    return nullptr;
  }
  if (operandIs(lhs, visible, compute))
    return rhs;
  if (operandIs(rhs, visible, compute))
    return lhs;
  return nullptr;
}

bool operandIs(Value operand, Value visible, Value compute) {
  if (operand == visible || operand == compute)
    return true;
  if (auto collapse = operand.getDefiningOp<mlir::tensor::CollapseShapeOp>())
    return collapse.getSrc() == visible || collapse.getSrc() == compute;
  return false;
}

mlir::LogicalResult fuseLayerNorm(Operation *root, Value rawD, Value rawA,
                                  llvm::SmallVectorImpl<Operation *> &chain,
                                  llvm::SmallPtrSetImpl<Operation *> &seen,
                                  mlir::PatternRewriter &rewriter) {
  Value ctx = root->getOperand(0);
  Value dCompute = definingCompute(rawD, chain, seen);
  auto sub = mlir::dyn_cast_or_null<mlir::hip::SubOp>(dCompute.getDefiningOp());
  if (!sub || !sameContext(sub, ctx))
    return rewriter.notifyMatchFailure(root, "numerator not Sub");
  keep(chain, seen, sub);

  Value x = definingCompute(sub.getLhs(), chain, seen);
  // The LayerNorm runtime reads any 2-byte element as fp16, so bf16 stays
  // unfused.
  auto xType = mlir::dyn_cast<mlir::RankedTensorType>(x.getType());
  if (!xType || xType.getRank() < 1 ||
      !(xType.getElementType().isF16() || xType.getElementType().isF32()) ||
      xType.isDynamicDim(xType.getRank() - 1))
    return rewriter.notifyMatchFailure(root, "x type");
  int64_t rank = xType.getRank();
  int64_t dim = xType.getDimSize(rank - 1);

  Value mean = definingCompute(sub.getRhs(), chain, seen);
  Operation *meanOp = mean.getDefiningOp();
  if (!isLastAxisMean(meanOp, x, rank) || !sameContext(meanOp, ctx))
    return rewriter.notifyMatchFailure(root, "mean not ReduceMean(x, -1)");
  keep(chain, seen, meanOp);

  Value aCompute = definingCompute(rawA, chain, seen);
  auto epsAdd =
      mlir::dyn_cast_or_null<mlir::hip::AddOp>(aCompute.getDefiningOp());
  if (!epsAdd || !sameContext(epsAdd, ctx))
    return rewriter.notifyMatchFailure(root, "sqrt input not Add");
  keep(chain, seen, epsAdd);
  std::optional<double> eps;
  Value var;
  if ((eps = epsilonOf(epsAdd.getLhs())))
    var = epsAdd.getRhs();
  else if ((eps = epsilonOf(epsAdd.getRhs())))
    var = epsAdd.getLhs();
  else
    return rewriter.notifyMatchFailure(root, "epsilon not a scalar const");

  Value varCompute = definingCompute(var, chain, seen);
  auto varOp = mlir::dyn_cast_or_null<mlir::hip::ReduceMeanOp>(
      varCompute.getDefiningOp());
  if (!varOp || !sameContext(varOp, ctx))
    return rewriter.notifyMatchFailure(root, "var not ReduceMean");
  Value sqVisible = varOp.getData();
  Value sqCompute = definingCompute(sqVisible, chain, seen);
  Operation *sqOp = sqCompute.getDefiningOp();
  bool isSquare = false;
  if (auto mul = mlir::dyn_cast_or_null<mlir::hip::MulOp>(sqOp))
    isSquare = definingCompute(mul.getLhs(), chain, seen) == dCompute &&
               definingCompute(mul.getRhs(), chain, seen) == dCompute &&
               sameContext(mul, ctx);
  else if (auto pow = mlir::dyn_cast_or_null<mlir::hip::PowOp>(sqOp))
    isSquare = definingCompute(pow.getInput(), chain, seen) == dCompute &&
               pow.getExponent().convertToDouble() == 2.0 &&
               sameContext(pow, ctx);
  if (!isSquare || !isLastAxisMean(varOp, sqVisible, rank))
    return rewriter.notifyMatchFailure(root, "var input not (x-mean)^2");
  keep(chain, seen, sqOp);
  keep(chain, seen, varOp);

  // The root may be the packed rank-4 op. The value consumers see is the
  // expand after the optional affine tail, and that is what must match x.
  auto [produced, n] = producedOf(root);

  Value gamma, beta;
  Operation *last = produced;
  Value normCompute = root->getResult(0);
  // Views of gamma / beta: the pack right-aligns [H] to [1, ..., 1, H]. They
  // are not x-shaped, and isPerChannel checks the vector they view.
  llvm::SmallVector<Operation *> affineViews;
  auto nextAffine = [&](Value current) -> Operation * {
    Operation *user = onlyUser(current);
    if (auto collapse =
            mlir::dyn_cast_or_null<mlir::tensor::CollapseShapeOp>(user)) {
      keep(chain, seen, collapse);
      user = onlyUser(collapse.getResult());
    }
    return user;
  };

  if (Operation *user = nextAffine(n)) {
    if (auto mul = mlir::dyn_cast<mlir::hip::MulOp>(user)) {
      Value gRaw = affineOther(mul, n, normCompute);
      if (gRaw && sameContext(mul, ctx)) {
        Value g = definingCompute(gRaw, affineViews, seen);
        auto [mulProduced, mulN] = producedOf(mul);
        if (isPerChannel(g, dim) && mulN.getType() == xType) {
          gamma = g;
          last = mulProduced;
          n = mulN;
          keep(chain, seen, mul);
        }
      }
    }
  }
  if (gamma) {
    if (Operation *user = nextAffine(n)) {
      if (auto add = mlir::dyn_cast<mlir::hip::AddOp>(user)) {
        Value bRaw = affineOther(add, n, n);
        if (bRaw && sameContext(add, ctx)) {
          Value b = definingCompute(bRaw, affineViews, seen);
          auto [addProduced, addN] = producedOf(add);
          if (isPerChannel(b, dim) && addN.getType() == xType) {
            beta = b;
            last = addProduced;
            keep(chain, seen, add);
          }
        }
      }
    }
  }

  if (last != root)
    keep(chain, seen, root);
  if (n.getType() != xType)
    return rewriter.notifyMatchFailure(root, "norm result type");
  if (!reshapesArePacks(chain, xType))
    return rewriter.notifyMatchFailure(root, "reshape regroups rows");
  chain.append(affineViews.begin(), affineViews.end());

  rewriter.setInsertionPoint(last);
  if (!gamma) {
    auto elemType = mlir::cast<mlir::FloatType>(xType.getElementType());
    auto gammaType = mlir::RankedTensorType::get({dim}, elemType);
    llvm::APFloat one = llvm::APFloat::getOne(elemType.getFloatSemantics());
    gamma = mlir::hip::ConstantOp::create(
                rewriter, last->getLoc(), gammaType,
                mlir::DenseElementsAttr::get(gammaType, one))
                .getResult();
  }

  Value outInit;
  if (auto dps = mlir::dyn_cast<mlir::DestinationStyleOpInterface>(last)) {
    mlir::SmallVector<Value> inits(dps.getDpsInits());
    if (!inits.empty() && inits.front().getType() == xType)
      outInit = inits.front();
  }
  if (!outInit)
    outInit = mlir::tensor::EmptyOp::create(
        rewriter, last->getLoc(),
        mlir::tensor::getMixedSizes(rewriter, last->getLoc(), x),
        xType.getElementType());

  mlir::OperationState state(last->getLoc(), "hip.layer_norm");
  llvm::SmallVector<Value> operands = {ctx, x, gamma};
  if (beta)
    operands.push_back(beta);
  operands.push_back(outInit);
  state.addOperands(operands);
  state.addTypes(xType);
  state.addAttribute("axis", rewriter.getI64IntegerAttr(-1));
  state.addAttribute("epsilon",
                     rewriter.getF32FloatAttr(static_cast<float>(*eps)));
  state.addAttribute("stash_type", rewriter.getI64IntegerAttr(1));
  state.addAttribute("operand_segment_sizes",
                     rewriter.getDenseI32ArrayAttr({1, 1, 1, beta ? 1 : 0, 1}));
  Operation *ln = rewriter.create(state);

  for (Operation *op : chain)
    if (op && op != last && op->getNumResults() == 1)
      retargetChainShapeQueries(rewriter, op->getResult(0), x);
  for (Operation *&op : chain)
    if (op == last)
      op = nullptr;
  rewriter.replaceOp(last, ln->getResult(0));

  llvm::SmallVector<Operation *> side;
  for (Operation *op : chain) {
    if (!op || op == last)
      continue;
    if (auto dps = mlir::dyn_cast<mlir::DestinationStyleOpInterface>(op))
      for (Value init : dps.getDpsInits())
        if (init != outInit)
          if (Operation *def = init.getDefiningOp())
            keep(side, seen, def);
    for (Value operand : op->getOperands()) {
      Operation *def = operand.getDefiningOp();
      if (def &&
          (mlir::isa<mlir::hip::ConstantOp, mlir::arith::ConstantOp>(def)))
        keep(side, seen, def);
    }
  }

  eraseDeadOps(rewriter, chain);
  eraseDeadOps(rewriter, side);
  return mlir::success();
}

struct LayerNormDivFusion : public mlir::OpRewritePattern<mlir::hip::DivOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::hip::DivOp div,
                  mlir::PatternRewriter &rewriter) const override {
    llvm::SmallVector<Operation *> chain;
    llvm::SmallPtrSet<Operation *, 16> seen;
    Value a = matchStd(div.getRhs(), chain, seen);
    if (!a)
      return rewriter.notifyMatchFailure(div, "denominator not sqrt");
    return fuseLayerNorm(div, div.getLhs(), a, chain, seen, rewriter);
  }
};

struct LayerNormMulFusion : public mlir::OpRewritePattern<mlir::hip::MulOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::hip::MulOp mul,
                  mlir::PatternRewriter &rewriter) const override {
    for (int i = 0; i < 2; ++i) {
      Value d = i == 0 ? mul.getLhs() : mul.getRhs();
      Value inv = i == 0 ? mul.getRhs() : mul.getLhs();
      Value dCompute = d;
      if (auto collapse = d.getDefiningOp<mlir::tensor::CollapseShapeOp>())
        dCompute = collapse.getSrc();
      if (auto expand = dCompute.getDefiningOp<mlir::tensor::ExpandShapeOp>())
        dCompute = expand.getSrc();
      if (!mlir::isa_and_nonnull<mlir::hip::SubOp>(dCompute.getDefiningOp()))
        continue;
      llvm::SmallVector<Operation *> chain;
      llvm::SmallPtrSet<Operation *, 16> seen;
      Value a = matchInvStd(inv, chain, seen);
      if (a)
        return fuseLayerNorm(mul, d, a, chain, seen, rewriter);
    }
    return rewriter.notifyMatchFailure(mul, "not (x - mean) * rsqrt");
  }
};

bool canUseSkip(Value input, Value skip, Value gamma, Value beta,
                mlir::Type outputType) {
  auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(input.getType());
  if (!inputType || inputType.getRank() < 1 || outputType != inputType)
    return false;
  // Equal dynamic types do not prove equal extents: [3, 8] and [1, 8] are both
  // tensor<?x8xf32>, and the Add would broadcast where the kernel reads rows.
  if (!sameRuntimeShape(input, skip))
    return false;
  mlir::Type elemType = inputType.getElementType();
  if (!elemType.isF16() && !elemType.isF32())
    return false;
  int64_t hidden = inputType.getDimSize(inputType.getRank() - 1);
  if (mlir::ShapedType::isDynamic(hidden))
    return false;
  auto isHiddenVector = [&](Value v) {
    if (!v)
      return true;
    auto type = mlir::dyn_cast<mlir::RankedTensorType>(v.getType());
    return type && type.hasStaticShape() && type.getElementType() == elemType &&
           type.getNumElements() == hidden;
  };
  return isHiddenVector(gamma) && isHiddenVector(beta);
}

struct ResidualLayerNormFusion
    : public mlir::OpRewritePattern<mlir::hip::LayerNormOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::hip::LayerNormOp ln,
                  mlir::PatternRewriter &rewriter) const override {
    if (ln.getOutputs().size() != 1 || ln->getNumResults() != 1)
      return rewriter.notifyMatchFailure(ln, "mean / inv_std requested");
    auto resultType = mlir::dyn_cast<mlir::RankedTensorType>(ln.getType(0));
    if (!resultType || resultType.getRank() < 1)
      return rewriter.notifyMatchFailure(ln, "result not ranked");
    int64_t axis = ln.getAxis();
    if (axis != -1 && axis != resultType.getRank() - 1)
      return rewriter.notifyMatchFailure(ln, "not the last axis");

    llvm::SmallVector<Operation *> pack;
    llvm::SmallPtrSet<Operation *, 8> seen;
    Value sumValue = ln.getInput();
    Value sumCompute = definingCompute(sumValue, pack, seen);
    auto add =
        mlir::dyn_cast_or_null<mlir::hip::AddOp>(sumCompute.getDefiningOp());
    if (!add || !sameContext(add, ln.getCtx()))
      return rewriter.notifyMatchFailure(ln, "input not an Add");
    Value lhs = definingCompute(add.getLhs(), pack, seen);
    Value rhs = definingCompute(add.getRhs(), pack, seen);
    if (lhs.getType() != resultType || rhs.getType() != resultType)
      return rewriter.notifyMatchFailure(ln, "Add broadcasts");
    if (!canUseSkip(lhs, rhs, ln.getScale(), ln.getBias(), resultType))
      return rewriter.notifyMatchFailure(
          ln, "unsupported types, or dynamic extents not known to match");

    bool withSum = false;
    for (mlir::OpOperand &use : sumValue.getUses()) {
      Operation *user = use.getOwner();
      if (user == ln)
        continue;
      if (mlir::isa<mlir::tensor::DimOp>(user))
        continue;
      if (user->getBlock() != ln->getBlock() || !ln->isBeforeInBlock(user))
        return rewriter.notifyMatchFailure(ln, "sum used before the norm");
      withSum = true;
    }

    rewriter.setInsertionPoint(ln);
    Value outInit = ln.getOutputs().front();
    Value sumInit;
    if (withSum) {
      if (sumValue.getDefiningOp() == add &&
          add.getOutput().getType() == resultType)
        sumInit = add.getOutput();
      else
        sumInit = mlir::tensor::EmptyOp::create(
            rewriter, ln.getLoc(),
            mlir::tensor::getMixedSizes(rewriter, ln.getLoc(), lhs),
            resultType.getElementType());
    }

    llvm::SmallVector<Value> operands = {ln.getCtx(), lhs, rhs, ln.getScale()};
    if (ln.getBias())
      operands.push_back(ln.getBias());
    operands.push_back(outInit);
    if (withSum)
      operands.push_back(sumInit);
    mlir::OperationState state(ln.getLoc(), "hip.skip_layer_norm");
    state.addOperands(operands);
    state.addTypes(withSum ? mlir::TypeRange{resultType, resultType}
                           : mlir::TypeRange{resultType});
    state.addAttribute("epsilon", ln.getEpsilonAttr());
    state.addAttribute(
        "operand_segment_sizes",
        rewriter.getDenseI32ArrayAttr(
            {1, 1, 1, 1, ln.getBias() ? 1 : 0, 0, withSum ? 2 : 1}));
    Operation *skip = rewriter.create(state);

    retargetShapeQueries(rewriter, sumValue, lhs);
    rewriter.replaceOp(ln, skip->getResult(0));
    if (withSum)
      rewriter.replaceAllUsesWith(sumValue, skip->getResult(1));
    keep(pack, seen, add);
    if (Operation *init = add.getOutput().getDefiningOp())
      keep(pack, seen, init);
    eraseDeadOps(rewriter, pack);
    return mlir::success();
  }
};

} // namespace

void populateLayerNormFusionPatterns(mlir::RewritePatternSet &patterns,
                                     mlir::PatternBenefit benefit) {
  patterns.add<LayerNormDivFusion, LayerNormMulFusion, ResidualLayerNormFusion>(
      patterns.getContext(), benefit);
}

} // namespace fusion_transform
} // namespace hip
