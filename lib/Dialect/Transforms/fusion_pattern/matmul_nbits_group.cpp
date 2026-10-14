/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- matmul_nbits_group.cpp - Projections of one int8 activation --------===//
//
// Why this exists and which projections qualify is in the header; what
// follows is the IR it emits.
//
// Projections that share an int8 activation (q/k/v):
//
//   Before:
//     %q = hip.matmul_nbits(%ctx) ins(%x, %Bq, %Sq) quantized(%xq, %xs)
//         outs(%eq)
//     %k = hip.matmul_nbits(%ctx) ins(%x, %Bk, %Sk) quantized(%xq, %xs)
//         outs(%ek)
//     %v = hip.matmul_nbits(%ctx) ins(%x, %Bv, %Sv) quantized(%xq, %xs)
//         outs(%ev)
//   After:
//     %q, %k, %v = hip.matmul_nbits_group(%ctx) ins(%x)
//         quantized(%xq, %xs) weights(%Bq, %Bk, %Bv) scales(%Sq, %Sk, %Sv)
//         outs(%eq, %ek, %ev) {N = [..], epilogue = 0}
//
// Asymmetric members carry their zero points over the same way:
// `zero_points(%Zq, %Zk, %Zv)` after `scales(...)`.
//
// Two projections whose only consumer is SiLU(gate) * up:
//
//   Before:
//     %g = hip.matmul_nbits(%ctx) ins(%x, %Bg, %Sg) quantized(..) outs(%eg)
//     %u = hip.matmul_nbits(%ctx) ins(%x, %Bu, %Su) quantized(..) outs(%eu)
//     %s = hip.sigmoid(%ctx) ins(%g) outs(...)
//     %m = hip.mul(%ctx) ins(%g, %s) outs(...)
//     %y = hip.mul(%ctx) ins(%m, %u) outs(...)
//   After:
//     %y, %unused = hip.matmul_nbits_group(%ctx) ins(%x) quantized(..)
//         weights(%Bg, %Bu) scales(%Sg, %Su) outs(%eg) up_tmp(%eu)
//         {N = [n, n], epilogue = 1}
//
// The group is emitted at its first member. Operands of later members that
// are defined after that point (their `tensor.empty` inits and the constants
// feeding them) are side-effect free and are moved up; a member whose
// operands cannot be moved is left out of the group.
//===----------------------------------------------------------------------===//

#include "matmul_nbits_group.hpp"

#include "hip/Dialect/IR/HipDialect.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <optional>

namespace hip {
namespace fusion_transform {
namespace {

constexpr size_t kMaxGroupSize = 3;
constexpr int64_t kEpilogueConcat = 0;
constexpr int64_t kEpilogueSiluMul = 1;

bool isF16Tensor(mlir::Type type) {
  auto t = mlir::dyn_cast<mlir::RankedTensorType>(type);
  return t && t.getElementType().isF16();
}

/// A projection the grouped kernel can serve.
bool isGroupable(mlir::hip::MatMulNBitsOp op) {
  if (!op.getAQuant() || op.getGIdx() || op.getBias())
    return false;
  if (op.getZeroPoints() && op.getZpElemSize() != 1)
    return false;
  if (op.getBits() != 4 || op.getScaleElemSize() != 2 ||
      op->getNumResults() != 1)
    return false;
  return isF16Tensor(op.getA().getType()) &&
         isF16Tensor(op.getOutput().getType());
}

bool sameGrouping(mlir::hip::MatMulNBitsOp a, mlir::hip::MatMulNBitsOp b) {
  return a.getA() == b.getA() && a.getAQuant() == b.getAQuant() &&
         a.getAScale() == b.getAScale() && a.getK() == b.getK() &&
         a.getBlockSize() == b.getBlockSize() &&
         a.getAccuracyLevel() == b.getAccuracyLevel() &&
         static_cast<bool>(a.getZeroPoints()) ==
             static_cast<bool>(b.getZeroPoints());
}

/// Whether `v` is available at `point`, possibly after moving its
/// side-effect-free producers in `point`'s block up to just before it.
bool canMakeAvailableBefore(mlir::Value v, mlir::Operation *point,
                            unsigned depth = 0) {
  mlir::Operation *def = v.getDefiningOp();
  if (!def || def->getBlock() != point->getBlock() ||
      def->isBeforeInBlock(point))
    return true;
  // A value computed from `point` itself can never be moved above it.
  if (def == point)
    return false;
  if (depth > 8 || def->getNumRegions() != 0 || !mlir::isMemoryEffectFree(def))
    return false;
  return llvm::all_of(def->getOperands(), [&](mlir::Value operand) {
    return canMakeAvailableBefore(operand, point, depth + 1);
  });
}

/// Every hip.matmul_nbits that reads `root`'s int8 activation and can join its
/// group, in program order, starting with `root`.
llvm::SmallVector<mlir::hip::MatMulNBitsOp>
collectGroup(mlir::hip::MatMulNBitsOp root) {
  llvm::SmallVector<mlir::hip::MatMulNBitsOp> members;
  for (mlir::Operation *user : root.getAQuant().getUsers()) {
    auto mm = mlir::dyn_cast<mlir::hip::MatMulNBitsOp>(user);
    if (!mm || mm->getBlock() != root->getBlock() ||
        llvm::is_contained(members, mm) || !isGroupable(mm) ||
        !sameGrouping(root, mm))
      continue;
    members.push_back(mm);
  }
  llvm::sort(members,
             [](mlir::hip::MatMulNBitsOp x, mlir::hip::MatMulNBitsOp y) {
               return x->isBeforeInBlock(y);
             });
  // Only the first member forms the group, so it is formed exactly once and
  // the result is independent of the order the driver visits the members in.
  if (members.empty() || members.front() != root)
    return {};

  llvm::SmallVector<mlir::hip::MatMulNBitsOp> group{root};
  for (mlir::hip::MatMulNBitsOp mm : llvm::drop_begin(members)) {
    if (group.size() == kMaxGroupSize)
      break;
    if (llvm::all_of(mm->getOperands(), [&](mlir::Value v) {
          return canMakeAvailableBefore(v, root);
        }))
      group.push_back(mm);
  }
  return group;
}

/// The SiLU(gate) * up chain consuming a gate/up pair.
struct SiluMulChain {
  mlir::hip::MatMulNBitsOp gate;
  mlir::hip::MatMulNBitsOp up;
  mlir::hip::SigmoidOp sigmoid;
  mlir::hip::MulOp gateMul; // gate * sigmoid(gate)
  mlir::hip::MulOp upMul;   // gateMul * up
};

bool isMulOf(mlir::hip::MulOp mul, mlir::Value x, mlir::Value y) {
  return (mul.getLhs() == x && mul.getRhs() == y) ||
         (mul.getLhs() == y && mul.getRhs() == x);
}

/// Users of `v` other than `tensor.dim`. The conversions size every init
/// from `tensor.dim` of a producer, so those only read the shape, which the
/// fused result shares.
llvm::SmallVector<mlir::Operation *> valueUsers(mlir::Value v) {
  llvm::SmallVector<mlir::Operation *> users;
  for (mlir::Operation *user : v.getUsers())
    if (!mlir::isa<mlir::tensor::DimOp>(user) &&
        !llvm::is_contained(users, user))
      users.push_back(user);
  return users;
}

template <typename OpT> OpT soleUserOf(mlir::Value v) {
  llvm::SmallVector<mlir::Operation *> users = valueUsers(v);
  if (users.size() != 1)
    return nullptr;
  return mlir::dyn_cast<OpT>(users.front());
}

std::optional<SiluMulChain> matchSiluMul(mlir::hip::MatMulNBitsOp gate,
                                         mlir::hip::MatMulNBitsOp up) {
  mlir::Value g = gate->getResult(0);
  mlir::Value u = up->getResult(0);
  if (g.getType() != u.getType() || valueUsers(u).size() != 1)
    return std::nullopt;
  llvm::SmallVector<mlir::Operation *> gUsers = valueUsers(g);
  if (gUsers.size() != 2)
    return std::nullopt;

  mlir::hip::SigmoidOp sigmoid;
  for (mlir::Operation *user : gUsers)
    if (auto s = mlir::dyn_cast<mlir::hip::SigmoidOp>(user))
      sigmoid = s;
  if (!sigmoid || sigmoid.getX() != g || sigmoid->getNumResults() != 1)
    return std::nullopt;
  mlir::Value s = sigmoid->getResult(0);
  auto gateMul = soleUserOf<mlir::hip::MulOp>(s);
  if (!gateMul || gateMul->getNumResults() != 1 || !isMulOf(gateMul, g, s) ||
      !llvm::is_contained(gUsers, gateMul.getOperation()))
    return std::nullopt;
  mlir::Value m = gateMul->getResult(0);
  auto upMul = soleUserOf<mlir::hip::MulOp>(m);
  if (!upMul || upMul->getNumResults() != 1 || !isMulOf(upMul, m, u) ||
      soleUserOf<mlir::hip::MulOp>(u) != upMul)
    return std::nullopt;
  // No broadcasting anywhere in the chain: every value is gate-shaped.
  if (s.getType() != g.getType() || m.getType() != g.getType() ||
      upMul->getResult(0).getType() != g.getType())
    return std::nullopt;
  return SiluMulChain{gate, up, sigmoid, gateMul, upMul};
}

struct MatMulNBitsGroup final
    : public mlir::OpRewritePattern<mlir::hip::MatMulNBitsOp> {
  using mlir::OpRewritePattern<mlir::hip::MatMulNBitsOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::hip::MatMulNBitsOp root,
                  mlir::PatternRewriter &rewriter) const override {
    if (!isGroupable(root))
      return rewriter.notifyMatchFailure(root, "not a groupable projection");
    llvm::SmallVector<mlir::hip::MatMulNBitsOp> group = collectGroup(root);
    if (group.size() < 2)
      return rewriter.notifyMatchFailure(root, "no sibling projections");

    std::optional<SiluMulChain> silu;
    if (group.size() == 2) {
      silu = matchSiluMul(group[0], group[1]);
      if (!silu)
        silu = matchSiluMul(group[1], group[0]);
    }

    // collectGroup admitted only members whose operands can be made available
    // at `root`, so these moves are expected to succeed. If one does not, the
    // ops already moved are side-effect free and stay valid where they are.
    for (mlir::hip::MatMulNBitsOp mm : llvm::drop_begin(group))
      if (mlir::failed(mlir::moveOperationDependencies(rewriter, mm, root)))
        return rewriter.notifyMatchFailure(mm, "cannot hoist operands");

    llvm::SmallVector<mlir::hip::MatMulNBitsOp> ordered =
        silu ? llvm::SmallVector<mlir::hip::MatMulNBitsOp>{silu->gate, silu->up}
             : group;
    llvm::SmallVector<mlir::Value> weights, scales, zeroPoints, outputs;
    llvm::SmallVector<mlir::Attribute> ns;
    llvm::SmallVector<mlir::Type> resultTypes;
    for (mlir::hip::MatMulNBitsOp mm : ordered) {
      weights.push_back(mm.getB());
      scales.push_back(mm.getScales());
      if (mlir::Value zp = mm.getZeroPoints())
        zeroPoints.push_back(zp);
      ns.push_back(rewriter.getI64IntegerAttr(mm.getN()));
      resultTypes.push_back(mm->getResult(0).getType());
      if (!silu || mm == silu->gate)
        outputs.push_back(mm.getOutput());
    }
    mlir::Value upTmp = silu ? silu->up.getOutput() : mlir::Value();

    rewriter.setInsertionPoint(root);
    auto groupOp = mlir::hip::MatMulNBitsGroupOp::create(
        rewriter, root.getLoc(), resultTypes, root.getHandle(), root.getA(),
        root.getAQuant(), root.getAScale(), weights, scales, zeroPoints,
        outputs, upTmp, root.getKAttr(), rewriter.getArrayAttr(ns),
        root.getBitsAttr(), root.getBlockSizeAttr(),
        root.getAccuracyLevelAttr(),
        rewriter.getI64IntegerAttr(silu ? kEpilogueSiluMul : kEpilogueConcat));

    if (!silu) {
      for (auto [i, mm] : llvm::enumerate(group))
        rewriter.replaceOp(mm, groupOp->getResult(i));
      return mlir::success();
    }
    mlir::Value fused = groupOp->getResult(0);
    for (mlir::Value v :
         {silu->gate->getResult(0), silu->up->getResult(0),
          silu->sigmoid->getResult(0), silu->gateMul->getResult(0)})
      for (mlir::OpOperand &use : llvm::make_early_inc_range(v.getUses()))
        if (auto dim = mlir::dyn_cast<mlir::tensor::DimOp>(use.getOwner()))
          rewriter.modifyOpInPlace(dim, [&] { use.set(fused); });
    rewriter.replaceOp(silu->upMul, fused);
    rewriter.eraseOp(silu->gateMul);
    rewriter.eraseOp(silu->sigmoid);
    rewriter.eraseOp(silu->up);
    rewriter.eraseOp(silu->gate);
    return mlir::success();
  }
};

} // namespace

void populateMatMulNBitsGroupPatterns(mlir::RewritePatternSet &patterns,
                                      mlir::PatternBenefit benefit) {
  patterns.add<MatMulNBitsGroup>(patterns.getContext(), benefit);
}

} // namespace fusion_transform
} // namespace hip
