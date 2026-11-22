/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- matmul_nbits_group.cpp - MatMulNBits ops sharing one activation ----===//
//
// Why this exists and when it declines is in the header; what follows is the
// IR it emits.
//
// Projections that share an activation (q/k/v):
//
//   Before:
//     %q = hip.matmul_nbits(%ctx) ins(%x, %Bq, %Sq) outs(%eq)
//     %k = hip.matmul_nbits(%ctx) ins(%x, %Bk, %Sk) outs(%ek)
//     %v = hip.matmul_nbits(%ctx) ins(%x, %Bv, %Sv) outs(%ev)
//   After:
//     %q, %k, %v = hip.matmul_nbits_group(%ctx) ins(%x)
//         weights(%Bq, %Bk, %Bv) scales(%Sq, %Sk, %Sv) outs(%eq, %ek, %ev)
//         {epilogue = 0}
//
// Asymmetric members carry their zero points over the same way:
// `zero_points(%Zq, %Zk, %Zv)` after `scales(...)`.
//
// Two projections whose only consumer is SiLU(gate) * up:
//
//   Before:
//     %g = hip.matmul_nbits(%ctx) ins(%x, %Bg, %Sg) outs(%eg)
//     %u = hip.matmul_nbits(%ctx) ins(%x, %Bu, %Su) outs(%eu)
//     %s = hip.sigmoid(%ctx) ins(%g) outs(...)
//     %m = hip.mul(%ctx) ins(%g, %s) outs(...)
//     %y = hip.mul(%ctx) ins(%m, %u) outs(...)
//   After:
//     %y = hip.matmul_nbits_group(%ctx) ins(%x) weights(%Bg, %Bu)
//         scales(%Sg, %Su) outs(%eg) {epilogue = 1}
//
// The group is emitted at the first member. Operands of later members that
// are defined after that point (their `tensor.empty` inits and the constants
// feeding them) are side-effect free and are moved up; a member whose
// operands cannot be moved is left out of the group.
//
// A group whose activation comes straight from hip.skip_rms_norm then takes
// the norm's quantized copy, so the row is quantized inside the norm kernel
// instead of by a separate launch:
//
//   Before:
//     %y, %sum = hip.skip_rms_norm(%ctx) ins(...) outs(%ey, %es)
//     %q, %k, %v = hip.matmul_nbits_group(%ctx) ins(%y) ...
//   After:
//     %y, %sum, %yq, %ys = hip.skip_rms_norm(%ctx) ins(...) outs(%ey, %es)
//         quant_outs(%eq, %esc) {quant_block_size = 32}
//     %q, %k, %v = hip.matmul_nbits_group(%ctx) ins(%y) ...
//         quantized(%yq, %ys)
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

constexpr size_t kMaxGroupSize = 4;
constexpr int64_t kEpilogueConcat = 0;
constexpr int64_t kEpilogueSiluMul = 1;

/// A projection the grouped kernel can serve: int4 weights, symmetric or with
/// packed uint8 zero points, an fp16 activation and a single tensor result.
bool isGroupable(mlir::hip::MatMulNBitsOp op) {
  if (op.getGIdx() || op.getBias())
    return false;
  if (op.getZeroPoints() && op.getZpElemSize() != 1)
    return false;
  if (op.getBits() != 4 || op->getNumResults() != 1)
    return false;
  auto aType = mlir::dyn_cast<mlir::RankedTensorType>(op.getA().getType());
  auto outType =
      mlir::dyn_cast<mlir::RankedTensorType>(op.getOutput().getType());
  return aType && outType && aType.getElementType().isF16() &&
         outType.getElementType().isF16();
}

bool sameGrouping(mlir::hip::MatMulNBitsOp a, mlir::hip::MatMulNBitsOp b) {
  return a.getK() == b.getK() && a.getBits() == b.getBits() &&
         a.getBlockSize() == b.getBlockSize() &&
         a.getScaleElemSize() == b.getScaleElemSize() &&
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

/// Every groupable hip.matmul_nbits in `op`'s block that reads its activation
/// with the same grouping attributes, `op` included, in program order.
llvm::SmallVector<mlir::hip::MatMulNBitsOp>
groupCandidates(mlir::hip::MatMulNBitsOp op) {
  llvm::SmallVector<mlir::hip::MatMulNBitsOp> members;
  mlir::Value a = op.getA();
  for (mlir::Operation *user : a.getUsers()) {
    auto mm = mlir::dyn_cast<mlir::hip::MatMulNBitsOp>(user);
    if (!mm || mm.getA() != a || mm->getBlock() != op->getBlock() ||
        llvm::is_contained(members, mm) || !isGroupable(mm) ||
        !sameGrouping(op, mm))
      continue;
    members.push_back(mm);
  }
  llvm::sort(members,
             [](mlir::hip::MatMulNBitsOp x, mlir::hip::MatMulNBitsOp y) {
               return x->isBeforeInBlock(y);
             });
  return members;
}

/// The group the first candidate forms: itself and up to kMaxGroupSize - 1
/// later candidates whose operands can be made available before it.
llvm::SmallVector<mlir::hip::MatMulNBitsOp>
greedyGroup(llvm::ArrayRef<mlir::hip::MatMulNBitsOp> candidates) {
  mlir::hip::MatMulNBitsOp root = candidates.front();
  llvm::SmallVector<mlir::hip::MatMulNBitsOp> group{root};
  for (mlir::hip::MatMulNBitsOp mm : llvm::drop_begin(candidates)) {
    if (group.size() == kMaxGroupSize)
      break;
    if (llvm::all_of(mm->getOperands(), [&](mlir::Value v) {
          return canMakeAvailableBefore(v, root);
        }))
      group.push_back(mm);
  }
  return group;
}

/// Every hip.matmul_nbits that reads `root`'s activation and can join its
/// group, in program order, starting with `root`.
llvm::SmallVector<mlir::hip::MatMulNBitsOp>
collectGroup(mlir::hip::MatMulNBitsOp root) {
  llvm::SmallVector<mlir::hip::MatMulNBitsOp> candidates =
      groupCandidates(root);
  // Only the first candidate forms the group, so it is formed exactly once and
  // the result is independent of the order the driver visits the members in.
  if (candidates.empty() || candidates.front() != root)
    return {};
  return greedyGroup(candidates);
}

/// Whether MatMulNBitsGroup will put `op` in a group of two or more. Groups
/// form from the first remaining candidate on, each taking what it can, so a
/// candidate whose operands cannot move above an earlier one, or one past a
/// full group with no partner left, stays single.
bool joinsGroup(mlir::hip::MatMulNBitsOp op) {
  llvm::SmallVector<mlir::hip::MatMulNBitsOp> remaining = groupCandidates(op);
  while (remaining.size() >= 2) {
    llvm::SmallVector<mlir::hip::MatMulNBitsOp> group = greedyGroup(remaining);
    if (group.size() >= 2 && llvm::is_contained(group, op))
      return true;
    llvm::erase_if(remaining, [&](mlir::hip::MatMulNBitsOp mm) {
      return llvm::is_contained(group, mm);
    });
  }
  return false;
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

/// The hip.swiglu that SwigluFusion makes of the same chain when it reaches the
/// sigmoid and multiplies first. The promoted form keeps f32 intermediates the
/// grouped epilogue does not promise, so it stays as it is.
mlir::hip::SwigluOp matchSwiglu(mlir::hip::MatMulNBitsOp gate,
                                mlir::hip::MatMulNBitsOp up) {
  mlir::Value g = gate->getResult(0);
  mlir::Value u = up->getResult(0);
  if (g.getType() != u.getType())
    return nullptr;
  auto swiglu = soleUserOf<mlir::hip::SwigluOp>(g);
  if (!swiglu || soleUserOf<mlir::hip::SwigluOp>(u) != swiglu ||
      swiglu.getGate() != g || swiglu.getUp() != u ||
      swiglu.getPromoteActivation() || swiglu->getNumResults() != 1 ||
      swiglu->getResult(0).getType() != g.getType())
    return nullptr;
  return swiglu;
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
    mlir::hip::SwigluOp swiglu;
    if (group.size() == 2) {
      silu = matchSiluMul(group[0], group[1]);
      if (!silu)
        silu = matchSiluMul(group[1], group[0]);
      if (!silu) {
        if ((swiglu = matchSwiglu(group[0], group[1])))
          silu = SiluMulChain{group[0], group[1], {}, {}, {}};
        else if ((swiglu = matchSwiglu(group[1], group[0])))
          silu = SiluMulChain{group[1], group[0], {}, {}, {}};
      }
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
      if (!silu || mm == silu->gate) {
        outputs.push_back(mm.getOutput());
        resultTypes.push_back(mm->getResult(0).getType());
      }
    }

    rewriter.setInsertionPoint(root);
    auto groupOp = mlir::hip::MatMulNBitsGroupOp::create(
        rewriter, root.getLoc(), resultTypes, root.getHandle(), root.getA(),
        weights, scales, zeroPoints, /*a_quant=*/mlir::Value(),
        /*a_scale=*/mlir::Value(), outputs, root.getKAttr(),
        rewriter.getArrayAttr(ns), root.getBitsAttr(), root.getBlockSizeAttr(),
        root.getAccuracyLevelAttr(), root.getScaleElemSizeAttr(),
        rewriter.getI64IntegerAttr(silu ? kEpilogueSiluMul : kEpilogueConcat),
        /*row_offsets=*/mlir::DenseI64ArrayAttr());

    if (!silu) {
      for (auto [i, mm] : llvm::enumerate(group))
        rewriter.replaceOp(mm, groupOp->getResult(i));
      return mlir::success();
    }
    mlir::Value fused = groupOp->getResult(0);
    llvm::SmallVector<mlir::Value, 4> shapeSources{silu->gate->getResult(0),
                                                   silu->up->getResult(0)};
    if (!swiglu) {
      shapeSources.push_back(silu->sigmoid->getResult(0));
      shapeSources.push_back(silu->gateMul->getResult(0));
    }
    for (mlir::Value v : shapeSources)
      for (mlir::OpOperand &use : llvm::make_early_inc_range(v.getUses()))
        if (auto dim = mlir::dyn_cast<mlir::tensor::DimOp>(use.getOwner()))
          rewriter.modifyOpInPlace(dim, [&] { use.set(fused); });
    if (swiglu) {
      rewriter.replaceOp(swiglu, fused);
    } else {
      rewriter.replaceOp(silu->upMul, fused);
      rewriter.eraseOp(silu->gateMul);
      rewriter.eraseOp(silu->sigmoid);
    }
    rewriter.eraseOp(silu->up);
    rewriter.eraseOp(silu->gate);
    return mlir::success();
  }
};

/// A group whose activation is the output of a producer that can also write
/// it quantized (hip.skip_rms_norm's output, hip.hyper_connection_pre_mix's
/// output, flattened hip.hyper_connection_post_mix_norm's normed) reads that
/// quantized copy instead of quantizing the row itself. The group's
/// block_size has to be the 32 the producer kernels quantize with.
struct MatMulNBitsGroupReadsNormQuant final
    : public mlir::OpRewritePattern<mlir::hip::MatMulNBitsGroupOp> {
  using mlir::OpRewritePattern<mlir::hip::MatMulNBitsGroupOp>::OpRewritePattern;

  static constexpr int64_t kQuantBlockSize = 32;

  static bool canRead(mlir::hip::MatMulNBitsGroupOp group) {
    return !group.getAQuant() && group.getBlockSize() == kQuantBlockSize &&
           group.getK() % kQuantBlockSize == 0;
  }

  /// The norm kernel quantizes only on its uniform-fp16 entry; an fp32
  /// residual with fp16 skip, gamma or output takes the mixed entry, which
  /// the lowering refuses to give quant_outputs.
  static bool allF16(mlir::hip::SkipRmsNormOp norm) {
    auto isF16 = [](mlir::Value v) {
      auto type = mlir::dyn_cast<mlir::ShapedType>(v.getType());
      return type && type.getElementType().isF16();
    };
    return isF16(norm.getInput()) && isF16(norm.getSkip()) &&
           isF16(norm.getGamma()) &&
           (!norm.getBias() || isF16(norm.getBias())) &&
           llvm::all_of(norm.getOutputs(), isF16);
  }

  /// The producer of `a` if `a` is the result it can quantize and it does not
  /// write a quantized copy yet.
  static mlir::Operation *quantProducer(mlir::Value a) {
    auto result = mlir::dyn_cast<mlir::OpResult>(a);
    if (!result)
      return nullptr;
    mlir::Operation *op = result.getOwner();
    unsigned idx = result.getResultNumber();
    if (auto norm = mlir::dyn_cast<mlir::hip::SkipRmsNormOp>(op))
      return norm.getQuantOutputs().empty() && idx == 0 && allF16(norm)
                 ? op
                 : nullptr;
    if (auto pre = mlir::dyn_cast<mlir::hip::HyperConnectionPreMixOp>(op))
      return pre.getQuantOutputs().empty() && idx == 0 ? op : nullptr;
    if (auto post = mlir::dyn_cast<mlir::hip::HyperConnectionPostMixNormOp>(op))
      return post.getQuantOutputs().empty() && post.getFlattened() == 1 &&
                     idx == 1
                 ? op
                 : nullptr;
    return nullptr;
  }

  /// Rebuilds `producer` with [qInit, sInit] as its quant_outputs.
  static mlir::Operation *withQuantOutputs(mlir::PatternRewriter &rewriter,
                                           mlir::Operation *producer,
                                           mlir::Value qInit,
                                           mlir::Value sInit) {
    llvm::SmallVector<mlir::Type> resultTypes(producer->getResultTypes());
    resultTypes.push_back(qInit.getType());
    resultTypes.push_back(sInit.getType());
    llvm::SmallVector<mlir::Value, 2> quant{qInit, sInit};
    mlir::IntegerAttr block = rewriter.getI64IntegerAttr(kQuantBlockSize);
    mlir::Location loc = producer->getLoc();
    if (auto norm = mlir::dyn_cast<mlir::hip::SkipRmsNormOp>(producer))
      return mlir::hip::SkipRmsNormOp::create(
          rewriter, loc, resultTypes, norm.getCtx(), norm.getInput(),
          norm.getSkip(), norm.getGamma(), norm.getBias(), norm.getOutputs(),
          quant, norm.getEpsilonAttr(), block);
    if (auto pre = mlir::dyn_cast<mlir::hip::HyperConnectionPreMixOp>(producer))
      return mlir::hip::HyperConnectionPreMixOp::create(
          rewriter, loc, resultTypes, pre.getCtx(), pre.getStreams(),
          pre.getPreMix(), pre.getOutput(), quant, pre.getBranchesAttr(),
          pre.getHiddenAttr(), pre.getFlattenedAttr(), pre.getGateLayoutAttr(),
          pre.getReductionScaleAttr(), pre.getGateSigmoidAttr(), block);
    auto post = mlir::cast<mlir::hip::HyperConnectionPostMixNormOp>(producer);
    return mlir::hip::HyperConnectionPostMixNormOp::create(
        rewriter, loc, resultTypes, post.getCtx(), post.getStreams(),
        post.getBlockOutput(), post.getGate(), post.getNormScale(),
        post.getStreamMix(), post.getOutputs(), quant, post.getBranchesAttr(),
        post.getHiddenAttr(), post.getFlattenedAttr(), post.getGateLayoutAttr(),
        post.getGateActivationAttr(), post.getEpsilonAttr(), block);
  }

  mlir::LogicalResult
  matchAndRewrite(mlir::hip::MatMulNBitsGroupOp group,
                  mlir::PatternRewriter &rewriter) const override {
    if (!canRead(group))
      return rewriter.notifyMatchFailure(group, "not a norm-quant reader");
    mlir::Value a = group.getA();
    mlir::Operation *producer = quantProducer(a);
    if (!producer)
      return rewriter.notifyMatchFailure(group,
                                         "A is not a quantizable producer");
    auto aType = mlir::dyn_cast<mlir::RankedTensorType>(a.getType());
    if (!aType || !aType.getElementType().isF16() || aType.getRank() < 1 ||
        aType.getShape().back() != group.getK())
      return rewriter.notifyMatchFailure(group, "unsupported activation type");

    llvm::SmallVector<mlir::hip::MatMulNBitsGroupOp> readers;
    for (mlir::Operation *user : a.getUsers())
      if (auto g = mlir::dyn_cast<mlir::hip::MatMulNBitsGroupOp>(user))
        if (g.getA() == a && canRead(g) && !llvm::is_contained(readers, g))
          readers.push_back(g);

    mlir::Location loc = producer->getLoc();
    rewriter.setInsertionPoint(producer);
    mlir::Value outInit =
        mlir::cast<mlir::DestinationStyleOpInterface>(producer)
            .getTiedOpOperand(mlir::cast<mlir::OpResult>(a))
            ->get();
    llvm::SmallVector<mlir::Value> dynSizes;
    for (auto [i, d] : llvm::enumerate(aType.getShape()))
      if (mlir::ShapedType::isDynamic(d))
        dynSizes.push_back(
            mlir::tensor::DimOp::create(rewriter, loc, outInit, i).getResult());
    auto qType = mlir::RankedTensorType::get(aType.getShape(),
                                             rewriter.getIntegerType(8));
    llvm::SmallVector<int64_t> scaleShape(aType.getShape());
    scaleShape.back() = group.getK() / kQuantBlockSize;
    auto sType = mlir::RankedTensorType::get(scaleShape, rewriter.getF32Type());
    mlir::Value qInit = mlir::tensor::EmptyOp::create(
        rewriter, loc, qType.getShape(), qType.getElementType(), dynSizes);
    mlir::Value sInit = mlir::tensor::EmptyOp::create(
        rewriter, loc, sType.getShape(), sType.getElementType(), dynSizes);

    mlir::Operation *quantProducerOp =
        withQuantOutputs(rewriter, producer, qInit, sInit);
    unsigned numOld = producer->getNumResults();
    rewriter.replaceOp(producer,
                       quantProducerOp->getResults().take_front(numOld));

    mlir::Value aQuant = quantProducerOp->getResult(numOld);
    mlir::Value aScale = quantProducerOp->getResult(numOld + 1);
    for (mlir::hip::MatMulNBitsGroupOp g : readers)
      rewriter.modifyOpInPlace(g, [&] {
        g.getAQuantMutable().assign(aQuant);
        g.getAScaleMutable().assign(aScale);
      });
    return mlir::success();
  }
};

/// Whether `slice` keeps every dim of its source but the last whole, with unit
/// strides and a static range along the last.
bool isLastAxisSlice(mlir::tensor::ExtractSliceOp slice) {
  mlir::RankedTensorType src = slice.getSourceType();
  int64_t rank = src.getRank();
  if (rank < 1 || slice.getType().getRank() != rank || !slice.hasUnitStride())
    return false;
  if (mlir::ShapedType::isDynamic(slice.getStaticOffsets().back()) ||
      mlir::ShapedType::isDynamic(slice.getStaticSizes().back()))
    return false;
  llvm::SmallVector<mlir::OpFoldResult> sizes = slice.getMixedSizes();
  for (int64_t d = 0; d + 1 < rank; ++d) {
    if (slice.getStaticOffsets()[d] != 0)
      return false;
    if (auto attr = mlir::dyn_cast<mlir::Attribute>(sizes[d])) {
      if (src.isDynamicDim(d) ||
          mlir::cast<mlir::IntegerAttr>(attr).getInt() != src.getDimSize(d))
        return false;
      continue;
    }
    auto dim =
        mlir::cast<mlir::Value>(sizes[d]).getDefiningOp<mlir::tensor::DimOp>();
    if (!dim || dim.getSource() != slice.getSource() ||
        dim.getConstantIndex() != d)
      return false;
  }
  return true;
}

/// A column range [offset, offset + size) of a projection's output and the
/// slices that read it.
struct Piece {
  int64_t offset;
  int64_t size;
  llvm::SmallVector<mlir::tensor::ExtractSliceOp> slices;
};

/// The column ranges `result` (N columns) is read through, in order, when its
/// only readers are last-axis slices that together cover [0, N) without
/// overlap, plus `tensor.dim` of its leading dims. Empty otherwise.
llvm::SmallVector<Piece> columnPieces(mlir::Value result, int64_t n) {
  auto type = mlir::dyn_cast<mlir::RankedTensorType>(result.getType());
  if (!type || type.getRank() < 1)
    return {};
  llvm::SmallVector<Piece> pieces;
  for (mlir::Operation *user : result.getUsers()) {
    if (auto dim = mlir::dyn_cast<mlir::tensor::DimOp>(user)) {
      std::optional<int64_t> index = dim.getConstantIndex();
      if (!index || *index + 1 >= type.getRank())
        return {};
      continue;
    }
    auto slice = mlir::dyn_cast<mlir::tensor::ExtractSliceOp>(user);
    if (!slice || !isLastAxisSlice(slice))
      return {};
    int64_t offset = slice.getStaticOffsets().back();
    int64_t size = slice.getStaticSizes().back();
    auto it = llvm::find_if(pieces, [&](const Piece &p) {
      return p.offset == offset && p.size == size;
    });
    if (it == pieces.end())
      pieces.push_back({offset, size, {slice}});
    else
      it->slices.push_back(slice);
  }
  llvm::sort(pieces, [](const Piece &a, const Piece &b) {
    return a.offset < b.offset;
  });
  int64_t next = 0;
  for (const Piece &p : pieces) {
    if (p.offset != next || p.size <= 0)
      return {};
    next += p.size;
  }
  if (pieces.size() < 2 || next != n)
    return {};
  return pieces;
}

/// One member of the group being built.
struct SplitMember {
  mlir::Value weight, scales, zeroPoints;
  int64_t n, rowOffset;
  mlir::Value result, init;
  llvm::SmallVector<Piece> pieces; // empty: kept whole
};

/// Rewrites `op` (a hip.matmul_nbits or a concat hip.matmul_nbits_group) as a
/// group with one member per column slice of its outputs.
mlir::LogicalResult splitByColumns(mlir::Operation *op,
                                   llvm::MutableArrayRef<SplitMember> members,
                                   mlir::hip::MatMulNBitsGroupOp like,
                                   mlir::hip::MatMulNBitsOp likeSingle,
                                   mlir::PatternRewriter &rewriter) {
  size_t count = 0;
  bool split = false;
  for (SplitMember &m : members) {
    m.pieces = columnPieces(m.result, m.n);
    split |= !m.pieces.empty();
    count += m.pieces.empty() ? 1 : m.pieces.size();
  }
  if (!split)
    return rewriter.notifyMatchFailure(op, "no output read in column slices");
  if (count > kMaxGroupSize)
    return rewriter.notifyMatchFailure(op, "too many slices for one group");

  mlir::Location loc = op->getLoc();
  rewriter.setInsertionPoint(op);
  llvm::SmallVector<mlir::Value> weights, scales, zeroPoints, outputs;
  llvm::SmallVector<mlir::Attribute> ns;
  llvm::SmallVector<int64_t> rowOffsets;
  llvm::SmallVector<mlir::Type> resultTypes;
  auto addMember = [&](const SplitMember &m, int64_t offset, int64_t n,
                       mlir::Value init) {
    weights.push_back(m.weight);
    scales.push_back(m.scales);
    if (m.zeroPoints)
      zeroPoints.push_back(m.zeroPoints);
    ns.push_back(rewriter.getI64IntegerAttr(n));
    rowOffsets.push_back(m.rowOffset + offset);
    outputs.push_back(init);
    resultTypes.push_back(init.getType());
  };
  for (const SplitMember &m : members) {
    if (m.pieces.empty()) {
      addMember(m, 0, m.n, m.init);
      continue;
    }
    auto type = mlir::cast<mlir::RankedTensorType>(m.result.getType());
    llvm::SmallVector<mlir::Value> dynSizes;
    for (int64_t d = 0; d + 1 < type.getRank(); ++d)
      if (type.isDynamicDim(d))
        dynSizes.push_back(
            mlir::tensor::DimOp::create(rewriter, loc, m.init, d).getResult());
    for (const Piece &p : m.pieces) {
      llvm::SmallVector<int64_t> shape(type.getShape());
      shape.back() = p.size;
      mlir::Value init = mlir::tensor::EmptyOp::create(
          rewriter, loc, shape, type.getElementType(), dynSizes);
      addMember(m, p.offset, p.size, init);
    }
  }

  mlir::hip::MatMulNBitsGroupOp groupOp;
  if (like) {
    groupOp = mlir::hip::MatMulNBitsGroupOp::create(
        rewriter, loc, resultTypes, like.getHandle(), like.getA(), weights,
        scales, zeroPoints, like.getAQuant(), like.getAScale(), outputs,
        like.getKAttr(), rewriter.getArrayAttr(ns), like.getBitsAttr(),
        like.getBlockSizeAttr(), like.getAccuracyLevelAttr(),
        like.getScaleElemSizeAttr(), like.getEpilogueAttr(),
        rewriter.getDenseI64ArrayAttr(rowOffsets));
  } else {
    groupOp = mlir::hip::MatMulNBitsGroupOp::create(
        rewriter, loc, resultTypes, likeSingle.getHandle(), likeSingle.getA(),
        weights, scales, zeroPoints, /*a_quant=*/mlir::Value(),
        /*a_scale=*/mlir::Value(), outputs, likeSingle.getKAttr(),
        rewriter.getArrayAttr(ns), likeSingle.getBitsAttr(),
        likeSingle.getBlockSizeAttr(), likeSingle.getAccuracyLevelAttr(),
        likeSingle.getScaleElemSizeAttr(),
        rewriter.getI64IntegerAttr(kEpilogueConcat),
        rewriter.getDenseI64ArrayAttr(rowOffsets));
  }

  unsigned next = 0;
  for (const SplitMember &m : members) {
    if (m.pieces.empty()) {
      rewriter.replaceAllUsesWith(m.result, groupOp->getResult(next++));
      continue;
    }
    for (const Piece &p : m.pieces) {
      for (mlir::tensor::ExtractSliceOp slice : p.slices)
        rewriter.replaceOp(slice, groupOp->getResult(next));
      ++next;
    }
    // Only leading-dim tensor.dim readers remain; the init has the same shape.
    for (mlir::OpOperand &use : llvm::make_early_inc_range(m.result.getUses()))
      rewriter.modifyOpInPlace(use.getOwner(), [&] { use.set(m.init); });
  }
  rewriter.eraseOp(op);
  return mlir::success();
}

/// A projection whose output is consumed only in last-axis slices -- a fused
/// projection split back into its parts -- writes each slice as its own group
/// member instead. A strided slice of the full output would otherwise be
/// copied out before its consumer runs, one memref.copy launch per slice.
///
///   Before:
///     %y = hip.matmul_nbits(%ctx) ins(%x, %B, %S) outs(%e :
///     tensor<?x?x324xf16>) %a = tensor.extract_slice %y[0, 0, 0] [%d0, %d1,
///     320] [1, 1, 1] %b = tensor.extract_slice %y[0, 0, 320] [%d0, %d1, 4] [1,
///     1, 1]
///   After:
///     %a, %b = hip.matmul_nbits_group(%ctx) ins(%x) weights(%B, %B)
///         scales(%S, %S) outs(%ea, %eb : tensor<?x?x320xf16>,
///         tensor<?x?x4xf16>) {N = [320, 4], row_offsets = array<i64: 0, 320>}
struct MatMulNBitsSplitSlicedOutput final
    : public mlir::OpRewritePattern<mlir::hip::MatMulNBitsOp> {
  using mlir::OpRewritePattern<mlir::hip::MatMulNBitsOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::hip::MatMulNBitsOp op,
                  mlir::PatternRewriter &rewriter) const override {
    if (!isGroupable(op))
      return rewriter.notifyMatchFailure(op, "not a groupable projection");
    // Projections sharing the activation group first; the group is split.
    if (joinsGroup(op))
      return rewriter.notifyMatchFailure(op, "joins a sibling group");
    SplitMember m{op.getB(), op.getScales(),   op.getZeroPoints(), op.getN(),
                  0,         op->getResult(0), op.getOutput(),     {}};
    return splitByColumns(op, m, {}, op, rewriter);
  }
};

struct MatMulNBitsGroupSplitSlicedOutputs final
    : public mlir::OpRewritePattern<mlir::hip::MatMulNBitsGroupOp> {
  using mlir::OpRewritePattern<mlir::hip::MatMulNBitsGroupOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::hip::MatMulNBitsGroupOp group,
                  mlir::PatternRewriter &rewriter) const override {
    if (group.getEpilogue() != kEpilogueConcat ||
        group->getNumResults() != group.getOutputs().size())
      return rewriter.notifyMatchFailure(group,
                                         "not a concat group on tensors");
    std::optional<llvm::ArrayRef<int64_t>> offsets = group.getRowOffsets();
    llvm::SmallVector<SplitMember> members;
    for (auto [i, w] : llvm::enumerate(group.getWeights()))
      members.push_back(
          {w,
           group.getScales()[i],
           group.getZeroPoints().empty() ? mlir::Value()
                                         : group.getZeroPoints()[i],
           mlir::cast<mlir::IntegerAttr>(group.getN()[i]).getInt(),
           offsets ? (*offsets)[i] : 0,
           group->getResult(i),
           group.getOutputs()[i],
           {}});
    return splitByColumns(group, members, group, {}, rewriter);
  }
};

} // namespace

void populateMatMulNBitsGroupPattern(mlir::RewritePatternSet &patterns,
                                     mlir::PatternBenefit benefit) {
  patterns
      .add<MatMulNBitsGroup, MatMulNBitsGroupReadsNormQuant,
           MatMulNBitsSplitSlicedOutput, MatMulNBitsGroupSplitSlicedOutputs>(
          patterns.getContext(), benefit);
}

} // namespace fusion_transform
} // namespace hip
