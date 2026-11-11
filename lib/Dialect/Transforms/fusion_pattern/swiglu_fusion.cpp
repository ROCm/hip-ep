/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- swiglu_fusion.cpp - hip.sigmoid + hip.mul chain to hip.swiglu ------===//
//
// Any frontend that lowers a gated MLP activation to the primitive HIP ops
// gets the fused kernel. The match lives here, not in an ONNX conversion,
// so a later frontend does not repeat it.
//
// Both multiplies commute. The outer product's destination is reused, so
// the fused op keeps the shape the chain already computed.
//
//   Before:
//     %s = hip.sigmoid(%ctx) ins(%gate : t) outs(%e0 : t) : t
//     %a = hip.mul(%ctx) ins(%gate, %s : t, t) outs(%e1 : t) -> t
//     %d0 = tensor.dim %a, %c0 : t
//     %y = hip.mul(%ctx) ins(%a, %up : t, t) outs(%e2 : t) -> t
//   After:
//     %d0 = tensor.dim %gate, %c0 : t
//     %y = hip.swiglu(%ctx) ins(%gate, %up : t, t) outs(%e2 : t) : t
//
// A dynamic export sizes the outer init with tensor.dim of the inner
// product. That query does not read activation values, and the pattern
// already requires the inner product's type to equal the gate's, so the
// query is retargeted to the gate and does not keep the intermediate alive.
//
// Identical types do not prove a dynamic axis has one runtime extent:
// tensor<?x?x14336xf16> matches tensor<?x?x14336xf16> when one side is
// [4, S, 14336] and the other is [1, S, 14336], and hip.mul would
// broadcast. Each dynamic axis must trace to the same extent. A block
// argument is its own extent. hip.matmul and hip.qmatmul contribute batch
// and M from the activation (N from the weight); hip.matmul_nbits
// contributes every axis but the last from the activation and the last
// from its N attribute. That is the same source choice as
// reifyMatmulLikeShape and MatMulNBitsOp::reifyResultShapes.
//
//===----------------------------------------------------------------------===//

#include "swiglu_fusion.hpp"

#include "hip/Dialect/IR/HipDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallPtrSet.h"

namespace hip {
namespace fusion_transform {
namespace {

// True when every use is either `consumer` or a tensor.dim. tensor.dim only
// reads the type, which this pattern requires to match the gate.
bool hasSingleValueUse(mlir::Value value, mlir::Operation *consumer) {
  mlir::Operation *seen = nullptr;
  for (mlir::OpOperand &use : value.getUses()) {
    if (mlir::isa<mlir::tensor::DimOp>(use.getOwner()))
      continue;
    if (seen)
      return false;
    seen = use.getOwner();
  }
  return seen == consumer;
}

void retargetShapeQueries(mlir::PatternRewriter &rewriter, mlir::Value from,
                          mlir::Value to) {
  rewriter.replaceUsesWithIf(from, to, [](mlir::OpOperand &operand) {
    return mlir::isa<mlir::tensor::DimOp>(operand.getOwner());
  });
}

// One dynamic axis after projection producers have been followed.
struct Extent {
  bool isStatic = false;
  int64_t staticSize = 0;
  mlir::Value value;
  int64_t dim = 0;

  bool operator==(const Extent &other) const {
    if (isStatic || other.isStatic)
      return isStatic && other.isStatic && staticSize == other.staticSize;
    return value == other.value && dim == other.dim;
  }
};

// Follow one matmul-like result axis to the operand axis that supplies it.
// A static result is returned directly. std::nullopt means this op does not
// reveal the axis, so the caller keeps the op result as the extent.
std::optional<Extent> stepProjection(mlir::Operation *op, int64_t outDim) {
  if (op->getNumResults() == 0)
    return std::nullopt;
  auto mapMatmul = [&](mlir::Value a, mlir::Value b, int64_t transA,
                       int64_t transB,
                       int64_t resultRank) -> std::optional<Extent> {
    auto aType = mlir::dyn_cast<mlir::RankedTensorType>(a.getType());
    auto bType = mlir::dyn_cast<mlir::RankedTensorType>(b.getType());
    if (!aType || !bType || aType.getRank() < 2 || bType.getRank() < 2)
      return std::nullopt;
    int64_t aRank = aType.getRank();
    int64_t bRank = bType.getRank();
    int64_t batchRank = std::max(aRank - 2, bRank - 2);
    int64_t outRank = batchRank + 2;
    if (outDim < 0 || outDim >= outRank || outRank != resultRank)
      return std::nullopt;

    if (outDim + 2 == outRank) {
      int64_t aDim = transA ? aRank - 1 : aRank - 2;
      return Extent{false, 0, a, aDim};
    }
    if (outDim + 1 == outRank) {
      int64_t bDim = transB ? bRank - 2 : bRank - 1;
      return Extent{false, 0, b, bDim};
    }

    int64_t aPad = batchRank - (aRank - 2);
    int64_t bPad = batchRank - (bRank - 2);
    int64_t aExtent =
        outDim < aPad ? 1
                      : aType.getDimSize(static_cast<unsigned>(outDim - aPad));
    int64_t bExtent =
        outDim < bPad ? 1
                      : bType.getDimSize(static_cast<unsigned>(outDim - bPad));
    bool aCanonical = outDim >= aPad && aExtent != 1;
    bool bCanonical = outDim >= bPad && bExtent != 1;
    bool pickA = aCanonical || (!bCanonical && outDim >= aPad);
    if (pickA && outDim >= aPad)
      return Extent{false, 0, a, outDim - aPad};
    if (!pickA && outDim >= bPad)
      return Extent{false, 0, b, outDim - bPad};
    return std::nullopt;
  };

  auto resultType =
      mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
  if (!resultType)
    return std::nullopt;
  int64_t resultRank = resultType.getRank();

  if (auto matmul = mlir::dyn_cast<mlir::hip::MatmulOp>(op))
    return mapMatmul(matmul.getA(), matmul.getB(), matmul.getTransA(),
                     matmul.getTransB(), resultRank);
  if (auto qmatmul = mlir::dyn_cast<mlir::hip::QMatMulOp>(op))
    return mapMatmul(qmatmul.getA(), qmatmul.getB(), qmatmul.getTransA(),
                     qmatmul.getTransB(), resultRank);
  if (auto nbits = mlir::dyn_cast<mlir::hip::MatMulNBitsOp>(op)) {
    auto aType = mlir::dyn_cast<mlir::RankedTensorType>(nbits.getA().getType());
    if (!aType || aType.getRank() != resultRank || outDim < 0 ||
        outDim >= resultRank)
      return std::nullopt;
    // The trailing axis is the N attribute, not A's contraction axis.
    if (outDim + 1 == resultRank)
      return Extent{true, static_cast<int64_t>(nbits.getN()), {}, 0};
    return Extent{false, 0, nbits.getA(), outDim};
  }
  return std::nullopt;
}

// Follow casts, projections and explicit destination extents through the
// gated-normalization graph. Keep the walk bounded; unsupported or cyclic
// provenance must decline rather than assume equal dynamic dimensions.
constexpr int kMaxExtentHops = 32;

// A declared size is either a constant, an existing tensor dimension, or an
// identical scalar SSA value. No arithmetic equivalence is assumed.
std::optional<Extent> shapeSizeExtent(mlir::OpFoldResult size) {
  if (auto constant = mlir::getConstantIntValue(size))
    return Extent{true, *constant, {}, 0};
  auto value = mlir::dyn_cast<mlir::Value>(size);
  if (!value)
    return std::nullopt;
  for (int hop = 0; hop < kMaxExtentHops; ++hop) {
    if (auto cast = value.getDefiningOp<mlir::arith::IndexCastOp>()) {
      // Only peel the 64-bit index/integer bridge used by shape reification.
      // A narrower integer can truncate an extent and is not equivalent.
      auto integer = mlir::dyn_cast<mlir::IntegerType>(
          cast.getIn().getType().isIndex() ? cast.getType()
                                           : cast.getIn().getType());
      if (integer && integer.getWidth() == 64) {
        value = cast.getIn();
        continue;
      }
    }
    if (auto query = value.getDefiningOp<mlir::tensor::DimOp>())
      if (auto index = query.getConstantIndex())
        return Extent{false, 0, query.getSource(), *index};
    return Extent{false, 0, value, -1};
  }
  return std::nullopt;
}

// Bounds the nesting of select arms traced by selectExtent.
constexpr int kMaxSelectDepth = 4;

llvm::SmallVector<Extent> traceExtents(mlir::Value value, int64_t dim,
                                       int selectDepth = 0);

// A select of two extents proven equal is that extent whatever the
// condition. This is how an elementwise op sizes its destination when its
// operands could broadcast: select(lhs == 1, rhs, lhs) per axis.
std::optional<Extent> selectExtent(mlir::Value value, int selectDepth) {
  auto select = value.getDefiningOp<mlir::arith::SelectOp>();
  if (!select || selectDepth >= kMaxSelectDepth)
    return std::nullopt;
  std::optional<Extent> lhs = shapeSizeExtent(select.getTrueValue());
  std::optional<Extent> rhs = shapeSizeExtent(select.getFalseValue());
  if (!lhs || !rhs)
    return std::nullopt;
  auto expand = [&](const Extent &e) {
    if (e.isStatic || e.dim < 0)
      return llvm::SmallVector<Extent>{e};
    return traceExtents(e.value, e.dim, selectDepth + 1);
  };
  for (const Extent &l : expand(*lhs))
    for (const Extent &r : expand(*rhs))
      if (l == r)
        return l;
  return std::nullopt;
}

// Keep every proven anchor, so two branches can meet at a local shared input
// without having to reach the beginning of an arbitrarily deep residual graph.
llvm::SmallVector<Extent> traceExtents(mlir::Value value, int64_t dim,
                                       int selectDepth) {
  llvm::SmallVector<Extent> extents;
  mlir::Value current = value;
  int64_t axis = dim;
  for (int hop = 0; hop < kMaxExtentHops; ++hop) {
    auto type = mlir::dyn_cast<mlir::RankedTensorType>(current.getType());
    if (!type || axis < 0 || axis >= type.getRank())
      break;
    if (!type.isDynamicDim(static_cast<unsigned>(axis))) {
      extents.push_back(Extent{true, type.getDimSize(axis), {}, 0});
      break;
    }
    extents.push_back(Extent{false, 0, current, axis});
    mlir::Operation *def = current.getDefiningOp();
    if (!def)
      break;
    if (auto cast = mlir::dyn_cast<mlir::tensor::CastOp>(def)) {
      current = cast.getSource();
      continue;
    }
    if (auto cast = mlir::dyn_cast<mlir::hip::CastOp>(def)) {
      current = cast.getInput();
      continue;
    }
    if (std::optional<Extent> stepped = stepProjection(def, axis)) {
      if (stepped->isStatic) {
        extents.push_back(*stepped);
        break;
      }
      current = stepped->value;
      axis = stepped->dim;
      continue;
    }
    if (auto dps = mlir::dyn_cast<mlir::DestinationStyleOpInterface>(def)) {
      if (auto result = mlir::dyn_cast<mlir::OpResult>(current)) {
        current = dps.getTiedOpOperand(result)->get();
        continue;
      }
    }
    if (auto collapse = mlir::dyn_cast<mlir::tensor::CollapseShapeOp>(def)) {
      auto groups = collapse.getReassociationIndices();
      // A singleton reassociation preserves this axis. Merged dynamic axes
      // require a product proof and are deliberately left unsupported.
      if (groups[axis].size() != 1)
        break;
      current = collapse.getSrc();
      axis = groups[axis].front();
      continue;
    }
    std::optional<Extent> sizeExtent;
    if (auto expand = mlir::dyn_cast<mlir::tensor::ExpandShapeOp>(def)) {
      sizeExtent = shapeSizeExtent(expand.getMixedOutputShape()[axis]);
    } else if (auto slice = mlir::dyn_cast<mlir::tensor::ExtractSliceOp>(def)) {
      auto sourceType =
          mlir::dyn_cast<mlir::RankedTensorType>(slice.getSource().getType());
      // Restrict to rank-preserving slices; their explicit sizes describe
      // each output axis directly, independently of offsets and data values.
      if (!sourceType || sourceType.getRank() != type.getRank())
        break;
      sizeExtent = shapeSizeExtent(slice.getMixedSizes()[axis]);
    } else if (auto empty = mlir::dyn_cast<mlir::tensor::EmptyOp>(def)) {
      unsigned dynamicIndex = 0;
      for (int64_t i = 0; i < axis; ++i)
        dynamicIndex += type.isDynamicDim(i);
      sizeExtent = shapeSizeExtent(empty.getDynamicSizes()[dynamicIndex]);
    }
    if (!sizeExtent)
      break;
    if (!sizeExtent->isStatic && sizeExtent->dim == -1)
      if (std::optional<Extent> common =
              selectExtent(sizeExtent->value, selectDepth))
        sizeExtent = common;
    if (sizeExtent->isStatic || sizeExtent->dim == -1) {
      extents.push_back(*sizeExtent);
      break;
    }
    current = sizeExtent->value;
    axis = sizeExtent->dim;
  }
  return extents;
}

// True when each dynamic axis has a common, proven runtime extent. Static
// axes are settled by type equality; equal type spelling alone proves no
// equality between independent dynamic arguments.
bool sameRuntimeShape(mlir::Value lhs, mlir::Value rhs) {
  auto lhsType = mlir::dyn_cast<mlir::RankedTensorType>(lhs.getType());
  auto rhsType = mlir::dyn_cast<mlir::RankedTensorType>(rhs.getType());
  if (!lhsType || lhsType != rhsType)
    return false;
  for (int64_t dim : llvm::seq<int64_t>(0, lhsType.getRank())) {
    if (!lhsType.isDynamicDim(static_cast<unsigned>(dim)))
      continue;
    auto left = traceExtents(lhs, dim);
    auto right = traceExtents(rhs, dim);
    bool equal = false;
    for (const auto &l : left)
      for (const auto &r : right)
        equal |= l == r;
    if (!equal)
      return false;
  }
  return true;
}

struct SwigluFusion : public mlir::OpRewritePattern<mlir::hip::MulOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::hip::MulOp outer,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::hip::MulOp silu;
    mlir::Value up;
    if (auto candidate = outer.getLhs().getDefiningOp<mlir::hip::MulOp>()) {
      silu = candidate;
      up = outer.getRhs();
    } else if (auto candidate =
                   outer.getRhs().getDefiningOp<mlir::hip::MulOp>()) {
      silu = candidate;
      up = outer.getLhs();
    } else {
      return rewriter.notifyMatchFailure(outer, "no inner hip.mul");
    }

    mlir::hip::SigmoidOp sigmoid;
    mlir::Value gate;
    auto matchSiLU = [&](mlir::Value sigmoidResult, mlir::Value other) {
      auto candidate = sigmoidResult.getDefiningOp<mlir::hip::SigmoidOp>();
      if (!candidate || other != candidate.getX())
        return false;
      sigmoid = candidate;
      gate = candidate.getX();
      return true;
    };
    if (!matchSiLU(silu.getLhs(), silu.getRhs()) &&
        !matchSiLU(silu.getRhs(), silu.getLhs()))
      return rewriter.notifyMatchFailure(
          outer, "inner mul is not gate * sigmoid(gate)");

    if (!sigmoid.getCtx() || sigmoid.getCtx() != outer.getCtx() ||
        silu.getCtx() != outer.getCtx())
      return rewriter.notifyMatchFailure(outer,
                                         "SwiGLU ops must share a context");

    if (!hasSingleValueUse(sigmoid->getResult(0), silu) ||
        !hasSingleValueUse(silu->getResult(0), outer))
      return rewriter.notifyMatchFailure(
          outer, "SwiGLU intermediates must have one value use");

    auto asTensor = [](mlir::Value value) {
      return mlir::dyn_cast<mlir::RankedTensorType>(value.getType());
    };
    auto gateType = asTensor(gate);
    auto upType = asTensor(up);
    auto sigmoidType = asTensor(sigmoid->getResult(0));
    auto siluType = asTensor(silu->getResult(0));
    auto resultType = asTensor(outer->getResult(0));
    if (!gateType || gateType != upType || gateType != sigmoidType ||
        gateType != siluType || gateType != resultType)
      return rewriter.notifyMatchFailure(
          outer, "SwiGLU does not support broadcasting");

    mlir::Type elemType = gateType.getElementType();
    if (!elemType.isF16() && !elemType.isF32() && !elemType.isBF16() &&
        !elemType.isF64())
      return rewriter.notifyMatchFailure(
          outer, "SwiGLU supports only f16, f32, bf16, and f64");

    if (!sameRuntimeShape(gate, up))
      return rewriter.notifyMatchFailure(
          outer, "dynamic dimensions are not known to match");

    mlir::Operation *siluInitDef = silu.getOutput().getDefiningOp();
    mlir::Operation *sigmoidInitDef = sigmoid.getY().getDefiningOp();
    mlir::Value siluResult = silu->getResult(0);
    mlir::Value sigmoidResult = sigmoid->getResult(0);
    retargetShapeQueries(rewriter, siluResult, gate);
    retargetShapeQueries(rewriter, sigmoidResult, gate);
    mlir::hip::SwigluOp swiglu = mlir::hip::SwigluOp::create(
        rewriter, outer.getLoc(), resultType, outer.getCtx(), gate, up,
        outer.getOutput());
    rewriter.replaceOp(outer, swiglu.getResult(0));
    rewriter.eraseOp(silu);
    rewriter.eraseOp(sigmoid);
    // Drop a destination that exists only to feed an erased op. A producer
    // with side effects, such as a call that records, stays in the graph.
    // Both destinations may come from one producer, which must be erased once.
    auto eraseIfDead = [&](mlir::Operation *def) {
      if (def && mlir::isOpTriviallyDead(def))
        rewriter.eraseOp(def);
    };
    eraseIfDead(siluInitDef);
    if (sigmoidInitDef != siluInitDef)
      eraseIfDead(sigmoidInitDef);
    return mlir::success();
  }
};

// Cast-to-f32(gate/up) -> f32 SwiGLU -> cast-to-f16. The f32 operation
// establishes the intermediate rounding contract; the promoted kernel keeps
// that contract while reading and writing only the original half buffers.
struct PromotedSwigluFusion : public mlir::OpRewritePattern<mlir::hip::CastOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::hip::CastOp outputCast,
                  mlir::PatternRewriter &rewriter) const override {
    auto swiglu = outputCast.getInput().getDefiningOp<mlir::hip::SwigluOp>();
    if (!swiglu || swiglu.getPromoteActivation())
      return mlir::failure();
    auto gateCast = swiglu.getGate().getDefiningOp<mlir::hip::CastOp>();
    auto upCast = swiglu.getUp().getDefiningOp<mlir::hip::CastOp>();
    if (!gateCast || !upCast || gateCast == upCast)
      return mlir::failure();
    auto gateType =
        mlir::dyn_cast<mlir::RankedTensorType>(gateCast.getInput().getType());
    auto upType =
        mlir::dyn_cast<mlir::RankedTensorType>(upCast.getInput().getType());
    auto outputType = mlir::dyn_cast<mlir::RankedTensorType>(
        outputCast.getResult(0).getType());
    auto floatType =
        mlir::dyn_cast<mlir::RankedTensorType>(swiglu.getResult(0).getType());
    if (!gateType || !floatType || !gateType.getElementType().isF16() ||
        !floatType.getElementType().isF32() || gateType != upType ||
        gateType != outputType || floatType.getShape() != gateType.getShape() ||
        swiglu.getGate().getType() != floatType ||
        swiglu.getUp().getType() != floatType)
      return mlir::failure();
    if (outputCast.getCtx() != swiglu.getCtx() ||
        gateCast.getCtx() != swiglu.getCtx() ||
        upCast.getCtx() != swiglu.getCtx() ||
        !hasSingleValueUse(swiglu.getResult(0), outputCast) ||
        !hasSingleValueUse(gateCast.getResult(0), swiglu) ||
        !hasSingleValueUse(upCast.getResult(0), swiglu))
      return mlir::failure();

    // SwiGLU itself has a no-broadcast shape contract. The standard fusion
    // above proves it before creating the f32 operation.
    llvm::SmallVector<mlir::Operation *> initDefs{
        swiglu.getOutput().getDefiningOp(),
        gateCast.getOutput().getDefiningOp(),
        upCast.getOutput().getDefiningOp()};
    retargetShapeQueries(rewriter, swiglu.getResult(0), gateCast.getInput());
    retargetShapeQueries(rewriter, gateCast.getResult(0), gateCast.getInput());
    retargetShapeQueries(rewriter, upCast.getResult(0), upCast.getInput());
    auto promoted = mlir::hip::SwigluOp::create(
        rewriter, outputCast.getLoc(), outputType, outputCast.getCtx(),
        gateCast.getInput(), upCast.getInput(), outputCast.getOutput(), true);
    rewriter.replaceOp(outputCast, promoted.getResult(0));
    rewriter.eraseOp(swiglu);
    rewriter.eraseOp(gateCast);
    rewriter.eraseOp(upCast);
    llvm::SmallPtrSet<mlir::Operation *, 4> erased;
    for (mlir::Operation *def : initDefs)
      if (def && erased.insert(def).second && mlir::isOpTriviallyDead(def))
        rewriter.eraseOp(def);
    return mlir::success();
  }
};

// hip.gelu(gate) * up -> hip.geglu, under the SwiGLU shape contract.
//
//   Before:
//     %g = hip.gelu(%ctx) ins(%gate : t) outs(%e0 : t) {approximate = "tanh"}
//     %y = hip.mul(%ctx) ins(%g, %up : t, t) outs(%e1 : t) -> t
//   After:
//     %y = hip.geglu(%ctx) ins(%gate, %up : t, t) outs(%e1 : t)
//                          {approximate = "tanh"}
struct GegluFusion : public mlir::OpRewritePattern<mlir::hip::MulOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::hip::MulOp mul,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::hip::GeluOp gelu = mul.getLhs().getDefiningOp<mlir::hip::GeluOp>();
    mlir::Value up = mul.getRhs();
    if (!gelu) {
      gelu = mul.getRhs().getDefiningOp<mlir::hip::GeluOp>();
      up = mul.getLhs();
    }
    if (!gelu)
      return rewriter.notifyMatchFailure(mul, "no hip.gelu operand");
    llvm::StringRef approximate = gelu.getApproximate();
    if (approximate != "none" && approximate != "tanh")
      return rewriter.notifyMatchFailure(mul, "unknown gelu approximation");
    if (!gelu.getCtx() || gelu.getCtx() != mul.getCtx())
      return rewriter.notifyMatchFailure(mul, "GeGLU ops must share a context");
    if (!hasSingleValueUse(gelu->getResult(0), mul))
      return rewriter.notifyMatchFailure(
          mul, "GeGLU intermediate must have one value use");

    mlir::Value gate = gelu.getInput();
    auto gateType = mlir::dyn_cast<mlir::RankedTensorType>(gate.getType());
    auto resultType =
        mlir::dyn_cast<mlir::RankedTensorType>(mul->getResult(0).getType());
    if (!gateType || gateType != up.getType() ||
        gateType != gelu->getResult(0).getType() || gateType != resultType)
      return rewriter.notifyMatchFailure(mul,
                                         "GeGLU does not support broadcasting");
    mlir::Type elemType = gateType.getElementType();
    if (!elemType.isF16() && !elemType.isF32() && !elemType.isBF16() &&
        !elemType.isF64())
      return rewriter.notifyMatchFailure(
          mul, "GeGLU supports only f16, f32, bf16, and f64");
    if (!sameRuntimeShape(gate, up))
      return rewriter.notifyMatchFailure(
          mul, "dynamic dimensions are not known to match");

    mlir::Operation *geluInitDef = gelu.getOutput().getDefiningOp();
    retargetShapeQueries(rewriter, gelu->getResult(0), gate);
    auto geglu = mlir::hip::GegluOp::create(
        rewriter, mul.getLoc(), resultType, mul.getCtx(), gate, up,
        mul.getOutput(), rewriter.getStringAttr(approximate));
    rewriter.replaceOp(mul, geglu.getResult(0));
    rewriter.eraseOp(gelu);
    if (geluInitDef && mlir::isOpTriviallyDead(geluInitDef))
      rewriter.eraseOp(geluInitDef);
    return mlir::success();
  }
};

// hip.rms_norm(x) + r -> hip.rms_norm_add, a post-norm residual connection,
// under the same shape contract.
//
//   Before:
//     %n = hip.rms_norm(%ctx) ins(%x, %g : t, tg) outs(%e0 : t) {...}
//     %y = hip.add(%ctx) ins(%r, %n : t, t) outs(%e1 : t) -> t
//   After:
//     %y = hip.rms_norm_add(%ctx) ins(%x, %g, %r : t, tg, t) outs(%e1 : t)
//
// stash_type is dropped: hip.rms_norm accumulates in fp32 whatever it says.
struct RmsNormAddFusion : public mlir::OpRewritePattern<mlir::hip::AddOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::hip::AddOp add,
                  mlir::PatternRewriter &rewriter) const override {
    auto norm = add.getLhs().getDefiningOp<mlir::hip::RmsNormOp>();
    mlir::Value residual = add.getRhs();
    if (!norm) {
      norm = add.getRhs().getDefiningOp<mlir::hip::RmsNormOp>();
      residual = add.getLhs();
    }
    if (!norm)
      return rewriter.notifyMatchFailure(add, "no hip.rms_norm operand");
    if (!norm.getCtx() || norm.getCtx() != add.getCtx())
      return rewriter.notifyMatchFailure(add, "ops must share a context");
    if (!hasSingleValueUse(norm->getResult(0), add))
      return rewriter.notifyMatchFailure(
          add, "rms_norm result must have one value use");

    auto type =
        mlir::dyn_cast<mlir::RankedTensorType>(add->getResult(0).getType());
    if (!type || norm->getResult(0).getType() != type ||
        norm.getInput().getType() != type || residual.getType() != type)
      return rewriter.notifyMatchFailure(add,
                                         "residual add does not broadcast");
    if (!type.getElementType().isF16())
      return rewriter.notifyMatchFailure(add, "fused residual add is f16 only");
    if (!sameRuntimeShape(norm.getInput(), residual))
      return rewriter.notifyMatchFailure(
          add, "dynamic dimensions are not known to match");

    mlir::Operation *normInitDef = norm.getOutput().getDefiningOp();
    retargetShapeQueries(rewriter, norm->getResult(0), norm.getInput());
    auto fused = mlir::hip::RmsNormAddOp::create(
        rewriter, add.getLoc(), type, add.getCtx(), norm.getInput(),
        norm.getScale(), residual, add.getOutput(), norm.getAxisAttr(),
        norm.getEpsilonAttr());
    rewriter.replaceOp(add, fused->getResult(0));
    rewriter.eraseOp(norm);
    if (normInitDef && mlir::isOpTriviallyDead(normInitDef))
      rewriter.eraseOp(normInitDef);
    return mlir::success();
  }
};

// An RMS norm without a scale, written out in f32 primitives, to hip.rms_norm
// with a scale of ones. Gemma's value norm is exported this way; the f32
// round trip is the contract hip.rms_norm already keeps (fp32 accumulation,
// result rounded once to f16).
//
//   Before:
//     %x32 = hip.cast(%ctx) ins(%x : t16) outs(..) {to = 1}
//     %sq  = hip.mul(%ctx) ins(%x32, %x32 : t32, t32) outs(..)
//     %m   = hip.reduce_mean(%ctx) ins(%sq, %axes) outs(..)
//            {reduced_axes = array<i64: last>}                  // keeps dims
//     %a   = hip.add(%ctx) ins(%m, %eps : r32, tensor<1xf32>) outs(..)
//     %s   = hip.sqrt(%ctx) ins(%a) outs(..)
//     %r   = hip.reciprocal(%ctx) ins(%s) outs(..)
//     %y32 = hip.mul(%ctx) ins(%x32, %r : t32, r32) outs(..)
//     %y   = hip.cast(%ctx) ins(%y32 : t32) outs(%e : t16) {to = 10}
//   After:
//     %ones = hip.constant {value = dense<1.0> : tensor<Dxf16>}
//     %y = hip.rms_norm(%ctx) ins(%x, %ones) outs(%e : t16)
//          {axis = -1, epsilon = eps, stash_type = 1}
//
// The chain's results also size the inits that follow them through
// tensor.dim. Those queries move to %x: a full-width result has its shape,
// and a reduced one shares every axis but the last, which only a static 1
// stands for. A query of the last axis of a reduced result declines.
struct UnscaledRmsNormFusion
    : public mlir::OpRewritePattern<mlir::hip::CastOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::hip::CastOp outCast,
                  mlir::PatternRewriter &rewriter) const override {
    auto outType =
        mlir::dyn_cast<mlir::RankedTensorType>(outCast.getResult(0).getType());
    if (!outType || !outType.getElementType().isF16() || outType.getRank() < 1)
      return rewriter.notifyMatchFailure(outCast, "not an f16 result");
    const int64_t rank = outType.getRank();
    const int64_t width = outType.getShape().back();
    if (mlir::ShapedType::isDynamic(width))
      return rewriter.notifyMatchFailure(outCast, "normalized axis is dynamic");

    auto yMul = outCast.getInput().getDefiningOp<mlir::hip::MulOp>();
    if (!yMul)
      return rewriter.notifyMatchFailure(outCast, "no hip.mul producer");
    mlir::Value x32 = yMul.getLhs();
    auto recip = yMul.getRhs().getDefiningOp<mlir::hip::ReciprocalOp>();
    if (!recip) {
      x32 = yMul.getRhs();
      recip = yMul.getLhs().getDefiningOp<mlir::hip::ReciprocalOp>();
    }
    if (!recip)
      return rewriter.notifyMatchFailure(outCast, "no hip.reciprocal operand");
    auto inCast = x32.getDefiningOp<mlir::hip::CastOp>();
    auto sqrt = recip->getOperand(1).getDefiningOp<mlir::hip::SqrtOp>();
    if (!inCast || !sqrt)
      return rewriter.notifyMatchFailure(outCast, "not cast(x) * 1/sqrt(.)");
    auto add = sqrt->getOperand(1).getDefiningOp<mlir::hip::AddOp>();
    if (!add)
      return rewriter.notifyMatchFailure(outCast, "no hip.add under sqrt");
    mlir::Value epsValue = add.getRhs();
    auto mean = add.getLhs().getDefiningOp<mlir::hip::ReduceMeanOp>();
    if (!mean) {
      epsValue = add.getLhs();
      mean = add.getRhs().getDefiningOp<mlir::hip::ReduceMeanOp>();
    }
    if (!mean)
      return rewriter.notifyMatchFailure(outCast, "no hip.reduce_mean");
    auto sqMul = mean->getOperand(1).getDefiningOp<mlir::hip::MulOp>();
    if (!sqMul || sqMul.getLhs() != x32 || sqMul.getRhs() != x32)
      return rewriter.notifyMatchFailure(outCast, "mean is not over x * x");

    mlir::Value x = inCast.getInput();
    auto xType = mlir::dyn_cast<mlir::RankedTensorType>(x.getType());
    auto x32Type = mlir::dyn_cast<mlir::RankedTensorType>(x32.getType());
    auto meanType =
        mlir::dyn_cast<mlir::RankedTensorType>(mean->getResult(0).getType());
    if (xType != outType || !x32Type || !x32Type.getElementType().isF32() ||
        x32Type.getShape() != outType.getShape() ||
        yMul->getResult(0).getType() != x32Type ||
        sqMul->getResult(0).getType() != x32Type || !meanType ||
        meanType.getRank() != rank || meanType.getShape().back() != 1)
      return rewriter.notifyMatchFailure(outCast, "unexpected chain types");
    for (mlir::Value reduced :
         {add->getResult(0), sqrt->getResult(0), recip->getResult(0)})
      if (reduced.getType() != meanType)
        return rewriter.notifyMatchFailure(outCast, "unexpected reduced types");
    auto axes = mean->getAttrOfType<mlir::DenseI64ArrayAttr>("reduced_axes");
    if (!axes || axes.size() != 1 || (axes[0] != rank - 1 && axes[0] != -1))
      return rewriter.notifyMatchFailure(outCast, "not a last-axis mean");
    auto epsConst = epsValue.getDefiningOp<mlir::hip::ConstantOp>();
    auto epsPayload = epsConst
                          ? mlir::dyn_cast_or_null<mlir::DenseElementsAttr>(
                                epsConst->getAttr("value"))
                          : mlir::DenseElementsAttr();
    if (!epsPayload || epsPayload.getNumElements() != 1 ||
        !epsPayload.getElementType().isF32())
      return rewriter.notifyMatchFailure(outCast,
                                         "epsilon is not an f32 scalar");
    const float eps =
        epsPayload.getSplatValue<llvm::APFloat>().convertToFloat();
    if (!(eps >= 0.0f))
      return rewriter.notifyMatchFailure(outCast, "negative epsilon");

    mlir::Value ctx = outCast.getCtx();
    for (mlir::Operation *op :
         {yMul.getOperation(), recip.getOperation(), sqrt.getOperation(),
          add.getOperation(), mean.getOperation(), sqMul.getOperation(),
          inCast.getOperation()})
      if (op->getOperand(0) != ctx)
        return rewriter.notifyMatchFailure(outCast, "ops must share a context");

    // Every value use stays inside the chain; the rest are shape queries.
    if (!hasSingleValueUse(yMul->getResult(0), outCast) ||
        !hasSingleValueUse(recip->getResult(0), yMul) ||
        !hasSingleValueUse(sqrt->getResult(0), recip) ||
        !hasSingleValueUse(add->getResult(0), sqrt) ||
        !hasSingleValueUse(mean->getResult(0), add) ||
        !hasSingleValueUse(sqMul->getResult(0), mean))
      return rewriter.notifyMatchFailure(outCast, "intermediate escapes");
    for (mlir::OpOperand &use : x32.getUses()) {
      mlir::Operation *owner = use.getOwner();
      if (owner != sqMul && owner != yMul &&
          !mlir::isa<mlir::tensor::DimOp>(owner))
        return rewriter.notifyMatchFailure(outCast, "cast result escapes");
    }
    for (mlir::Value reduced : {mean->getResult(0), add->getResult(0),
                                sqrt->getResult(0), recip->getResult(0)})
      for (mlir::Operation *user : reduced.getUsers())
        if (auto dim = mlir::dyn_cast<mlir::tensor::DimOp>(user)) {
          std::optional<int64_t> index = dim.getConstantIndex();
          if (!index || *index < 0 || *index >= rank - 1)
            return rewriter.notifyMatchFailure(outCast,
                                               "query of the reduced axis");
        }

    auto scaleType =
        mlir::RankedTensorType::get({width}, outType.getElementType());
    auto ones = mlir::hip::ConstantOp::create(
        rewriter, outCast.getLoc(), scaleType,
        mlir::DenseElementsAttr::get(
            scaleType, llvm::ArrayRef<mlir::Attribute>{rewriter.getFloatAttr(
                           outType.getElementType(), 1.0)}));
    auto norm = mlir::hip::RmsNormOp::create(
        rewriter, outCast.getLoc(), outType, ctx, x, ones.getResult(),
        outCast.getOutput(), rewriter.getI64IntegerAttr(-1),
        rewriter.getF32FloatAttr(eps), rewriter.getI64IntegerAttr(1));
    rewriter.replaceOp(outCast, norm->getResult(0));

    // Last consumer first, so each op has only shape queries left when it
    // goes.
    for (mlir::Operation *op :
         {yMul.getOperation(), recip.getOperation(), sqrt.getOperation(),
          add.getOperation(), mean.getOperation(), sqMul.getOperation(),
          inCast.getOperation()}) {
      mlir::Operation *initDef =
          op->getOperand(op->getNumOperands() - 1).getDefiningOp();
      retargetShapeQueries(rewriter, op->getResult(0), x);
      rewriter.eraseOp(op);
      if (initDef && mlir::isOpTriviallyDead(initDef))
        rewriter.eraseOp(initDef);
    }
    return mlir::success();
  }
};

} // namespace

void populateRmsNormAddFusionPattern(mlir::RewritePatternSet &patterns,
                                     mlir::PatternBenefit benefit) {
  patterns.add<RmsNormAddFusion>(patterns.getContext(), benefit);
}

void populateUnscaledRmsNormFusionPattern(mlir::RewritePatternSet &patterns,
                                          mlir::PatternBenefit benefit) {
  patterns.add<UnscaledRmsNormFusion>(patterns.getContext(), benefit);
}

void populateSwigluFusionPattern(mlir::RewritePatternSet &patterns,
                                 mlir::PatternBenefit benefit) {
  patterns.add<SwigluFusion, PromotedSwigluFusion>(patterns.getContext(),
                                                   benefit);
}

void populateGegluFusionPattern(mlir::RewritePatternSet &patterns,
                                mlir::PatternBenefit benefit) {
  patterns.add<GegluFusion>(patterns.getContext(), benefit);
}

} // namespace fusion_transform
} // namespace hip
