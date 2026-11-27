/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- GegluFusion.cpp - Fold the GEGLU primitive chain into hip.geglu ----===//
//
// diffusers' feed-forward (GEGLU) chunks a projection P = [.., 2F] into a value
// half and a gate half and multiplies the value by the exact GELU of the gate.
// Exported, that is
//
//   h = Slice(P, [0], [F], axes = [-1])
//   g = Slice(P, [F], [2F], axes = [-1])
//   y = Mul(h, Mul(Mul(g, Add(Erf(Div(g, sqrt 2)), 1)), 0.5))
//
// with the 0.5 and the 1 in either association and order. Lowered as written
// each Slice becomes a strided copy of half the projection and each of the five
// elementwise ops a pass over [.., F]: at 64x64 latents that is about 300 MB of
// traffic per block for an op that has to read the projection once and write
// its output once. hip.geglu does exactly that.
//
// The Div may already have been turned into Mul(g, Reciprocal(sqrt 2)) by the
// broadcasting-Div rewrite, and the GELU may already be an onnx.Gelu(
// approximate = "none") from ErfGeluFusion; both forms are accepted.
//
// Rooted on onnx.Mul (the gating multiply) and run in the pre-lowering set,
// where the slice bounds and the GELU literals are still onnx.Constant.
//
//===----------------------------------------------------------------------===//

#include "OnnxToHipUtils.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"

#include <cmath>
#include <limits>
#include <optional>

#undef DEBUG_TYPE
#define DEBUG_TYPE "geglu-fusion"

STATISTIC(NumGegluFused,
          "Number of GEGLU primitive chains folded to hip.geglu");

namespace mlir {
namespace hip {

namespace {

// fp16 literals: sqrt(2) is stored as 1.4140625.
constexpr double kConstantTolerance = 1e-3;

bool isOnnxOp(mlir::Operation *op, llvm::StringRef name) {
  return op && op->getName().getStringRef() == name;
}

std::optional<double> getScalarConstant(mlir::Value v) {
  mlir::Operation *def = v.getDefiningOp();
  if (!def)
    return std::nullopt;
  if (isOnnxOp(def, "onnx.Constant")) {
    auto attr =
        mlir::dyn_cast_or_null<mlir::DenseElementsAttr>(def->getAttr("value"));
    if (!attr || attr.getNumElements() != 1 ||
        !mlir::isa<mlir::FloatType>(attr.getElementType()))
      return std::nullopt;
    return (*attr.getValues<llvm::APFloat>().begin()).convertToDouble();
  }
  if ((isOnnxOp(def, "onnx.Cast") || isOnnxOp(def, "onnx.CastLike")) &&
      def->getNumOperands() >= 1)
    return getScalarConstant(def->getOperand(0));
  if (isOnnxOp(def, "onnx.Reciprocal") && def->getNumOperands() == 1) {
    auto inner = getScalarConstant(def->getOperand(0));
    if (!inner || *inner == 0.0)
      return std::nullopt;
    return 1.0 / *inner;
  }
  if (isOnnxOp(def, "onnx.Sqrt") && def->getNumOperands() == 1) {
    auto inner = getScalarConstant(def->getOperand(0));
    if (!inner || *inner < 0.0)
      return std::nullopt;
    return std::sqrt(*inner);
  }
  return std::nullopt;
}

bool isNear(std::optional<double> v, double expected) {
  return v && std::abs(*v - expected) < kConstantTolerance;
}

/// For a binary elementwise op with one scalar operand near `expected`, the
/// other operand.
mlir::Value operandBesideScalar(mlir::Operation *op, double expected) {
  if (!op || op->getNumOperands() != 2)
    return nullptr;
  for (unsigned i : {0u, 1u})
    if (isNear(getScalarConstant(op->getOperand(1 - i)), expected) &&
        op->getOperand(i).getType() == op->getResult(0).getType())
      return op->getOperand(i);
  return nullptr;
}

/// The operand of `op` that is not `known`, for a binary op that has it.
mlir::Value otherOperand(mlir::Operation *op, mlir::Value known) {
  if (!op || op->getNumOperands() != 2)
    return nullptr;
  if (op->getOperand(0) == known)
    return op->getOperand(1);
  if (op->getOperand(1) == known)
    return op->getOperand(0);
  return nullptr;
}

bool isMul(mlir::Operation *op) {
  return isOnnxOp(op, "onnx.Mul") && op->getNumOperands() == 2;
}

/// The single scalar i64 of a Slice bound, normalised against `extent`.
std::optional<int64_t> getSliceBound(mlir::Value v, int64_t extent) {
  mlir::Operation *def = v.getDefiningOp();
  if (!isOnnxOp(def, "onnx.Constant"))
    return std::nullopt;
  auto attr =
      mlir::dyn_cast_or_null<mlir::DenseIntElementsAttr>(def->getAttr("value"));
  if (!attr || attr.getNumElements() != 1)
    return std::nullopt;
  int64_t b = (*attr.getValues<llvm::APInt>().begin()).getSExtValue();
  if (b < 0)
    b += extent;
  return std::clamp<int64_t>(b, 0, extent);
}

/// A unit-step onnx.Slice of the last axis of a static-last-dim tensor, as
/// (source, start, end).
struct LastAxisSlice {
  mlir::Value source;
  int64_t start, end;
};
std::optional<LastAxisSlice> matchLastAxisSlice(mlir::Value v) {
  mlir::Operation *slice = v.getDefiningOp();
  if (!isOnnxOp(slice, "onnx.Slice") || slice->getNumOperands() < 3)
    return std::nullopt;
  mlir::Value source = slice->getOperand(0);
  auto type = mlir::dyn_cast<mlir::RankedTensorType>(source.getType());
  if (!type || type.getRank() < 1 || type.isDynamicDim(type.getRank() - 1))
    return std::nullopt;
  const int64_t rank = type.getRank();
  const int64_t extent = type.getDimSize(rank - 1);
  if (slice->getNumOperands() >= 4 &&
      !mlir::isa<mlir::NoneType>(slice->getOperand(3).getType())) {
    mlir::Operation *axesDef = slice->getOperand(3).getDefiningOp();
    auto axes = isOnnxOp(axesDef, "onnx.Constant")
                    ? mlir::dyn_cast_or_null<mlir::DenseIntElementsAttr>(
                          axesDef->getAttr("value"))
                    : nullptr;
    if (!axes || axes.getNumElements() != 1)
      return std::nullopt;
    int64_t axis = (*axes.getValues<llvm::APInt>().begin()).getSExtValue();
    if (axis < 0)
      axis += rank;
    if (axis != rank - 1)
      return std::nullopt;
  } else if (rank != 1) {
    return std::nullopt;
  }
  if (slice->getNumOperands() >= 5 &&
      !mlir::isa<mlir::NoneType>(slice->getOperand(4).getType())) {
    mlir::Operation *stepDef = slice->getOperand(4).getDefiningOp();
    auto steps = isOnnxOp(stepDef, "onnx.Constant")
                     ? mlir::dyn_cast_or_null<mlir::DenseIntElementsAttr>(
                           stepDef->getAttr("value"))
                     : nullptr;
    if (!steps || steps.getNumElements() != 1 ||
        (*steps.getValues<llvm::APInt>().begin()).getSExtValue() != 1)
      return std::nullopt;
  }
  auto start = getSliceBound(slice->getOperand(1), extent);
  auto end = getSliceBound(slice->getOperand(2), extent);
  if (!start || !end)
    return std::nullopt;
  return LastAxisSlice{source, *start, *end};
}

/// Whether `v` is the exact GELU of `g`, as onnx.Gelu or as the primitive
/// chain. Appends the chain's ops to `chain` on success.
bool matchErfGelu(mlir::Value v, mlir::Value &g,
                  llvm::SmallVectorImpl<mlir::Operation *> &chain) {
  mlir::Operation *outer = v.getDefiningOp();
  if (isOnnxOp(outer, "onnx.Gelu") && outer->getNumOperands() == 1) {
    auto approx = outer->getAttrOfType<mlir::StringAttr>("approximate");
    if (approx && approx.getValue() != "none")
      return false;
    g = outer->getOperand(0);
    chain.push_back(outer);
    return true;
  }
  if (!isMul(outer))
    return false;

  // 0.5 * g * a in any association: find the 0.5 and the (1 + erf) factor.
  // a = Add(Erf(g / sqrt 2), 1) or the same with Sum.
  auto matchOnePlusErf = [&](mlir::Value a, mlir::Value &gate,
                             mlir::Operation *&addOp, mlir::Operation *&erfOp,
                             mlir::Operation *&scaleOp) -> bool {
    addOp = a.getDefiningOp();
    if (!(isOnnxOp(addOp, "onnx.Add") || isOnnxOp(addOp, "onnx.Sum")))
      return false;
    mlir::Value e = operandBesideScalar(addOp, 1.0);
    if (!e)
      return false;
    erfOp = e.getDefiningOp();
    if (!isOnnxOp(erfOp, "onnx.Erf") || erfOp->getNumOperands() != 1)
      return false;
    scaleOp = erfOp->getOperand(0).getDefiningOp();
    if (isOnnxOp(scaleOp, "onnx.Div") && scaleOp->getNumOperands() == 2 &&
        isNear(getScalarConstant(scaleOp->getOperand(1)), std::sqrt(2.0))) {
      gate = scaleOp->getOperand(0);
      return true;
    }
    if (isMul(scaleOp)) {
      gate = operandBesideScalar(scaleOp, 1.0 / std::sqrt(2.0));
      return static_cast<bool>(gate);
    }
    return false;
  };

  mlir::Operation *addOp = nullptr, *erfOp = nullptr, *scaleOp = nullptr;
  mlir::Operation *inner = nullptr;
  mlir::Value gate;
  if (mlir::Value m = operandBesideScalar(outer, 0.5)) {
    // (g * a) * 0.5
    inner = m.getDefiningOp();
    if (!isMul(inner))
      return false;
    bool found = false;
    for (unsigned i : {0u, 1u}) {
      mlir::Value cand;
      if (matchOnePlusErf(inner->getOperand(i), cand, addOp, erfOp, scaleOp) &&
          inner->getOperand(1 - i) == cand) {
        gate = cand;
        found = true;
        break;
      }
    }
    if (!found)
      return false;
  } else {
    // (g * 0.5) * a, or g * (a * 0.5): one operand of `outer` carries the
    // 0.5 together with g or with a.
    bool found = false;
    for (unsigned i : {0u, 1u}) {
      mlir::Operation *half = outer->getOperand(i).getDefiningOp();
      mlir::Value rest = outer->getOperand(1 - i);
      if (!isMul(half))
        continue;
      mlir::Value x = operandBesideScalar(half, 0.5);
      if (!x)
        continue;
      mlir::Value cand;
      if (matchOnePlusErf(rest, cand, addOp, erfOp, scaleOp) && x == cand) {
        gate = cand; // (g * 0.5) * a
      } else if (matchOnePlusErf(x, cand, addOp, erfOp, scaleOp) &&
                 rest == cand) {
        gate = cand; // g * (a * 0.5)
      } else {
        continue;
      }
      inner = half;
      found = true;
      break;
    }
    if (!found)
      return false;
  }
  g = gate;
  chain.append({outer, inner, addOp, erfOp, scaleOp});
  return true;
}

struct GegluToHip : public mlir::RewritePattern {
  GegluToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Mul", /*benefit=*/2, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *mul,
                  mlir::PatternRewriter &rewriter) const override {
    if (!isMul(mul))
      return rewriter.notifyMatchFailure(mul, "mul.arity");
    auto outType =
        mlir::dyn_cast<mlir::RankedTensorType>(mul->getResult(0).getType());
    if (!outType || outType.getRank() < 1 ||
        !(outType.getElementType().isF16() || outType.getElementType().isF32()))
      return rewriter.notifyMatchFailure(mul, "output.type");

    for (unsigned i : {0u, 1u}) {
      mlir::Value value = mul->getOperand(i);
      llvm::SmallVector<mlir::Operation *> chain;
      mlir::Value gate;
      if (!matchErfGelu(mul->getOperand(1 - i), gate, chain))
        continue;
      auto valueSlice = matchLastAxisSlice(value);
      auto gateSlice = matchLastAxisSlice(gate);
      if (!valueSlice || !gateSlice || valueSlice->source != gateSlice->source)
        continue;
      mlir::Value proj = valueSlice->source;
      auto projType = mlir::cast<mlir::RankedTensorType>(proj.getType());
      const int64_t rank = projType.getRank();
      const int64_t twoF = projType.getDimSize(rank - 1);
      if (twoF % 2 != 0)
        continue;
      const int64_t f = twoF / 2;
      if (valueSlice->start != 0 || valueSlice->end != f ||
          gateSlice->start != f || gateSlice->end != twoF)
        continue;
      if (projType.getElementType() != outType.getElementType() ||
          outType.getRank() != rank || outType.isDynamicDim(rank - 1) ||
          outType.getDimSize(rank - 1) != f)
        continue;
      // The output must have the projection's leading extents (no
      // broadcast through the multiply).
      bool sameLeading = true;
      for (int64_t d : llvm::seq<int64_t>(0, rank - 1))
        sameLeading =
            sameLeading && outType.getDimSize(d) == projType.getDimSize(d);
      if (!sameLeading)
        continue;
      // Every intermediate must feed only the chain, or the fold would keep
      // it alive anyway and gain nothing.
      chain.push_back(value.getDefiningOp());
      chain.push_back(gate.getDefiningOp());
      bool private_ = true;
      for (mlir::Operation *op : chain)
        for (mlir::Operation *user : op->getResult(0).getUsers())
          private_ =
              private_ && (user == mul || llvm::is_contained(chain, user));
      if (!private_)
        continue;

      auto ctx = getContextArg(mul, rewriter);
      if (mlir::failed(ctx))
        return rewriter.notifyMatchFailure(mul, "no context argument");
      rewriter.setInsertionPoint(mul);
      mlir::Location loc = mul->getLoc();
      mlir::Value init = createEmptyTensor(rewriter, loc, outType, proj);
      auto geglu = GegluOp::create(rewriter, loc, *ctx, proj, init);
      rewriter.replaceOp(mul, geglu->getResult(0));
      // Users before producers: the GELU chain, then the two slices.
      for (mlir::Operation *op : chain)
        if (op->use_empty())
          rewriter.eraseOp(op);
      ++NumGegluFused;
      return mlir::success();
    }
    return rewriter.notifyMatchFailure(mul, "not a GEGLU gating multiply");
  }
};

} // namespace

void populateGegluFusionPatterns(RewritePatternSet &patterns,
                                 MLIRContext *ctx) {
  patterns.add<GegluToHip>(ctx);
}

} // namespace hip
} // namespace mlir
