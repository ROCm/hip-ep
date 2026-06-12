/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- ErfGeluFusion.cpp - Fold erf-form GELU primitive chain
//---------------===//
//
// Some ONNX exports inline the function body of `Gelu(approximate="none")`
// (the exact erf-based definition) into primitive ops:
//
//   y = 0.5 * x * (1 + erf(x / sqrt(2)))
//
// expressed as the chain produced by e.g. the ConvNeXt export path
// (constants are wrapped in onnx.CastLike <f32-const>, <activation> so the
// scalar inherits the activation's dtype):
//
//   Before (chain rooted on onnx.Erf, leaves -> root):
//     %one      = onnx.CastLike(%c1_f32,   %x) : (f32, ...f16) -> f16
//     %two      = onnx.CastLike(%c2_f32,   %x) : (f32, ...f16) -> f16
//     %sqrt2    = onnx.Sqrt(%two)              : f16        // ≈ 1.4142
//     %xsqrt    = onnx.Div(%x, %sqrt2)         : ...xf16
//     %erf      = onnx.Erf(%xsqrt)             : ...xf16
//     %phi      = onnx.Sum(%one, %erf)         : ...xf16    // 1 + erf
//     %half     = onnx.CastLike(%c05_f32,  %x) : (f32, ...f16) -> f16
//     %halfX    = onnx.Mul(%half, %x)          : ...xf16    // 0.5 * x
//     %y        = onnx.Mul(%halfX, %phi)       : ...xf16
//
//   After (entire chain replaced; primitive constants left dangling for the
//   global `onnx.* use_empty` DCE walk in OnnxToHip.cpp):
//     %y = onnx.Gelu(%x) {approximate = "none"} : tensor<...xf16>
//
// MorphiZen has a converter for `onnx.Gelu(approximate="none")` (lowered to
// `hip.gelu`) but no converters for the primitive ops (onnx.Erf, onnx.Sqrt,
// onnx.Sum, onnx.Div in this scalar-divisor form). Without this fusion the
// inlined primitives survive to OneShotBufferize and fail with
// `error: op was not bufferized`, aborting the whole compile.
//
// Matching is value-based against the canonical literals (1.0, 2.0, 0.5) with
// the same tolerance / constant-peek-through helpers used by FastGeluFusion
// (Cast / CastLike / Sqrt-of-const), so the `Sqrt(2.0)` and `CastLike` wrappers
// fold transparently into the scalar comparison.
//
// Rooted on `onnx.Erf`; run BEFORE `lowerOnnxConstants` so literal values are
// still inline in `onnx.Constant` value attributes.
//
//===----------------------------------------------------------------------===//

#include "OnnxToHipUtils.h"

#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Debug.h"

#include <cmath>

#define DEBUG_TYPE "erf-gelu-fusion"

STATISTIC(NumErfGeluFused,
          "Number of inlined erf-form GELU primitive chains folded back to "
          "onnx.Gelu(approximate=none)");

namespace mlir {
namespace hip {

namespace {

// Same tolerance rationale as FastGeluFusion: fp16 round-trip of 1.0/2.0/0.5
// is exact, but sqrt(2.0) -> 1.4142 etc. drift slightly; 1e-3 covers both.
constexpr double kConstantTolerance = 1e-3;

/// Recursively evaluate a single fp scalar value defined by an `onnx.Constant`
/// possibly wrapped in value-preserving ops (Cast / CastLike for fp dtype
/// changes; Sqrt for `sqrt(2)` style encodings). Mirrors FastGeluFusion's
/// helper of the same name.
static std::optional<double> getScalarFloatConstant(mlir::Value v) {
  mlir::Operation *def = v.getDefiningOp();
  if (!def)
    return std::nullopt;
  llvm::StringRef opName = def->getName().getStringRef();

  if (opName == "onnx.Constant") {
    auto valueAttr = def->getAttr("value");
    auto denseAttr = mlir::dyn_cast_or_null<mlir::DenseElementsAttr>(valueAttr);
    if (!denseAttr || denseAttr.getNumElements() != 1)
      return std::nullopt;
    mlir::Type et = denseAttr.getElementType();
    if (et.isF32())
      return static_cast<double>(*denseAttr.getValues<float>().begin());
    if (et.isF64())
      return *denseAttr.getValues<double>().begin();
    if (et.isF16() || et.isBF16())
      return (*denseAttr.getValues<llvm::APFloat>().begin()).convertToDouble();
    return std::nullopt;
  }

  if ((opName == "onnx.Cast" || opName == "onnx.CastLike") &&
      def->getNumOperands() >= 1)
    return getScalarFloatConstant(def->getOperand(0));

  if (opName == "onnx.Sqrt" && def->getNumOperands() >= 1) {
    auto inner = getScalarFloatConstant(def->getOperand(0));
    if (!inner)
      return std::nullopt;
    return std::sqrt(*inner);
  }

  return std::nullopt;
}

static bool isScalarFloatNear(mlir::Value v, double expected) {
  auto val = getScalarFloatConstant(v);
  return val && std::abs(*val - expected) < kConstantTolerance;
}

/// For a 2-operand commutative op, return (constant, other) when exactly one
/// operand is a scalar float near `expected`.
static std::pair<mlir::Value, mlir::Value>
matchCommutativeConst(mlir::Operation *op, double expected) {
  if (!op || op->getNumOperands() != 2)
    return {nullptr, nullptr};
  mlir::Value a = op->getOperand(0), b = op->getOperand(1);
  bool aIsConst = isScalarFloatNear(a, expected);
  bool bIsConst = isScalarFloatNear(b, expected);
  if (aIsConst && !bIsConst)
    return {a, b};
  if (bIsConst && !aIsConst)
    return {b, a};
  return {nullptr, nullptr};
}

struct InlinedErfGeluToGelu : public mlir::RewritePattern {
  InlinedErfGeluToGelu(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Erf", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *erfOp,
                  mlir::PatternRewriter &rewriter) const override {
    if (erfOp->getNumOperands() != 1 || erfOp->getNumResults() != 1)
      return rewriter.notifyMatchFailure(erfOp, "erf.arity");

    // ── back-walk ────────────────────────────────────────────────────────
    // Erf's input must be Div(x, c) where c ≈ sqrt(2). Not commutative —
    // operand 0 is the numerator (x), operand 1 the divisor.
    mlir::Operation *divOp = erfOp->getOperand(0).getDefiningOp();
    if (!divOp)
      return rewriter.notifyMatchFailure(erfOp, "div.defop_null");
    if (divOp->getName().getStringRef() != "onnx.Div" ||
        divOp->getNumOperands() != 2)
      return rewriter.notifyMatchFailure(erfOp, [&](mlir::Diagnostic &d) {
        d << "div.op (observed: " << divOp->getName() << ")";
      });
    mlir::Value x = divOp->getOperand(0);
    if (!isScalarFloatNear(divOp->getOperand(1), std::sqrt(2.0)))
      return rewriter.notifyMatchFailure(erfOp, [&](mlir::Diagnostic &d) {
        auto v = getScalarFloatConstant(divOp->getOperand(1));
        d << "div.divisor_not_sqrt2 (observed: "
          << (v ? std::to_string(*v) : std::string("<not_const>")) << ")";
      });

    // ── forward-walk ─────────────────────────────────────────────────────
    // erf.result feeds exactly one Sum: phi = Sum(1, erf)
    mlir::Value erfRes = erfOp->getResult(0);
    if (!erfRes.hasOneUse())
      return rewriter.notifyMatchFailure(erfOp, "erf.not_one_use");
    mlir::Operation *phiOp = *erfRes.getUsers().begin();
    if (phiOp->getName().getStringRef() != "onnx.Sum" ||
        phiOp->getNumOperands() != 2)
      return rewriter.notifyMatchFailure(erfOp, [&](mlir::Diagnostic &d) {
        d << "phi.op (observed: " << phiOp->getName() << ")";
      });
    mlir::Value c1, erfInPhi;
    {
      auto [c, other] = matchCommutativeConst(phiOp, 1.0);
      if (!c)
        return rewriter.notifyMatchFailure(erfOp, "phi.sum_one");
      c1 = c;
      erfInPhi = other;
    }
    if (erfInPhi != erfRes)
      return rewriter.notifyMatchFailure(erfOp, "phi.erf_mismatch");

    // phi must feed exactly one Mul: y = Mul(halfX, phi)
    if (!phiOp->getResult(0).hasOneUse())
      return rewriter.notifyMatchFailure(erfOp, "phi.not_one_use");
    mlir::Operation *finalMulOp = *phiOp->getResult(0).getUsers().begin();
    if (finalMulOp->getName().getStringRef() != "onnx.Mul" ||
        finalMulOp->getNumOperands() != 2)
      return rewriter.notifyMatchFailure(erfOp, [&](mlir::Diagnostic &d) {
        d << "final.mul (observed: " << finalMulOp->getName() << ")";
      });

    // The non-phi operand of finalMul is halfX = Mul(0.5, x).
    mlir::Value halfX;
    for (mlir::Value cand : finalMulOp->getOperands()) {
      if (cand != phiOp->getResult(0)) {
        halfX = cand;
        break;
      }
    }
    if (!halfX)
      return rewriter.notifyMatchFailure(erfOp, "halfx.missing");
    mlir::Operation *halfXOp = halfX.getDefiningOp();
    mlir::Value cHalf, xInHalf;
    {
      if (!halfXOp || halfXOp->getName().getStringRef() != "onnx.Mul" ||
          halfXOp->getNumOperands() != 2)
        return rewriter.notifyMatchFailure(erfOp, "halfx.not_mul");
      auto [c, other] = matchCommutativeConst(halfXOp, 0.5);
      if (!c)
        return rewriter.notifyMatchFailure(erfOp, "halfx.mul_half");
      cHalf = c;
      xInHalf = other;
    }
    if (xInHalf != x)
      return rewriter.notifyMatchFailure(erfOp, "halfx.x_mismatch");

    // ── all matched; rewrite final Mul to onnx.Gelu(x, "none") ───────────
    mlir::Location loc = finalMulOp->getLoc();
    mlir::OperationState state(loc, "onnx.Gelu");
    state.addOperands(x);
    state.addTypes(finalMulOp->getResult(0).getType());
    state.addAttribute("approximate", rewriter.getStringAttr("none"));
    if (auto outputs =
            finalMulOp->getAttrOfType<mlir::ArrayAttr>("node.outputs"))
      state.addAttribute("node.outputs", outputs);
    if (auto nodeName =
            finalMulOp->getAttrOfType<mlir::StringAttr>("onnx_node_name"))
      state.addAttribute("onnx_node_name", nodeName);
    rewriter.setInsertionPoint(finalMulOp);
    mlir::Operation *geluOp = rewriter.create(state);

    // replaceOp transfers all uses; same cascade-erase discipline as
    // FastGeluFusion. Each intermediate has exactly one user along the
    // matched chain, so erasure cascades cleanly. Shared constants
    // (CastLike-wrapped 1.0 / 2.0 / 0.5 scalars, and any `onnx.Sqrt` of
    // the 2.0 scalar) are intentionally left to the global `onnx.*
    // use_empty` DCE walk at the end of ConvertOnnxToHipPass: they may
    // be shared across many GELU sites, so erasing them eagerly here
    // would race against sibling fusions.
    rewriter.replaceOp(finalMulOp, geluOp->getResult(0));
    auto eraseIfDead = [&rewriter](mlir::Operation *op) {
      if (op && op->use_empty())
        rewriter.eraseOp(op);
    };
    eraseIfDead(halfXOp);
    eraseIfDead(phiOp);
    eraseIfDead(erfOp);
    eraseIfDead(divOp);

    LLVM_DEBUG(llvm::dbgs()
               << "[" DEBUG_TYPE "] fused chain at " << loc << "\n");
    ++NumErfGeluFused;
    return mlir::success();
  }
};

} // namespace

void populateErfGeluFusionPatterns(mlir::RewritePatternSet &patterns,
                                   mlir::MLIRContext *ctx) {
  patterns.add<InlinedErfGeluToGelu>(ctx);
}

} // namespace hip
} // namespace mlir
