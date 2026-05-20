/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- FastGeluFusion.cpp - Fold inlined FastGelu chain into onnx.Gelu ----===//
//
// ORT inlines the function body of `Gelu(approximate="tanh")` for some loading
// paths (notably dynamic-shape models — fixed-shape variants typically keep
// the function intact). The inlined chain is the standard FastGelu identity:
//
//   y = 0.5 * x * (1 + tanh( sqrt(2/π) * (x + 0.044715 * x³) ))
//
// expressed as primitive onnx ops:
//
//   pow      = onnx.Pow(x, 3)
//   scaled   = onnx.Mul(0.044715, pow)
//   inner    = onnx.Sum(x, scaled)
//   tanh_in  = onnx.Mul(sqrt(2/π), inner)
//   tanh     = onnx.Tanh(tanh_in)
//   phi      = onnx.Sum(1, tanh)
//   half_x   = onnx.Mul(0.5, x)
//   y        = onnx.Mul(half_x, phi)
//
// MorphiZen has a converter for `onnx.Gelu(approximate="tanh")` (lowered to
// `hip.gelu`) but **no converters for the primitive ops** (onnx.Pow, onnx.Sum,
// onnx.Tanh). Without this fusion the inlined primitives survive to
// OneShotBufferize and fail with `error: op was not bufferized`, causing
// silent CPU fallback for the affected subgraphs.
//
// Matching is purely structural — operand graph + literal constant values
// (0.044715, sqrt(2/π) ≈ 0.7978845608, 0.5, 1.0, exponent 3) — with a small
// tolerance to absorb fp32→fp16 round-trip drift. Constants may be wrapped
// in `onnx.Cast` (f32 → f16); we peek through one level of Cast.
//
// Implemented as a RewritePattern rooted on `onnx.Tanh` and run BEFORE
// `lowerOnnxConstants` so the literal float values are still inline in
// `onnx.Constant` `value` attributes (post-lowering, constants become
// `arith.constant` / `memref.get_global`, defeating value-based matching).
// Per-Tanh failure diagnostics flow through MLIR's standard
// `notifyMatchFailure` mechanism (toggle via `-debug-only=fast-gelu-fusion`
// or `-debug-only=greedy-rewriter`).
//
//===----------------------------------------------------------------------===//

#include "OnnxToHipUtils.h"

#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Debug.h"

#include <cmath>

#define DEBUG_TYPE "fast-gelu-fusion"

STATISTIC(NumFastGeluFused,
          "Number of inlined FastGelu primitive chains folded back to "
          "onnx.Gelu(approximate=tanh)");

namespace mlir {
namespace hip {

namespace {

/// fp16 round-trip leaves sqrt(2/π) at ~0.79785 and 0.044715 at ~0.04471.
/// 1e-3 absolute tolerance covers the worst case with margin.
constexpr double kConstantTolerance = 1e-3;

/// Recursively evaluate a single fp scalar value defined by an `onnx.Constant`
/// possibly wrapped in any chain of value-preserving ops (`onnx.Cast` for
/// fp32→fp16 conversion, `onnx.Sqrt` for `sqrt(2/π)` which ORT's inlined
/// FastGelu encodes as `Sqrt(Cast(Constant(2/π)))`). Returns `std::nullopt`
/// when `v` is not derived from a single scalar Constant via this restricted
/// op set.
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
    // Integer constants with a single element appear in the wild for the
    // Pow exponent (some ORT exports ship `Pow(x, 3 : i64)` instead of
    // `Pow(x, 3.0)`). Convert to double for value-based matching.
    if (et.isIntOrIndex()) {
      auto first = *denseAttr.getValues<llvm::APInt>().begin();
      return static_cast<double>(first.getSExtValue());
    }
    return std::nullopt;
  }

  // Cast is value-preserving for fp→fp downconversion modulo rounding,
  // which is absorbed by `kConstantTolerance` in the caller. CastLike
  // behaves identically — its second operand only supplies the target
  // dtype; the value being cast is operand 0. Some ORT export paths
  // emit CastLike (notably the inlined Gelu chain in Gemma-3, where every
  // FastGelu literal — sqrt(2/π), 0.044715, 0.5, 1.0, exponent 3 — is
  // wrapped CastLike(<f32 const>, <f16 activation>) so the literal is
  // promoted to whatever dtype the surrounding tensor uses).
  if ((opName == "onnx.Cast" || opName == "onnx.CastLike") &&
      def->getNumOperands() >= 1)
    return getScalarFloatConstant(def->getOperand(0));

  // Sqrt of a constant is itself a constant — FastGelu encodes sqrt(2/π)
  // this way to keep the on-disk model compact.
  if (opName == "onnx.Sqrt" && def->getNumOperands() >= 1) {
    auto inner = getScalarFloatConstant(def->getOperand(0));
    if (!inner)
      return std::nullopt;
    return std::sqrt(*inner);
  }

  return std::nullopt;
}

/// True if `v` is a scalar float constant within `kConstantTolerance` of
/// `expected`.
static bool isScalarFloatNear(mlir::Value v, double expected) {
  auto val = getScalarFloatConstant(v);
  return val && std::abs(*val - expected) < kConstantTolerance;
}

/// For a 2-operand commutative op, return the operand pair (constant, other)
/// when exactly one operand is a scalar float constant near `expected` and
/// the other is anything else. Returns null pair otherwise.
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

/// True if `op` is a binary onnx op of the given name with the given operands
/// (in some order); on success returns through `outConst`/`outOther`.
static bool isBinaryOpWithConst(mlir::Operation *op, llvm::StringRef opName,
                                double expectedConst, mlir::Value &outConst,
                                mlir::Value &outOther) {
  if (!op || op->getName().getStringRef() != opName)
    return false;
  auto [c, other] = matchCommutativeConst(op, expectedConst);
  if (!c)
    return false;
  outConst = c;
  outOther = other;
  return true;
}

struct InlinedFastGeluToGelu : public mlir::RewritePattern {
  InlinedFastGeluToGelu(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Tanh", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *tanhOp,
                  mlir::PatternRewriter &rewriter) const override {
    if (tanhOp->getNumOperands() != 1 || tanhOp->getNumResults() != 1)
      return rewriter.notifyMatchFailure(tanhOp, "tanh.arity");

    // ── back-walk ────────────────────────────────────────────────────────
    // tanh_in = Mul(sqrt(2/π), inner)
    mlir::Operation *tanhInputMul = tanhOp->getOperand(0).getDefiningOp();
    if (!tanhInputMul)
      return rewriter.notifyMatchFailure(tanhOp, "tanh.in.defop_null");
    mlir::Value c2pi, innerSum;
    if (!isBinaryOpWithConst(tanhInputMul, "onnx.Mul",
                             /*sqrt(2/π)*/ 0.7978845608, c2pi, innerSum))
      return rewriter.notifyMatchFailure(tanhOp, [&](mlir::Diagnostic &d) {
        d << "tanh.in.mul_sqrt2pi (observed: " << tanhInputMul->getName()
          << ")";
      });

    // inner = Sum(x, scaled_pow)  (commutative; scaled_pow is the
    // Mul-of-Pow leg)
    mlir::Operation *innerSumOp = innerSum.getDefiningOp();
    if (!innerSumOp)
      return rewriter.notifyMatchFailure(tanhOp, "inner.defop_null");
    if (innerSumOp->getName().getStringRef() != "onnx.Sum" ||
        innerSumOp->getNumOperands() != 2)
      return rewriter.notifyMatchFailure(tanhOp, [&](mlir::Diagnostic &d) {
        d << "inner.sum (observed: " << innerSumOp->getName() << ")";
      });

    // Find which operand of innerSum is the `0.044715 * x³` leg.
    mlir::Value scaledPow, x;
    for (mlir::Value cand : innerSumOp->getOperands()) {
      if (mlir::Operation *def = cand.getDefiningOp()) {
        if (def->getName().getStringRef() == "onnx.Mul" &&
            def->getNumOperands() == 2) {
          auto [c, other] = matchCommutativeConst(def, 0.044715);
          if (c) {
            scaledPow = cand;
            // The other innerSum operand is `x`.
            x = (cand == innerSumOp->getOperand(0)) ? innerSumOp->getOperand(1)
                                                    : innerSumOp->getOperand(0);
            break;
          }
        }
      }
    }
    if (!scaledPow || !x)
      return rewriter.notifyMatchFailure(tanhOp, "scaled_pow.044715");

    // scaled_pow = Mul(0.044715, pow); pow = Pow(x, 3)
    mlir::Operation *scaledPowOp = scaledPow.getDefiningOp();
    mlir::Value c044715, powVal;
    if (!isBinaryOpWithConst(scaledPowOp, "onnx.Mul", 0.044715, c044715,
                             powVal))
      return rewriter.notifyMatchFailure(tanhOp, "scaled_pow.mul");

    mlir::Operation *powOp = powVal.getDefiningOp();
    if (!powOp)
      return rewriter.notifyMatchFailure(tanhOp, "pow.defop_null");
    if (powOp->getName().getStringRef() != "onnx.Pow" ||
        powOp->getNumOperands() != 2)
      return rewriter.notifyMatchFailure(tanhOp, [&](mlir::Diagnostic &d) {
        d << "pow.op (observed: " << powOp->getName() << ")";
      });
    if (powOp->getOperand(0) != x)
      return rewriter.notifyMatchFailure(tanhOp, "pow.base_neq_x");
    if (!isScalarFloatNear(powOp->getOperand(1), 3.0))
      return rewriter.notifyMatchFailure(tanhOp, [&](mlir::Diagnostic &d) {
        auto v = getScalarFloatConstant(powOp->getOperand(1));
        d << "pow.exponent_not_3 (observed: "
          << (v ? std::to_string(*v) : std::string("<not_const>")) << ")";
      });

    // ── forward-walk ─────────────────────────────────────────────────────
    // tanh.result must feed exactly one Sum: phi = Sum(1, tanh)
    mlir::Value tanhRes = tanhOp->getResult(0);
    if (!tanhRes.hasOneUse())
      return rewriter.notifyMatchFailure(tanhOp, "tanh.not_one_use");
    mlir::Operation *phiOp = *tanhRes.getUsers().begin();
    mlir::Value c1, tanhInPhi;
    if (!isBinaryOpWithConst(phiOp, "onnx.Sum", 1.0, c1, tanhInPhi))
      return rewriter.notifyMatchFailure(tanhOp, [&](mlir::Diagnostic &d) {
        d << "phi.sum_one (observed: " << phiOp->getName() << ")";
      });
    if (tanhInPhi != tanhRes)
      return rewriter.notifyMatchFailure(tanhOp, "phi.tanh_mismatch");

    // phi must feed exactly one Mul: y = Mul(half_x, phi)
    if (!phiOp->getResult(0).hasOneUse())
      return rewriter.notifyMatchFailure(tanhOp, "phi.not_one_use");
    mlir::Operation *finalMulOp = *phiOp->getResult(0).getUsers().begin();
    if (finalMulOp->getName().getStringRef() != "onnx.Mul" ||
        finalMulOp->getNumOperands() != 2)
      return rewriter.notifyMatchFailure(tanhOp, [&](mlir::Diagnostic &d) {
        d << "final.mul (observed: " << finalMulOp->getName() << ")";
      });

    // The non-phi operand of finalMul is half_x = Mul(0.5, x).
    mlir::Value halfX;
    for (mlir::Value cand : finalMulOp->getOperands()) {
      if (cand != phiOp->getResult(0)) {
        halfX = cand;
        break;
      }
    }
    if (!halfX)
      return rewriter.notifyMatchFailure(tanhOp, "half_x.missing");
    mlir::Operation *halfXOp = halfX.getDefiningOp();
    mlir::Value cHalf, xInHalf;
    if (!isBinaryOpWithConst(halfXOp, "onnx.Mul", 0.5, cHalf, xInHalf))
      return rewriter.notifyMatchFailure(tanhOp, "half_x.mul_half");
    if (xInHalf != x)
      return rewriter.notifyMatchFailure(tanhOp, "half_x.x_mismatch");

    // ── all matched; rewrite final Mul to onnx.Gelu(x, "tanh") ───────────
    mlir::Location loc = finalMulOp->getLoc();
    mlir::OperationState state(loc, "onnx.Gelu");
    state.addOperands(x);
    state.addTypes(finalMulOp->getResult(0).getType());
    state.addAttribute("approximate", rewriter.getStringAttr("tanh"));
    // Preserve the original output name attribute so downstream IR dumps
    // and metadata stay readable.
    if (auto outputs =
            finalMulOp->getAttrOfType<mlir::ArrayAttr>("node.outputs"))
      state.addAttribute("node.outputs", outputs);
    if (auto nodeName =
            finalMulOp->getAttrOfType<mlir::StringAttr>("onnx_node_name"))
      state.addAttribute("onnx_node_name", nodeName);
    rewriter.setInsertionPoint(finalMulOp);
    mlir::Operation *geluOp = rewriter.create(state);

    // replaceOp transfers all uses of finalMulOp to the new Gelu and
    // erases finalMulOp. In the canonical FastGelu chain each remaining
    // intermediate has exactly one user, so erasure cascades in
    // reverse-topological order: halfXOp and phiOp lose their last user
    // immediately, phi's removal makes tanh use-empty, then tanhInputMul,
    // innerSum, scaledPow, and powOp follow. The `eraseIfDead` guard is
    // defensive — if upstream CSE has merged any of these intermediates
    // across multiple FastGelu sites, the not-yet-dead op is silently
    // skipped here and picked up later when the last sibling fires (or by
    // the post-conversion `onnx.* use_empty` DCE walk in
    // ConvertOnnxToHipPass). The shared `x` activation and any shared
    // constants (0.044715, sqrt(2/π), 0.5, 1.0, exponent 3) are not
    // erased here at all — they fall to that same global sweep.
    rewriter.replaceOp(finalMulOp, geluOp->getResult(0));
    auto eraseIfDead = [&rewriter](mlir::Operation *op) {
      if (op && op->use_empty())
        rewriter.eraseOp(op);
    };
    eraseIfDead(halfXOp);
    eraseIfDead(phiOp);
    eraseIfDead(tanhOp);
    eraseIfDead(tanhInputMul);
    eraseIfDead(innerSumOp);
    eraseIfDead(scaledPowOp);
    eraseIfDead(powOp);

    LLVM_DEBUG(llvm::dbgs()
               << "[" DEBUG_TYPE "] fused chain at " << loc << "\n");
    ++NumFastGeluFused;
    return mlir::success();
  }
};

} // namespace

void populateFastGeluFusionPatterns(mlir::RewritePatternSet &patterns,
                                    mlir::MLIRContext *ctx) {
  patterns.add<InlinedFastGeluToGelu>(ctx);
}

} // namespace hip
} // namespace mlir
