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
//   Before (inlined chain rooted on onnx.Tanh, leaves to root):
//     %three     = onnx.Constant{value=3}              : tensor<i64>
//     %pow       = onnx.Pow(%x, %three)                : tensor<...xf16>
//     %c044715   = onnx.Constant{value=0.044715}       : tensor<f16>
//     %scaled    = onnx.Mul(%c044715, %pow)            : tensor<...xf16>
//     %inner     = onnx.Sum(%x, %scaled)               : tensor<...xf16>
//     %cSqrt2pi  = onnx.Constant{value=0.7978845608}   : tensor<f16>
//     %tanh_in   = onnx.Mul(%cSqrt2pi, %inner)         : tensor<...xf16>
//     %tanh      = onnx.Tanh(%tanh_in)                 : tensor<...xf16>
//     %one       = onnx.Constant{value=1.0}            : tensor<f16>
//     %phi       = onnx.Sum(%one, %tanh)               : tensor<...xf16>
//     %half      = onnx.Constant{value=0.5}            : tensor<f16>
//     %half_x    = onnx.Mul(%half, %x)                 : tensor<...xf16>
//     %y         = onnx.Mul(%half_x, %phi)             : tensor<...xf16>
//
//   After (entire chain replaced; primitive constants left dangling for the
//   downstream DCE walk in OnnxToHip.cpp to clean up):
//     %y = onnx.Gelu(%x) {approximate = "tanh"} : tensor<...xf16>
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

  // Reciprocal of a constant. Exact-Gelu inlined chains in HF Qwen-VL-style
  // exports encode 1/sqrt(2) as `Reciprocal(Sqrt(Constant(2)))` rather than
  // a baked literal, so the chain to a constant goes through both Sqrt and
  // Reciprocal before reaching the leaf onnx.Constant.
  if (opName == "onnx.Reciprocal" && def->getNumOperands() >= 1) {
    auto inner = getScalarFloatConstant(def->getOperand(0));
    if (!inner || *inner == 0.0)
      return std::nullopt;
    return 1.0 / *inner;
  }

  // Div of two constants — sometimes 1/sqrt(2) ships as Div(1, Sqrt(2)).
  if (opName == "onnx.Div" && def->getNumOperands() == 2) {
    auto num = getScalarFloatConstant(def->getOperand(0));
    auto den = getScalarFloatConstant(def->getOperand(1));
    if (num && den && *den != 0.0)
      return *num / *den;
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

/// Vision variant: Gemma-3 SigLIP MLP inlines the FastGelu chain with three
/// substitutions vs. the canonical pattern matched by `InlinedFastGeluToGelu`:
///
///   1. `Pow(x, 3)`              -> `Mul(x, Mul(x, x))`   (cube via repeated
///   Mul)
///   2. `Sum(a, b)`              -> `Add(a, b)`           (Add instead of Sum)
///   3. `Mul(Mul(0.5, x), phi)`  -> `Mul(0.5, Mul(x, phi))` (constant on the
///                                                          outer Mul)
///
/// Walk (all op names use `onnx.Add`/`onnx.Mul`, NOT `onnx.Sum`):
///
///   xx       = Mul(x, x)                 // x^2
///   xxx      = Mul(x, xx)                // x^3 (operand order may vary)
///   scaled   = Mul(0.044715, xxx)
///   inner    = Add(x, scaled)
///   tanh_in  = Mul(sqrt(2/π), inner)
///   tanh     = Tanh(tanh_in)
///   phi      = Add(1.0, tanh)
///   x_phi    = Mul(x, phi)               // not yet * 0.5
///   y        = Mul(0.5, x_phi)
///
/// Implemented separately rather than polymorphically with the canonical
/// pattern so each match path stays readable and its diagnostics keep their
/// per-step granularity (the `tanh.in.mul_sqrt2pi :: onnx.Mul` line documented
/// in CLAUDE.md is the regression signal for that pattern; this pattern's
/// failure histogram doesn't conflate the two).
struct InlinedFastGeluVisionToGelu : public mlir::RewritePattern {
  InlinedFastGeluVisionToGelu(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Tanh", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *tanhOp,
                  mlir::PatternRewriter &rewriter) const override {
    if (tanhOp->getNumOperands() != 1 || tanhOp->getNumResults() != 1)
      return rewriter.notifyMatchFailure(tanhOp, "tanh.arity");

    // tanh_in = Mul(sqrt(2/π), inner)
    mlir::Operation *tanhInputMul = tanhOp->getOperand(0).getDefiningOp();
    if (!tanhInputMul)
      return rewriter.notifyMatchFailure(tanhOp, "tanh.in.defop_null");
    mlir::Value c2pi, inner;
    if (!isBinaryOpWithConst(tanhInputMul, "onnx.Mul",
                             /*sqrt(2/π)*/ 0.7978845608, c2pi, inner))
      return rewriter.notifyMatchFailure(tanhOp, [&](mlir::Diagnostic &d) {
        d << "tanh.in.mul_sqrt2pi.vision (observed: " << tanhInputMul->getName()
          << ")";
      });

    // inner = Add(x, scaled);  scaled = Mul(0.044715, xxx);  xxx = Mul(x,
    // Mul(x, x))
    mlir::Operation *innerOp = inner.getDefiningOp();
    if (!innerOp || innerOp->getName().getStringRef() != "onnx.Add" ||
        innerOp->getNumOperands() != 2)
      return rewriter.notifyMatchFailure(tanhOp, [&](mlir::Diagnostic &d) {
        d << "inner.add.vision (observed: "
          << (innerOp ? innerOp->getName().getStringRef() : "<null>") << ")";
      });

    // Probe each Add operand: the non-`x` operand is `scaled = Mul(0.044715,
    // xxx)`
    mlir::Value scaled, x;
    for (mlir::Value cand : innerOp->getOperands()) {
      mlir::Operation *def = cand.getDefiningOp();
      if (def && def->getName().getStringRef() == "onnx.Mul" &&
          def->getNumOperands() == 2) {
        auto [c, other] = matchCommutativeConst(def, 0.044715);
        if (c) {
          scaled = cand;
          x = (cand == innerOp->getOperand(0)) ? innerOp->getOperand(1)
                                               : innerOp->getOperand(0);
          break;
        }
      }
    }
    if (!scaled || !x)
      return rewriter.notifyMatchFailure(tanhOp, "scaled_pow.044715.vision");

    mlir::Operation *scaledOp = scaled.getDefiningOp();
    mlir::Value c044715, xxx;
    if (!isBinaryOpWithConst(scaledOp, "onnx.Mul", 0.044715, c044715, xxx))
      return rewriter.notifyMatchFailure(tanhOp, "scaled_pow.mul.vision");

    // xxx = Mul(x, xx);  xx = Mul(x, x). The outer Mul is commutative — the
    // canonical Gemma-3 export is Mul(x, xx) but accept either operand order.
    mlir::Operation *xxxOp = xxx.getDefiningOp();
    if (!xxxOp || xxxOp->getName().getStringRef() != "onnx.Mul" ||
        xxxOp->getNumOperands() != 2)
      return rewriter.notifyMatchFailure(tanhOp, "xxx.not_mul");
    mlir::Value xx;
    if (xxxOp->getOperand(0) == x)
      xx = xxxOp->getOperand(1);
    else if (xxxOp->getOperand(1) == x)
      xx = xxxOp->getOperand(0);
    else
      return rewriter.notifyMatchFailure(tanhOp, "xxx.x_not_in_operands");
    mlir::Operation *xxOp = xx.getDefiningOp();
    if (!xxOp || xxOp->getName().getStringRef() != "onnx.Mul" ||
        xxOp->getNumOperands() != 2)
      return rewriter.notifyMatchFailure(tanhOp, "xx.not_mul");
    if (xxOp->getOperand(0) != x || xxOp->getOperand(1) != x)
      return rewriter.notifyMatchFailure(tanhOp, "xx.not_x_squared");

    // Forward walk:
    //   phi   = Add(1.0, tanh)        (Add not Sum)
    //   x_phi = Mul(x, phi)
    //   y     = Mul(0.5, x_phi)
    mlir::Value tanhRes = tanhOp->getResult(0);
    if (!tanhRes.hasOneUse())
      return rewriter.notifyMatchFailure(tanhOp, "tanh.not_one_use");
    mlir::Operation *phiOp = *tanhRes.getUsers().begin();
    mlir::Value c1, tanhInPhi;
    if (!isBinaryOpWithConst(phiOp, "onnx.Add", 1.0, c1, tanhInPhi))
      return rewriter.notifyMatchFailure(tanhOp, [&](mlir::Diagnostic &d) {
        d << "phi.add_one.vision (observed: " << phiOp->getName() << ")";
      });
    if (tanhInPhi != tanhRes)
      return rewriter.notifyMatchFailure(tanhOp, "phi.tanh_mismatch.vision");

    if (!phiOp->getResult(0).hasOneUse())
      return rewriter.notifyMatchFailure(tanhOp, "phi.not_one_use");
    mlir::Operation *xPhiOp = *phiOp->getResult(0).getUsers().begin();
    if (xPhiOp->getName().getStringRef() != "onnx.Mul" ||
        xPhiOp->getNumOperands() != 2)
      return rewriter.notifyMatchFailure(tanhOp, "x_phi.not_mul.vision");
    // The non-phi operand must be `x`.
    mlir::Value xPhiOther;
    for (mlir::Value cand : xPhiOp->getOperands()) {
      if (cand != phiOp->getResult(0)) {
        xPhiOther = cand;
        break;
      }
    }
    if (xPhiOther != x)
      return rewriter.notifyMatchFailure(tanhOp, "x_phi.x_mismatch.vision");

    if (!xPhiOp->getResult(0).hasOneUse())
      return rewriter.notifyMatchFailure(tanhOp, "x_phi.not_one_use");
    mlir::Operation *finalMulOp = *xPhiOp->getResult(0).getUsers().begin();
    mlir::Value cHalf, xPhiInFinal;
    if (!isBinaryOpWithConst(finalMulOp, "onnx.Mul", 0.5, cHalf, xPhiInFinal))
      return rewriter.notifyMatchFailure(tanhOp, [&](mlir::Diagnostic &d) {
        d << "final.mul_half.vision (observed: " << finalMulOp->getName()
          << ")";
      });
    if (xPhiInFinal != xPhiOp->getResult(0))
      return rewriter.notifyMatchFailure(tanhOp, "final.xphi_mismatch.vision");

    // ── all matched; rewrite final Mul to onnx.Gelu(x, "tanh") ───────────
    mlir::Location loc = finalMulOp->getLoc();
    mlir::OperationState state(loc, "onnx.Gelu");
    state.addOperands(x);
    // Gelu output type == input type (same-shape unary). For an unranked
    // input (rare on HF exports) the final Mul has the same shape as x by
    // construction; any residual dynamic dims are resolved post-conversion
    // by the HIP-dialect `--hip-infer-shapes` pass.
    mlir::Type geluResultType = mlir::isa<mlir::RankedTensorType>(x.getType())
                                    ? x.getType()
                                    : finalMulOp->getResult(0).getType();
    state.addTypes(geluResultType);
    state.addAttribute("approximate", rewriter.getStringAttr("tanh"));
    if (auto outputs =
            finalMulOp->getAttrOfType<mlir::ArrayAttr>("node.outputs"))
      state.addAttribute("node.outputs", outputs);
    if (auto nodeName =
            finalMulOp->getAttrOfType<mlir::StringAttr>("onnx_node_name"))
      state.addAttribute("onnx_node_name", nodeName);
    rewriter.setInsertionPoint(finalMulOp);
    mlir::Operation *geluOp = rewriter.create(state);

    rewriter.replaceOp(finalMulOp, geluOp->getResult(0));
    auto eraseIfDead = [&rewriter](mlir::Operation *op) {
      if (op && op->use_empty())
        rewriter.eraseOp(op);
    };
    eraseIfDead(xPhiOp);
    eraseIfDead(phiOp);
    eraseIfDead(tanhOp);
    eraseIfDead(tanhInputMul);
    eraseIfDead(innerOp);
    eraseIfDead(scaledOp);
    eraseIfDead(xxxOp);
    eraseIfDead(xxOp);

    LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] fused vision-variant chain at "
                            << loc << "\n");
    ++NumFastGeluFused;
    return mlir::success();
  }
};

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
    // Gelu output type == input type (same-shape unary). For an unranked
    // input (rare on HF exports) the final Mul has the same shape as x by
    // construction; any residual dynamic dims are resolved post-conversion
    // by the HIP-dialect `--hip-infer-shapes` pass.
    mlir::Type geluResultType = mlir::isa<mlir::RankedTensorType>(x.getType())
                                    ? x.getType()
                                    : finalMulOp->getResult(0).getType();
    state.addTypes(geluResultType);
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

/// Exact-Gelu (approximate="none") inlined chain — Erf-based.
///
/// Some ORT exports inline `Gelu(approximate="none")` instead of the Tanh
/// approximation. Canonical inlined form:
///
///   scaled = Mul(x, 1/sqrt(2))      // 0.70710678...
///   erf    = Erf(scaled)
///   phi    = Sum(1.0, erf)          // or Add(1.0, erf); commutative
///   half_x = Mul(0.5, x)            // or Mul(x, 0.5); commutative
///   y      = Mul(half_x, phi)       // or Mul(phi, half_x); commutative
///
/// Rewritten to `onnx.Gelu(x) {approximate = "none"}` which is then handled
/// by the existing `GeluToHip` converter (lowers to `hip.gelu` with the
/// non-tanh runtime kernel).
///
/// Anchored on `onnx.Erf` to keep the matcher cheap — Erf is rare in non-
/// Gelu graphs. The pattern walks forward from Erf to its Sum/Add consumer
/// and then to the final Mul, checking the half-x branch resolves to the
/// same SSA `x` as the `Mul(x, 1/sqrt(2))` operand. Multi-use Erf or Sum
/// results break the pattern (we'd then duplicate the chain when rewriting
/// to Gelu — fine for correctness but pessimises the IR; leave it for now).
///
/// Before:
///   %scaled = onnx.Mul(%x, %const_invsqrt2)
///   %erf    = onnx.Erf(%scaled)
///   %phi    = onnx.Sum(%const_one, %erf)
///   %hx     = onnx.Mul(%const_half, %x)
///   %y      = onnx.Mul(%hx, %phi) : tensor<?xf16>
///
/// After:
///   %y = onnx.Gelu(%x) {approximate = "none"} : tensor<?xf16>
struct InlinedExactGeluToGelu : public mlir::RewritePattern {
  InlinedExactGeluToGelu(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Erf", /*benefit=*/2, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *erfOp,
                  mlir::PatternRewriter &rewriter) const override {
    if (erfOp->getNumOperands() != 1 || erfOp->getNumResults() != 1)
      return rewriter.notifyMatchFailure(erfOp, "erf.arity");

    // scaled = Mul(x, 1/sqrt(2))
    mlir::Operation *scaledOp = erfOp->getOperand(0).getDefiningOp();
    mlir::Value invSqrt2, x;
    if (!isBinaryOpWithConst(scaledOp, "onnx.Mul",
                             /*1/sqrt(2)*/ 0.7071067811865, invSqrt2, x))
      return rewriter.notifyMatchFailure(erfOp, "erf.in.mul_invsqrt2");

    // erf has exactly one user, which must be the Sum/Add producing phi.
    if (!erfOp->getResult(0).hasOneUse())
      return rewriter.notifyMatchFailure(erfOp, "erf.multi_use");
    mlir::Operation *phiOp = *erfOp->getResult(0).getUsers().begin();
    if (!phiOp ||
        (phiOp->getName().getStringRef() != "onnx.Sum" &&
         phiOp->getName().getStringRef() != "onnx.Add") ||
        phiOp->getNumOperands() != 2)
      return rewriter.notifyMatchFailure(erfOp, [&](mlir::Diagnostic &d) {
        d << "phi.sum_or_add (observed: "
          << (phiOp ? phiOp->getName().getStringRef() : "<null>") << ")";
      });
    mlir::Value oneConst, erfVal;
    // The non-erf operand must be the constant 1.0.
    mlir::Value lhs = phiOp->getOperand(0), rhs = phiOp->getOperand(1);
    if (lhs == erfOp->getResult(0) && isScalarFloatNear(rhs, 1.0)) {
      erfVal = lhs;
      oneConst = rhs;
    } else if (rhs == erfOp->getResult(0) && isScalarFloatNear(lhs, 1.0)) {
      erfVal = rhs;
      oneConst = lhs;
    } else {
      return rewriter.notifyMatchFailure(erfOp, "phi.no_one_constant");
    }

    // phi has exactly one user, the final Mul.
    if (!phiOp->getResult(0).hasOneUse())
      return rewriter.notifyMatchFailure(erfOp, "phi.multi_use");
    mlir::Operation *finalMul = *phiOp->getResult(0).getUsers().begin();
    if (!finalMul || finalMul->getName().getStringRef() != "onnx.Mul" ||
        finalMul->getNumOperands() != 2)
      return rewriter.notifyMatchFailure(erfOp, "final_mul.shape");

    // Other operand of finalMul must be a `Mul(0.5, x)` or `Mul(x, 0.5)`
    // where x matches the scaled-input x.
    mlir::Value otherOperand = (finalMul->getOperand(0) == phiOp->getResult(0))
                                   ? finalMul->getOperand(1)
                                   : finalMul->getOperand(0);
    if (otherOperand == phiOp->getResult(0))
      return rewriter.notifyMatchFailure(erfOp, "final_mul.no_phi_operand");
    mlir::Operation *halfMulOp = otherOperand.getDefiningOp();
    mlir::Value halfConst, halfX;
    if (!isBinaryOpWithConst(halfMulOp, "onnx.Mul", /*0.5*/ 0.5, halfConst,
                             halfX))
      return rewriter.notifyMatchFailure(erfOp, "half_mul.no_half_constant");
    if (halfX != x)
      return rewriter.notifyMatchFailure(erfOp, "half_mul.x_mismatch_scaled_x");

    // All checks passed — replace finalMul with onnx.Gelu(x, "none").
    mlir::Type resultTy = finalMul->getResult(0).getType();
    mlir::OperationState gelu(finalMul->getLoc(), "onnx.Gelu");
    gelu.addOperands(x);
    gelu.addTypes(resultTy);
    gelu.addAttribute("approximate", rewriter.getStringAttr("none"));
    mlir::Value y = rewriter.create(gelu)->getResult(0);
    rewriter.replaceOp(finalMul, y);
    // Erase the now-dead chain explicitly in reverse-topological order. The
    // module-level DCE walk in OnnxToHip is a backstop but doesn't always
    // catch arith subtrees emitted by other patterns once their consumers
    // get rewritten — explicit erasure here mirrors the canonical FastGelu
    // pattern's discipline and keeps the IR clean for the structural
    // "no onnx.* should survive" check.
    if (halfMulOp && halfMulOp->use_empty())
      rewriter.eraseOp(halfMulOp);
    if (phiOp && phiOp->use_empty())
      rewriter.eraseOp(phiOp);
    if (erfOp && erfOp->use_empty())
      rewriter.eraseOp(erfOp);
    if (scaledOp && scaledOp->use_empty())
      rewriter.eraseOp(scaledOp);
    ++NumFastGeluFused;
    LLVM_DEBUG(llvm::dbgs()
               << "[" DEBUG_TYPE "] fused exact (Erf-based) Gelu at "
               << finalMul->getLoc() << "\n");
    return mlir::success();
  }
};

} // namespace

void populateFastGeluFusionPatterns(mlir::RewritePatternSet &patterns,
                                    mlir::MLIRContext *ctx) {
  patterns.add<InlinedFastGeluToGelu, InlinedFastGeluVisionToGelu,
               InlinedExactGeluToGelu>(ctx);
}

} // namespace hip
} // namespace mlir
