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
// Operates as a function-level walk run inside ConvertOnnxToHipPass, BEFORE
// `lowerOnnxConstants` so the constant values are still inline `onnx.Constant`
// attributes (post-lowering, constants become arith.constant /
// memref.get_global, defeating value-based matching). The intermediate ops
// become dead and are removed by the existing unused-onnx-op DCE in
// `ConvertOnnxToHipPass::runOnOperation`.
//
//===----------------------------------------------------------------------===//

#include "OnnxToHipUtils.h"

#include "llvm/ADT/StringMap.h"

#include <cmath>

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

/// One-line description of why a `tryFuseFastGeluAt` call gave up. Used only
/// for diagnostics when fused < total.
struct FuseAttemptResult {
  bool success = false;
  const char *failedStep = "ok";
  std::string
      observed; // op name we actually saw at the failed step (when useful)
};

static FuseAttemptResult fail(const char *step, llvm::StringRef obs = "") {
  FuseAttemptResult r;
  r.success = false;
  r.failedStep = step;
  r.observed = obs.str();
  return r;
}

/// Try to recognise the FastGelu identity rooted at `tanhOp`. On success,
/// replaces the final output Mul with `onnx.Gelu(x){approximate="tanh"}`.
static FuseAttemptResult tryFuseFastGeluAt(mlir::Operation *tanhOp,
                                           mlir::OpBuilder &builder) {
  if (tanhOp->getNumOperands() != 1 || tanhOp->getNumResults() != 1)
    return fail("tanh.arity");

  // ── back-walk ──────────────────────────────────────────────────────────
  // tanh_in = Mul(sqrt(2/π), inner)
  mlir::Operation *tanhInputMul = tanhOp->getOperand(0).getDefiningOp();
  if (!tanhInputMul)
    return fail("tanh.in.defop_null");
  llvm::StringRef tanhInName = tanhInputMul->getName().getStringRef();
  mlir::Value c2pi, innerSum;
  if (!isBinaryOpWithConst(tanhInputMul, "onnx.Mul",
                           /*sqrt(2/π)*/ 0.7978845608, c2pi, innerSum))
    return fail("tanh.in.mul_sqrt2pi", tanhInName);

  // inner = Sum(x, scaled_pow)  (commutative; scaled_pow is the Mul-of-Pow leg)
  mlir::Operation *innerSumOp = innerSum.getDefiningOp();
  if (!innerSumOp)
    return fail("inner.defop_null");
  llvm::StringRef innerName = innerSumOp->getName().getStringRef();
  if (innerName != "onnx.Sum" || innerSumOp->getNumOperands() != 2)
    return fail("inner.sum", innerName);

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
    return fail("scaled_pow.044715");

  // scaled_pow = Mul(0.044715, pow); pow = Pow(x, 3)
  mlir::Operation *scaledPowOp = scaledPow.getDefiningOp();
  mlir::Value c044715, powVal;
  if (!isBinaryOpWithConst(scaledPowOp, "onnx.Mul", 0.044715, c044715, powVal))
    return fail("scaled_pow.mul");

  mlir::Operation *powOp = powVal.getDefiningOp();
  if (!powOp)
    return fail("pow.defop_null");
  llvm::StringRef powName = powOp->getName().getStringRef();
  if (powName != "onnx.Pow" || powOp->getNumOperands() != 2)
    return fail("pow.op", powName);
  if (powOp->getOperand(0) != x)
    return fail("pow.base_neq_x");
  if (!isScalarFloatNear(powOp->getOperand(1), 3.0)) {
    auto v = getScalarFloatConstant(powOp->getOperand(1));
    return fail("pow.exponent_not_3", v ? std::to_string(*v) : "<not_const>");
  }

  // ── forward-walk ───────────────────────────────────────────────────────
  // tanh.result must feed exactly one Sum: phi = Sum(1, tanh)
  mlir::Value tanhRes = tanhOp->getResult(0);
  if (!tanhRes.hasOneUse())
    return fail("tanh.not_one_use");
  mlir::Operation *phiOp = *tanhRes.getUsers().begin();
  llvm::StringRef phiName = phiOp->getName().getStringRef();
  mlir::Value c1, tanhInPhi;
  if (!isBinaryOpWithConst(phiOp, "onnx.Sum", 1.0, c1, tanhInPhi))
    return fail("phi.sum_one", phiName);
  if (tanhInPhi != tanhRes)
    return fail("phi.tanh_mismatch");

  // phi must feed exactly one Mul: y = Mul(half_x, phi)
  if (!phiOp->getResult(0).hasOneUse())
    return fail("phi.not_one_use");
  mlir::Operation *finalMulOp = *phiOp->getResult(0).getUsers().begin();
  llvm::StringRef finalName = finalMulOp->getName().getStringRef();
  if (finalName != "onnx.Mul" || finalMulOp->getNumOperands() != 2)
    return fail("final.mul", finalName);

  // The non-phi operand of finalMul is half_x = Mul(0.5, x).
  mlir::Value halfX;
  for (mlir::Value cand : finalMulOp->getOperands()) {
    if (cand != phiOp->getResult(0)) {
      halfX = cand;
      break;
    }
  }
  if (!halfX)
    return fail("half_x.missing");
  mlir::Operation *halfXOp = halfX.getDefiningOp();
  mlir::Value cHalf, xInHalf;
  if (!isBinaryOpWithConst(halfXOp, "onnx.Mul", 0.5, cHalf, xInHalf))
    return fail("half_x.mul_half");
  if (xInHalf != x)
    return fail("half_x.x_mismatch");

  // ── all matched; rewrite final Mul to onnx.Gelu(x, "tanh") ────────────
  builder.setInsertionPoint(finalMulOp);
  mlir::OperationState state(finalMulOp->getLoc(), "onnx.Gelu");
  state.addOperands(x);
  state.addTypes(finalMulOp->getResult(0).getType());
  state.addAttribute("approximate", builder.getStringAttr("tanh"));
  // Preserve the original output name attribute so downstream IR dumps and
  // metadata stay readable.
  if (auto outputs = finalMulOp->getAttrOfType<mlir::ArrayAttr>("node.outputs"))
    state.addAttribute("node.outputs", outputs);
  if (auto nodeName =
          finalMulOp->getAttrOfType<mlir::StringAttr>("onnx_node_name"))
    state.addAttribute("onnx_node_name", nodeName);
  mlir::Operation *geluOp = builder.create(state);
  finalMulOp->getResult(0).replaceAllUsesWith(geluOp->getResult(0));

  // Erase the matched chain in reverse-topological order so each op is
  // `use_empty` when we erase it. Avoids relying on a cascade of post-pass
  // DCE iterations to clean up the dead chain (relevant when the chain
  // mixes onnx.* and other dialects after later conversions). Constants
  // shared across many Gelu instances become dead naturally only when the
  // last instance is fused; any leftover dead onnx.* op is picked up by
  // the per-pass `onnx.*` use-empty DCE walk in ConvertOnnxToHipPass.
  auto eraseIfDead = [](mlir::Operation *op) {
    if (op && op->use_empty())
      op->erase();
  };
  finalMulOp->erase();
  eraseIfDead(halfXOp);
  eraseIfDead(phiOp);
  eraseIfDead(tanhOp);
  eraseIfDead(tanhInputMul);
  eraseIfDead(innerSumOp);
  eraseIfDead(scaledPowOp);
  eraseIfDead(powOp);

  FuseAttemptResult ok;
  ok.success = true;
  return ok;
}

} // namespace

mlir::LogicalResult fuseInlinedFastGelu(mlir::func::FuncOp funcOp) {
  // Snapshot Tanh ops first (don't mutate during the walk).
  llvm::SmallVector<mlir::Operation *> tanhOps;
  funcOp.walk([&](mlir::Operation *op) {
    if (op->getName().getStringRef() == "onnx.Tanh")
      tanhOps.push_back(op);
  });

  mlir::OpBuilder builder(funcOp.getContext());
  int fused = 0;
  llvm::SmallVector<FuseAttemptResult> failures;
  for (mlir::Operation *tanh : tanhOps) {
    // The op may have been erased by an earlier fusion in this loop (for
    // structurally overlapping chains — not expected for FastGelu but cheap
    // to guard against).
    if (tanh->use_empty() && tanh->getNumResults() > 0)
      continue;
    auto r = tryFuseFastGeluAt(tanh, builder);
    if (r.success)
      ++fused;
    else
      failures.push_back(std::move(r));
  }

  // Diagnostic — only when at least one Tanh failed to fuse. Quiet on the
  // expected "0 of 0" tiny-shape-graph compile pass and on full-success.
  // The histogram tells us at a glance whether all 34 layer Tanhs failed at
  // the same step (structural mismatch at a known node) or fanned out across
  // multiple steps (e.g. some constants encoded differently). Remove this
  // block when the matcher is known to be complete for all production
  // models in CI.
  if (!failures.empty()) {
    llvm::StringMap<int> stepHist;
    llvm::StringMap<int> detailHist;
    for (auto &f : failures) {
      ++stepHist[f.failedStep];
      if (!f.observed.empty()) {
        std::string key = std::string(f.failedStep) + " :: " + f.observed;
        ++detailHist[key];
      }
    }
    llvm::errs() << "[FastGeluFusion] @" << funcOp.getName() << ": fused "
                 << fused << " of " << tanhOps.size()
                 << " Tanh chains; failure step histogram:\n";
    for (auto &kv : stepHist)
      llvm::errs() << "  " << kv.second << "x  " << kv.first() << "\n";
    if (!detailHist.empty()) {
      llvm::errs() << "  detail (step :: observed):\n";
      for (auto &kv : detailHist)
        llvm::errs() << "    " << kv.second << "x  " << kv.first() << "\n";
    }
  }
  return mlir::success();
}

} // namespace hip
} // namespace mlir
