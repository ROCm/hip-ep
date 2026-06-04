/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

#include <cmath>
#include <optional>

namespace mlir {
namespace hip {
namespace {

/// Maximum positive exponent we unroll into a chain of multiplications. Real
/// models only ever use tiny integer powers (2 for variance, 3 for the
/// tanh-Gelu chain), so a small cap keeps the IR bounded while covering every
/// case seen in practice.
static constexpr int64_t kMaxPowUnroll = 16;

/// Extract a scalar (single-element / splat) float/integer constant value as
/// a double, peeking through value-preserving cast chains to find either an
/// `onnx.Constant` (the production form, before `lowerOnnxConstants` runs) or
/// an `arith.constant` (test fixtures that hand-write the exponent). Returns
/// std::nullopt when no such constant feeds \p v.
///
/// The exponent is frequently wrapped in a Cast: ORT exports emit the literal
/// as f32 and Cast/CastLike it to the activation dtype, e.g.
/// `Pow(x_f16, Cast(2.0_f32 -> f16))` (notably the inlined-Gelu chain in
/// Gemma-3, and SAM's LayerNorm2d variance `Pow(x, 2)`). fp->fp rounding is
/// immaterial for the integer/half-integer exponents we decompose
/// (2, 3, 0.5, -1, ...).
///
/// Why this peek runs PRE-lowering: with externalization enabled, every
/// `onnx.Constant` — including 1-element scalars — is replaced by
/// `bufferization.to_tensor(memref.get_global)` whose value lives in the
/// constants sidecar, NOT in IR. A post-lowering matcher cannot recover the
/// scalar value from that form. Running before `lowerOnnxConstants` keeps the
/// `onnx.Constant` (and any wrapping Cast) in IR, so the value is readable.
static std::optional<double> getScalarConstantValue(mlir::Value v) {
  // Peek through value-preserving casts back to the underlying constant. At
  // pre-lowering time `onnx.Cast`/`onnx.CastLike` have not yet been converted
  // to `hip.cast`, so this short chain covers the production case.
  while (mlir::Operation *def = v.getDefiningOp()) {
    llvm::StringRef name = def->getName().getStringRef();
    if ((name == "onnx.Cast" || name == "onnx.CastLike") &&
        def->getNumOperands() >= 1) {
      v = def->getOperand(0);
      continue;
    }
    break;
  }

  mlir::Operation *def = v.getDefiningOp();
  if (!def)
    return std::nullopt;

  mlir::DenseElementsAttr dense;
  if (def->getName().getStringRef() == "onnx.Constant") {
    dense =
        mlir::dyn_cast_or_null<mlir::DenseElementsAttr>(def->getAttr("value"));
  } else if (auto cst = mlir::dyn_cast<mlir::arith::ConstantOp>(def)) {
    dense = mlir::dyn_cast<mlir::DenseElementsAttr>(cst.getValue());
  }
  if (!dense)
    return std::nullopt;
  if (!dense.isSplat() && dense.getNumElements() != 1)
    return std::nullopt;

  mlir::Type et = dense.getElementType();
  if (et.isF32())
    return static_cast<double>(*dense.getValues<float>().begin());
  if (et.isF64())
    return *dense.getValues<double>().begin();
  if (et.isF16() || et.isBF16())
    return (*dense.getValues<llvm::APFloat>().begin()).convertToDouble();
  // Some ORT exports ship integer exponents (e.g. `Pow(x, 3 : i64)`).
  if (et.isIntOrIndex())
    return static_cast<double>(
        (*dense.getValues<llvm::APInt>().begin()).getSExtValue());
  return std::nullopt;
}

/// onnx.Reciprocal -> hip.reciprocal
/// Converts ONNX reciprocal (y = 1/x, full signed IEEE domain) to HIP.
/// Runtime uses a HIP elementwise kernel (not MIOpen POWER activation).
struct ReciprocalToHip : public mlir::RewritePattern {
  ReciprocalToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Reciprocal", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override;
};

/// onnx.Sqrt -> hip.sqrt
/// ONNX element-wise sqrt; lowered to @wrap_power(0, 1, 0.5) and executed
/// with hip_elementwise_sqrt at runtime (not MIOpen POWER).
struct SqrtToHip : public mlir::RewritePattern {
  SqrtToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Sqrt", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override;
};

mlir::LogicalResult
ReciprocalToHip::matchAndRewrite(mlir::Operation *op,
                                 mlir::PatternRewriter &rewriter) const {
  auto ctxOrFailure = getContextArg(op, rewriter);
  if (mlir::failed(ctxOrFailure))
    return mlir::failure();
  mlir::Value context = *ctxOrFailure;

  mlir::Location loc = op->getLoc();
  mlir::Value input = op->getOperand(0);
  if (!mlir::isa<mlir::RankedTensorType>(input.getType()))
    return rewriter.notifyMatchFailure(
        op, "onnx.Reciprocal lowering expects a ranked tensor input");
  if (!mlir::isa<mlir::RankedTensorType>(op->getResult(0).getType()))
    return rewriter.notifyMatchFailure(
        op, "onnx.Reciprocal lowering expects a ranked tensor result");
  auto resultType =
      mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());
  mlir::Value init = createEmptyTensor(rewriter, loc, resultType, input);
  auto hipOp = mlir::hip::ReciprocalOp::create(rewriter, loc, resultType,
                                               context, input, init);
  rewriter.replaceOp(op, hipOp->getResult(0));
  return mlir::success();
}

mlir::LogicalResult
SqrtToHip::matchAndRewrite(mlir::Operation *op,
                           mlir::PatternRewriter &rewriter) const {
  auto ctxOrFailure = getContextArg(op, rewriter);
  if (mlir::failed(ctxOrFailure))
    return mlir::failure();
  mlir::Value context = *ctxOrFailure;

  mlir::Location loc = op->getLoc();
  mlir::Value input = op->getOperand(0);
  if (!mlir::isa<mlir::RankedTensorType>(input.getType()))
    return rewriter.notifyMatchFailure(
        op, "onnx.Sqrt lowering expects a ranked tensor input");
  if (!mlir::isa<mlir::RankedTensorType>(op->getResult(0).getType()))
    return rewriter.notifyMatchFailure(
        op, "onnx.Sqrt lowering expects a ranked tensor result");
  auto resultType =
      mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());
  mlir::Value init = createEmptyTensor(rewriter, loc, resultType, input);
  auto hipOp = mlir::hip::SqrtOp::create(rewriter, loc, resultType, context,
                                         input, init);
  rewriter.replaceOp(op, hipOp->getResult(0));
  return mlir::success();
}

/// onnx.Neg -> hip.neg
/// Negation: y = -x. Lowered via wrap_power(alpha=0, beta=-1, gamma=1).
struct NegToHip : public mlir::RewritePattern {
  NegToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Neg", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto ctxOrFailure = getContextArg(op, rewriter);
    if (mlir::failed(ctxOrFailure))
      return mlir::failure();
    mlir::Value context = *ctxOrFailure;

    mlir::Location loc = op->getLoc();
    mlir::Value input = op->getOperand(0);
    if (!mlir::isa<mlir::RankedTensorType>(input.getType()))
      return rewriter.notifyMatchFailure(
          op, "onnx.Neg lowering expects a ranked tensor input");
    auto resultType =
        mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());
    mlir::Value init = createEmptyTensor(rewriter, loc, resultType, input);
    auto hipOp = mlir::hip::NegOp::create(rewriter, loc, resultType, context,
                                          input, init);
    rewriter.replaceOp(op, hipOp->getResult(0));
    return mlir::success();
  }
};

/// onnx.Pow(X, Y) -> decomposition into ONNX primitives, for a constant
/// scalar exponent Y. Runs as a PRE-lowering rewrite (before
/// `lowerOnnxConstants` and `convertComputeOps`) so the exponent is still
/// readable as `onnx.Constant` (or an `onnx.Cast`/`onnx.CastLike` chain to
/// one). With externalization enabled in production every `onnx.Constant`
/// — including 1-element scalars — gets replaced by
/// `bufferization.to_tensor(memref.get_global)` whose value is buried in the
/// constants sidecar; matching post-lowering would silently miss every Pow.
///
/// In practice the exponent is always a constant scalar (variance
/// `Pow(x, 2)` in RMS/LayerNorm and SAM's LayerNorm2d, `Pow(x, 3)` in the
/// tanh-Gelu chain). We decompose into existing ONNX ops, which are then
/// lowered by their own ONNX→HIP converters in `convertComputeOps`:
///   * e == 1            -> X (identity)
///   * integer e >= 2    -> chained `onnx.Mul` (x*x*...). Correct for ALL
///                          input signs, unlike MIOpen POWER `(a+bx)^g`
///                          which treats a negative base as non-negative —
///                          that path would silently miscompute the
///                          dominant `Pow(x, 2)` case whenever x < 0.
///   * e == 0.5          -> `onnx.Sqrt`
///   * e == -1           -> `onnx.Reciprocal`
///   * e == -0.5         -> `onnx.Reciprocal(onnx.Sqrt(x))`
///   * negative integer  -> `onnx.Reciprocal(x^|e|)`
/// Anything else (non-constant exponent, or a fractional value we cannot
/// express losslessly) is left unmatched, so a genuinely unsupported Pow
/// surfaces downstream (bufferization error) instead of being silently wrong.
///
/// Before (production, externalized constant + Cast wrap):
///   %c   = "onnx.Constant"() {value = dense<2.0> : tensor<f32>}
///        : () -> tensor<f32>
///   %ec  = "onnx.Cast"(%c) {to = 10 : si64}
///        : (tensor<f32>) -> tensor<f16>
///   %y   = "onnx.Pow"(%x, %ec)
///        : (tensor<1x64x128x128xf16>, tensor<f16>)
///        -> tensor<1x64x128x128xf16>
/// After:
///   %y   = "onnx.Mul"(%x, %x)
///        : (tensor<1x64x128x128xf16>, tensor<1x64x128x128xf16>)
///        -> tensor<1x64x128x128xf16>
struct PowDecompose : public mlir::RewritePattern {
  PowDecompose(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Pow", /*benefit=*/1, ctx) {}

  // Build an unregistered onnx.* binary/unary elementwise op. The result type
  // is taken from the original Pow result (the scalar exponent broadcasts
  // away, so the output shape is identical to the base).
  static mlir::Value createOnnxOp(mlir::PatternRewriter &rewriter,
                                  mlir::Location loc, llvm::StringRef opName,
                                  llvm::ArrayRef<mlir::Value> operands,
                                  mlir::Type resultType) {
    mlir::OperationState state(loc, opName);
    state.addOperands(operands);
    state.addTypes(resultType);
    return rewriter.create(state)->getResult(0);
  }

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::Location loc = op->getLoc();
    mlir::Value base = op->getOperand(0);
    mlir::Value expVal = op->getOperand(1);

    auto resultType =
        mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(
          op, "onnx.Pow decompose expects a ranked tensor result");

    std::optional<double> expOpt = getScalarConstantValue(expVal);
    if (!expOpt)
      return rewriter.notifyMatchFailure(
          op, "onnx.Pow exponent is not a constant scalar");
    double e = *expOpt;

    if (e == 1.0) {
      rewriter.replaceOp(op, base);
      return mlir::success();
    }

    auto buildIntPow = [&](int64_t n) -> mlir::Value {
      mlir::Value acc = base;
      for (int64_t i : llvm::seq<int64_t>(1, n))
        acc = createOnnxOp(rewriter, loc, "onnx.Mul", {acc, base}, resultType);
      return acc;
    };

    bool isInt = std::isfinite(e) && (e == std::floor(e));
    int64_t n = static_cast<int64_t>(e);

    if (isInt && n >= 2 && n <= kMaxPowUnroll) {
      rewriter.replaceOp(op, buildIntPow(n));
      return mlir::success();
    }
    if (e == 0.5) {
      rewriter.replaceOp(
          op, createOnnxOp(rewriter, loc, "onnx.Sqrt", {base}, resultType));
      return mlir::success();
    }
    if (e == -1.0) {
      rewriter.replaceOp(op, createOnnxOp(rewriter, loc, "onnx.Reciprocal",
                                          {base}, resultType));
      return mlir::success();
    }
    if (e == -0.5) {
      mlir::Value sqrt =
          createOnnxOp(rewriter, loc, "onnx.Sqrt", {base}, resultType);
      rewriter.replaceOp(op, createOnnxOp(rewriter, loc, "onnx.Reciprocal",
                                          {sqrt}, resultType));
      return mlir::success();
    }
    if (isInt && n <= -2 && n >= -kMaxPowUnroll) {
      mlir::Value posPow = buildIntPow(-n);
      rewriter.replaceOp(op, createOnnxOp(rewriter, loc, "onnx.Reciprocal",
                                          {posPow}, resultType));
      return mlir::success();
    }

    return rewriter.notifyMatchFailure(
        op, "onnx.Pow exponent not representable via mul/sqrt/reciprocal");
  }
};

} // namespace

void populatePowerConversionPatterns(RewritePatternSet &patterns,
                                     MLIRContext *ctx) {
  patterns.add<ReciprocalToHip, SqrtToHip, NegToHip>(ctx);
}

void populatePowDecompositionPatterns(RewritePatternSet &patterns,
                                      MLIRContext *ctx) {
  patterns.add<PowDecompose>(ctx);
}

} // namespace hip
} // namespace mlir
