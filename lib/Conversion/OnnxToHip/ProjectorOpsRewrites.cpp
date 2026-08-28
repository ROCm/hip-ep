/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- ProjectorOpsRewrites.cpp - Pre-lowering rewrites for projector ops --==//
//
// Some vision-encoder projector blocks (e.g. a `multi_modal_projector`) use
// two ops here that have no direct HIP converters: `onnx.Pow` and
// `onnx.AveragePool`. Rather than introduce new HIP dialect ops + runtime
// kernels for each, we rewrite them in pre-lowering to compositions of ops
// that ALREADY have converters:
//
//   1. `Pow(x, c)` for small integer constants c -> chain of `Mul`s
//      (c==2 -> Mul(x,x); c==3 -> Mul(x, Mul(x,x)); etc.). Sqrt special-cased
//      for c==0.5.
//   2. `AveragePool(NCHW, kernel=K, stride=K, no pad)` -> Reshape ->
//      Transpose -> ReduceMean. The 4-D NCHW input is first reshaped to
//      6-D `[B, C, H/K, K, W/K, K]`, then transposed with perm
//      `[0, 1, 2, 4, 3, 5]` to `[B, C, H/K, W/K, K, K]` so the K-K patch
//      axes end up TRAILING, then ReduceMean over the two trailing K dims
//      collapses them with `keepdims=0`. The transpose is required because
//      `hip_reduce_sum` only handles reductions over a contiguous suffix
//      of the input (it sums consecutive `reduce_size`-blocks); skipping
//      it silently misroutes the reduction to dims [4,5] (the K-stride
//      and innermost-K dims), producing wrong AvgPool output values that
//      cascade into NaN downstream. This is mathematically equivalent to
//      non-overlapping AveragePool with `K == stride`. Coverage is
//      intentionally narrow — typical site is a Gemma-3-style projector
//      (`kernel=4, stride=4` on a `[B, 1152, 64, 64]` tensor).
//
//      Overlap, padding, dynamic spatial dims, and other shapes outside
//      this narrow case are NOT handled here — `notifyMatchFailure` leaves
//      the original `onnx.AveragePool` alive for `PoolConversion.cpp`
//      (`hip.pool`, pool_mode = AVERAGE) in the subsequent compute-op
//      conversion round. Benefit = 2 here vs benefit = 1 on `PoolToHip`
//      ensures the fast decomposition wins only when its preconditions
//      are satisfied.
//
// These rewrites run in the pre-lowering block of `ConvertOnnxToHipPass`,
// alongside `FastGeluFusionPatterns` in a fixed-point round loop (an emitted
// op from one round may be the root of a rewrite in the next). These
// ONNX-rooted patterns run before lowerOnnxConstants so generic
// `onnx.Constant` producers and scalar Pow values are directly matchable;
// standalone externalization happens after compute conversion.
//
// Note: `onnx.ReduceMean` is NOT rewritten here. It lowers directly to the
// first-class `hip.reduce_mean` op (ReduceMeanConversion.cpp), whose runtime
// kernel divides by the reduced-element count in-kernel -- so it needs no
// compile-time-static reduce dim and no ONNX-level shape refinement. The
// AveragePool decomposition below still emits an `onnx.ReduceMean`, which
// then takes that direct path.
//
// Failure to rewrite returns `notifyMatchFailure` (silent at this layer).
// For Pow that leaves a dead-end `onnx.*` op. For AveragePool the surviving
// op is converted to `hip.pool` by `PoolConversion.cpp` unless it falls
// outside both the decomposition and the pool runtime's coverage.
//
//===----------------------------------------------------------------------===//

#include "OnnxToHipUtils.h"

#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Debug.h"

#include <cmath>

#define DEBUG_TYPE "projector-ops-rewrites"

STATISTIC(NumPowRewrites, "onnx.Pow rewrites into Mul chains / Sqrt");
STATISTIC(NumAveragePoolRewrites,
          "onnx.AveragePool rewrites into Reshape + ReduceMean");
STATISTIC(NumBroadcastDivRewrites,
          "onnx.Div rewrites into Mul(x, Reciprocal(y)) when broadcasting");

namespace mlir {
namespace hip {

namespace {

/// Read a single-element scalar from an onnx.Constant. Accepts f16/f32/f64
/// and integer types.
static std::optional<double> getOnnxConstantScalar(mlir::Operation *constOp) {
  if (!constOp || constOp->getName().getStringRef() != "onnx.Constant")
    return std::nullopt;
  auto attr = mlir::dyn_cast_or_null<mlir::DenseElementsAttr>(
      constOp->getAttr("value"));
  if (!attr || attr.getNumElements() != 1)
    return std::nullopt;
  mlir::Type et = attr.getElementType();
  if (et.isF32())
    return static_cast<double>(*attr.getValues<float>().begin());
  if (et.isF64())
    return *attr.getValues<double>().begin();
  if (et.isF16() || et.isBF16())
    return (*attr.getValues<llvm::APFloat>().begin()).convertToDouble();
  if (et.isIntOrIndex())
    return static_cast<double>(
        (*attr.getValues<llvm::APInt>().begin()).getSExtValue());
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// Pow(x, c) -> Mul chains (small integer exponents) or Sqrt (c=0.5)
//===----------------------------------------------------------------------===//

struct PowToMul : public mlir::RewritePattern {
  PowToMul(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Pow", /*benefit=*/2, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 2 || op->getNumResults() != 1)
      return rewriter.notifyMatchFailure(op, "pow.arity");
    mlir::Value x = op->getOperand(0);
    auto expConst = getOnnxConstantScalar(op->getOperand(1).getDefiningOp());
    if (!expConst)
      return rewriter.notifyMatchFailure(op, "pow.exp_not_scalar_const");

    auto outputType =
        mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
    if (!outputType)
      return rewriter.notifyMatchFailure(op, "pow.output_not_ranked");

    mlir::Location loc = op->getLoc();
    double e = *expConst;

    auto emitMul = [&](mlir::Value a, mlir::Value b) -> mlir::Value {
      mlir::OperationState state(loc, "onnx.Mul");
      state.addOperands({a, b});
      state.addTypes(outputType);
      return rewriter.create(state)->getResult(0);
    };

    // Small integer exponents: explicit Mul chain.
    int64_t ie = static_cast<int64_t>(std::round(e));
    if (std::abs(e - static_cast<double>(ie)) < 1e-6) {
      if (ie == 0) {
        // x^0 = 1 — emit a constant tensor of ones with the same shape.
        // Skip: extremely rare in practice; bail out to surface the
        // original op (and let the developer reconsider).
        return rewriter.notifyMatchFailure(op, "pow.exp_zero");
      }
      if (ie == 1) {
        rewriter.replaceOp(op, x);
        ++NumPowRewrites;
        return mlir::success();
      }
      if (ie >= 2 && ie <= 6) {
        mlir::Value acc = x;
        // x^N as repeated multiply (N-1 multiplies). For N=2: Mul(x, x).
        // For N=3: Mul(x, Mul(x, x)). Etc. Simple and correct.
        for (int64_t i = 1; i < ie; ++i)
          acc = emitMul(x, acc);
        rewriter.replaceOp(op, acc);
        ++NumPowRewrites;
        return mlir::success();
      }
    }

    // Pow(x, 0.5) == Sqrt(x). Useful for RMSNorm's normalization step in
    // some exports.
    if (std::abs(e - 0.5) < 1e-6) {
      mlir::OperationState state(loc, "onnx.Sqrt");
      state.addOperands(x);
      state.addTypes(outputType);
      mlir::Value sqrt = rewriter.create(state)->getResult(0);
      rewriter.replaceOp(op, sqrt);
      ++NumPowRewrites;
      return mlir::success();
    }

    return rewriter.notifyMatchFailure(op, "pow.exp_not_supported");
  }
};

//===----------------------------------------------------------------------===//
// AveragePool(NCHW, kernel=K, stride=K, no pad) -> Reshape + ReduceMean
//===----------------------------------------------------------------------===//

struct AveragePoolToReshapeMean : public mlir::RewritePattern {
  AveragePoolToReshapeMean(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.AveragePool", /*benefit=*/2, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 1 || op->getNumResults() != 1)
      return rewriter.notifyMatchFailure(op, "avgpool.arity");
    mlir::Value x = op->getOperand(0);
    auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(x.getType());
    auto outputType =
        mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
    if (!inputType || !outputType)
      return rewriter.notifyMatchFailure(op, "avgpool.not_ranked");
    // NCHW only.
    if (inputType.getRank() != 4)
      return rewriter.notifyMatchFailure(op, "avgpool.rank_not_4");

    auto getInt = [&](mlir::ArrayAttr a, int64_t i) -> int64_t {
      auto ia = mlir::dyn_cast<mlir::IntegerAttr>(a.getValue()[i]);
      return ia ? ia.getValue().getSExtValue() : -1;
    };

    auto kAttr = op->getAttrOfType<mlir::ArrayAttr>("kernel_shape");
    auto sAttr = op->getAttrOfType<mlir::ArrayAttr>("strides");
    auto pAttr = op->getAttrOfType<mlir::ArrayAttr>("pads");
    if (!kAttr || kAttr.size() != 2)
      return rewriter.notifyMatchFailure(op, "avgpool.no_kernel_2d");
    int64_t kH = getInt(kAttr, 0), kW = getInt(kAttr, 1);
    int64_t sH = kH, sW = kW;
    if (sAttr) {
      if (sAttr.size() != 2)
        return rewriter.notifyMatchFailure(op, "avgpool.strides_not_2");
      sH = getInt(sAttr, 0);
      sW = getInt(sAttr, 1);
    }
    if (sH != kH || sW != kW)
      return rewriter.notifyMatchFailure(op, "avgpool.kernel_neq_stride");
    if (pAttr) {
      for (mlir::Attribute a : pAttr) {
        auto ia = mlir::dyn_cast<mlir::IntegerAttr>(a);
        if (!ia || ia.getValue().getSExtValue() != 0)
          return rewriter.notifyMatchFailure(op, "avgpool.has_padding");
      }
    }
    // auto_pad must be NOTSET or VALID (no implicit padding).
    if (auto ap = op->getAttrOfType<mlir::StringAttr>("auto_pad")) {
      llvm::StringRef v = ap.getValue();
      if (v != "NOTSET" && v != "VALID")
        return rewriter.notifyMatchFailure(op, "avgpool.auto_pad_other");
    }
    // ceil_mode must be 0 (default).
    if (auto a = op->getAttrOfType<mlir::IntegerAttr>("ceil_mode"))
      if (a.getValue().getSExtValue() != 0)
        return rewriter.notifyMatchFailure(op, "avgpool.ceil_mode");

    // Spatial dims must be static and divisible by kernel size. (Vision
    // encoder ships [B,1152,64,64] so this is comfortable.)
    if (inputType.isDynamicDim(2) || inputType.isDynamicDim(3))
      return rewriter.notifyMatchFailure(op, "avgpool.spatial_dynamic");
    int64_t H = inputType.getDimSize(2);
    int64_t W = inputType.getDimSize(3);
    if (H % kH != 0 || W % kW != 0)
      return rewriter.notifyMatchFailure(op, "avgpool.not_evenly_divisible");
    int64_t outH = H / kH;
    int64_t outW = W / kW;

    int64_t B = inputType.getDimSize(0); // possibly dynamic
    int64_t C = inputType.getDimSize(1); // expected static
    if (inputType.isDynamicDim(1))
      return rewriter.notifyMatchFailure(op, "avgpool.c_dynamic");

    mlir::Location loc = op->getLoc();
    mlir::Type elemType = inputType.getElementType();

    // Step 1: Reshape [B, C, H, W] -> [B, C, outH, kH, outW, kW].
    // Shape operand built as: Concat(Shape(x)[0:1], const[C, outH, kH,
    // outW, kW]) when B is dynamic, or a single onnx.Constant when B is
    // static.
    auto i64Type = rewriter.getI64Type();
    auto buildShapeConst = [&](mlir::ArrayRef<int64_t> dims) -> mlir::Value {
      auto t = mlir::RankedTensorType::get({static_cast<int64_t>(dims.size())},
                                           i64Type);
      auto attr = mlir::DenseElementsAttr::get(t, dims);
      mlir::OperationState s(loc, "onnx.Constant");
      s.addTypes(t);
      s.addAttribute("value", attr);
      return rewriter.create(s)->getResult(0);
    };

    mlir::Value reshapeShape;
    if (B != mlir::ShapedType::kDynamic) {
      reshapeShape = buildShapeConst({B, C, outH, kH, outW, kW});
    } else {
      // Build Shape(x)[0:1] -> Concat with const tail.
      // First emit Shape(x).
      auto shapeRetType = mlir::RankedTensorType::get({4}, i64Type);
      mlir::OperationState shapeState(loc, "onnx.Shape");
      shapeState.addOperands(x);
      shapeState.addTypes(shapeRetType);
      mlir::Value shp = rewriter.create(shapeState)->getResult(0);
      // Slice to [0:1].
      mlir::Value zero = buildShapeConst({0});
      mlir::Value one = buildShapeConst({1});
      mlir::Value axes0 = buildShapeConst({0});
      auto sliceRetType = mlir::RankedTensorType::get({1}, i64Type);
      mlir::OperationState sliceState(loc, "onnx.Slice");
      sliceState.addOperands({shp, zero, one, axes0});
      sliceState.addTypes(sliceRetType);
      mlir::Value sliceB = rewriter.create(sliceState)->getResult(0);
      // Tail const.
      mlir::Value tail = buildShapeConst({C, outH, kH, outW, kW});
      auto concatType = mlir::RankedTensorType::get({6}, i64Type);
      mlir::OperationState concatState(loc, "onnx.Concat");
      concatState.addOperands({sliceB, tail});
      concatState.addTypes(concatType);
      concatState.addAttribute(
          "axis", mlir::IntegerAttr::get(
                      mlir::IntegerType::get(rewriter.getContext(), 64,
                                             mlir::IntegerType::Signed),
                      0));
      reshapeShape = rewriter.create(concatState)->getResult(0);
    }
    // The result type is built explicitly from the dims we just computed
    // ({B, C, outH, kH, outW, kW}) — the rewriter already knows every
    // static dim, so there is no need to re-derive it by tracing the
    // shape-operand SSA chain. B may be dynamic; any such dim is resolved
    // post-conversion by the HIP-dialect `--hip-infer-shapes` pass.
    auto reshapedType =
        mlir::RankedTensorType::get({B, C, outH, kH, outW, kW}, elemType);
    mlir::OperationState reshapeState(loc, "onnx.Reshape");
    reshapeState.addOperands({x, reshapeShape});
    reshapeState.addTypes(reshapedType);
    reshapeState.addAttribute(
        "allowzero", mlir::IntegerAttr::get(
                         mlir::IntegerType::get(rewriter.getContext(), 64,
                                                mlir::IntegerType::Signed),
                         0));
    mlir::Value reshaped = rewriter.create(reshapeState)->getResult(0);

    // Step 2: Transpose [B, C, outH, kH, outW, kW] -> [B, C, outH, outW, kH,
    // kW] (perm [0, 1, 2, 4, 3, 5]) so that the K-K patch axes end up trailing
    // and contiguous in memory. Required because `hip_reduce_sum` only
    // handles reductions over a contiguous suffix; without this transpose
    // the subsequent ReduceMean over the K-sized dims would silently sum
    // the wrong contiguous block (dims [4,5] instead of dims [3,5]) and
    // produce wrong values that cascade into NaN downstream.
    auto transposedType =
        mlir::RankedTensorType::get({B, C, outH, outW, kH, kW}, elemType);
    mlir::OperationState transposeState(loc, "onnx.Transpose");
    transposeState.addOperands(reshaped);
    transposeState.addTypes(transposedType);
    transposeState.addAttribute(
        "perm",
        rewriter.getI64ArrayAttr(llvm::ArrayRef<int64_t>{0, 1, 2, 4, 3, 5}));
    mlir::Value transposed = rewriter.create(transposeState)->getResult(0);

    // Step 3: ReduceMean over dims 4 and 5 (the now-trailing K dims),
    // keepdims=0. axes_const = [4, 5]
    auto axesType = mlir::RankedTensorType::get({2}, i64Type);
    auto axesAttr =
        mlir::DenseElementsAttr::get(axesType, llvm::ArrayRef<int64_t>{4, 5});
    mlir::OperationState axesState(loc, "onnx.Constant");
    axesState.addTypes(axesType);
    axesState.addAttribute("value", axesAttr);
    mlir::Value axes = rewriter.create(axesState)->getResult(0);

    // ReduceMean output type is built explicitly from the known dims,
    // keeping the decomposition self-contained.
    auto pooledType = mlir::RankedTensorType::get({B, C, outH, outW}, elemType);
    mlir::OperationState meanState(loc, "onnx.ReduceMean");
    meanState.addOperands({transposed, axes});
    meanState.addTypes(pooledType);
    meanState.addAttribute(
        "keepdims", mlir::IntegerAttr::get(
                        mlir::IntegerType::get(rewriter.getContext(), 64,
                                               mlir::IntegerType::Signed),
                        0));
    meanState.addAttribute(
        "noop_with_empty_axes",
        mlir::IntegerAttr::get(
            mlir::IntegerType::get(rewriter.getContext(), 64,
                                   mlir::IntegerType::Signed),
            0));
    mlir::Value pooled = rewriter.create(meanState)->getResult(0);

    rewriter.replaceOp(op, pooled);
    ++NumAveragePoolRewrites;
    LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] decomposed AveragePool "
                            << "K=" << kH << "x" << kW << " on " << inputType
                            << " -> Reshape + ReduceMean\n");
    return mlir::success();
  }
};

//===----------------------------------------------------------------------===//
// Broadcasting Div(x, y) -> Mul(x, Reciprocal(y))
//
// hip.div lowers to a plain element-wise kernel that does NOT broadcast:
// it reads `num_elements` from the OUTPUT shape and indexes lhs/rhs with
// the same linear index, which silently reads out-of-bounds (uninitialized
// memory → NaN/Inf) when rhs is smaller than lhs. hip.mul (and other
// ElementwiseOpLowering ops) DO broadcast via miopenOpTensor's 4D
// descriptor API.
//
// The Gemma-3 multi-modal projector's RMSNorm shape divides
// `x[1, 256, 1152]` by `sqrt(mean(x^2)+eps)[1, 256, 1]` — a textbook
// broadcast. Without this rewrite the Div produces NaN for most output
// rows and the projector's final MatMul cascades NaN to the output
// `image_features`.
//
// Rewrite is conservative: only fires when lhs and rhs have different
// shapes (i.e. the broadcast case). Same-shape Div continues to use the
// existing element-wise path. Reciprocal is element-wise (no broadcast
// concern) — it produces the same shape as rhs, then Mul broadcasts
// rhs-shape against lhs.
//
// Before:
//   %y = onnx.Div(%x, %d) : (tensor<?x256x1152xf16>, tensor<?x256x1xf16>)
//                       -> tensor<?x256x1152xf16>
// After:
//   %r = onnx.Reciprocal(%d) : tensor<?x256x1xf16>
//   %y = onnx.Mul(%x, %r)    : (tensor<?x256x1152xf16>, tensor<?x256x1xf16>)
//                           -> tensor<?x256x1152xf16>
//===----------------------------------------------------------------------===//

struct BroadcastDivToMulReciprocal : public mlir::RewritePattern {
  BroadcastDivToMulReciprocal(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Div", /*benefit=*/2, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 2 || op->getNumResults() != 1)
      return rewriter.notifyMatchFailure(op, "div.arity");
    mlir::Value lhs = op->getOperand(0);
    mlir::Value rhs = op->getOperand(1);
    auto lhsType = mlir::dyn_cast<mlir::RankedTensorType>(lhs.getType());
    auto rhsType = mlir::dyn_cast<mlir::RankedTensorType>(rhs.getType());
    auto outType =
        mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
    if (!lhsType || !rhsType || !outType)
      return rewriter.notifyMatchFailure(op, "div.not_ranked");
    if (lhsType.getElementType() != rhsType.getElementType())
      return rewriter.notifyMatchFailure(op, "div.mixed_types");

    // Skip when shapes already match — the existing element-wise Div path
    // is correct for that case.
    if (lhsType.getShape() == rhsType.getShape())
      return rewriter.notifyMatchFailure(op, "div.same_shape");

    // Integer broadcast Div has no decomposition into Mul x Reciprocal:
    // ONNX Reciprocal is fp-only, and hip.div in the runtime is a flat
    // element-wise kernel that doesn't broadcast — on batch > 0 it
    // reads past the end of the smaller operand and pulls garbage into
    // the GEMM/RMSNorm chain downstream (NaN cascade visible only at
    // model-output cosine, after the rewriter has long returned). Surface
    // the unsupported case at compile time so it can't pass silently.
    mlir::Type elemType = lhsType.getElementType();
    if (elemType.isIntOrIndex()) {
      op->emitWarning()
          << "onnx.Div with mismatched integer-type operands has no "
             "broadcasting decomposition; runtime hip.div will read OOB "
             "on batch > 0. Extend BroadcastDivToMulReciprocal to integer "
             "Div or add broadcast support to hip_elementwise_div.";
      return rewriter.notifyMatchFailure(op,
                                         "div.integer_broadcast_unsupported");
    }

    // Only fp16/fp32/bf16 — Reciprocal supports the same set.
    if (!elemType.isF16() && !elemType.isF32() && !elemType.isBF16())
      return rewriter.notifyMatchFailure(op, "div.unsupported_dtype");

    mlir::Location loc = op->getLoc();

    // 1) Reciprocal(rhs) — same shape as rhs.
    mlir::OperationState recState(loc, "onnx.Reciprocal");
    recState.addOperands(rhs);
    recState.addTypes(rhsType);
    mlir::Value rec = rewriter.create(recState)->getResult(0);

    // 2) Mul(lhs, Reciprocal(rhs)) — broadcasts via existing Mul path.
    mlir::OperationState mulState(loc, "onnx.Mul");
    mulState.addOperands({lhs, rec});
    mulState.addTypes(outType);
    mlir::Value mul = rewriter.create(mulState)->getResult(0);

    rewriter.replaceOp(op, mul);
    ++NumBroadcastDivRewrites;
    LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] decomposed broadcasting "
                            << "Div " << lhsType << " / " << rhsType
                            << " into Mul(x, Reciprocal(y))\n");
    return mlir::success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// PatchEmbedConvToGemm — Conv with kernel covering entire spatial input.
//
// Vision-encoder "patch embedding" Convs have the unusual shape that the
// kernel matches the input spatial dims exactly, so each output spatial dim
// is 1: e.g. input `[N, 3, 2, 16, 16]`, weight `[1152, 3, 2, 16, 16]`,
// strides `[2, 16, 16]`, output `[N, 1152, 1, 1, 1]`. Mathematically this is
// a per-batch dot product of the flattened input row with each weight row:
//
//   out[n, m] = bias[m] + sum_{c, kd, kh, kw}
//                 input[n, c, kd, kh, kw] * W[m, c, kd, kh, kw]
//
// which is exactly an `onnx.Gemm` with `transB=1`:
//
//   input_flat = Reshape(input, [-1, C*kD*kH*kW])
//   weight_flat = Reshape(W, [M, C*kD*kH*kW])
//   y_flat = Gemm(input_flat, weight_flat, bias, transA=0, transB=1)
//   y = Reshape(y_flat, [-1, M, 1, 1, ..., 1])
//
// Rewriting at the ONNX level (before bufferization) avoids needing a new
// 5D-Conv runtime / lowering path: the existing MatMul/Gemm path handles it
// natively. Restricted to:
//   - input rank >= 4 (covers 2D Conv with kernel == input spatial too)
//   - rank-5 in practice today (3D Conv from typical ViT patch embeddings)
//   - all output spatial dims static and equal to 1
//   - all input spatial dims static (so the flatten K is compile-time known)
//   - pads all zero
//   - group = 1
//   - weight has fully static shape (so we can flatten it with a Constant
//     shape operand)
//
// Bias is optional. When absent we still emit Gemm with a zero C operand
// because the converter requires three inputs; α=1 β=1.
//
// Before (ONNX dialect snippet, 3-D patch embed):
//   %y = onnx.Conv(%x, %w, %b)
//     {pads=[0,0,0,0,0,0], strides=[2,16,16], dilations=[1,1,1], group=1}
//     : (tensor<?x3x2x16x16xf16>, tensor<1152x3x2x16x16xf16>,
//        tensor<1152xf16>) -> tensor<?x1152x1x1x1xf16>
//
// After:
//   %xf = onnx.Reshape(%x,  const [-1, 1536]) : -> tensor<?x1536xf16>
//   %wf = onnx.Reshape(%w,  const [1152, 1536]) : -> tensor<1152x1536xf16>
//   %y2 = onnx.Gemm(%xf, %wf, %b) {transA=0, transB=1, alpha=1.0, beta=1.0}
//           : (tensor<?x1536xf16>, tensor<1152x1536xf16>, tensor<1152xf16>)
//           -> tensor<?x1152xf16>
//   %y  = onnx.Reshape(%y2, const [-1, 1152, 1, 1, 1])
//           : -> tensor<?x1152x1x1x1xf16>
struct PatchEmbedConvToGemm : public mlir::RewritePattern {
  PatchEmbedConvToGemm(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Conv", /*benefit=*/2, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    if (op->getNumOperands() < 2 || op->getNumOperands() > 3 ||
        op->getNumResults() != 1)
      return rewriter.notifyMatchFailure(op, "conv.arity");
    mlir::Value x = op->getOperand(0);
    mlir::Value w = op->getOperand(1);
    mlir::Value bias = op->getNumOperands() == 3 ? op->getOperand(2) : nullptr;

    auto xType = mlir::dyn_cast<mlir::RankedTensorType>(x.getType());
    auto wType = mlir::dyn_cast<mlir::RankedTensorType>(w.getType());
    auto yType =
        mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
    if (!xType || !wType || !yType)
      return rewriter.notifyMatchFailure(op, "conv.not_ranked");
    int64_t rank = xType.getRank();
    if (rank < 4 || rank != wType.getRank() || rank != yType.getRank())
      return rewriter.notifyMatchFailure(op, "conv.rank_mismatch");

    // group must be 1 (default).
    if (auto g = op->getAttrOfType<mlir::IntegerAttr>("group"))
      if (g.getValue().getSExtValue() != 1)
        return rewriter.notifyMatchFailure(op, "conv.group_ne_1");

    // pads all zero (any padding would make patch embed non-trivial).
    if (auto pads = op->getAttrOfType<mlir::ArrayAttr>("pads"))
      for (mlir::Attribute a : pads) {
        auto ia = mlir::dyn_cast<mlir::IntegerAttr>(a);
        if (!ia || ia.getValue().getSExtValue() != 0)
          return rewriter.notifyMatchFailure(op, "conv.has_padding");
      }

    int64_t nSpatial = rank - 2;
    // Input/weight spatial dims must be fully static AND equal to each other
    // (kernel covers entire spatial), and output spatial dims must all be 1.
    int64_t K = 1; // C * product(spatial)
    int64_t Cin = xType.getDimSize(1);
    if (Cin == mlir::ShapedType::kDynamic)
      return rewriter.notifyMatchFailure(op, "conv.cin_dynamic");
    K = Cin;
    for (int64_t i : llvm::seq<int64_t>(0, nSpatial)) {
      int64_t xs = xType.getDimSize(2 + i);
      int64_t ws = wType.getDimSize(2 + i);
      int64_t ys = yType.getDimSize(2 + i);
      if (xs == mlir::ShapedType::kDynamic || ws == mlir::ShapedType::kDynamic)
        return rewriter.notifyMatchFailure(op, "conv.spatial_dynamic");
      if (xs != ws)
        return rewriter.notifyMatchFailure(op, "conv.kernel_neq_input");
      if (ys != 1)
        return rewriter.notifyMatchFailure(op, "conv.out_spatial_ne_1");
      K *= ws;
    }
    int64_t M = wType.getDimSize(0); // out_channels
    if (M == mlir::ShapedType::kDynamic)
      return rewriter.notifyMatchFailure(op, "conv.m_dynamic");

    mlir::Location loc = op->getLoc();
    mlir::Type elemType = xType.getElementType();
    auto i64Type = rewriter.getI64Type();
    auto buildShapeConst = [&](mlir::ArrayRef<int64_t> dims) -> mlir::Value {
      auto t = mlir::RankedTensorType::get({static_cast<int64_t>(dims.size())},
                                           i64Type);
      auto attr = mlir::DenseElementsAttr::get(t, dims);
      mlir::OperationState s(loc, "onnx.Constant");
      s.addTypes(t);
      s.addAttribute("value", attr);
      return rewriter.create(s)->getResult(0);
    };

    // Reshape input to [N, K] using -1 for dynamic batch.
    int64_t N = xType.getDimSize(0);
    auto xFlatType = mlir::RankedTensorType::get({N, K}, elemType);
    mlir::OperationState rIn(loc, "onnx.Reshape");
    rIn.addOperands({x, buildShapeConst({-1, K})});
    rIn.addTypes(xFlatType);
    rIn.addAttribute("allowzero",
                     mlir::IntegerAttr::get(
                         mlir::IntegerType::get(rewriter.getContext(), 64,
                                                mlir::IntegerType::Signed),
                         0));
    mlir::Value xFlat = rewriter.create(rIn)->getResult(0);

    // Reshape weight to [M, K] (fully static).
    auto wFlatType =
        mlir::RankedTensorType::get({M, K}, wType.getElementType());
    mlir::OperationState rW(loc, "onnx.Reshape");
    rW.addOperands({w, buildShapeConst({M, K})});
    rW.addTypes(wFlatType);
    rW.addAttribute("allowzero",
                    mlir::IntegerAttr::get(
                        mlir::IntegerType::get(rewriter.getContext(), 64,
                                               mlir::IntegerType::Signed),
                        0));
    mlir::Value wFlat = rewriter.create(rW)->getResult(0);

    // Build Gemm. Bias is required by the GemmConversion (which calls
    // ConvertOnnxGemm with 3 operands). When absent, synthesize a zero
    // bias of shape [M]. The bias-add overhead is one element-wise
    // op on a small tensor — negligible vs the GEMM itself.
    mlir::Value cOp;
    if (bias) {
      cOp = bias;
    } else {
      // Zero bias constant. Use the input element type (Gemm expects all
      // operands to share the dtype on this path).
      auto bType = mlir::RankedTensorType::get({M}, elemType);
      mlir::Attribute zeroAttr;
      if (auto ft = mlir::dyn_cast<mlir::FloatType>(elemType)) {
        llvm::SmallVector<llvm::APFloat> zeros(
            M, llvm::APFloat::getZero(ft.getFloatSemantics()));
        zeroAttr = mlir::DenseElementsAttr::get(bType, zeros);
      } else {
        return rewriter.notifyMatchFailure(op, "conv.bias_unsupported_dtype");
      }
      mlir::OperationState zState(loc, "onnx.Constant");
      zState.addTypes(bType);
      zState.addAttribute("value", zeroAttr);
      cOp = rewriter.create(zState)->getResult(0);
    }

    auto gemmType = mlir::RankedTensorType::get({N, M}, elemType);
    mlir::OperationState gemm(loc, "onnx.Gemm");
    gemm.addOperands({xFlat, wFlat, cOp});
    gemm.addTypes(gemmType);
    gemm.addAttribute("transA",
                      mlir::IntegerAttr::get(
                          mlir::IntegerType::get(rewriter.getContext(), 64,
                                                 mlir::IntegerType::Signed),
                          0));
    gemm.addAttribute("transB",
                      mlir::IntegerAttr::get(
                          mlir::IntegerType::get(rewriter.getContext(), 64,
                                                 mlir::IntegerType::Signed),
                          1));
    gemm.addAttribute("alpha", rewriter.getF32FloatAttr(1.0f));
    gemm.addAttribute("beta", rewriter.getF32FloatAttr(1.0f));
    mlir::Value gemmOut = rewriter.create(gemm)->getResult(0);

    // Reshape Gemm output [N, M] back to the Conv output shape (which has
    // trailing 1s for every spatial dim).
    llvm::SmallVector<int64_t> outShape;
    outShape.push_back(-1); // batch
    outShape.push_back(M);
    for (int64_t i : llvm::seq<int64_t>(0, nSpatial))
      (void)i, outShape.push_back(1);
    mlir::OperationState rOut(loc, "onnx.Reshape");
    rOut.addOperands({gemmOut, buildShapeConst(outShape)});
    rOut.addTypes(yType);
    rOut.addAttribute("allowzero",
                      mlir::IntegerAttr::get(
                          mlir::IntegerType::get(rewriter.getContext(), 64,
                                                 mlir::IntegerType::Signed),
                          0));
    mlir::Value reshaped = rewriter.create(rOut)->getResult(0);

    rewriter.replaceOp(op, reshaped);
    LLVM_DEBUG(llvm::dbgs()
               << "[" DEBUG_TYPE "] PatchEmbedConvToGemm: rank-" << rank
               << " Conv " << xType << " * " << wType
               << " -> Gemm + Reshape (K=" << K << ", M=" << M << ")\n");
    return mlir::success();
  }
};

void populateProjectorOpsRewritePatterns(mlir::RewritePatternSet &patterns,
                                         mlir::MLIRContext *ctx) {
  patterns.add<PowToMul, AveragePoolToReshapeMean, BroadcastDivToMulReciprocal,
               PatchEmbedConvToGemm>(ctx);
}

} // namespace hip
} // namespace mlir
