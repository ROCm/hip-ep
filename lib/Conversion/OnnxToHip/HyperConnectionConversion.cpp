/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- HyperConnectionConversion.cpp - HyperConnection mix to HIP --------===//
//
//   onnx.Custom(com.microsoft.HyperConnectionPreMix)
//       -> hip.hyper_connection_pre_mix
//   onnx.Custom(com.microsoft.HyperConnectionPostMix)
//       -> hip.hyper_connection_post_mix
//
// The math follows the ORT CPU kernels: FP32 products, one rounding back to
// the element type, and no activation inside the op. Stream layout is
// (..., C, H), or (..., C * H) when num_branches is set.
//
//===----------------------------------------------------------------------===//

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

mlir::Value optionalOperand(mlir::Operation *op, unsigned idx) {
  if (idx >= op->getNumOperands())
    return nullptr;
  mlir::Value value = op->getOperand(idx);
  if (mlir::isa<mlir::NoneType>(value.getType()))
    return nullptr;
  return value;
}

bool isHyperFloat(mlir::Type type) { return type.isF16() || type.isF32(); }

bool dimsMatch(int64_t a, int64_t b) {
  return mlir::ShapedType::isDynamic(a) || mlir::ShapedType::isDynamic(b) ||
         a == b;
}

struct StreamParams {
  int64_t branches = 0;
  int64_t hidden = 0;
  bool flattened = false;
  size_t prefixRank = 0;
};

mlir::LogicalResult resolveStreams(mlir::Operation *op,
                                   mlir::RankedTensorType streams,
                                   int64_t numBranches, StreamParams &params) {
  int rank = streams.getRank();
  params.flattened = numBranches != 0;
  if (params.flattened) {
    if (numBranches < 1 || rank < 1)
      return mlir::failure();
    int64_t width = streams.getDimSize(rank - 1);
    if (mlir::ShapedType::isDynamic(width) || width <= 0 ||
        width % numBranches != 0)
      return mlir::failure();
    params.branches = numBranches;
    params.hidden = width / numBranches;
    params.prefixRank = static_cast<size_t>(rank - 1);
    return mlir::success();
  }
  if (rank < 2)
    return mlir::failure();
  int64_t branches = streams.getDimSize(rank - 2);
  int64_t hidden = streams.getDimSize(rank - 1);
  if (mlir::ShapedType::isDynamic(branches) ||
      mlir::ShapedType::isDynamic(hidden) || branches <= 0 || hidden <= 0)
    return mlir::failure();
  params.branches = branches;
  params.hidden = hidden;
  params.prefixRank = static_cast<size_t>(rank - 2);
  (void)op;
  return mlir::success();
}

// layout 0: gate suffix (C) or (C, 1). layout 1: (C, H), or (C * H) when
// allowFlattenedFeature is set. Branch is tried first, matching ORT.
mlir::LogicalResult resolveGate(mlir::RankedTensorType gate,
                                mlir::RankedTensorType streams,
                                const StreamParams &params,
                                bool allowFlattenedFeature, int64_t &layout) {
  if (gate.getRank() < static_cast<int>(params.prefixRank) ||
      gate.getElementType() != streams.getElementType())
    return mlir::failure();
  for (size_t i = 0; i < params.prefixRank; ++i) {
    if (!dimsMatch(gate.getDimSize(i), streams.getDimSize(i)))
      return mlir::failure();
  }
  size_t suffix = static_cast<size_t>(gate.getRank()) - params.prefixRank;
  auto suffixDim = [&](size_t i) {
    return gate.getDimSize(params.prefixRank + i);
  };
  auto isStatic = [](int64_t value, int64_t expected) {
    return !mlir::ShapedType::isDynamic(value) && value == expected;
  };
  if (suffix == 1 && isStatic(suffixDim(0), params.branches)) {
    layout = 0;
    return mlir::success();
  }
  if (allowFlattenedFeature && suffix == 1 &&
      isStatic(suffixDim(0), params.branches * params.hidden)) {
    layout = 1;
    return mlir::success();
  }
  if (suffix == 2 && isStatic(suffixDim(0), params.branches) &&
      isStatic(suffixDim(1), 1)) {
    layout = 0;
    return mlir::success();
  }
  if (suffix == 2 && isStatic(suffixDim(0), params.branches) &&
      isStatic(suffixDim(1), params.hidden)) {
    layout = 1;
    return mlir::success();
  }
  return mlir::failure();
}

bool isReduced(mlir::RankedTensorType reduced, mlir::RankedTensorType streams,
               const StreamParams &params) {
  if (reduced.getElementType() != streams.getElementType() ||
      reduced.getRank() != static_cast<int>(params.prefixRank) + 1)
    return false;
  if (!dimsMatch(reduced.getDimSize(reduced.getRank() - 1), params.hidden))
    return false;
  for (size_t i = 0; i < params.prefixRank; ++i) {
    if (!dimsMatch(reduced.getDimSize(i), streams.getDimSize(i)))
      return false;
  }
  return true;
}

mlir::StringAttr layoutAttr(mlir::PatternRewriter &rewriter, int64_t layout) {
  return rewriter.getStringAttr(layout == 0 ? "branch" : "feature");
}

struct HyperConnectionPreMixToHip : public mlir::RewritePattern {
  HyperConnectionPreMixToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Custom", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto funcName = op->getAttrOfType<mlir::StringAttr>("function_name");
    if (!funcName || funcName.getValue() != "HyperConnectionPreMix")
      return rewriter.notifyMatchFailure(op, "not HyperConnectionPreMix");
    auto domain = op->getAttrOfType<mlir::StringAttr>("domain_name");
    if (!domain || domain.getValue() != "com.microsoft")
      return rewriter.notifyMatchFailure(op, "domain must be com.microsoft");
    if (op->getNumOperands() != 2 || op->getNumResults() != 1)
      return rewriter.notifyMatchFailure(
          op, "HyperConnectionPreMix expects 2 inputs and 1 result");

    mlir::Value gate = op->getOperand(1);
    int64_t gateSigmoid = 0;
    mlir::Operation *sigmoidOp = gate.getDefiningOp();
    if (sigmoidOp && sigmoidOp->getName().getStringRef() == "onnx.Sigmoid" &&
        sigmoidOp->hasOneUse()) {
      gate = sigmoidOp->getOperand(0);
      gateSigmoid = 1;
    } else {
      sigmoidOp = nullptr;
    }
    auto streamsType =
        mlir::dyn_cast<mlir::RankedTensorType>(op->getOperand(0).getType());
    auto gateType = mlir::dyn_cast<mlir::RankedTensorType>(gate.getType());
    auto resultType =
        mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
    if (!streamsType || !gateType || !resultType ||
        !isHyperFloat(streamsType.getElementType()))
      return rewriter.notifyMatchFailure(
          op, "HyperConnectionPreMix supports ranked f16 and f32");

    int64_t numBranches = 0;
    if (auto attr = op->getAttrOfType<mlir::IntegerAttr>("num_branches"))
      numBranches = attr.getValue().getSExtValue();
    StreamParams params;
    if (failed(resolveStreams(op, streamsType, numBranches, params)))
      return rewriter.notifyMatchFailure(
          op, "HyperConnectionPreMix stream shape is not static (C, H)");
    int64_t gateLayout = 0;
    if (failed(resolveGate(gateType, streamsType, params,
                           /*allowFlattenedFeature=*/true, gateLayout)))
      return rewriter.notifyMatchFailure(
          op, "HyperConnectionPreMix gate shape is not (C), (C, 1), (C, H), "
              "or flattened (C * H)");
    if (!isReduced(resultType, streamsType, params))
      return rewriter.notifyMatchFailure(
          op, "HyperConnectionPreMix output must be (..., H)");

    float reductionScale = 1.0f;
    if (auto attr = op->getAttrOfType<mlir::FloatAttr>("reduction_scale"))
      reductionScale = attr.getValue().convertToFloat();

    auto ctxOrFailure = getContextArg(op, rewriter);
    if (mlir::failed(ctxOrFailure))
      return rewriter.notifyMatchFailure(op, "missing context argument");
    mlir::Location loc = op->getLoc();
    mlir::Value init =
        createEmptyTensor(rewriter, loc, resultType, op->getOperand(0));
    auto hipOp = mlir::hip::HyperConnectionPreMixOp::create(
        rewriter, loc, *ctxOrFailure, op->getOperand(0), gate, init,
        rewriter.getI64IntegerAttr(params.branches),
        rewriter.getI64IntegerAttr(params.hidden),
        rewriter.getI64IntegerAttr(params.flattened ? 1 : 0),
        layoutAttr(rewriter, gateLayout),
        rewriter.getF32FloatAttr(reductionScale),
        rewriter.getI64IntegerAttr(gateSigmoid));
    rewriter.replaceOp(op, hipOp->getResult(0));
    if (sigmoidOp)
      rewriter.eraseOp(sigmoidOp);
    return mlir::success();
  }
};

bool isOnnx(mlir::Operation *op, llvm::StringRef name) {
  return op && op->getName().getStringRef() == name;
}

bool isScalarConstant(mlir::Value value, float expected) {
  mlir::Operation *op = value.getDefiningOp();
  if (!op)
    return false;
  llvm::StringRef name = op->getName().getStringRef();
  // Constants are rewritten to hip.constant before the main conversion.
  if (name != "onnx.Constant" && name != "hip.constant")
    return false;
  auto attr =
      mlir::dyn_cast_or_null<mlir::DenseElementsAttr>(op->getAttr("value"));
  if (!attr || attr.getNumElements() != 1)
    return false;
  mlir::Type elementType = attr.getElementType();
  double got = 0.0;
  if (elementType.isF32())
    got = *attr.getValues<float>().begin();
  else if (elementType.isF64())
    got = *attr.getValues<double>().begin();
  else if (elementType.isF16() || elementType.isBF16())
    got = (*attr.getValues<llvm::APFloat>().begin()).convertToDouble();
  else
    return false;
  double delta = got - static_cast<double>(expected);
  if (delta < 0.0)
    delta = -delta;
  return delta < 1e-4;
}

bool isEither(mlir::Operation *op, llvm::StringRef a, llvm::StringRef b) {
  if (!op)
    return false;
  llvm::StringRef name = op->getName().getStringRef();
  return name == a || name == b;
}

// onnx binaries are (lhs, rhs). hip DPS binaries are (ctx, lhs, rhs, init).
bool binaryInputs(mlir::Operation *op, mlir::Value &lhs, mlir::Value &rhs) {
  llvm::StringRef name = op->getName().getStringRef();
  if (name.starts_with("onnx.")) {
    if (op->getNumOperands() != 2)
      return false;
    lhs = op->getOperand(0);
    rhs = op->getOperand(1);
    return true;
  }
  if (name.starts_with("hip.") && op->getNumOperands() >= 3) {
    lhs = op->getOperand(1);
    rhs = op->getOperand(2);
    return true;
  }
  return false;
}

mlir::Value unaryInput(mlir::Operation *op) {
  llvm::StringRef name = op->getName().getStringRef();
  if (name.starts_with("onnx.") && op->getNumOperands() >= 1)
    return op->getOperand(0);
  if (name.starts_with("hip.") && op->getNumOperands() >= 2)
    return op->getOperand(1);
  return nullptr;
}

struct InjectChain {
  mlir::Value inject;
  mlir::Operation *outer = nullptr;
  mlir::Operation *sigmoid = nullptr;
  mlir::Operation *div = nullptr;
  mlir::Operation *recip = nullptr;
};

bool matchInjectChain(mlir::Value gate, InjectChain &chain) {
  mlir::Operation *outer = gate.getDefiningOp();
  if (!isEither(outer, "onnx.Mul", "hip.mul") || !outer->hasOneUse())
    return false;
  mlir::Value lhs, rhs;
  if (!binaryInputs(outer, lhs, rhs))
    return false;
  mlir::Value sigmoidVal;
  if (isScalarConstant(rhs, 2.f))
    sigmoidVal = lhs;
  else if (isScalarConstant(lhs, 2.f))
    sigmoidVal = rhs;
  else
    return false;
  mlir::Operation *sigmoid = sigmoidVal.getDefiningOp();
  if (!isEither(sigmoid, "onnx.Sigmoid", "hip.sigmoid") ||
      !sigmoid->hasOneUse())
    return false;
  mlir::Value divided = unaryInput(sigmoid);
  mlir::Operation *div = divided ? divided.getDefiningOp() : nullptr;
  if (!div || !div->hasOneUse())
    return false;
  llvm::StringRef divName = div->getName().getStringRef();
  if (divName == "onnx.Div") {
    mlir::Value input, scale;
    if (!binaryInputs(div, input, scale) || !isScalarConstant(scale, 4.f))
      return false;
    chain.inject = input;
  } else if (divName == "onnx.Mul" || divName == "hip.mul") {
    // Broadcast Div(x, scalar) is rewritten to Mul(x, Reciprocal(scalar))
    // before this pattern runs. The reciprocal may be shared.
    mlir::Value input, scale;
    if (!binaryInputs(div, input, scale))
      return false;
    mlir::Operation *recip = scale.getDefiningOp();
    mlir::Value raw = input;
    if (!isEither(recip, "onnx.Reciprocal", "hip.reciprocal")) {
      recip = input.getDefiningOp();
      raw = scale;
    }
    if (!isEither(recip, "onnx.Reciprocal", "hip.reciprocal") ||
        !isScalarConstant(unaryInput(recip), 4.f))
      return false;
    chain.inject = raw;
    chain.recip = recip;
  } else {
    return false;
  }
  chain.outer = outer;
  chain.sigmoid = sigmoid;
  chain.div = div;
  return true;
}

// Fold a following BranchwiseRMSNorm into the post-mix. stream_mix and a
// feature gate are kept. The 2 * sigmoid(x / 4) chain is folded only when
// that chain is actually present. applicable is set once the norm user is
// the fusion target, so a shape mismatch does not fall through to an
// unfused post-mix and leave the norm without a lowering.
mlir::LogicalResult fusePostMixNorm(mlir::Operation *op,
                                    mlir::PatternRewriter &rewriter,
                                    const StreamParams &params,
                                    mlir::Value streamMix, int64_t gateLayout,
                                    bool &applicable) {
  applicable = false;
  mlir::Operation *norm = nullptr;
  for (mlir::Operation *user : op->getResult(0).getUsers()) {
    auto funcName = user->getAttrOfType<mlir::StringAttr>("function_name");
    auto domain = user->getAttrOfType<mlir::StringAttr>("domain_name");
    if (!isOnnx(user, "onnx.Custom") || !funcName || !domain ||
        funcName.getValue() != "BranchwiseRMSNorm" ||
        domain.getValue() != "com.microsoft")
      continue;
    if (norm)
      return mlir::failure();
    norm = user;
  }
  if (!norm || norm->getNumOperands() != 2 || norm->getNumResults() != 1 ||
      norm->getOperand(0) != op->getResult(0))
    return mlir::failure();
  applicable = true;

  mlir::Value gate = op->getOperand(2);
  int64_t gateActivation = 0;
  InjectChain chain;
  if (matchInjectChain(gate, chain)) {
    gate = chain.inject;
    gateActivation = 1;
  }
  mlir::Value scale = norm->getOperand(1);
  auto streamsType =
      mlir::dyn_cast<mlir::RankedTensorType>(op->getOperand(0).getType());
  auto scaleType = mlir::dyn_cast<mlir::RankedTensorType>(scale.getType());
  auto normType =
      mlir::dyn_cast<mlir::RankedTensorType>(norm->getResult(0).getType());
  if (!streamsType || !scaleType || !normType || scaleType.getRank() != 1 ||
      scaleType.getElementType() != streamsType.getElementType() ||
      normType != streamsType)
    return mlir::failure();
  int64_t scaleLen = scaleType.getDimSize(0);
  if (mlir::ShapedType::isDynamic(scaleLen) ||
      scaleLen != params.branches * params.hidden)
    return mlir::failure();
  if (auto attr = norm->getAttrOfType<mlir::IntegerAttr>("num_branches")) {
    if (attr.getValue().getSExtValue() != params.branches)
      return mlir::failure();
  }
  float epsilon = 1e-6f;
  if (auto attr = norm->getAttrOfType<mlir::FloatAttr>("epsilon"))
    epsilon = attr.getValue().convertToFloat();

  auto ctxOrFailure = getContextArg(op, rewriter);
  if (mlir::failed(ctxOrFailure))
    return mlir::failure();
  mlir::Location loc = op->getLoc();
  mlir::Value residualInit =
      createEmptyTensor(rewriter, loc, streamsType, op->getOperand(0));
  mlir::Value normInit =
      createEmptyTensor(rewriter, loc, normType, op->getOperand(0));
  auto hipOp = mlir::hip::HyperConnectionPostMixNormOp::create(
      rewriter, loc, mlir::TypeRange{streamsType, normType}, *ctxOrFailure,
      op->getOperand(0), op->getOperand(1), gate, scale, streamMix,
      mlir::ValueRange{residualInit, normInit},
      rewriter.getI64IntegerAttr(params.branches),
      rewriter.getI64IntegerAttr(params.hidden),
      rewriter.getI64IntegerAttr(params.flattened ? 1 : 0),
      layoutAttr(rewriter, gateLayout),
      rewriter.getI64IntegerAttr(gateActivation),
      rewriter.getF32FloatAttr(epsilon));
  rewriter.replaceOp(norm, hipOp->getResult(1));
  rewriter.replaceOp(op, hipOp->getResult(0));
  if (gateActivation) {
    auto eraseIfDead = [&](mlir::Operation *dead) {
      if (dead && dead->use_empty())
        rewriter.eraseOp(dead);
    };
    eraseIfDead(chain.outer);
    eraseIfDead(chain.sigmoid);
    eraseIfDead(chain.div);
    eraseIfDead(chain.recip);
  }
  return mlir::success();
}

struct HyperConnectionPostMixToHip : public mlir::RewritePattern {
  HyperConnectionPostMixToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Custom", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto funcName = op->getAttrOfType<mlir::StringAttr>("function_name");
    if (!funcName || funcName.getValue() != "HyperConnectionPostMix")
      return rewriter.notifyMatchFailure(op, "not HyperConnectionPostMix");
    auto domain = op->getAttrOfType<mlir::StringAttr>("domain_name");
    if (!domain || domain.getValue() != "com.microsoft")
      return rewriter.notifyMatchFailure(op, "domain must be com.microsoft");
    if (op->getNumOperands() < 3 || op->getNumOperands() > 4 ||
        op->getNumResults() != 1)
      return rewriter.notifyMatchFailure(
          op, "HyperConnectionPostMix expects 3-4 inputs and 1 result");

    mlir::Value streams = op->getOperand(0);
    mlir::Value blockOutput = op->getOperand(1);
    mlir::Value postMix = op->getOperand(2);
    mlir::Value streamMix = optionalOperand(op, 3);
    auto streamsType =
        mlir::dyn_cast<mlir::RankedTensorType>(streams.getType());
    auto blockType =
        mlir::dyn_cast<mlir::RankedTensorType>(blockOutput.getType());
    auto gateType = mlir::dyn_cast<mlir::RankedTensorType>(postMix.getType());
    auto resultType =
        mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
    if (!streamsType || !blockType || !gateType || !resultType ||
        !isHyperFloat(streamsType.getElementType()) ||
        resultType != streamsType)
      return rewriter.notifyMatchFailure(
          op, "HyperConnectionPostMix supports ranked f16 and f32 streams");

    int64_t numBranches = 0;
    if (auto attr = op->getAttrOfType<mlir::IntegerAttr>("num_branches"))
      numBranches = attr.getValue().getSExtValue();
    StreamParams params;
    if (failed(resolveStreams(op, streamsType, numBranches, params)))
      return rewriter.notifyMatchFailure(
          op, "HyperConnectionPostMix stream shape is not static (C, H)");
    int64_t gateLayout = 0;
    if (failed(resolveGate(gateType, streamsType, params,
                           /*allowFlattenedFeature=*/false, gateLayout)))
      return rewriter.notifyMatchFailure(
          op,
          "HyperConnectionPostMix gate shape is not (C), (C, 1), or (C, H)");
    if (!isReduced(blockType, streamsType, params))
      return rewriter.notifyMatchFailure(
          op, "HyperConnectionPostMix block_output must be (..., H)");
    if (streamMix) {
      auto mixType =
          mlir::dyn_cast<mlir::RankedTensorType>(streamMix.getType());
      if (!mixType ||
          mixType.getElementType() != streamsType.getElementType() ||
          mixType.getRank() != static_cast<int>(params.prefixRank) + 2 ||
          !dimsMatch(mixType.getDimSize(params.prefixRank), params.branches) ||
          !dimsMatch(mixType.getDimSize(params.prefixRank + 1),
                     params.branches))
        return rewriter.notifyMatchFailure(
            op, "HyperConnectionPostMix stream_mix must be (..., C, C)");
      for (size_t i = 0; i < params.prefixRank; ++i) {
        if (!dimsMatch(mixType.getDimSize(i), streamsType.getDimSize(i)))
          return rewriter.notifyMatchFailure(
              op,
              "HyperConnectionPostMix stream_mix prefix must match streams");
      }
    }

    bool applicable = false;
    if (mlir::succeeded(fusePostMixNorm(op, rewriter, params, streamMix,
                                        gateLayout, applicable)))
      return mlir::success();
    if (applicable)
      return rewriter.notifyMatchFailure(
          op, "HyperConnectionPostMix norm fusion inputs do not match");

    auto ctxOrFailure = getContextArg(op, rewriter);
    if (mlir::failed(ctxOrFailure))
      return rewriter.notifyMatchFailure(op, "missing context argument");
    mlir::Location loc = op->getLoc();
    mlir::Value init = createEmptyTensor(rewriter, loc, resultType, streams);
    auto hipOp = mlir::hip::HyperConnectionPostMixOp::create(
        rewriter, loc, *ctxOrFailure, streams, blockOutput, postMix, streamMix,
        init, rewriter.getI64IntegerAttr(params.branches),
        rewriter.getI64IntegerAttr(params.hidden),
        rewriter.getI64IntegerAttr(params.flattened ? 1 : 0),
        layoutAttr(rewriter, gateLayout));
    rewriter.replaceOp(op, hipOp->getResult(0));
    return mlir::success();
  }
};

} // namespace

void populateHyperConnectionConversionPatterns(RewritePatternSet &patterns,
                                               MLIRContext *ctx) {
  patterns.add<HyperConnectionPreMixToHip, HyperConnectionPostMixToHip>(ctx);
}

} // namespace hip
} // namespace mlir
