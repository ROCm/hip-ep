/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- ImageScalerConversion.cpp - onnx.ImageScaler -> hip.add/mul -------===//
//
// ONNX ImageScaler (NCHW):
//   output = scale * (input + bias)
// `scale` is a scalar float attribute (default 1.0). `bias` is a per-channel
// float list whose length is C; it broadcasts as tensor<1xCx1x1xT>.
//
// The HIP dialect already has broadcasting add and mul, so this is a
// decomposition rather than a new op. `hip.add` / `hip.mul` are emitted
// directly: convert-onnx-to-hip runs with ExistingOps, so a synthesized
// `onnx.Add` / `onnx.Mul` would not be rewritten in the same pass. The
// literal constants stay `onnx.Constant` and are lowered by the phase-2
// constant pass.
//
//   Before:
//     %y = "onnx.ImageScaler"(%x) {bias = [...], scale = s}
//         : (tensor<NxCxHxWxT>) -> tensor<NxCxHxWxT>
//
//   After (non-zero bias):
//     %b = "onnx.Constant"() {value = dense<...> : tensor<1xCx1x1xT>}
//     %s = "onnx.Constant"() {value = dense<s> : tensor<T>}
//     %added = hip.add(%ctx) ins(%x, %b) outs(%init1)
//     %y = hip.mul(%ctx) ins(%added, %s) outs(%init2)
//
// A zero bias drops the add (TinyYOLOv2's 1/255 scale). A scale of exactly
// 1 drops the mul. Both together are an identity when the types match.
//
//===----------------------------------------------------------------------===//

#include "OnnxToHipUtils.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace hip {
namespace {

/// Convert `value` to the semantics of `elemType`, keeping an exact f32
/// attribute bit-identical when the input element type is f32.
static llvm::APFloat toElementFloat(llvm::APFloat value,
                                    mlir::FloatType elemType) {
  bool losesInfo = false;
  value.convert(elemType.getFloatSemantics(),
                llvm::APFloat::rmNearestTiesToEven, &losesInfo);
  return value;
}

/// 0-D `onnx.Constant` of `elemType`. Phase-2 constant lowering consumes it.
static mlir::Value buildScalarConstant(mlir::PatternRewriter &rewriter,
                                       mlir::Location loc,
                                       mlir::FloatType elemType,
                                       llvm::APFloat value) {
  auto scalarType = mlir::RankedTensorType::get({}, elemType);
  auto attr = mlir::DenseElementsAttr::get(scalarType, value);
  mlir::OperationState state(loc, "onnx.Constant");
  state.addTypes(scalarType);
  state.addAttribute("value", attr);
  return rewriter.create(state)->getResult(0);
}

/// Channel bias as `tensor<1xCx1x1xT>` so `hip.add` aligns C, not H/W.
/// Right-aligned `tensor<Cx1x1>` would broadcast onto the spatial axes.
static mlir::Value buildChannelBias(mlir::PatternRewriter &rewriter,
                                    mlir::Location loc,
                                    mlir::FloatType elemType,
                                    llvm::ArrayRef<llvm::APFloat> bias) {
  auto biasType = mlir::RankedTensorType::get(
      {1, static_cast<int64_t>(bias.size()), 1, 1}, elemType);
  auto attr = mlir::DenseElementsAttr::get(biasType, bias);
  mlir::OperationState state(loc, "onnx.Constant");
  state.addTypes(biasType);
  state.addAttribute("value", attr);
  return rewriter.create(state)->getResult(0);
}

/// Read the `bias` attribute into `out`. Accepts an array of float attrs
/// (the ONNX importer form) or a dense float attribute.
static mlir::LogicalResult readBias(mlir::Operation *op,
                                    mlir::FloatType elemType,
                                    llvm::SmallVectorImpl<llvm::APFloat> &out) {
  out.clear();
  if (auto array = op->getAttrOfType<mlir::ArrayAttr>("bias")) {
    if (array.empty())
      return mlir::failure();
    out.reserve(array.size());
    for (mlir::Attribute element : array) {
      auto floatAttr = mlir::dyn_cast<mlir::FloatAttr>(element);
      if (!floatAttr)
        return mlir::failure();
      out.push_back(toElementFloat(floatAttr.getValue(), elemType));
    }
    return mlir::success();
  }
  if (auto dense = op->getAttrOfType<mlir::DenseElementsAttr>("bias")) {
    if (!mlir::isa<mlir::FloatType>(dense.getElementType()) || dense.empty())
      return mlir::failure();
    out.reserve(dense.getNumElements());
    for (llvm::APFloat value : dense.getValues<llvm::APFloat>())
      out.push_back(toElementFloat(value, elemType));
    return mlir::success();
  }
  return mlir::failure();
}

struct ImageScalerToElementwise : public mlir::RewritePattern {
  ImageScalerToElementwise(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.ImageScaler", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 1 || op->getNumResults() != 1)
      return rewriter.notifyMatchFailure(
          op, "onnx.ImageScaler expects 1 operand and 1 result");

    auto ctxOrFailure = getContextArg(op, rewriter);
    if (mlir::failed(ctxOrFailure))
      return mlir::failure();
    mlir::Value context = *ctxOrFailure;

    mlir::Value input = op->getOperand(0);
    auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(input.getType());
    auto resultType =
        mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
    if (!inputType || !resultType || inputType.getRank() != 4 ||
        resultType.getRank() != 4)
      return rewriter.notifyMatchFailure(
          op, "onnx.ImageScaler requires rank-4 NCHW tensors");
    auto elemType = mlir::dyn_cast<mlir::FloatType>(inputType.getElementType());
    if (!elemType || resultType.getElementType() != elemType)
      return rewriter.notifyMatchFailure(
          op, "onnx.ImageScaler requires a float element type");
    for (int64_t axis : llvm::seq<int64_t>(0, 4)) {
      if (inputType.isDynamicDim(axis) || resultType.isDynamicDim(axis))
        continue;
      if (inputType.getDimSize(axis) != resultType.getDimSize(axis))
        return rewriter.notifyMatchFailure(
            op, "onnx.ImageScaler result shape must match the input");
    }

    llvm::SmallVector<llvm::APFloat> bias;
    if (mlir::failed(readBias(op, elemType, bias)))
      return rewriter.notifyMatchFailure(
          op, "onnx.ImageScaler bias must be a non-empty float list");
    // Channel is axis 1. A static C that disagrees with the bias length
    // cannot broadcast as 1xCx1x1.
    for (mlir::RankedTensorType type : {inputType, resultType}) {
      if (type.isDynamicDim(1))
        continue;
      if (type.getDimSize(1) != static_cast<int64_t>(bias.size()))
        return rewriter.notifyMatchFailure(
            op, "onnx.ImageScaler bias length must equal the channel count");
    }

    llvm::APFloat scale(elemType.getFloatSemantics(), 1);
    if (auto scaleAttr = op->getAttrOfType<mlir::FloatAttr>("scale"))
      scale = toElementFloat(scaleAttr.getValue(), elemType);
    else if (op->getAttr("scale"))
      return rewriter.notifyMatchFailure(
          op, "onnx.ImageScaler scale must be a float attribute");
    bool biasIsZero = llvm::all_of(
        bias, [](const llvm::APFloat &value) { return value.isZero(); });
    bool scaleIsOne = scale.isExactlyValue(1.0);

    mlir::Location loc = op->getLoc();
    // scale == 1 and a zero bias copy the input. A result type that does
    // not match the input is not a legal ImageScaler, so leave it alone.
    if (biasIsZero && scaleIsOne) {
      if (inputType != resultType)
        return rewriter.notifyMatchFailure(
            op, "onnx.ImageScaler identity requires matching types");
      rewriter.replaceOp(op, input);
      return mlir::success();
    }

    mlir::Value current = input;
    if (!biasIsZero) {
      mlir::Value biasConst = buildChannelBias(rewriter, loc, elemType, bias);
      mlir::Value init = createEmptyTensor(rewriter, loc, resultType, input);
      auto addOp = mlir::hip::AddOp::create(rewriter, loc, context, current,
                                            biasConst, init);
      current = addOp->getResult(0);
    }
    if (!scaleIsOne) {
      mlir::Value scaleConst =
          buildScalarConstant(rewriter, loc, elemType, scale);
      // Shape source stays the original input: `current` may be the add
      // result, whose dynamic dims are not a tensor.dim source.
      mlir::Value init = createEmptyTensor(rewriter, loc, resultType, input);
      auto mulOp = mlir::hip::MulOp::create(rewriter, loc, context, current,
                                            scaleConst, init);
      current = mulOp->getResult(0);
    }
    rewriter.replaceOp(op, current);
    return mlir::success();
  }
};

} // namespace

void populateImageScalerConversionPatterns(RewritePatternSet &patterns,
                                           MLIRContext *ctx) {
  patterns.add<ImageScalerToElementwise>(ctx);
}

} // namespace hip
} // namespace mlir
