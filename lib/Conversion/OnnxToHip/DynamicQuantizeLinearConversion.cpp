/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- DynamicQuantizeLinearConversion.cpp - onnx.DynamicQuantizeLinear --===//
//
// ONNX DynamicQuantizeLinear (opset 11), f32 input, uint8 output:
//   x_min = min(min(x), 0)
//   x_max = max(max(x), 0)
//   y_scale = (x_max - x_min) / 255          // 0 becomes 1
//   y_zero_point = saturate(round((0 - x_min) / y_scale), 0, 255)  // uint8
//   y = saturate(round(x / y_scale) + y_zero_point)                // uint8
//
// Decomposition into ops that already bufferize and already lower through
// both HipToLLVM and HipToTosa. The range is a chain of single-axis
// reductions (the runtime reduce path is defined for one axis). The data
// quantization is hip.quantize_linear, whose kernel and tosa lowering both
// round ties-to-even and saturate. There is no f32->ui8 cast kernel, so the
// zero point is rounded and clipped in f32, then cast f32->i32->ui8.
//
//   Before:
//     %y, %scale, %zp = "onnx.DynamicQuantizeLinear"(%x)
//         : (tensor<NxCxHxWxf32>) -> (tensor<NxCxHxWxui8>, tensor<f32>,
//         tensor<ui8>)
//
//   After:
//     %xmin = hip.reduce_min ...          // one axis at a time, down to rank 0
//     %xmax = hip.reduce_max ...
//     %scale = hip.where(%eq, %one, %raw)  // raw == 0 selects 1
//     %zp = hip.cast(hip.cast(clip(hip.round(...))))
//     %y = hip.quantize_linear(%x, %scale, %zp)
//
//===----------------------------------------------------------------------===//

#include "OnnxToHipUtils.h"

#include "mlir/IR/BuiltinAttributes.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace hip {
namespace {

/// 0-D `onnx.Constant`. Phase-2 constant lowering consumes it; this pass runs
/// with ExistingOps, so a synthesized onnx elementwise op would not be
/// rewritten here.
static mlir::Value buildScalarConstant(mlir::PatternRewriter &rewriter,
                                       mlir::Location loc,
                                       llvm::APFloat value) {
  auto scalarType = mlir::RankedTensorType::get({}, rewriter.getF32Type());
  auto attr = mlir::DenseElementsAttr::get(scalarType, value);
  mlir::OperationState state(loc, "onnx.Constant");
  state.addTypes(scalarType);
  state.addAttribute("value", attr);
  return rewriter.create(state)->getResult(0);
}

/// True when \p type is a static tensor of one element (rank 0, or a
/// degenerate shape such as `tensor<1x1xf32>`).
static bool isSingleElement(mlir::RankedTensorType type) {
  return type.hasStaticShape() && type.getNumElements() == 1;
}

/// Fold the last axis, repeating until the value is rank 0. Each reduce
/// carries that axis in `reduced_axes`; the LLVM lowering uses it for the
/// inner stride. One axis per reduce matches the HIP reduce lowering, which
/// infers a single reduced axis from the trailing shape.
template <typename ReduceOp>
static mlir::Value reduceToScalar(mlir::PatternRewriter &rewriter,
                                  mlir::Location loc, mlir::Value context,
                                  mlir::Value input) {
  mlir::Value current = input;
  auto elemType =
      mlir::cast<mlir::RankedTensorType>(input.getType()).getElementType();
  while (mlir::cast<mlir::RankedTensorType>(current.getType()).getRank() > 0) {
    auto curType = mlir::cast<mlir::RankedTensorType>(current.getType());
    int64_t axis = curType.getRank() - 1;
    llvm::SmallVector<int64_t> outShape(curType.getShape().drop_back());
    auto outType = mlir::RankedTensorType::get(outShape, elemType);
    mlir::Value init = createEmptyTensor(rewriter, loc, outType, current);

    auto axesType = mlir::RankedTensorType::get({1}, rewriter.getI64Type());
    llvm::SmallVector<int64_t, 1> axesVec = {axis};
    auto axesAttr = mlir::DenseIntElementsAttr::get(axesType, axesVec);
    mlir::Value axes =
        mlir::arith::ConstantOp::create(rewriter, loc, axesType, axesAttr);

    auto reduced =
        ReduceOp::create(rewriter, loc, context, current, axes, init,
                         rewriter.getI64IntegerAttr(/*keepdims=*/0),
                         rewriter.getI64IntegerAttr(/*noop_with_empty_axes=*/0),
                         getConstantReducedAxesAttr(rewriter, current, axes,
                                                    /*noopWithEmptyAxes=*/0));
    current = reduced->getResult(0);
  }
  return current;
}

static mlir::Value reshapeTo(mlir::PatternRewriter &rewriter,
                             mlir::Location loc, mlir::Value src,
                             mlir::RankedTensorType dstType) {
  if (src.getType() == dstType)
    return src;
  llvm::SmallVector<mlir::Value> dims;
  for (int64_t dim : dstType.getShape())
    dims.push_back(mlir::arith::ConstantIndexOp::create(rewriter, loc, dim));
  auto shapeType =
      mlir::RankedTensorType::get({dstType.getRank()}, rewriter.getIndexType());
  mlir::Value shape =
      mlir::tensor::FromElementsOp::create(rewriter, loc, shapeType, dims);
  return mlir::tensor::ReshapeOp::create(rewriter, loc, dstType, src, shape)
      .getResult();
}

static mlir::Value castSameShape(mlir::PatternRewriter &rewriter,
                                 mlir::Location loc, mlir::Value context,
                                 mlir::Value input, mlir::Type elemType,
                                 int64_t onnxTo) {
  auto srcType = mlir::cast<mlir::RankedTensorType>(input.getType());
  auto dstType = mlir::RankedTensorType::get(srcType.getShape(), elemType);
  mlir::Value init = createEmptyTensor(rewriter, loc, dstType, input);
  auto casted = mlir::hip::CastOp::create(rewriter, loc, context, input, init,
                                          rewriter.getI64IntegerAttr(onnxTo));
  return casted->getResult(0);
}

/// onnx.DynamicQuantizeLinear -> reductions + hip.quantize_linear.
struct DynamicQuantizeLinearToHip : public mlir::RewritePattern {
  DynamicQuantizeLinearToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.DynamicQuantizeLinear", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override;
};

mlir::LogicalResult DynamicQuantizeLinearToHip::matchAndRewrite(
    mlir::Operation *op, mlir::PatternRewriter &rewriter) const {
  if (op->getNumOperands() != 1 || op->getNumResults() != 3)
    return rewriter.notifyMatchFailure(
        op, "DynamicQuantizeLinear expects 1 input and 3 results");

  auto inputType =
      mlir::dyn_cast<mlir::RankedTensorType>(op->getOperand(0).getType());
  auto yType =
      mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
  auto scaleType =
      mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(1).getType());
  auto zpType =
      mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(2).getType());
  if (!inputType || !yType || !scaleType || !zpType)
    return rewriter.notifyMatchFailure(
        op, "DynamicQuantizeLinear requires ranked tensors");
  if (!inputType.getElementType().isF32())
    return rewriter.notifyMatchFailure(
        op, "DynamicQuantizeLinear input must be f32");
  if (!yType.getElementType().isUnsignedInteger(8) ||
      yType.getShape() != inputType.getShape())
    return rewriter.notifyMatchFailure(
        op, "DynamicQuantizeLinear y must be ui8 with the input shape");
  if (!scaleType.getElementType().isF32() || !isSingleElement(scaleType))
    return rewriter.notifyMatchFailure(
        op, "DynamicQuantizeLinear y_scale must be a single f32 element");
  if (!zpType.getElementType().isUnsignedInteger(8) || !isSingleElement(zpType))
    return rewriter.notifyMatchFailure(
        op, "DynamicQuantizeLinear y_zero_point must be a single ui8 element");

  auto ctxOrFailure = getContextArg(op, rewriter);
  if (mlir::failed(ctxOrFailure))
    return mlir::failure();
  mlir::Value context = *ctxOrFailure;
  mlir::Location loc = op->getLoc();
  mlir::Value input = op->getOperand(0);

  mlir::Value xMin =
      reduceToScalar<mlir::hip::ReduceMinOp>(rewriter, loc, context, input);
  mlir::Value xMax =
      reduceToScalar<mlir::hip::ReduceMaxOp>(rewriter, loc, context, input);

  auto scalarF32 = mlir::RankedTensorType::get({}, rewriter.getF32Type());
  mlir::Value zero = buildScalarConstant(rewriter, loc, llvm::APFloat(0.0f));
  mlir::Value one = buildScalarConstant(rewriter, loc, llvm::APFloat(1.0f));
  mlir::Value c255 = buildScalarConstant(rewriter, loc, llvm::APFloat(255.0f));

  auto scalarInit = [&](mlir::Value like) {
    return createEmptyTensor(rewriter, loc, scalarF32, like);
  };

  // Force the range through 0, matching the ONNX reference (an all-positive
  // tensor still has x_min 0, an all-negative tensor still has x_max 0).
  mlir::Value minAdj =
      mlir::hip::MinOp::create(rewriter, loc, scalarF32, context, xMin, zero,
                               scalarInit(xMin))
          ->getResult(0);
  mlir::Value maxAdj =
      mlir::hip::MaxOp::create(rewriter, loc, scalarF32, context, xMax, zero,
                               scalarInit(xMax))
          ->getResult(0);
  mlir::Value span = mlir::hip::SubOp::create(rewriter, loc, context, maxAdj,
                                              minAdj, scalarInit(maxAdj))
                         ->getResult(0);
  mlir::Value scaleRaw = mlir::hip::DivOp::create(rewriter, loc, context, span,
                                                  c255, scalarInit(span))
                             ->getResult(0);

  // A zero span (every element is 0) is defined as scale 1, not a division
  // by zero in the zero-point formula.
  auto i1Type = mlir::RankedTensorType::get({}, rewriter.getI1Type());
  mlir::Value eqInit = createEmptyTensor(rewriter, loc, i1Type, scaleRaw);
  mlir::Value isZero =
      mlir::hip::EqualOp::create(rewriter, loc, context, scaleRaw, zero, eqInit)
          ->getResult(0);
  mlir::Value scale =
      mlir::hip::WhereOp::create(rewriter, loc, context, isZero, one, scaleRaw,
                                 scalarInit(scaleRaw))
          ->getResult(0);

  mlir::Value negMin = mlir::hip::SubOp::create(rewriter, loc, context, zero,
                                                minAdj, scalarInit(minAdj))
                           ->getResult(0);
  mlir::Value zpDiv = mlir::hip::DivOp::create(rewriter, loc, context, negMin,
                                               scale, scalarInit(negMin))
                          ->getResult(0);
  mlir::Value zpRound =
      mlir::hip::RoundOp::create(rewriter, loc, scalarF32, context, zpDiv,
                                 scalarInit(zpDiv))
          ->getResult(0);
  mlir::Value zpHi =
      mlir::hip::MinOp::create(rewriter, loc, scalarF32, context, zpRound, c255,
                               scalarInit(zpRound))
          ->getResult(0);
  mlir::Value zpLo = mlir::hip::MaxOp::create(rewriter, loc, scalarF32, context,
                                              zpHi, zero, scalarInit(zpHi))
                         ->getResult(0);

  // ONNX UINT8 is datatype 2, INT32 is 6. hip.cast selects the kernel from
  // the buffer element types; `to` records the ONNX destination code.
  mlir::Value zpI32 =
      castSameShape(rewriter, loc, context, zpLo, rewriter.getI32Type(),
                    /*onnxTo=*/6);
  mlir::Value zpUi8 = castSameShape(rewriter, loc, context, zpI32,
                                    zpType.getElementType(), /*onnxTo=*/2);

  mlir::Value yInit = createEmptyTensor(rewriter, loc, yType, input);
  // Per-tensor quantization. Axis 0 is in range for every non-scalar input,
  // including rank 1. HipToTosa checks the axis before it notices that the
  // scale and zero point are scalars, so axis 1 rejects a vector.
  auto quantized = mlir::hip::QuantizeLinearOp::create(
      rewriter, loc, mlir::TypeRange{yType}, context, input, scale, zpUi8,
      yInit, rewriter.getI64IntegerAttr(/*axis=*/0),
      rewriter.getI64IntegerAttr(/*block_size=*/0),
      rewriter.getI64IntegerAttr(/*precision=*/0),
      rewriter.getI64IntegerAttr(/*saturate=*/1));

  mlir::Value scaleOut = reshapeTo(rewriter, loc, scale, scaleType);
  mlir::Value zpOut = reshapeTo(rewriter, loc, zpUi8, zpType);
  rewriter.replaceOp(op, {quantized->getResult(0), scaleOut, zpOut});
  return mlir::success();
}

} // namespace

void populateDynamicQuantizeLinearConversionPatterns(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.add<DynamicQuantizeLinearToHip>(ctx);
}

} // namespace hip
} // namespace mlir
