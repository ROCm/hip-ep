/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- HardSigmoidConversion.cpp - onnx.HardSigmoid decomposition --------===//
//
// ONNX HardSigmoid: y = max(0, min(1, alpha*x + beta)), with alpha defaulting
// to 0.2 and beta to 0.5.
//
// Decomposed into ONNX primitives rather than given a HIP op of its own: Mul,
// Add and Clip all already lower, and Clip's own decomposition is hip.max then
// hip.min -- exactly the clamp this needs. A dedicated op would add a kernel
// for arithmetic the existing ones already express.
//
// Emitted from the pre-lowering loop rather than convertComputeOps. That loop
// runs to a fixed point, so the primitives below are "existing ops" by the next
// round and reach their own converters normally; convertComputeOps runs with
// GreedyRewriteStrictness::ExistingOps, where anything synthesized is never
// revisited and would trip "op was not bufferized" instead (see the note at the
// top of ClipConversion.cpp).
//
// MobileNetV3 is the motivating model: 9 HardSigmoid nodes with alpha = 1/6 and
// beta left at its default, forming the h-swish gate. It was the only
// unconverted op in the fp32 graph.
//
//   Before:
//     %y = "onnx.HardSigmoid"(%x) {alpha = 0.166666672 : f32}
//            : (tensor<1x64x14x14xf32>) -> tensor<1x64x14x14xf32>
//   After:
//     %a = onnx.Constant {value = dense<0.166666672> : tensor<f32>}
//     %t = "onnx.Mul"(%x, %a)
//     %b = onnx.Constant {value = dense<5.000000e-01> : tensor<f32>}
//     %u = "onnx.Add"(%t, %b)
//     %lo = onnx.Constant {value = dense<0.000000e+00> : tensor<f32>}
//     %hi = onnx.Constant {value = dense<1.000000e+00> : tensor<f32>}
//     %y = "onnx.Clip"(%u, %lo, %hi)
//
//===----------------------------------------------------------------------===//

#include "OnnxToHipUtils.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "convert-onnx-to-hip"

STATISTIC(NumHardSigmoidRewrites,
          "Number of onnx.HardSigmoid ops decomposed into Mul/Add/Clip");

namespace mlir {
namespace hip {

namespace {

struct HardSigmoidDecompose : public mlir::RewritePattern {
  HardSigmoidDecompose(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.HardSigmoid", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 1 || op->getNumResults() != 1)
      return rewriter.notifyMatchFailure(
          op, "onnx.HardSigmoid expects 1 input, 1 output");

    auto resultType =
        mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(op, "expected a ranked tensor result");
    auto elemType =
        mlir::dyn_cast<mlir::FloatType>(resultType.getElementType());
    if (!elemType)
      return rewriter.notifyMatchFailure(
          op, "onnx.HardSigmoid expects a float element type");

    // ONNX schema defaults, applied when the exporter omitted the attribute --
    // MobileNetV3 writes `alpha` but leaves `beta` implicit.
    double alpha = 0.2;
    double beta = 0.5;
    if (auto attr = op->getAttrOfType<mlir::FloatAttr>("alpha"))
      alpha = attr.getValueAsDouble();
    if (auto attr = op->getAttrOfType<mlir::FloatAttr>("beta"))
      beta = attr.getValueAsDouble();

    mlir::Location loc = op->getLoc();
    mlir::Value x = op->getOperand(0);

    // Rank-0, matching how ONNX spells a scalar: Clip's converter takes its
    // bounds as 0-rank operands, and Mul/Add broadcast the scalar against x.
    auto scalar = [&](double value) -> mlir::Value {
      llvm::APFloat converted(value);
      bool losesInfo = false;
      converted.convert(elemType.getFloatSemantics(),
                        llvm::APFloat::rmNearestTiesToEven, &losesInfo);
      auto type = mlir::RankedTensorType::get({}, elemType);
      mlir::OperationState state(loc, "onnx.Constant");
      state.addTypes(type);
      state.addAttribute("value",
                         mlir::DenseElementsAttr::get(
                             type, llvm::ArrayRef<llvm::APFloat>{converted}));
      return rewriter.create(state)->getResult(0);
    };

    auto onnxOp = [&](llvm::StringRef name,
                      llvm::ArrayRef<mlir::Value> operands) -> mlir::Value {
      mlir::OperationState state(loc, name);
      state.addOperands(operands);
      state.addTypes(resultType);
      return rewriter.create(state)->getResult(0);
    };

    mlir::Value scaled = onnxOp("onnx.Mul", {x, scalar(alpha)});
    mlir::Value shifted = onnxOp("onnx.Add", {scaled, scalar(beta)});
    rewriter.replaceOp(
        op, onnxOp("onnx.Clip", {shifted, scalar(0.0), scalar(1.0)}));

    LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] decomposed HardSigmoid alpha="
                            << alpha << " beta=" << beta << "\n");
    ++NumHardSigmoidRewrites;
    return mlir::success();
  }
};

} // namespace

void populateHardSigmoidConversionPatterns(RewritePatternSet &patterns,
                                           MLIRContext *ctx) {
  patterns.add<HardSigmoidDecompose>(ctx);
}

} // namespace hip
} // namespace mlir
