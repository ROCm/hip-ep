/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- FuseConvRelu6.cpp - hip.conv + hip.max/min -> fused conv -----------===//
//
// MobileNet ReLU6 lowers as Clip(0, 6) -> hip.max(x, 0) + hip.min(x, 6).
// When this chain is the sole consumer of a conv, fold it into hip.conv with
// activation=1 so the runtime can execute MIOpen FusionPlan (Conv+Bias+Act).
//
//===----------------------------------------------------------------------===//

#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/Statistic.h"

#define DEBUG_TYPE "fuse-conv-relu6"

STATISTIC(NumConvRelu6Fused, "Number of conv+ReLU6 chains fused");

namespace mlir {
namespace hip {

#define GEN_PASS_DEF_FUSECONVRELU6PASS
#include "hip/Dialect/Transforms/Passes.h.inc"

namespace {

static constexpr int64_t kConvActivationRelu6 = 1;

static std::optional<double> getScalarConstant(mlir::Value v) {
  while (v) {
    if (auto castOp = v.getDefiningOp<mlir::UnrealizedConversionCastOp>())
      v = castOp.getOperand(0);
    else
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
  } else if (auto hc = mlir::dyn_cast<mlir::hip::ConstantOp>(def)) {
    if (auto val = hc.getValue())
      dense = mlir::dyn_cast<mlir::DenseElementsAttr>(*val);
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
  if (et.isIntOrIndex())
    return static_cast<double>(
        (*dense.getValues<llvm::APInt>().begin()).getSExtValue());
  return std::nullopt;
}

struct FuseConvRelu6Pattern : public OpRewritePattern<MinOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(MinOp minOp,
                                PatternRewriter &rewriter) const override {
    auto maxOp = minOp.getLhs().getDefiningOp<MaxOp>();
    if (!maxOp || !maxOp->hasOneUse())
      return failure();

    auto convOp = maxOp.getLhs().getDefiningOp<ConvOp>();
    if (!convOp || convOp.getActivation() != 0)
      return failure();
    if (!convOp.getResults().front().hasOneUse())
      return failure();

    std::optional<double> lo = getScalarConstant(maxOp.getRhs());
    std::optional<double> hi = getScalarConstant(minOp.getRhs());
    if (!lo || !hi)
      return failure();
    if (std::abs(*lo) > 1e-3 || std::abs(*hi - 6.0) > 1e-3)
      return failure();

    Value finalOutput = minOp.getOutput();
    if (finalOutput == convOp.getOutput()) {
      rewriter.modifyOpInPlace(
          convOp, [&]() { convOp.setActivation(kConvActivationRelu6); });
      rewriter.replaceOp(minOp, convOp.getResult(0));
      rewriter.eraseOp(maxOp);
      ++NumConvRelu6Fused;
      return success();
    }

    // The final output buffer is allocated after the conv in the typical
    // conv -> max -> min chain. Reassigning conv's outs in-place would make
    // conv reference a value that does not dominate it.
    Operation *outputDef = finalOutput.getDefiningOp();
    if (!outputDef)
      return failure();

    SmallVector<Value> operands = {convOp.getCtx(), convOp.getInput(),
                                   convOp.getWeights()};
    if (Value bias = convOp.getBias())
      operands.push_back(bias);
    operands.push_back(finalOutput);

    SmallVector<NamedAttribute> attrs;
    for (NamedAttribute attr : convOp->getAttrs()) {
      if (attr.getName() == "activation")
        continue;
      attrs.push_back(attr);
    }
    attrs.push_back(rewriter.getNamedAttr(
        "activation", rewriter.getI64IntegerAttr(kConvActivationRelu6)));

    rewriter.setInsertionPointAfter(outputDef);
    auto fusedConv = ConvOp::create(rewriter, convOp.getLoc(), operands, attrs);

    rewriter.replaceOp(minOp, fusedConv.getResult(0));
    rewriter.eraseOp(maxOp);
    rewriter.eraseOp(convOp);
    ++NumConvRelu6Fused;
    return success();
  }
};

struct FuseConvRelu6Pass
    : public impl::FuseConvRelu6PassBase<FuseConvRelu6Pass> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    RewritePatternSet patterns(&getContext());
    patterns.add<FuseConvRelu6Pattern>(&getContext());
    if (failed(applyPatternsGreedily(func, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
} // namespace hip
} // namespace mlir
