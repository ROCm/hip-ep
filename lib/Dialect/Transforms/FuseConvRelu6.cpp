/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- FuseConvRelu6.cpp - hip.conv + hip.max/min -> fused conv -----------===//
//
// MobileNet ReLU6 lowers as Clip(0, 6) -> hip.max(x, 0) + hip.min(x, 6).
// ResNet ReLU lowers as hip.max(x, 0). When either chain is the sole consumer
// of a convolution, fold it into hip.conv with fused_activation and clip bounds
// so the runtime can apply MIOpen ReLU / ClippedReLU after conv+bias.
//
//===----------------------------------------------------------------------===//

#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/Statistic.h"

#include <cstring>

#define DEBUG_TYPE "fuse-conv-relu6"

STATISTIC(NumConvRelu6Fused, "Number of conv+ReLU6 chains fused");
STATISTIC(NumConvReluFused, "Number of conv+ReLU chains fused");

namespace mlir {
namespace hip {

#define GEN_PASS_DEF_FUSECONVRELU6PASS
#include "hip/Dialect/Transforms/Passes.h.inc"

namespace {

static constexpr float kClipLoZero = 0.0f;
static constexpr float kClipHiNone = 0.0f;
static constexpr float kClipHiRelu6 = 6.0f;

static FloatAttr f32Attr(PatternRewriter &rewriter, float value) {
  return rewriter.getF32FloatAttr(value);
}

static std::optional<double> decodeSplatBytes(Type elemType,
                                              ArrayRef<uint8_t> bytes) {
  if (elemType.isF32()) {
    if (bytes.size() < sizeof(float))
      return std::nullopt;
    float value;
    std::memcpy(&value, bytes.data(), sizeof(float));
    return static_cast<double>(value);
  }
  if (elemType.isF64()) {
    if (bytes.size() < sizeof(double))
      return std::nullopt;
    double value;
    std::memcpy(&value, bytes.data(), sizeof(double));
    return value;
  }
  if (elemType.isF16()) {
    if (bytes.size() < sizeof(uint16_t))
      return std::nullopt;
    uint16_t bits;
    std::memcpy(&bits, bytes.data(), sizeof(uint16_t));
    llvm::APFloat value(llvm::APFloat::IEEEhalf(), llvm::APInt(16, bits));
    return value.convertToDouble();
  }
  if (elemType.isBF16()) {
    if (bytes.size() < sizeof(uint16_t))
      return std::nullopt;
    uint16_t bits;
    std::memcpy(&bits, bytes.data(), sizeof(uint16_t));
    llvm::APFloat value(llvm::APFloat::BFloat(), llvm::APInt(16, bits));
    return value.convertToDouble();
  }
  if (elemType.isIntOrIndex()) {
    if (bytes.empty() || bytes.size() > 8)
      return std::nullopt;
    int64_t raw = 0;
    std::memcpy(&raw, bytes.data(), bytes.size());
    const unsigned bitWidth = static_cast<unsigned>(bytes.size() * 8);
    return static_cast<double>(llvm::APInt(bitWidth, raw, true).getSExtValue());
  }
  return std::nullopt;
}

static std::optional<double>
getExternalizedSplatScalar(bufferization::ToTensorOp toTensor) {
  auto getGlobal = toTensor.getBuffer().getDefiningOp<memref::GetGlobalOp>();
  if (!getGlobal)
    return std::nullopt;

  auto module = getGlobal->getParentOfType<ModuleOp>();
  if (!module)
    return std::nullopt;

  auto global = module.lookupSymbol<memref::GlobalOp>(getGlobal.getName());
  if (!global)
    return std::nullopt;

  auto externalData =
      dyn_cast_or_null<DictionaryAttr>(global->getAttr("hip.external_data"));
  if (!externalData)
    return std::nullopt;

  auto indexAttr = dyn_cast_or_null<IntegerAttr>(externalData.get("index"));
  if (!indexAttr)
    return std::nullopt;
  int64_t index = indexAttr.getInt();

  auto sourceKinds =
      module->getAttrOfType<DenseI32ArrayAttr>("hipdnn.constant_source_kinds");
  auto splatValues = module->getAttrOfType<DenseI64ArrayAttr>(
      "hipdnn.constant_splat_elem_values");
  auto splatElemSizes = module->getAttrOfType<DenseI64ArrayAttr>(
      "hipdnn.constant_splat_elem_sizes");
  if (!sourceKinds || !splatValues || !splatElemSizes || index < 0 ||
      index >= static_cast<int64_t>(sourceKinds.size()))
    return std::nullopt;

  if (static_cast<ConstantMetadataSourceKind>(sourceKinds[index]) !=
      ConstantMetadataSourceKind::Splat)
    return std::nullopt;

  int64_t elemSize = splatElemSizes[index];
  if (elemSize <= 0 || elemSize > 8)
    return std::nullopt;

  int64_t rawValue = splatValues[index];
  const auto *bytes = reinterpret_cast<const uint8_t *>(&rawValue);
  return decodeSplatBytes(
      global.getType().getElementType(),
      ArrayRef<uint8_t>(bytes, static_cast<size_t>(elemSize)));
}

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

  if (auto toTensor = dyn_cast<bufferization::ToTensorOp>(def)) {
    if (auto splat = getExternalizedSplatScalar(toTensor))
      return splat;
  }

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

static bool convHasFusedActivation(ConvOp convOp) {
  return convOp.getFusedActivation();
}

static bool isRelu6MinChain(MaxOp maxOp) {
  if (!maxOp->hasOneUse())
    return false;
  auto minOp = dyn_cast<MinOp>(*maxOp->getUsers().begin());
  if (!minOp)
    return false;
  std::optional<double> lo = getScalarConstant(maxOp.getRhs());
  std::optional<double> hi = getScalarConstant(minOp.getRhs());
  return lo && hi && std::abs(*lo) <= 1e-3 && std::abs(*hi - 6.0) <= 1e-3;
}

static void setFusedClipActivation(ConvOp convOp, float clipLo, float clipHi) {
  convOp.setFusedActivation(true);
  convOp.setActivationClipLo(llvm::APFloat(clipLo));
  convOp.setActivationClipHi(llvm::APFloat(clipHi));
}

static LogicalResult fuseConvWithActivation(ConvOp convOp, Value finalOutput,
                                            float clipLo, float clipHi,
                                            PatternRewriter &rewriter,
                                            Operation *tailOp, MaxOp maxOp,
                                            MinOp minOp) {
  if (finalOutput == convOp.getOutput()) {
    rewriter.modifyOpInPlace(
        convOp, [&]() { setFusedClipActivation(convOp, clipLo, clipHi); });
    rewriter.replaceOp(tailOp, convOp.getResult(0));
    if (maxOp && maxOp.getOperation() != tailOp)
      rewriter.eraseOp(maxOp);
    if (minOp && minOp.getOperation() != tailOp)
      rewriter.eraseOp(minOp);
    return success();
  }

  Operation *outputDef = finalOutput.getDefiningOp();
  if (!outputDef)
    return failure();

  rewriter.setInsertionPointAfter(outputDef);
  auto fusedConv = ConvOp::create(
      rewriter, convOp.getLoc(), convOp.getCtx(), convOp.getInput(),
      convOp.getWeights(), convOp.getBias(), finalOutput,
      convOp.getKernelShapeAttr(), convOp.getStridesAttr(),
      convOp.getPadsAttr(), convOp.getDilationsAttr(), convOp.getGroupAttr(),
      rewriter.getBoolAttr(true), f32Attr(rewriter, clipLo),
      f32Attr(rewriter, clipHi), convOp.getActivationAlphaAttr());

  rewriter.replaceOp(tailOp, fusedConv.getResult(0));
  if (maxOp && maxOp.getOperation() != tailOp)
    rewriter.eraseOp(maxOp);
  if (minOp && minOp.getOperation() != tailOp)
    rewriter.eraseOp(minOp);
  rewriter.eraseOp(convOp);
  return success();
}

struct FuseConvRelu6Pattern : public OpRewritePattern<MinOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(MinOp minOp,
                                PatternRewriter &rewriter) const override {
    auto maxOp = minOp.getLhs().getDefiningOp<MaxOp>();
    if (!maxOp || !maxOp->hasOneUse())
      return failure();

    auto convOp = maxOp.getLhs().getDefiningOp<ConvOp>();
    if (!convOp || convHasFusedActivation(convOp))
      return failure();
    if (!convOp.getResults().front().hasOneUse())
      return failure();

    std::optional<double> lo = getScalarConstant(maxOp.getRhs());
    std::optional<double> hi = getScalarConstant(minOp.getRhs());
    if (!lo || !hi)
      return failure();
    if (std::abs(*lo) > 1e-3 || std::abs(*hi - 6.0) > 1e-3)
      return failure();

    if (failed(fuseConvWithActivation(convOp, minOp.getOutput(), kClipLoZero,
                                      kClipHiRelu6, rewriter, minOp, maxOp,
                                      minOp)))
      return failure();
    ++NumConvRelu6Fused;
    return success();
  }
};

struct FuseConvReluPattern : public OpRewritePattern<MaxOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(MaxOp maxOp,
                                PatternRewriter &rewriter) const override {
    if (isRelu6MinChain(maxOp))
      return failure();

    auto convOp = maxOp.getLhs().getDefiningOp<ConvOp>();
    if (!convOp || convHasFusedActivation(convOp))
      return failure();
    if (!convOp.getResults().front().hasOneUse())
      return failure();

    std::optional<double> lo = getScalarConstant(maxOp.getRhs());
    if (!lo || std::abs(*lo) > 1e-3)
      return failure();

    if (failed(fuseConvWithActivation(convOp, maxOp.getOutput(), kClipLoZero,
                                      kClipHiNone, rewriter, maxOp, maxOp,
                                      nullptr)))
      return failure();
    ++NumConvReluFused;
    return success();
  }
};

struct FuseConvRelu6Pass
    : public impl::FuseConvRelu6PassBase<FuseConvRelu6Pass> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    RewritePatternSet patterns(&getContext());
    patterns.add<FuseConvRelu6Pattern, FuseConvReluPattern>(&getContext());
    if (failed(applyPatternsGreedily(func, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
} // namespace hip
} // namespace mlir
