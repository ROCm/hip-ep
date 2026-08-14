/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- LoraWeightPackFusion.cpp - Fold LoRA weight_pack into MatMulNBits -===//
//
// Some LoRA exports insert a 5-op weight_pack chain before each
// LoRA MatMulNBits(bits=8):
//
//   weight_quantized [K,N] i8
//     -> Transpose [1,0]
//     -> Cast i16
//     -> Add(+128)
//     -> Cast ui8
//     -> Reshape [N, K/block, block]
//     -> MatMulNBits
//
// This pass rewires MatMulNBits to take the raw int8 graph input directly and
// sets lora_weight_pack=1 so the runtime packs once into a cached GPU buffer.
//
//===----------------------------------------------------------------------===//

#include "OnnxToHipUtils.h"

#include "mlir/IR/PatternMatch.h"

#include "llvm/ADT/Statistic.h"

STATISTIC(NumLoraWeightPackFused,
          "Number of LoRA weight_pack chains folded into MatMulNBits");

namespace mlir {
namespace hip {

namespace {

static bool is8BitInteger(mlir::Type ty) {
  if (auto intTy = mlir::dyn_cast<mlir::IntegerType>(ty))
    return intTy.getWidth() == 8;
  return false;
}

static bool valueHasMultipleElements(mlir::Value v) {
  auto shaped = mlir::dyn_cast<mlir::ShapedType>(v.getType());
  if (!shaped || !shaped.hasRank())
    return false;
  if (!shaped.hasStaticShape())
    return shaped.getRank() > 0;
  return shaped.getNumElements() > 1;
}

static mlir::Operation *getAddTensorInputDef(mlir::Operation *addOp) {
  if (!addOp)
    return nullptr;
  for (mlir::Value operand : addOp->getOperands()) {
    if (valueHasMultipleElements(operand))
      return operand.getDefiningOp();
  }
  return nullptr;
}

static bool isLoraWeightPackFusionCandidate(mlir::Operation *op) {
  if (!op || op->getName().getStringRef() != "onnx.Custom")
    return false;
  auto funcName = op->getAttrOfType<mlir::StringAttr>("function_name");
  if (!funcName || funcName.getValue() != "MatMulNBits")
    return false;
  auto domain = op->getAttrOfType<mlir::StringAttr>("domain_name");
  if (!domain || domain.getValue() != "com.microsoft")
    return false;
  auto bitsAttr = op->getAttrOfType<mlir::IntegerAttr>("bits");
  if (!bitsAttr || getOnnxIntegerAttrValue(bitsAttr) != 8)
    return false;
  if (op->getAttr("lora_weight_pack"))
    return false;
  if (op->getNumOperands() < 2)
    return false;
  mlir::Operation *reshapeOp = op->getOperand(1).getDefiningOp();
  return reshapeOp && reshapeOp->getName().getStringRef() == "onnx.Reshape";
}

static std::optional<mlir::Value>
resolveLoraWeightPackRawWeightFast(mlir::Operation *mnbOp) {
  if (!isLoraWeightPackFusionCandidate(mnbOp))
    return std::nullopt;

  auto *reshapeOp = mnbOp->getOperand(1).getDefiningOp();
  auto *castU8 = reshapeOp->getOperand(0).getDefiningOp();
  if (!castU8 || castU8->getName().getStringRef() != "onnx.Cast")
    return std::nullopt;

  auto *addOp = castU8->getOperand(0).getDefiningOp();
  if (!addOp || addOp->getName().getStringRef() != "onnx.Add")
    return std::nullopt;

  auto *castI16 = getAddTensorInputDef(addOp);
  if (!castI16 || castI16->getName().getStringRef() != "onnx.Cast")
    return std::nullopt;

  auto *transposeOp = castI16->getOperand(0).getDefiningOp();
  if (!transposeOp || transposeOp->getName().getStringRef() != "onnx.Transpose")
    return std::nullopt;
  // Do not introspect the Transpose `perm` attr on the fusion hot path.
  // MorphiZen-imported MLIR can attach perm in forms where even O(1) attr
  // reads stall the compiler; typical LoRA pack chains use [1,0] here.

  mlir::Value rawWeight = transposeOp->getOperand(0);
  auto rawTy = mlir::dyn_cast<mlir::RankedTensorType>(rawWeight.getType());
  if (!rawTy || rawTy.getRank() != 2 || !is8BitInteger(rawTy.getElementType()))
    return std::nullopt;

  return rawWeight;
}

static void eraseDeadOnnxPackChains(mlir::func::FuncOp funcOp) {
  while (true) {
    llvm::SmallVector<mlir::Operation *> toErase;
    funcOp.walk([&](mlir::Operation *op) {
      if (isDeadOnnxOpSafeToErase(op))
        toErase.push_back(op);
    });
    if (toErase.empty())
      break;
    for (mlir::Operation *op : toErase)
      op->erase();
  }
}

static bool applyLoraWeightPackFusion(mlir::Operation *mnbOp,
                                      mlir::RewriterBase &rewriter) {
  std::optional<mlir::Value> rawWeight =
      resolveLoraWeightPackRawWeightFast(mnbOp);
  if (!rawWeight)
    return false;

  rewriter.modifyOpInPlace(mnbOp, [&] {
    mnbOp->setOperand(1, *rawWeight);
    mnbOp->setAttr("lora_weight_pack", rewriter.getI64IntegerAttr(1));
  });

  ++NumLoraWeightPackFused;
  return true;
}

static void runLoraWeightPackFusionImpl(mlir::func::FuncOp funcOp) {
  llvm::SmallVector<mlir::Operation *, 64> candidates;
  funcOp.walk([&](mlir::Operation *op) {
    if (isLoraWeightPackFusionCandidate(op))
      candidates.push_back(op);
  });

  mlir::IRRewriter rewriter(funcOp.getContext());
  for (mlir::Operation *op : candidates)
    applyLoraWeightPackFusion(op, rewriter);

  eraseDeadOnnxPackChains(funcOp);
}

} // namespace

void runLoraWeightPackFusion(mlir::func::FuncOp funcOp) {
  runLoraWeightPackFusionImpl(funcOp);
}

} // namespace hip
} // namespace mlir
