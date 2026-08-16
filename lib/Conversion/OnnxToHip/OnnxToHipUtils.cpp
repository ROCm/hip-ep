/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- OnnxToHipUtils.cpp - ONNX-specific conversion helpers ------------===//

#include "OnnxToHipUtils.h"

namespace mlir::hip {

DenseElementsAttr getCompileTimeConstantTensor(Value value) {
  if (DenseElementsAttr dense = matchHipCompileTimeConstantTensor(value))
    return dense;
  if (!value)
    return {};

  Operation *defOp = value.getDefiningOp();
  // Semantic pre-lowering rewrites intentionally run before hip.constant
  // carrier creation. Recognize only the exact generic ONNX constant form.
  if (!defOp || defOp->getName().getStringRef() != "onnx.Constant" ||
      defOp->getNumResults() != 1 || defOp->getResult(0) != value)
    return {};
  auto dense = defOp->getAttrOfType<DenseElementsAttr>("value");
  return dense && dense.getType() == value.getType() ? dense
                                                     : DenseElementsAttr();
}

bool extractConstantIntTensor(Value value, llvm::SmallVectorImpl<int64_t> &out,
                              std::optional<int64_t> expectedRank) {
  return parseDenseIntElements(getCompileTimeConstantTensor(value), out,
                               expectedRank);
}

bool extractConstantIntVector(Value value,
                              llvm::SmallVectorImpl<int64_t> &out) {
  return extractConstantIntTensor(value, out, /*expectedRank=*/1);
}

bool isResultTypeCompatibleWithPayloadShape(
    mlir::RankedTensorType resultType, llvm::ArrayRef<int64_t> inferredShape) {
  return isResultTypeCompatibleWithInferredShape(resultType, inferredShape);
}

mlir::FailureOr<RotaryEmbeddingConfig>
resolveRotaryEmbeddingConfig(mlir::PatternRewriter &rewriter,
                             mlir::Operation *op, mlir::Value input,
                             mlir::Value cosCache) {
  auto interleavedAttr = op->getAttrOfType<mlir::IntegerAttr>("interleaved");
  auto numHeadsAttr = op->getAttrOfType<mlir::IntegerAttr>("num_heads");
  auto rotaryDimAttr =
      op->getAttrOfType<mlir::IntegerAttr>("rotary_embedding_dim");

  // ONNX INT attributes import as signed IntegerAttr, so getSInt() is required
  // here: IntegerAttr::getInt() asserts for signed integer types.
  RotaryEmbeddingConfig config{interleavedAttr ? interleavedAttr.getSInt() : 0,
                               numHeadsAttr ? numHeadsAttr.getSInt() : 0,
                               rotaryDimAttr ? rotaryDimAttr.getSInt() : 0};

  auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(input.getType());
  if (config.rotaryDim == 0) {
    auto cosCacheType =
        mlir::dyn_cast<mlir::RankedTensorType>(cosCache.getType());
    if (!cosCacheType || cosCacheType.getRank() < 2 ||
        cosCacheType.isDynamicDim(cosCacheType.getRank() - 1)) {
      rewriter.notifyMatchFailure(
          op, "Cannot infer rotary_embedding_dim: "
              "cos_cache last dim must be static with rank >= 2");
      return mlir::failure();
    }
    config.rotaryDim = cosCacheType.getDimSize(cosCacheType.getRank() - 1) * 2;
  }

  if (inputType && inputType.getRank() == 4) {
    int64_t shapeNumHeads = inputType.getShape()[1];
    if (!mlir::ShapedType::isDynamic(shapeNumHeads)) {
      if (config.numHeads == 0)
        config.numHeads = shapeNumHeads;
      else if (config.numHeads != shapeNumHeads) {
        rewriter.notifyMatchFailure(
            op, "RotaryEmbedding: num_heads attribute disagrees with 4D "
                "input shape (BNSH)");
        return mlir::failure();
      }
    }
    if (config.numHeads == 0) {
      rewriter.notifyMatchFailure(
          op, "Cannot infer num_heads: 4D input has dynamic num_heads dim "
              "and no num_heads attribute");
      return mlir::failure();
    }
  } else if (config.numHeads == 0 && config.rotaryDim > 0) {
    if (!inputType || inputType.getRank() < 1 ||
        inputType.isDynamicDim(inputType.getRank() - 1)) {
      rewriter.notifyMatchFailure(
          op, "Cannot infer num_heads: input last dim must be static");
      return mlir::failure();
    }
    config.numHeads =
        inputType.getDimSize(inputType.getRank() - 1) / config.rotaryDim;
  }

  return config;
}

std::optional<llvm::ArrayRef<int64_t>>
resolveReductionAxes(Operation *op, Value data, int64_t noopWithEmptyAxes,
                     llvm::SmallVectorImpl<int64_t> &storage) {
  storage.clear();
  bool hasAxesOperand =
      op->getNumOperands() > 1 && !isa<NoneType>(op->getOperand(1).getType());
  if (hasAxesOperand) {
    auto axesType = dyn_cast<RankedTensorType>(op->getOperand(1).getType());
    if (!axesType || (axesType.getRank() != 0 && axesType.getRank() != 1) ||
        !extractConstantIntTensor(op->getOperand(1), storage))
      return std::nullopt;
  } else if (auto axesAttr = op->getAttrOfType<ArrayAttr>("axes")) {
    for (Attribute entry : axesAttr)
      storage.push_back(cast<IntegerAttr>(entry).getValue().getSExtValue());
  }

  if (storage.empty() && noopWithEmptyAxes == 0) {
    auto dataType = dyn_cast<RankedTensorType>(data.getType());
    if (!dataType)
      return std::nullopt;
    llvm::append_range(storage, llvm::seq<int64_t>(0, dataType.getRank()));
  }
  return llvm::ArrayRef<int64_t>(storage);
}

FailureOr<RankedTensorType>
inferReduceResultType(Operation *op, Value data,
                      llvm::ArrayRef<int64_t> reducedAxes, int64_t keepdims) {
  auto ranked = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  auto inputType = dyn_cast<RankedTensorType>(data.getType());
  if (!inputType)
    return failure();
  FailureOr<llvm::SmallVector<int64_t>> outShape =
      inferReductionShape(inputType.getShape(), reducedAxes, keepdims);
  if (failed(outShape))
    return failure();
  if (ranked) {
    if (!isResultTypeCompatibleWithInferredShape(ranked, *outShape))
      return failure();
    return ranked;
  }
  return RankedTensorType::get(*outShape, inputType.getElementType());
}

} // namespace mlir::hip
