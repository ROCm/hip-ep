/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- OnnxToHipUtils.cpp - Shared helpers for ONNX-to-HIP patterns ------===//
//
// Out-of-line definitions for the destination-construction helpers declared in
// `OnnxToHipUtils.h`. That header is included by every per-operator conversion
// file, so only the pattern-facing templates stay inline there.
//
// See `docs/design/hip-shape-inference.md` for how these helpers relate to the
// shared `HipShapeUtils` shape rules and to `reifyResultShapes`.
//
//===----------------------------------------------------------------------===//

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

bool areCompatibleStaticExtents(int64_t lhs, int64_t rhs) {
  return mlir::ShapedType::isDynamic(lhs) || mlir::ShapedType::isDynamic(rhs) ||
         lhs == rhs;
}

mlir::Value indexDivByConstant(mlir::PatternRewriter &rewriter,
                               mlir::Location loc, mlir::Value extent,
                               int64_t divisor) {
  mlir::Value divisorValue =
      mlir::arith::ConstantIndexOp::create(rewriter, loc, divisor);
  return mlir::arith::DivUIOp::create(rewriter, loc, extent, divisorValue);
}

} // namespace

bool isLosslessShapeIndexCast(mlir::arith::IndexCastOp op) {
  constexpr unsigned kIndexBitWidth = 64;
  mlir::Type sourceType = op.getIn().getType();
  mlir::Type resultType = op.getType();
  if (sourceType.isIndex()) {
    auto integerType = mlir::dyn_cast<mlir::IntegerType>(resultType);
    return integerType && integerType.getWidth() >= kIndexBitWidth;
  }
  if (resultType.isIndex()) {
    auto integerType = mlir::dyn_cast<mlir::IntegerType>(sourceType);
    return integerType && integerType.getWidth() <= kIndexBitWidth;
  }
  return false;
}

mlir::DenseElementsAttr
getCompileTimeConstantTensor(mlir::Value value,
                             CompileTimeConstantScope scope) {
  // Optional ONNX operands arrive as a null Value; `getDefiningOp` would
  // dereference it.
  if (!value)
    return {};
  mlir::Operation *defOp = value.getDefiningOp();
  if (!defOp)
    return {};

  mlir::DenseElementsAttr dense;
  if (auto cst = mlir::dyn_cast<mlir::arith::ConstantOp>(defOp))
    dense = mlir::dyn_cast<mlir::DenseElementsAttr>(cst.getValue());
  if (!dense)
    if (auto attr = defOp->getAttr("value"))
      dense = mlir::dyn_cast<mlir::DenseElementsAttr>(attr);
  if (!dense && scope == CompileTimeConstantScope::IncludeExternalized) {
    // Externalized constant: to_tensor(get_global) whose global still carries
    // an initial value.
    if (auto toTensor = mlir::dyn_cast<mlir::bufferization::ToTensorOp>(defOp))
      if (auto getGlobal =
              toTensor.getBuffer().getDefiningOp<mlir::memref::GetGlobalOp>())
        if (auto module = getGlobal->getParentOfType<mlir::ModuleOp>())
          if (auto global = module.lookupSymbol<mlir::memref::GlobalOp>(
                  getGlobal.getNameAttr()))
            dense = mlir::dyn_cast_or_null<mlir::DenseElementsAttr>(
                global.getInitialValueAttr());
  }
  return dense;
}

bool extractConstantIntTensor(mlir::Value value,
                              llvm::SmallVectorImpl<int64_t> &out,
                              std::optional<int64_t> expectedRank,
                              CompileTimeConstantScope scope) {
  return mlir::hip::parseDenseIntElements(
      getCompileTimeConstantTensor(value, scope), out, expectedRank);
}

bool extractConstantIntVector(mlir::Value value,
                              llvm::SmallVectorImpl<int64_t> &out,
                              CompileTimeConstantScope scope) {
  return extractConstantIntTensor(value, out, /*expectedRank=*/1, scope);
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
resolveReductionAxes(mlir::Operation *op, mlir::Value data,
                     int64_t noopWithEmptyAxes,
                     llvm::SmallVectorImpl<int64_t> &storage) {
  storage.clear();
  bool hasAxesOperand = op->getNumOperands() > 1 &&
                        !mlir::isa<mlir::NoneType>(op->getOperand(1).getType());
  if (hasAxesOperand) {
    if (!extractConstantIntVector(op->getOperand(1), storage))
      return std::nullopt;
  } else if (auto axesAttr = op->getAttrOfType<mlir::ArrayAttr>("axes")) {
    for (mlir::Attribute entry : axesAttr)
      storage.push_back(
          mlir::cast<mlir::IntegerAttr>(entry).getValue().getSExtValue());
  }

  if (storage.empty() && noopWithEmptyAxes == 0) {
    // Empty axes with noop_with_empty_axes = 0 reduces every axis.
    auto dataType = mlir::dyn_cast<mlir::RankedTensorType>(data.getType());
    if (!dataType)
      return std::nullopt;
    llvm::append_range(storage, llvm::seq<int64_t>(0, dataType.getRank()));
  }
  return llvm::ArrayRef<int64_t>(storage);
}

mlir::Value materializeReductionAxes(mlir::OpBuilder &builder,
                                     mlir::Location loc, mlir::Operation *op,
                                     llvm::ArrayRef<int64_t> resolvedAxes) {
  if (op->getNumOperands() > 1 &&
      !mlir::isa<mlir::NoneType>(op->getOperand(1).getType()))
    return op->getOperand(1);

  auto axesType = mlir::RankedTensorType::get(
      {static_cast<int64_t>(resolvedAxes.size())}, builder.getI64Type());
  auto axesAttr = mlir::DenseIntElementsAttr::get(axesType, resolvedAxes);
  return mlir::arith::ConstantOp::create(builder, loc, axesType, axesAttr);
}

mlir::FailureOr<GqaSequenceExtents>
resolveGqaSequenceExtents(mlir::PatternRewriter &rewriter, mlir::Location loc,
                          mlir::Operation *op, mlir::Value totalSeqLen,
                          mlir::Value pastKey, mlir::Value pastValue,
                          mlir::RankedTensorType presentKeyType,
                          mlir::RankedTensorType presentValueType,
                          mlir::RankedTensorType outputQkType) {
  if (static_cast<bool>(pastKey) != static_cast<bool>(pastValue)) {
    rewriter.notifyMatchFailure(
        op, "past_key and past_value must both be present or both be absent");
    return mlir::failure();
  }
  if (!presentKeyType || !presentValueType || presentKeyType.getRank() != 4 ||
      presentValueType.getRank() != 4) {
    rewriter.notifyMatchFailure(
        op, "present_key and present_value must be rank-4 BNSH tensors");
    return mlir::failure();
  }
  if (outputQkType && outputQkType.getRank() != 4) {
    rewriter.notifyMatchFailure(op, "output_qk must be a rank-4 tensor");
    return mlir::failure();
  }

  if (pastKey) {
    auto pastKeyType =
        mlir::dyn_cast<mlir::RankedTensorType>(pastKey.getType());
    auto pastValueType =
        mlir::dyn_cast<mlir::RankedTensorType>(pastValue.getType());
    if (!pastKeyType || !pastValueType || pastKeyType.getRank() != 4 ||
        pastValueType.getRank() != 4) {
      rewriter.notifyMatchFailure(
          op, "past_key and past_value must be rank-4 BNSH tensors");
      return mlir::failure();
    }
  }

  GqaSequenceExtents extents;
  bool needsLogical = presentKeyType.isDynamicDim(2) ||
                      presentValueType.isDynamicDim(2) ||
                      (outputQkType && outputQkType.isDynamicDim(3));
  if (!needsLogical)
    return extents;

  auto totalSeqLenType =
      totalSeqLen
          ? mlir::dyn_cast<mlir::RankedTensorType>(totalSeqLen.getType())
          : mlir::RankedTensorType();
  if (!totalSeqLenType || totalSeqLenType.getRank() != 0 ||
      !mlir::isa<mlir::IntegerType, mlir::IndexType>(
          totalSeqLenType.getElementType())) {
    rewriter.notifyMatchFailure(
        op, "total_sequence_length must be a scalar integer tensor");
    return mlir::failure();
  }

  extents.logical =
      readbackScalarToIndexOrExtract(rewriter, loc, op, totalSeqLen);
  auto resolvePresent = [&](mlir::RankedTensorType resultType,
                            mlir::Value past) -> mlir::Value {
    if (!resultType.isDynamicDim(2))
      return {};
    if (!past)
      return extents.logical;
    mlir::Value pastExtent =
        mlir::tensor::DimOp::create(rewriter, loc, past, 2);
    return mlir::arith::MaxUIOp::create(rewriter, loc, pastExtent,
                                        extents.logical);
  };
  extents.presentKey = resolvePresent(presentKeyType, pastKey);
  extents.presentValue = resolvePresent(presentValueType, pastValue);
  return extents;
}

mlir::FailureOr<mlir::Value>
createGqaPresentEmpty(mlir::PatternRewriter &rewriter, mlir::Location loc,
                      mlir::RankedTensorType resultType, mlir::Value query,
                      mlir::Value key, mlir::Value totalSeqExtent,
                      int64_t numHeads, int64_t kvNumHeads) {
  auto queryType = mlir::dyn_cast<mlir::RankedTensorType>(query.getType());
  auto keyType = key ? mlir::dyn_cast<mlir::RankedTensorType>(key.getType())
                     : mlir::RankedTensorType();
  if (!queryType || queryType.getRank() != 3 || resultType.getRank() != 4 ||
      (totalSeqExtent && !totalSeqExtent.getType().isIndex()) ||
      (resultType.isDynamicDim(2) && !totalSeqExtent) || numHeads <= 0 ||
      kvNumHeads <= 0 || (key && !keyType) ||
      (keyType && keyType.getRank() != 3 && keyType.getRank() != 4))
    return mlir::failure();

  if (!areCompatibleStaticExtents(resultType.getDimSize(0),
                                  queryType.getDimSize(0)) ||
      (resultType.getDimSize(1) != mlir::ShapedType::kDynamic &&
       resultType.getDimSize(1) != kvNumHeads) ||
      (!keyType && resultType.isDynamicDim(3)))
    return mlir::failure();
  if (resultType.isDynamicDim(3) && keyType.getRank() == 3) {
    int64_t hidden = keyType.getDimSize(2);
    if (!mlir::ShapedType::isDynamic(hidden) && hidden % kvNumHeads != 0)
      return mlir::failure();
  }

  llvm::SmallVector<mlir::Value> dynamicSizes;
  dynamicSizes.reserve(resultType.getNumDynamicDims());
  for (int64_t dim : llvm::seq<int64_t>(0, resultType.getRank())) {
    if (!resultType.isDynamicDim(dim))
      continue;
    switch (dim) {
    case 0:
      dynamicSizes.push_back(
          mlir::tensor::DimOp::create(rewriter, loc, query, 0));
      break;
    case 1:
      dynamicSizes.push_back(
          mlir::arith::ConstantIndexOp::create(rewriter, loc, kvNumHeads));
      break;
    case 2:
      dynamicSizes.push_back(totalSeqExtent);
      break;
    case 3:
      if (keyType && keyType.getRank() == 4) {
        dynamicSizes.push_back(
            mlir::tensor::DimOp::create(rewriter, loc, key, 3));
      } else {
        mlir::Value source = key;
        mlir::Value hidden =
            mlir::tensor::DimOp::create(rewriter, loc, source, 2);
        dynamicSizes.push_back(
            indexDivByConstant(rewriter, loc, hidden, kvNumHeads));
      }
      break;
    default:
      llvm_unreachable("rank-4 GQA present shape has an invalid dimension");
    }
  }

  return mlir::Value(
      mlir::tensor::EmptyOp::create(rewriter, loc, resultType.getShape(),
                                    resultType.getElementType(), dynamicSizes));
}

mlir::FailureOr<mlir::Value>
createGqaQkEmpty(mlir::PatternRewriter &rewriter, mlir::Location loc,
                 mlir::RankedTensorType resultType, mlir::Value query,
                 mlir::Value totalSeqExtent, int64_t numHeads) {
  auto queryType = mlir::dyn_cast<mlir::RankedTensorType>(query.getType());
  if (!queryType || queryType.getRank() != 3 || resultType.getRank() != 4 ||
      (totalSeqExtent && !totalSeqExtent.getType().isIndex()) ||
      (resultType.isDynamicDim(3) && !totalSeqExtent) || numHeads <= 0 ||
      !areCompatibleStaticExtents(resultType.getDimSize(0),
                                  queryType.getDimSize(0)) ||
      (resultType.getDimSize(1) != mlir::ShapedType::kDynamic &&
       resultType.getDimSize(1) != numHeads) ||
      !areCompatibleStaticExtents(resultType.getDimSize(2),
                                  queryType.getDimSize(1)))
    return mlir::failure();

  llvm::SmallVector<mlir::Value> dynamicSizes;
  dynamicSizes.reserve(resultType.getNumDynamicDims());
  for (int64_t dim : llvm::seq<int64_t>(0, resultType.getRank())) {
    if (!resultType.isDynamicDim(dim))
      continue;
    switch (dim) {
    case 0:
      dynamicSizes.push_back(
          mlir::tensor::DimOp::create(rewriter, loc, query, 0));
      break;
    case 1:
      dynamicSizes.push_back(
          mlir::arith::ConstantIndexOp::create(rewriter, loc, numHeads));
      break;
    case 2:
      dynamicSizes.push_back(
          mlir::tensor::DimOp::create(rewriter, loc, query, 1));
      break;
    case 3:
      dynamicSizes.push_back(totalSeqExtent);
      break;
    default:
      llvm_unreachable("rank-4 GQA QK shape has an invalid dimension");
    }
  }

  return mlir::Value(
      mlir::tensor::EmptyOp::create(rewriter, loc, resultType.getShape(),
                                    resultType.getElementType(), dynamicSizes));
}

mlir::FailureOr<mlir::RankedTensorType>
inferReduceResultType(mlir::Operation *op, mlir::Value data,
                      std::optional<llvm::ArrayRef<int64_t>> reducedAxes,
                      int64_t keepdims) {
  if (auto ranked =
          mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType()))
    return ranked;
  auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(data.getType());
  if (!inputType || !reducedAxes)
    return mlir::failure();
  mlir::FailureOr<llvm::SmallVector<int64_t>> outShape =
      mlir::hip::inferReductionShape(inputType.getShape(), *reducedAxes,
                                     keepdims);
  if (mlir::failed(outShape))
    return mlir::failure();
  return mlir::RankedTensorType::get(*outShape, inputType.getElementType());
}

mlir::FailureOr<mlir::Value> createEmptyTensorFromReifiedShape(
    mlir::OpBuilder &builder, mlir::Location loc,
    mlir::RankedTensorType resultType,
    llvm::ArrayRef<mlir::OpFoldResult> reifiedShape) {
  if (static_cast<int64_t>(reifiedShape.size()) != resultType.getRank())
    return mlir::failure();

  // Check every dimension before materializing any index value. Bailing out
  // partway through the loop below would leave stray constants behind, which
  // the contract in HipShapeUtils.h forbids.
  for (int64_t dimIdx : llvm::seq<int64_t>(resultType.getRank())) {
    if (resultType.isDynamicDim(dimIdx))
      continue;
    std::optional<int64_t> reifiedStatic =
        mlir::getConstantIntValue(reifiedShape[dimIdx]);
    if (reifiedStatic && *reifiedStatic != resultType.getDimSize(dimIdx))
      return mlir::failure();
  }

  llvm::SmallVector<mlir::Value> dynSizes;
  for (int64_t dimIdx : llvm::seq<int64_t>(resultType.getRank()))
    if (resultType.isDynamicDim(dimIdx))
      dynSizes.push_back(mlir::getValueOrCreateConstantIndexOp(
          builder, loc, reifiedShape[dimIdx]));
  return mlir::Value(
      mlir::tensor::EmptyOp::create(builder, loc, resultType.getShape(),
                                    resultType.getElementType(), dynSizes));
}

mlir::FailureOr<mlir::Value>
createSameShapeEmptyTensor(mlir::OpBuilder &builder, mlir::Location loc,
                           mlir::RankedTensorType resultType,
                           mlir::Value source) {
  mlir::FailureOr<llvm::SmallVector<mlir::OpFoldResult>> shape =
      mlir::hip::reifyElementwiseSameShape(builder, loc, source);
  if (mlir::failed(shape))
    return mlir::failure();
  return createEmptyTensorFromReifiedShape(builder, loc, resultType, *shape);
}

mlir::RankedTensorType
getTensorTypeFromReifiedShape(llvm::ArrayRef<mlir::OpFoldResult> reifiedShape,
                              mlir::Type elementType,
                              mlir::Attribute encoding) {
  llvm::SmallVector<int64_t> shape;
  shape.reserve(reifiedShape.size());
  for (mlir::OpFoldResult dim : reifiedShape)
    shape.push_back(
        mlir::getConstantIntValue(dim).value_or(mlir::ShapedType::kDynamic));
  return mlir::RankedTensorType::get(shape, elementType, encoding);
}

mlir::FailureOr<mlir::Value>
createBroadcastEmptyTensor(mlir::OpBuilder &builder, mlir::Location loc,
                           mlir::RankedTensorType resultType,
                           mlir::ValueRange operands) {
  mlir::FailureOr<llvm::SmallVector<mlir::OpFoldResult>> shape =
      mlir::hip::reifyBroadcastResultShape(
          builder, loc, operands, [&] { return mlir::emitError(loc); });
  if (mlir::failed(shape))
    return mlir::failure();
  return createEmptyTensorFromReifiedShape(builder, loc, resultType, *shape);
}

mlir::FailureOr<mlir::Value>
createReductionEmptyTensor(mlir::OpBuilder &builder, mlir::Location loc,
                           mlir::RankedTensorType resultType, mlir::Value data,
                           std::optional<llvm::ArrayRef<int64_t>> reducedAxes,
                           int64_t keepdims) {
  if (!reducedAxes)
    return createEmptyTensor(builder, loc, resultType, data);
  mlir::FailureOr<llvm::SmallVector<mlir::OpFoldResult>> shape =
      mlir::hip::reifyReductionResultShape(builder, loc, data, *reducedAxes,
                                           keepdims);
  if (mlir::failed(shape))
    return mlir::failure();
  return createEmptyTensorFromReifiedShape(builder, loc, resultType, *shape);
}

} // namespace hip
} // namespace mlir
