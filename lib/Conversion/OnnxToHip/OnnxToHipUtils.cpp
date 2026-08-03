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
