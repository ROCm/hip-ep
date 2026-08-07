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

} // namespace hip
} // namespace mlir
