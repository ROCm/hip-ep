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
