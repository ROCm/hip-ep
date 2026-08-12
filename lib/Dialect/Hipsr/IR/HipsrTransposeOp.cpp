/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/Hipsr/IR/HipsrOps.h"

#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::hipsr;

MutableOperandRange TransposeOp::getDpsInitsMutable() {
  return getInitMutable();
}

LogicalResult TransposeOp::verify() {
  auto inputType = cast<ShapedType>(getInput().getType());
  auto outputType = cast<ShapedType>(getInit().getType());
  ArrayRef<int64_t> perm = getPerm();

  if (static_cast<int64_t>(perm.size()) != inputType.getRank()) {
    return emitOpError("perm length must equal the input rank; expected ")
           << inputType.getRank() << ", got " << perm.size();
  }
  if (!isPermutationVector(perm)) {
    return emitOpError("perm must be a permutation of [0, rank)");
  }
  if (inputType.getElementType() != outputType.getElementType()) {
    return emitOpError("input and output element types must match");
  }

  SmallVector<int64_t> permutedShape =
      applyPermutation(inputType.getShape(), perm);
  if (failed(verifyCompatibleShape(permutedShape, outputType.getShape()))) {
    return emitOpError("output shape must be the input shape permuted by perm");
  }
  return success();
}
