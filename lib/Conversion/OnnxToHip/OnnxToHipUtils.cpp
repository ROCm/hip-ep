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

} // namespace mlir::hip
