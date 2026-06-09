/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- HipInferTypeUtils.cpp - inferReturnTypes specialization helpers ----===//
//
// Implementation of the helpers declared in `HipInferTypeUtils.h`. These back
// the hand-written `inferReturnTypes` of HIP DPS ops whose result set the
// auto-generated inference cannot produce (currently the multi-output ops).
// Keeping the bodies here -- instead of inline in HipDialect.cpp -- collects
// every such specialization in one place as more are added.
//
//===----------------------------------------------------------------------===//

#include "hip/Dialect/IR/HipInferTypeUtils.h"

#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;
using namespace mlir::hip;

LogicalResult hip::inferDpsInitReturnTypes(TypeRange initTypes,
                                           SmallVectorImpl<Type> &results) {
  for (Type t : initTypes)
    if (auto ranked = dyn_cast<RankedTensorType>(t))
      results.push_back(ranked);
  return success();
}
