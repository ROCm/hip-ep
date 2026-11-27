/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===----------------------------------------------------------------------===//
// Helpers shared by the hip.transpose canonicalization patterns.
//===----------------------------------------------------------------------===//

#include "HipTransposeUtilsInternal.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"

using namespace mlir;
using namespace mlir::hip;

bool hip::detail::getPermutation(TransposeOp transpose,
                                 SmallVectorImpl<int64_t> &perm) {
  for (Attribute entry : transpose.getPerm()) {
    auto value = dyn_cast<IntegerAttr>(entry);
    if (!value)
      return false;
    perm.push_back(value.getInt());
  }
  return !perm.empty();
}

bool hip::detail::permutesIntoResultType(Value source, Value result,
                                         ArrayRef<int64_t> perm) {
  auto sourceType = dyn_cast<RankedTensorType>(source.getType());
  auto resultType = dyn_cast<RankedTensorType>(result.getType());
  if (!sourceType || !resultType ||
      sourceType.getElementType() != resultType.getElementType() ||
      sourceType.getRank() != resultType.getRank() ||
      static_cast<int64_t>(perm.size()) != resultType.getRank())
    return false;
  return llvm::all_of(llvm::seq<int64_t>(0, resultType.getRank()),
                      [&](int64_t i) {
                        int64_t from = sourceType.getDimSize(perm[i]);
                        int64_t to = resultType.getDimSize(i);
                        return ShapedType::isDynamic(from) ||
                               ShapedType::isDynamic(to) || from == to;
                      });
}
