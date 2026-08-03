/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- HipShapeUtilsInternal.cpp - Private shape utility details ----------===//

#include "HipShapeUtilsInternal.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/SmallBitVector.h"

using namespace mlir;

namespace {

enum class PermutationErrorKind {
  None,
  RankMismatch,
  OutOfRange,
  Duplicate,
};

struct PermutationError {
  PermutationErrorKind kind = PermutationErrorKind::None;
  int64_t value = 0;
};

PermutationError checkPermutation(ArrayRef<int64_t> perm, int64_t rank) {
  if (static_cast<int64_t>(perm.size()) != rank)
    return {PermutationErrorKind::RankMismatch, 0};

  llvm::SmallBitVector seen(rank);
  for (int64_t value : perm) {
    if (value < 0 || value >= rank)
      return {PermutationErrorKind::OutOfRange, value};
    if (seen.test(value))
      return {PermutationErrorKind::Duplicate, value};
    seen.set(value);
  }
  return {};
}

} // namespace

ArrayRef<int64_t> mlir::hip::detail::getShapeOf(Value value) {
  if (auto tensorType = dyn_cast<RankedTensorType>(value.getType()))
    return tensorType.getShape();
  if (auto memrefType = dyn_cast<MemRefType>(value.getType()))
    return memrefType.getShape();
  return {};
}

SmallVector<int64_t> mlir::hip::detail::getI64Array(ArrayAttr attr) {
  SmallVector<int64_t> values;
  values.reserve(attr.size());
  for (Attribute value : attr)
    values.push_back(cast<IntegerAttr>(value).getInt());
  return values;
}

LogicalResult mlir::hip::detail::validatePermutation(
    ArrayRef<int64_t> perm, int64_t rank,
    function_ref<InFlightDiagnostic()> emitError) {
  PermutationError error = checkPermutation(perm, rank);
  switch (error.kind) {
  case PermutationErrorKind::None:
    return success();
  case PermutationErrorKind::RankMismatch:
    emitError() << "perm length (" << perm.size() << ") must match input rank ("
                << rank << ")";
    return failure();
  case PermutationErrorKind::OutOfRange:
    emitError() << "perm value " << error.value << " is out of range";
    return failure();
  case PermutationErrorKind::Duplicate:
    emitError() << "perm must be a permutation (duplicate value " << error.value
                << ")";
    return failure();
  }
  llvm_unreachable("unknown permutation validation result");
}

LogicalResult mlir::hip::detail::validatePermutation(ArrayRef<int64_t> perm,
                                                     int64_t rank) {
  return success(checkPermutation(perm, rank).kind ==
                 PermutationErrorKind::None);
}

mlir::hip::detail::GatherBlockQuantizedStorageFlags
mlir::hip::detail::getGatherBlockQuantizedStorageFlags(int64_t bits,
                                                       Type dataElementType) {
  return {/*bytePackedInt4=*/bits == 4,
          /*uint8Storage=*/bits == 8 || dataElementType.isUnsignedInteger(8)};
}
