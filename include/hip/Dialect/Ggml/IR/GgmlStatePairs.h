/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===- GgmlStatePairs.h - Checked state metadata ------------------*- C++
//-*-===//

#ifndef HIP_DIALECT_GGML_IR_GGMLSTATEPAIRS_H
#define HIP_DIALECT_GGML_IR_GGMLSTATEPAIRS_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <string>

namespace mlir::ggml {

/// A structurally valid pair of function inputs and results. Aliasing, which
/// tensors are states and whether the result may reuse the input's
/// allocation are checked by the consumers, not implied by this record.
struct StatePair {
  DictionaryAttr attributes;
  StringAttr name;
  unsigned inputIndex;
  unsigned outputIndex;
};

/// Read the shared state-pair contract without indexing SSA values. An absent
/// attribute means no pairs; a malformed present attribute is an error. Bounds
/// refer to the function signature, so callers accessing a return operation
/// must also verify its structure. Diagnostics are left to the caller.
inline FailureOr<SmallVector<StatePair>> readStatePairs(func::FuncOp function,
                                                        std::string &reason) {
  reason.clear();
  Attribute attribute = function->getAttr("ggml.state_pairs");
  if (!attribute)
    return SmallVector<StatePair>{};
  auto pairs = dyn_cast<ArrayAttr>(attribute);
  if (!pairs) {
    reason = "ggml.state_pairs must be an array";
    return failure();
  }

  SmallVector<StatePair> result;
  result.reserve(pairs.size());
  for (auto [ordinal, entryAttribute] : llvm::enumerate(pairs)) {
    const std::string description =
        "ggml.state_pairs[" + std::to_string(ordinal) + "]";
    auto pair = dyn_cast<DictionaryAttr>(entryAttribute);
    if (!pair) {
      reason = description + " must be a dictionary";
      return failure();
    }
    auto name = pair.getAs<StringAttr>("name");
    if (!name || name.getValue().empty()) {
      reason = description + ".name must be a nonempty string";
      return failure();
    }

    // getInt() asserts for signed/unsigned types and for wide values that do
    // not fit int64_t. Validate the type and use checked APInt conversion
    // before comparing bounds or narrowing to the unsigned SSA index type.
    auto readIndex = [&](StringRef field, unsigned count,
                         unsigned &index) -> LogicalResult {
      auto integer = pair.getAs<IntegerAttr>(field);
      auto type =
          integer ? dyn_cast<IntegerType>(integer.getType()) : IntegerType();
      if (!type || !type.isSignless() || type.getWidth() == 1) {
        reason = description + "." + field.str() +
                 " must be a signless integer index";
        return failure();
      }
      auto value = integer.getValue().trySExtValue();
      if (!value || *value < 0 || static_cast<uint64_t>(*value) >= count) {
        reason = description + "." + field.str() + " must be in [0, " +
                 std::to_string(count) + ")";
        return failure();
      }
      index = static_cast<unsigned>(*value);
      return success();
    };

    unsigned inputIndex = 0;
    unsigned outputIndex = 0;
    if (failed(
            readIndex("input_index", function.getNumArguments(), inputIndex)) ||
        failed(
            readIndex("output_index", function.getNumResults(), outputIndex)))
      return failure();

    result.push_back({pair, name, inputIndex, outputIndex});
  }
  return result;
}

} // namespace mlir::ggml

#endif // HIP_DIALECT_GGML_IR_GGMLSTATEPAIRS_H
