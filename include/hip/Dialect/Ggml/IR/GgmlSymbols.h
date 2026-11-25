/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef HIP_DIALECT_GGML_IR_GGMLSYMBOLS_H
#define HIP_DIALECT_GGML_IR_GGMLSYMBOLS_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"

#include <cstdint>
#include <optional>
#include <string>

namespace mlir::ggml {

/// Bounds are independently optional. Consumers that need a bounded or positive
/// interval must establish that requirement themselves. Retain the original
/// dictionary so reading bounds does not discard product declarations, input
/// mappings, or any other fields owned by their consumers.
struct SymbolDeclaration {
  DictionaryAttr attributes;
  StringAttr name;
  std::optional<int64_t> minimum;
  std::optional<int64_t> maximum;
};

/// An absent attribute and an empty array both contain no declarations. Callers
/// that distinguish them must retain the original attribute's presence.
inline FailureOr<SmallVector<SymbolDeclaration>>
readSymbolDeclarations(Attribute attribute, std::string &reason) {
  reason.clear();
  if (!attribute)
    return SmallVector<SymbolDeclaration>{};
  auto symbols = dyn_cast<ArrayAttr>(attribute);
  if (!symbols) {
    reason = "ggml.symbols must be an array";
    return failure();
  }

  SmallVector<SymbolDeclaration> result;
  result.reserve(symbols.size());
  llvm::StringSet<> names;
  for (auto [ordinal, entryAttribute] : llvm::enumerate(symbols)) {
    const std::string description =
        "ggml.symbols[" + std::to_string(ordinal) + "]";
    auto entry = dyn_cast<DictionaryAttr>(entryAttribute);
    if (!entry) {
      reason = description + " must be a dictionary";
      return failure();
    }
    auto name = entry.getAs<StringAttr>("name");
    if (!name || name.getValue().empty()) {
      reason = description + ".name must be a nonempty string";
      return failure();
    }
    if (!names.insert(name.getValue()).second) {
      reason = description + ".name duplicates symbol '" +
               name.getValue().str() + "'";
      return failure();
    }

    const auto readBound = [&](StringRef field,
                               std::optional<int64_t> &bound) -> LogicalResult {
      Attribute raw = entry.get(field);
      if (!raw)
        return success();
      auto integer = dyn_cast<IntegerAttr>(raw);
      auto type =
          integer ? dyn_cast<IntegerType>(integer.getType()) : IntegerType();
      const bool validType =
          integer && (integer.getType().isIndex() ||
                      (type && type.isSignless() && type.getWidth() != 1));
      const auto value =
          validType ? integer.getValue().trySExtValue() : std::nullopt;
      if (!value || *value < 0) {
        reason = description + "." + field.str() +
                 " must be a nonnegative integer representable as int64";
        return failure();
      }
      bound = *value;
      return success();
    };

    std::optional<int64_t> minimum;
    std::optional<int64_t> maximum;
    if (failed(readBound("minimum", minimum)) ||
        failed(readBound("maximum", maximum)))
      return failure();
    if (minimum && maximum && *minimum > *maximum) {
      reason = description + ".minimum must not exceed maximum";
      return failure();
    }
    result.push_back({entry, name, minimum, maximum});
  }
  return result;
}

inline const SymbolDeclaration *
findSymbolDeclaration(ArrayRef<SymbolDeclaration> symbols, StringRef name) {
  for (const SymbolDeclaration &symbol : symbols) {
    if (symbol.name.getValue() == name)
      return &symbol;
  }
  return nullptr;
}

} // namespace mlir::ggml

#endif // HIP_DIALECT_GGML_IR_GGMLSYMBOLS_H
