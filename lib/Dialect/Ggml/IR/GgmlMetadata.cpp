/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/Ggml/IR/GgmlDialect.h"
#include "hip/Dialect/Ggml/IR/GgmlSymbols.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSwitch.h"

#include <optional>
#include <string>

using namespace mlir;
using namespace mlir::ggml;

namespace {

std::optional<int64_t> metadataInteger(Attribute attribute) {
  auto integer = dyn_cast_or_null<IntegerAttr>(attribute);
  if (!integer)
    return std::nullopt;
  auto type = dyn_cast<IntegerType>(integer.getType());
  if (!integer.getType().isIndex() &&
      (!type || !type.isSignless() || type.getWidth() == 1))
    return std::nullopt;
  return integer.getValue().trySExtValue();
}

// Bounds are optional, and producer-owned fields remain in their original
// dictionaries. Validate the declarations before shape or interface consumers
// read their integers or interpret their input mappings.
LogicalResult verifySymbols(Operation *op, Attribute attribute) {
  std::string reason;
  auto symbols = readSymbolDeclarations(attribute, reason);
  if (failed(symbols))
    return op->emitOpError(reason);
  for (auto [ordinal, symbol] : llvm::enumerate(*symbols)) {
    const std::string description =
        "ggml.symbols[" + std::to_string(ordinal) + "]";
    DictionaryAttr entry = symbol.attributes;

    if (Attribute rawProduct = entry.get("product")) {
      auto product = dyn_cast<ArrayAttr>(rawProduct);
      if (!product || product.size() != 2 ||
          llvm::any_of(product, [](Attribute factor) {
            auto name = dyn_cast<StringAttr>(factor);
            return !name || name.getValue().empty();
          }))
        return op->emitOpError()
               << description << ".product must name two symbols";
    }
    const auto verifyInput = [&](DictionaryAttr place,
                                 const std::string &where) -> LogicalResult {
      auto input = place.getAs<StringAttr>("input");
      if (!input || input.getValue().empty())
        return op->emitOpError() << where << ".input must be a nonempty string";
      // (Dimension 4: where the view begins.)
      auto dimension = metadataInteger(place.get("dimension"));
      if (!dimension || *dimension < 0 || *dimension > 4)
        return op->emitOpError()
               << where << ".dimension must be an integer in [0, 4]";
      if (Attribute rawScale = place.get("scale")) {
        auto scale = metadataInteger(rawScale);
        if (!scale || *scale <= 0)
          return op->emitOpError()
                 << where
                 << ".scale must be a positive integer representable as int64";
      }
      return success();
    };
    if (entry.get("input") && failed(verifyInput(entry, description)))
      return failure();
    if (Attribute rawInputs = entry.get("inputs")) {
      auto inputs = dyn_cast<ArrayAttr>(rawInputs);
      if (!inputs)
        return op->emitOpError() << description << ".inputs must be an array";
      for (auto [index, rawInput] : llvm::enumerate(inputs)) {
        const std::string where =
            description + ".inputs[" + std::to_string(index) + "]";
        auto input = dyn_cast<DictionaryAttr>(rawInput);
        if (!input)
          return op->emitOpError() << where << " must be a dictionary";
        if (failed(verifyInput(input, where)))
          return failure();
      }
    }
  }
  return success();
}

LogicalResult verifyValueMetadata(Operation *op, NamedAttribute attribute) {
  const StringRef name = attribute.getName().getValue();
  if (name == "ggml.value_id") {
    const auto integer = dyn_cast<IntegerAttr>(attribute.getValue());
    if (!integer || !integer.getType().isSignlessInteger(64) ||
        integer.getValue().isNegative())
      return op->emitOpError()
             << "requires '" << name << "' to be a nonnegative signless i64";
    return success();
  }
  const bool isStringField =
      llvm::StringSwitch<bool>(name)
          .Cases({"ggml.source_name", "ggml.source_kind"}, true)
          .Cases({"ggml.io_name", "ggml.io_kind"}, true)
          .Default(false);
  if (isStringField && !isa<StringAttr>(attribute.getValue()))
    return op->emitOpError() << "requires '" << name << "' to be a string";
  return success();
}

} // namespace

LogicalResult GgmlDialect::verifyOperationAttribute(Operation *op,
                                                    NamedAttribute attribute) {
  if (isa<ModuleOp>(op) && attribute.getName().getValue() == "ggml.symbols")
    return verifySymbols(op, attribute.getValue());
  return verifyValueMetadata(op, attribute);
}

LogicalResult GgmlDialect::verifyRegionArgAttribute(Operation *op, unsigned,
                                                    unsigned,
                                                    NamedAttribute attribute) {
  return verifyValueMetadata(op, attribute);
}

LogicalResult
GgmlDialect::verifyRegionResultAttribute(Operation *op, unsigned, unsigned,
                                         NamedAttribute attribute) {
  return verifyValueMetadata(op, attribute);
}
