/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once

#include "mlir/IR/Value.h"
#include <optional>
#include <string>
#include <unordered_map>

namespace mlir_impl {

class MLIRSymbolTable {
public:
  MLIRSymbolTable() = default;
  ~MLIRSymbolTable() = default;

  // Non-copyable but movable
  MLIRSymbolTable(const MLIRSymbolTable &) = delete;
  MLIRSymbolTable &operator=(const MLIRSymbolTable &) = delete;
  MLIRSymbolTable(MLIRSymbolTable &&) = default;
  MLIRSymbolTable &operator=(MLIRSymbolTable &&) = default;

  /// Insert a mapping from name to value
  /// Returns true if inserted, false if name already exists
  bool insert(const std::string &name, mlir::Value value);

  /// Lookup value by name
  /// Returns the mlir::Value if found, std::nullopt otherwise
  std::optional<mlir::Value> lookup(const std::string &name) const;

  /// Lookup name by value
  /// Returns the name if found, std::nullopt otherwise
  std::optional<std::string> lookup(mlir::Value value) const;

  /// Get value by name (throws if not found)
  mlir::Value getValue(const std::string &name) const;

  /// Get name by value (throws if not found)
  std::string getName(mlir::Value value) const;

  /// Check if name exists
  bool contains(const std::string &name) const;

  /// Check if value exists
  bool contains(mlir::Value value) const;

  /// Remove mapping by name
  /// Returns true if removed, false if not found
  bool erase(const std::string &name);

  /// Remove mapping by value
  /// Returns true if removed, false if not found
  bool erase(mlir::Value value);

  /// replace an value
  bool replace(mlir::Value old_value, mlir::Value new_value);

  /// Clear all mappings
  void clear();

  /// Get number of mappings
  size_t size() const;

  /// Check if empty
  bool empty() const;

private:
  // Map from name to value
  std::unordered_map<std::string, mlir::Value> name_to_value_;

  // Map from value (as opaque pointer) to name
  std::unordered_map<void *, std::string> value_to_name_;

  /// Helper to get opaque pointer from mlir::Value
  void *getOpaquePointer(mlir::Value value) const;
};

} // namespace mlir_impl
