/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "./symbol-table.hpp"
#include "mlir/Support/LogicalResult.h"
#include <glog/logging.h>
#include <stdexcept>

namespace mlir_impl {

bool MLIRSymbolTable::insert(const std::string& name, mlir::Value value) {
  // Check if name already exists
  if (name_to_value_.find(name) != name_to_value_.end()) {
    return false;
  }

  void* opaque_ptr = getOpaquePointer(value);

  // Insert both mappings
  name_to_value_[name] = value;
  value_to_name_[opaque_ptr] = name;

  return true;
}

std::optional<mlir::Value>
MLIRSymbolTable::lookup(const std::string& name) const {
  auto it = name_to_value_.find(name);
  if (it != name_to_value_.end()) {
    return it->second;
  }
  return std::nullopt;
}

std::optional<std::string> MLIRSymbolTable::lookup(mlir::Value value) const {
  void* opaque_ptr = getOpaquePointer(value);
  auto it = value_to_name_.find(opaque_ptr);
  if (it != value_to_name_.end()) {
    return it->second;
  }
  return std::nullopt;
}

mlir::Value MLIRSymbolTable::getValue(const std::string& name) const {
  auto result = lookup(name);
  if (!result) {
    throw std::runtime_error("Symbol not found: " + name);
  }
  return *result;
}

std::string MLIRSymbolTable::getName(mlir::Value value) const {
  auto result = lookup(value);
  if (!result) {
    throw std::runtime_error("Value not found in symbol table");
  }
  return *result;
}

bool MLIRSymbolTable::contains(const std::string& name) const {
  return name_to_value_.find(name) != name_to_value_.end();
}

bool MLIRSymbolTable::contains(mlir::Value value) const {
  void* opaque_ptr = getOpaquePointer(value);
  return value_to_name_.find(opaque_ptr) != value_to_name_.end();
}

bool MLIRSymbolTable::replace(mlir::Value old_value, mlir::Value new_value) {
  old_value.replaceAllUsesWith(new_value);
  // CHECK(result.succeeded()) << "Failed to replace old value with new value";
  void* old_opaque_ptr = getOpaquePointer(old_value);
  auto it = value_to_name_.find(old_opaque_ptr);
  CHECK(it != value_to_name_.end()) << "Old value not found in symbol table";
  std::string old_name = it->second;
  // Remove old mappings
  name_to_value_.erase(old_name);
  value_to_name_.erase(it);
  // Insert new mappings with the same name
  return insert(old_name, new_value);
}

bool MLIRSymbolTable::erase(const std::string& name) {
  auto it = name_to_value_.find(name);
  if (it == name_to_value_.end()) {
    return false;
  }

  // Remove from both maps
  void* opaque_ptr = getOpaquePointer(it->second);
  value_to_name_.erase(opaque_ptr);
  name_to_value_.erase(it);

  return true;
}

bool MLIRSymbolTable::erase(mlir::Value value) {
  void* opaque_ptr = getOpaquePointer(value);
  auto it = value_to_name_.find(opaque_ptr);
  if (it == value_to_name_.end()) {
    return false;
  }

  // Remove from both maps
  const std::string& name = it->second;
  name_to_value_.erase(name);
  value_to_name_.erase(it);

  return true;
}

void MLIRSymbolTable::clear() {
  name_to_value_.clear();
  value_to_name_.clear();
}

size_t MLIRSymbolTable::size() const { return name_to_value_.size(); }

bool MLIRSymbolTable::empty() const { return name_to_value_.empty(); }

void* MLIRSymbolTable::getOpaquePointer(mlir::Value value) const {
  return value.getAsOpaquePointer();
}

} // namespace mlir_impl
