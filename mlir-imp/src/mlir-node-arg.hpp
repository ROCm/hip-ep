/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once

#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallVector.h"
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace morphizen {
namespace mlir_impl {

class alignas(8) MLIRNodeArg {
public:
  using shape_t = llvm::SmallVector<int64_t>;
  // Non-copyable but movable
  MLIRNodeArg(const MLIRNodeArg&) = delete;
  MLIRNodeArg& operator=(const MLIRNodeArg&) = delete;
  MLIRNodeArg(MLIRNodeArg&&) = default;
  MLIRNodeArg& operator=(MLIRNodeArg&&) = default;

  /// Constructor for tensor argument (no data)
  MLIRNodeArg(const std::string& name, const shape_t& shape, int element_type);

  /// Constructor for concrete tensor (with data)
  MLIRNodeArg(const std::string& name, const shape_t& shape, int element_type,
              const void* data, size_t data_size);

  /// Constructor from MLIR Value (extracts shape and type from value)
  MLIRNodeArg(const std::string& name, mlir::Value value);

  /// Get the argument name
  const std::string& getName() const;

  /// Get the shape
  const shape_t& getShape() const;

  void setShape(const shape_t& shape);

  /// Get the element type
  int getElementType() const;
  void setElementType(int data_type);

  /// Get the MLIR value (mutable)
  const mlir::Value& getValue() const;
  mlir::Value& getValue();

  /// Set the MLIR value (mutable)
  void setValue(mlir::Value value) const;

  /// Get the MLIR type based on element type or value type
  mlir::Type getType(mlir::OpBuilder& builder) const;

  // === Data access methods (for concrete tensors) ===

  /// Get raw data pointer (returns nullptr for tensor arguments)
  const void* getData() const;

  /// Get data size in bytes
  size_t getDataSize() const;

  /// Check if this tensor has data storage
  bool hasData() const;

  // === Utility methods ===

  /// Get total number of elements
  int64_t getElementCount() const;

  /// Get size of a single element in bytes
  size_t getElementSize() const;

  /// Check if the held mlir::Value is a constant
  /// Returns true if the value is produced by a constant operation
  /// (e.g., arith.constant, std.constant, or has ConstantLike trait)
  bool isConstantValue() const;

  // === Template methods for typed data access ===

private:
  const std::string name_;
  // before Operation created , the NodeArg can be changed
  mutable shape_t shape_;
  mutable int element_type_;
  mutable mlir::Value value_;

  // Optional data storage - only present for concrete tensors
  mutable std::optional<std::vector<uint8_t>> data_store_;

  void validateElementType(int element_type) const;
  void copyData(const void* data, size_t size);
};

// === Type alias for backward compatibility ===
// This allows existing MLIRTensor code to work unchanged
using MLIRTensor = MLIRNodeArg;

} // namespace mlir_impl
} // namespace morphizen
