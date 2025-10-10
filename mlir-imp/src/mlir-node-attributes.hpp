/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#pragma once

#include "mlir-named-attribute.hpp"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/SmallVector.h"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace morphizen {
namespace mlir_impl {

/**
 * @brief MLIR-based implementation of node attributes for ONNX operations
 *
 * This class provides a lightweight reference wrapper for MLIR operation
 * attributes, allowing direct access to node attributes without copying.
 * Similar to MLIRNode, this class directly references the underlying MLIR
 * operation's attributes, providing zero-copy access to all attribute data.
 *
 * All attribute access methods directly delegate to the underlying MLIR
 * operation, ensuring consistency and performance.
 */
class MLIRNodeAttributes : public mlir::Op<MLIRNodeAttributes> {
public:
  using Op::Op;

  /**
   * @brief Construct an empty MLIRNodeAttributes object
   */
  static mlir::Operation* Create();
  bool has_attribute(const std::string& name) const;
  /**
   * @brief Add an attribute using MLIRNamedAttribute
   * @param named_attr MLIRNamedAttribute containing name and value
   */
  void add(const mlir::NamedAttribute& named_attr);

  /**
   * @brief Get all attribute names (excludes internal ones)
   * @return Vector of attribute names
   */
  std::vector<std::string> get_attribute_names() const;

  /**
   * @brief Get the MLIR attribute by name
   * @param name Attribute name
   * @return MLIR attribute or nullptr if not found
   */
  const MLIRNamedAttribute& get_mlir_attribute(const std::string& name) const;

  /**
   * @brief Get all MLIR attributes as a dictionary
   * @return MLIR DictionaryAttr containing all attributes
   */
  mlir::DictionaryAttr get_mlir_dictionary() const;

  std::vector<std::string>
  get_attribute_as_strings(const std::string& name) const;
};

} // namespace mlir_impl
} // namespace morphizen
