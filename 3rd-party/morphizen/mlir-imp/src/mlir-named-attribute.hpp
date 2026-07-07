/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#pragma once

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/StringRef.h"
#include <memory>
#include <string>

namespace morphizen {
namespace mlir_impl {

// Forward declarations
class MLIRNodeArg;
class MLIRGraph;

/**
 * @brief MLIR-based implementation of a named attribute
 *
 * This class provides a wrapper around MLIR's NamedAttribute, allowing
 * easy access to attribute names and values while maintaining compatibility
 * with the MorphiZen ORT API. It supports both owned and unowned MLIR contexts.
 */
class MLIRNamedAttribute : public mlir::NamedAttribute {
public:
  using NamedAttribute::NamedAttribute;

  /**
   * @brief Factory method to create attribute
   * @param name The attribute name
   * @param data values
   * @return Unique pointer to new NamedAttribute
   */
  // clang-format off
  static std::unique_ptr<mlir::NamedAttribute>
  create_int_array(const std::string& name, const std::vector<int64_t>& data);
  static std::unique_ptr<mlir::NamedAttribute>
  create_float_array(const std::string& name, const std::vector<float>& data);
  static std::unique_ptr<mlir::NamedAttribute>
  create_string_array(const std::string& name, const std::vector<std::string>& data);
  static std::unique_ptr<mlir::NamedAttribute>
  create_int(const std::string& name, int64_t value);
  static std::unique_ptr<mlir::NamedAttribute>
  create_float(const std::string& name, float value);
  static std::unique_ptr<mlir::NamedAttribute>
  create_string(const std::string& name, const std::string& value);
  static std::unique_ptr<mlir::NamedAttribute>
  create_tensor(const std::string& name, const MLIRNodeArg& value);
  // mlir-imp backend of attr_proto_new_graph; see ext header for the
  // caller contract. Encodes the raw MLIRGraph* in a DictionaryAttr
  // marker that add_node identifies and drops from the real op attrs.
  static std::unique_ptr<mlir::NamedAttribute>
  create_subgraph_ref(const std::string& name, MLIRGraph& sub);
  // clang-format on

  // gets
  int64_t get_int() const;
  double get_float() const;
  const std::string &get_string() const;
  std::vector<int64_t> get_ints() const;
  const std::vector<float> &get_floats() const;
  std::vector<std::string> get_strings() const;
  MLIRGraph *get_subgraph_ref() const;

  /**
   * @brief Get the ONNX attribute type for this MLIR attribute
   * @return ONNX attribute type enum value
   */
  int get_onnx_type() const;

  // sets
  void set_name(const std::string &name);
};

} // namespace mlir_impl
} // namespace morphizen
