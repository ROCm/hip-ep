/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once
#include "./mlir-graph-id.hpp"
#include "./mlir-node-arg.hpp"
#include <cstddef>
#include <functional>
#include <optional>
#include <string>
#include <vector>

// Forward declarations for MLIR types
namespace mlir {
class Operation;
}

namespace morphizen {
namespace mlir_impl {

// Forward declarations
class MLIRGraph;

/**
 * @brief Index for referencing node arguments (values) in a graph
 *
 * This class represents an index to a specific value/argument within a graph.
 * Values can be graph inputs, initializers, or outputs from nodes.
 */
class MLIRNodeArgIndex {
public:
  enum class Type : int {
    INVALID = 0, // Represents an invalid index
    GRAPH_INPUT =
        1,       // Represents a graph input value, index to GraphProto_.input
    INITIALIZER =
        2, // Represents a graph initializer, index to GraphProto_.value_info
    NODE_OUTPUT = 3, // Represents an output value from a node, index to
                     // GraphProto_.value_info
    GRAPH_OUTPUT = 4 // Represents a graph output, index to
                     // GraphProto_.output
  };
  // from_morphizen_core_node_arg_ptr is used to create a MLIRNodeArgIndex from
  // a morphizen::NodeArg pointer, which is used in the ORT C API
  static MLIRNodeArgIndex from_morphizen_core_node_arg_ptr(const void* ptr);

  // Round-trip the raw 64-bit payload, used when encoding/decoding an index
  // into an IntegerAttr (e.g. morphizen.node_inputs).
  static MLIRNodeArgIndex from_uint64(uint64_t v);

public:
  // Default constructor
  MLIRNodeArgIndex();
  // Constructor with arguments
  MLIRNodeArgIndex(unsigned int index, Type type);

  // Constructor with graph_id
  MLIRNodeArgIndex(unsigned int index, Type type, GraphId graph_id);

  // Copy constructor
  MLIRNodeArgIndex(const MLIRNodeArgIndex& other) = default;

  // Move constructor
  MLIRNodeArgIndex(MLIRNodeArgIndex&& other) noexcept = default;

  // Copy assignment operator
  MLIRNodeArgIndex& operator=(const MLIRNodeArgIndex& other) = default;

  // Move assignment operator
  MLIRNodeArgIndex& operator=(MLIRNodeArgIndex&& other) noexcept = default;
  // Destructor
  ~MLIRNodeArgIndex() = default;

  // Validity check
  bool is_valid() const;

  /// Check if the underlying node argument represents a constant value
  /// This method delegates to MLIRNodeArg::isConstantValue()
  bool is_constant() const;

  // Conversion to bool operator
  explicit operator bool() const { return is_valid(); }

  // Equality operator
  bool operator==(const MLIRNodeArgIndex& other) const;

  // Inequality operator
  bool operator!=(const MLIRNodeArgIndex& other) const;

  // Hash function
  std::size_t hash() const; // Getters
  unsigned int get_index() const;
  Type get_type() const;
  GraphId get_graph_id() const; // Member functions
  bool exists() const;

  std::optional<llvm::SmallVector<int64_t>> get_shape_i64() const;
  void set_shape_i64(const llvm::SmallVector<int64_t>& shape);
  std::vector<std::string>* get_denotation_unsafe() const;
  void set_denotation(const std::vector<std::string>& denotation);
  int get_element_type() const;
  void set_element_type(int element_type);
  int external_location(std::string& external_file, size_t& offset,
                        size_t& size, size_t& checksum) const;

  /**
   * @brief Get the producer node for this node argument
   * @return Pointer to the operation that produces this node argument, or
   * nullptr if none
   */
  mlir::Operation* get_producer_node() const;

  /**
   * @brief Get the MLIRNodeArg (throws if not holding MLIRNodeArg)
   * @return Reference to MLIRNodeArg
   */
  const MLIRNodeArg& get_node_arg() const;
  const MLIRNodeArg& get_const_data_as_tensor() const;
  // Get the graph this index belongs to
  const MLIRGraph& get_graph() const;

  const std::string& get_name() const;
  std::string to_string() const;
  // Static factory methods
  static MLIRNodeArgIndex invalid();
  static MLIRNodeArgIndex graph_input(unsigned int index, GraphId graph_id);
  static MLIRNodeArgIndex initializer(unsigned int index, GraphId graph_id);
  static MLIRNodeArgIndex node_output(unsigned int index, GraphId graph_id);
  static MLIRNodeArgIndex graph_output(unsigned int index, GraphId graph_id);

  // Conversion method
  const void* to_morphizen_core_node_arg_ptr() const;

  // Raw 64-bit payload, used when encoding into an IntegerAttr.
  uint64_t to_uint64() const;

private:
  union {
    struct {
      uint64_t index_ : 29;
      uint64_t type_ : 3;
      uint64_t graph_id_ : 32;
    } fields_;
    uint64_t value_; // 64 bits total: 29 for index, 3 for type, 32 for graph_id
  };
};                   // class MLIRNodeArgIndex

static_assert(8 == sizeof(MLIRNodeArgIndex),
              "MLIRNodeArgIndex type must be 8 bytes");

} // namespace mlir_impl
} // namespace morphizen

// Standard library hash specialization for MLIRNodeArgIndex
namespace std {
template <> struct hash<morphizen::mlir_impl::MLIRNodeArgIndex> {
  std::size_t
  operator()(const morphizen::mlir_impl::MLIRNodeArgIndex& index) const {
    return index.hash();
  }
};
} // namespace std
