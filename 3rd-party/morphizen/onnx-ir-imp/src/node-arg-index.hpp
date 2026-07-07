/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once
#include "./graph-id.hpp"
#include "./onnx-deps.hpp"

#include <cstddef>
#include <functional>
namespace morphizen {

// Forward declaration
class NodeIndex;
class Graph;
/**
 * @brief Index for referencing node arguments (values) in a graph
 *
 * This class represents an index to a specific value/argument within a graph.
 * Values can be graph inputs, initializers, or outputs from nodes.
 */
class NodeArgIndex {
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
  // from_morphizen_core_node_arg_ptr is used to create a NodeArgIndex from a
  // morphizen::NodeArg pointer, which is used in the ORT C API
  static NodeArgIndex from_morphizen_core_node_arg_ptr(const void* ptr);

public:
  // Default constructor
  NodeArgIndex();
  // Constructor with arguments
  NodeArgIndex(unsigned int index, Type type);

  // Constructor with graph_id
  NodeArgIndex(unsigned int index, Type type, GraphId graph_id);

  // Copy constructor
  NodeArgIndex(const NodeArgIndex& other) = default;

  // Move constructor
  NodeArgIndex(NodeArgIndex&& other) noexcept = default;

  // Copy assignment operator
  NodeArgIndex& operator=(const NodeArgIndex& other) = default;

  // Move assignment operator
  NodeArgIndex& operator=(NodeArgIndex&& other) noexcept = default;
  // Destructor
  ~NodeArgIndex() = default;

  // Validity check
  bool is_valid(const Graph& graph) const;
  bool is_valid() const;
  bool is_valid_graph_input() const;
  bool is_valid_initializer() const;
  bool is_valid_node_output() const;
  bool is_valid_graph_output() const;
  bool is_graph_input() const;
  bool is_initializer() const;
  bool is_node_output() const;
  bool is_graph_output() const;
  // Equality operator
  bool operator==(const NodeArgIndex& other) const;

  // Inequality operator
  bool operator!=(const NodeArgIndex& other) const;

  // Hash function
  std::size_t hash() const; // Getters
  unsigned int get_index() const;
  Type get_type() const;
  GraphId get_graph_id() const; // Member functions
  bool exists() const;
  std::vector<int64_t>* get_shape_i64_unsafe() const;
  void set_shape_i64(const std::vector<int64_t>& shape);
  std::vector<std::string>* get_denotation_unsafe() const;
  void set_denotation(const std::vector<std::string>& denotation);
  const std::string* get_name_unsafe() const;
  int get_element_type() const;
  int external_location(std::string& external_file, size_t& offset,
                        size_t& size, size_t& checksum) const;
  // Get the producer node for this node argument
  NodeIndex get_producer_node() const;
  std::string get_name() const;
  std::string to_string() const;
  // Get constant tensor data (only valid for initializers)
  const morphizen_onnx::TensorProto*
  get_const_data_as_tensor(const Graph& g) const;
  // Static factory methods
  static NodeArgIndex invalid();
  static NodeArgIndex graph_input(unsigned int index, GraphId graph_id);
  static NodeArgIndex initializer(unsigned int index, GraphId graph_id);
  static NodeArgIndex node_output(unsigned int index, GraphId graph_id);
  static NodeArgIndex graph_output(unsigned int index, GraphId graph_id);

  // Conversion method
  const void* to_morphizen_core_node_arg_ptr() const;
  const morphizen_onnx::ValueInfoProto& get_value_info() const;

private:
  // Helper method to extract shape from tensor type
  static std::vector<int64_t>* extract_shape_from_tensor_type(
      const ::morphizen_onnx::TensorShapeProto& shape);

  // Helper method to extract shape from value info
  static std::vector<int64_t>* extract_shape_from_value_info(
      const ::morphizen_onnx::ValueInfoProto& value_info);

  // Helper method to extract shape from initializer tensor
  static std::vector<int64_t>* extract_shape_from_initializer(
      const ::morphizen_onnx::TensorProto& initializer);

  // Helper method to extract denotation from value info
  static std::vector<std::string>* extract_denotation_from_value_info(
      const ::morphizen_onnx::ValueInfoProto& value_info);

  union {
    struct {
      unsigned int index_ : 29;
      unsigned int type_ : 3;
      unsigned int graph_id_ : 32;
    } fields_;
    uint64_t value_; // 64 bits total: 29 for index, 3 for type, 32 for graph_id
  };
};                   // class NodeArgIndex

static_assert(8 == sizeof(NodeArgIndex), "NodeArgIndex type must be 8 bytes");

} // namespace morphizen

// Standard library hash specialization for NodeArgIndex
namespace std {
template <> struct hash<morphizen::NodeArgIndex> {
  std::size_t operator()(const morphizen::NodeArgIndex& index) const {
    return index.hash();
  }
};
} // namespace std
