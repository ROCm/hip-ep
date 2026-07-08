/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

/// @file node.hpp
/// @brief C++ wrapper utilities for node operations over MORPHIZEN_ORT_API
///
/// This file provides C++ wrappers for node-related operations in the
/// computational graph. A node represents an operator/operation in the
/// graph (e.g., Conv, Relu, MatMul).
///
/// All operations are forwarded to the active backend implementation via
/// the MORPHIZEN_ORT_API function pointer table.
///
/// Example:
/// @code
///   const Node& node = ...;
///   std::string op = node_op_type(node);           // Get operation type
///   auto inputs = node_get_inputs(node);            // Get input nodes
///   auto attrs = node_get_attributes_ref(node);     // Get node attributes
///   bool is_conv = node_is_op(node, "Conv", "");   // Check if Conv op
/// @endcode

#pragma once
#include "./_sanity_check.hpp"
#include "./node_arg.hpp"
#include <morphizen/dll_safe.h>
#include <morphizen/morphizen_gsl.h>
#include <morphizen/my_ort.h>
#include <optional>
#include <vector>
namespace morphizen {
std::vector<const NodeArg *> node_get_input_node_args(const Node &node);
std::vector<const AttributeProto *> node_get_attributes(const Node &node);
MORPHIZEN_DLL_SPEC const NodeAttributes &
node_get_attributes_ref(const Node &node);
std::vector<int64_t> node_get_output_shape(const Node &node, int index);

// Node input/output accessors
std::vector<NodeInput> node_get_inputs(const Node &node);
const NodeArg &node_get_output_node_arg(const Node &node);
std::vector<const NodeArg *> node_get_output_node_args(const Node &node);
const std::string &node_get_first_output_name(const Node &node);

MORPHIZEN_DLL_SPEC const std::string &node_get_output_name(const Node &node);
MORPHIZEN_DLL_SPEC bool node_is_op(const Node &node, const std::string &op_type,
                                   const std::string &domain);
MORPHIZEN_DLL_SPEC int node_get_output_element_type(const Node &node);
MORPHIZEN_DLL_SPEC const AttributeProto *node_get_attr(const Node &node,
                                                       const std::string &name);
MORPHIZEN_DLL_SPEC bool node_has_attr(const Node &node,
                                      const std::string &name);

// Node attribute getters
MORPHIZEN_DLL_SPEC int64_t node_get_attr_int(const Node &node,
                                             const std::string &name);
MORPHIZEN_DLL_SPEC gsl::span<const int64_t>
node_get_attr_ints(const Node &node, const std::string &name);

// Node string representation
std::string node_as_string(const Node &node);

// Node op type and domain
MORPHIZEN_DLL_SPEC const std::string &node_op_type(const Node &node);
MORPHIZEN_DLL_SPEC const std::string &node_op_domain(const Node &node);

MORPHIZEN_DLL_SPEC const AttributeProto *
node_attributes_get(const NodeAttributes &attributes, const std::string &name);

MORPHIZEN_DLL_SPEC NodeAttributesPtr node_clone_attributes(const Node &node);

} // namespace morphizen

namespace morphizen_cxx {
class NodeInput;
/**
 * @class NodeConstRef
 * @brief A class that provides a constant reference wrapper around a node,
 * offering access to its inputs, outputs, and a string representation.
 *
 * This class acts as a reference or wrapper around another node-like
 * structure, possibly for graph manipulation or analysis purposes. It
 * provides methods to access the node's inputs and outputs, convert the node
 * to its underlying type, and obtain a string representation for output
 * purposes.
 */
class MORPHIZEN_DLL_SPEC NodeConstRef {
  friend class GraphConstRef;
  friend class NodeArgConstRef;
  friend class NodeInput;

public:
  static NodeConstRef from_node(const morphizen::Graph &graph,
                                const morphizen::Node &node) {
    return NodeConstRef(graph, node);
  }

  NodeConstRef() = delete;
  NodeConstRef &operator=(const NodeConstRef &) = default;
  /**
   * @brief Checks if this NodeConstRef is equal to another NodeConstRef.
   *
   * Two NodeConstRef objects are considered equal if they refer to the same
   * Node object.
   *
   * @param other The NodeConstRef to compare with.
   * @return True if the NodeConstRef objects are equal, false otherwise.
   */
  bool operator==(const NodeConstRef &other) const {
    return self_ == other.self_;
  }
  bool operator<(const NodeConstRef &other) const {
    return self_ < other.self_;
  }
  /**
   * @brief Get the name of the node.
   *
   * @return const std::string& The name of the node.
   *
   * @note The name of a node is a unique identifier assigned to it by the
   * graph. it is only optional so that it can be empty. it is only used for
   * diagnostic purposes. It most likely is not as same as the name of it's
   * first output node arg.
   */
  const std::string &name() const;
  /**
   * @brief Returns a vector of constant references to NodeArg objects,
   * representing the inputs to the node.
   * @return std::vector<NodeArgConstRef> A vector of constant references to
   * NodeArg objects.
   *
   * @note input could be optional, so it could be empty.
   */
  std::vector<std::optional<NodeArgConstRef>> inputs() const;

  /**
   * @brief Returns node inputs as NodeInput structs for compatibility.
   * @return std::vector<NodeInput> A vector of NodeInput structs.
   *
   * This method provides compatibility with code that expects NodeInput
   * structs. NodeInput contains {Node* producer, NodeArg* node_arg} for each
   * input.
   */
  std::vector<morphizen::NodeInput> inputs_as_node_input() const;

  /**
   * @brief Outer-scope SSA values used inside this node's nested regions
   * (Loop/If/Scan bodies). Empty vector on nodes without regions or
   * without captures.
   */
  std::vector<std::optional<NodeArgConstRef>> implicit_inputs() const;

  /**
   * @brief Explicit operands followed by implicit captures. Equivalent to
   * `inputs()` concatenated with `implicit_inputs()`.
   */
  std::vector<std::optional<NodeArgConstRef>> all_inputs() const;

  /**
   * @brief Gets the index of the node within the graph.
   * @return int The index of the node.
   *
   * The index of a node is a unique identifier assigned to it by the graph
   */
  size_t index() const;

  /**
   * @brief Gets the operation type of the node.
   * @return std::string The operation type of the node.
   *
   * the op type is the name of the operation that the node represents
   */
  const std::string &op_type() const;

  /**
   * @brief Gets the operation domain of the node.
   * @return std::string The operation domain of the node.
   *
   * the op domain is the domain of the operation that the node represents
   */
  const std::string &op_domain() const;

  /**
   * Checks if the node has the specified attribute.
   *
   * @param attr_name The name of the attribute to check.
   * @return True if the node has the attribute, false otherwise.
   */
  bool has_attr(const std::string &attr_name) const;
  /**
   * Retrieves the integer value of the specified attribute.
   *
   * @param attr_name The name of the attribute.
   * @return The integer value of the attribute.
   */
  int64_t get_attr_int(const std::string &attr_name) const;
  /**
   * Retrieves the value of an attribute as an integer.
   *
   * This function retrieves the value of the attribute specified by `attr_name`
   * as an integer. If the attribute does not exist or cannot be converted to an
   * integer, the `default_value` is returned.
   *
   * @param attr_name The name of the attribute to retrieve.
   * @param default_value The default value to return if the attribute does not
   * exist or cannot be converted to an integer.
   * @return The value of the attribute as an integer, or the `default_value` if
   * the attribute does not exist or cannot be converted.
   */
  int64_t get_attr_int(const std::string &attr_name,
                       int64_t default_value) const;
  /**
   * Retrieves a span of `int64_t` values associated with the specified
   * attribute name.
   *
   * @param attr_name The name of the attribute.
   * @return A `gsl::span<int64_t>` containing the attribute values.
   */
  gsl::span<const int64_t> get_attr_ints(const std::string &attr_name) const;
  /**
   * Retrieves the attribute values as a span of int64_t values.
   *
   * @param attr_name The name of the attribute.
   * @param default_value The default value to be returned if the attribute is
   * not found.
   * @return A span of int64_t values representing the attribute values.
   */
  gsl::span<const int64_t>
  get_attr_ints(const std::string &attr_name,
                const std::vector<int64_t> &default_value) const;
  /**
   * Retrieves the value of the specified attribute as a float.
   *
   * @param attr_name The name of the attribute to retrieve.
   * @return The value of the attribute as a float.
   */
  float get_attr_float(const std::string &attr_name) const;
  /**
   * Retrieves the value of the specified attribute as a float.
   *
   * @param attr_name The name of the attribute to retrieve.
   * @param default_value The default value to return if the attribute is not
   * found.
   * @return The value of the attribute as a float, or the default value if the
   * attribute is not found.
   */
  float get_attr_float(const std::string &attr_name, float default_value) const;

  /**
   * Retrieves the attribute values as a span of constant floats.
   *
   * @param attr_name The name of the attribute.
   * @return A gsl::span<const float> containing the attribute values.
   */
  gsl::span<const float> get_attr_floats(const std::string &attr_name) const;
  /**
   * Retrieves the attribute values as a span of constant floats.
   *
   * @param attr_name The name of the attribute.
   * @param default_value The default value to return if the attribute is not
   * found.
   * @return A span of constant floats representing the attribute values.
   */
  gsl::span<const float>
  get_attr_floats(const std::string &attr_name,
                  const std::vector<float> &default_value) const;

  /**
   * Retrieves the attribute value as a string.
   *
   * @param name The name of the attribute.
   * @return The attribute value as a string.
   */
  const std::string &get_attr_string(const std::string &name) const;
  /**
   * Retrieves the value of the specified attribute as a string.
   *
   * @param name The name of the attribute to retrieve.
   * @param default_value The default value to return if the attribute is not
   * found.
   * @return The value of the attribute as a string, or the default value if the
   * attribute is not found.
   */
  const std::string &get_attr_string(const std::string &name,
                                     const std::string &default_value) const;

  morphizen::DllSafe<std::string>
  release_attr_string(const std::string &name) const;

  std::vector<std::string> get_attr_strings(const std::string &name) const;
  std::vector<std::string>
  get_attr_strings(const std::string &name,
                   const std::vector<std::string> &default_value) const;
  /**
   * @brief Returns a vector of constant references to NodeArg objects,
   * representing the outputs of the node.
   * @return std::vector<NodeArgConstRef> A vector of constant references to
   * NodeArg objects.
   *
   * @note output could be optional, so it could be empty.
   */
  std::vector<std::optional<NodeArgConstRef>> outputs() const;

  /**
   * @brief Gets the first output NodeArg of the node.
   * @return const morphizen::NodeArg& Reference to the first output NodeArg.
   *
   * Checks that the node has at least one output. Throws if no outputs exist.
   */
  const morphizen::NodeArg &first_output_node_arg() const;

  /**
   *@brief Return the function body of the node.
   *@return GraphConstRef The function body of the node.
   */
  GraphConstRef get_function_body() const;

  /**
   * @brief Type conversion operator to const morphizen::Node&.
   * Allows instances of NodeConstRef to be implicitly converted to a constant
   * reference of morphizen::Node, providing read-only access to the underlying
   * node.
   * @return const morphizen::Node& A constant reference to the underlying
   * morphizen::Node object.
   */
  operator const morphizen::Node &() const { return *self_; }
  /**
   * Returns a pointer to the underlying `morphizen::Node` object.
   *
   * @return A const pointer to the `morphizen::Node` object.
   */
  const morphizen::Node *ptr() const { return self_; }
  /**
   * Returns a reference to the graph associated with this node.
   *
   * @return A const reference to the graph.
   */
  const morphizen::Graph &graph() const { return *graph_; }
  /**
   * @brief Converts the node's state or relevant information into a string
   * representation. Useful for debugging, logging, or any scenario where a
   * textual representation of the node is needed.
   * @return std::string The string representation of the node.
   */
  std::string to_string() const;

  /**
   * @brief Overloads the insertion (<<) operator for std::ostream, enabling the
   * direct output of the node's string representation to output streams.
   * @param os The output stream to which the string representation is written.
   * @param ref A constant reference to the instance of NodeConstRef.
   * @return std::ostream& The output stream, enabling the chaining of output
   * operations.
   */
  MORPHIZEN_DLL_SPEC friend std::ostream &operator<<(std::ostream &os,
                                                     const NodeConstRef &ref);

protected:
  NodeConstRef(const morphizen::Graph &graph, const morphizen::Node &self)
      : graph_{&graph}, self_{&self} {}

protected:
  const morphizen::Graph *graph_;
  const morphizen::Node *self_;
};
class NodeRef : public NodeConstRef {
  friend class GraphRef;
  friend class NodeArgRef;

public:
  static NodeRef from_node(morphizen::Graph &graph, morphizen::Node &node) {
    return NodeRef(graph, node);
  }
  operator morphizen::Node &() const {
    return const_cast<morphizen::Node &>(*self_);
  }
  /**
   * Returns a pointer to the underlying `morphizen::Node` object.
   *
   * @return A const pointer to the `morphizen::Node` object.
   */
  morphizen::Node *ptr() { return &(operator morphizen::Node &()); }

private:
  NodeRef(morphizen::Graph &graph, morphizen::Node &self)
      : NodeConstRef(graph, self) {}
  morphizen::Graph &graph() {
    return const_cast<morphizen::Graph &>(this->graph());
  }
};
} // namespace morphizen_cxx
