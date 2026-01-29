/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

/// @file node_arg.hpp
/// @brief C++ wrapper utilities for node argument operations over
/// MORPHIZEN_ORT_API
///
/// This file provides C++ wrappers for NodeArg-related operations. A NodeArg
/// represents a value/tensor in the computational graph (inputs and outputs
/// of nodes). Each NodeArg has:
/// - A unique name
/// - Type information (element type)
/// - Shape information (dimensions)
/// - Optional constant data (for initializers)
///
/// All operations are forwarded to the active backend implementation via
/// the MORPHIZEN_ORT_API function pointer table.
///
/// Example:
/// @code
///   const NodeArg& arg = ...;
///   std::string name = node_arg_get_name(arg);         // Get name
///   int elem_type = node_arg_get_element_type(arg);    // Get element type
///   auto shape = node_arg_get_shape_i64(arg);          // Get shape
///   bool is_dynamic = node_arg_is_dynamic_shape(arg);  // Check if dynamic
/// @endcode

#pragma once
#include <morphizen/morphizen_gsl.h>
#include <morphizen/my_ort.h>
#include <optional>
#include <ostream>
namespace morphizen {

MORPHIZEN_DLL_SPEC bool node_arg_exists(const NodeArg& node_arg);
MORPHIZEN_DLL_SPEC const std::string&
node_arg_get_name(const NodeArg& node_arg);
MORPHIZEN_DLL_SPEC std::unique_ptr<std::vector<int64_t>>
node_arg_get_shape_i64(const NodeArg& node_arg);
MORPHIZEN_DLL_SPEC std::unique_ptr<std::vector<std::string>>
node_arg_get_denotation(const NodeArg& node_arg);
MORPHIZEN_DLL_SPEC int node_arg_get_element_type(const NodeArg& node_arg);
MORPHIZEN_DLL_SPEC bool node_arg_is_unknown_shape(const NodeArg& node_arg);
MORPHIZEN_DLL_SPEC bool node_arg_is_scalar(const NodeArg& node_arg);
MORPHIZEN_DLL_SPEC bool node_arg_is_zero_shape(const NodeArg& node_arg);
MORPHIZEN_DLL_SPEC bool node_arg_is_dynamic_shape(const NodeArg& node_arg);
MORPHIZEN_DLL_SPEC const TensorProto&
node_arg_get_const_data_as_tensor(const Graph& graph, const NodeArg& node_arg);
MORPHIZEN_DLL_SPEC float
node_arg_get_const_data_as_float(const Graph& graph, const NodeArg& node_arg);
MORPHIZEN_DLL_SPEC uint8_t
node_arg_get_const_data_as_u8(const Graph& graph, const NodeArg& node_arg);
MORPHIZEN_DLL_SPEC int8_t
node_arg_get_const_data_as_i8(const Graph& graph, const NodeArg& node_arg);
MORPHIZEN_DLL_SPEC int32_t
node_arg_get_const_data_as_i32(const Graph& graph, const NodeArg& node_arg);
MORPHIZEN_DLL_SPEC uint16_t
node_arg_get_const_data_as_u16(const Graph& graph, const NodeArg& node_arg);
MORPHIZEN_DLL_SPEC int16_t
node_arg_get_const_data_as_bf16(const Graph& graph, const NodeArg& node_arg);
MORPHIZEN_DLL_SPEC int16_t
node_arg_get_const_data_as_fp16(const Graph& graph, const NodeArg& node_arg);
MORPHIZEN_DLL_SPEC gsl::span<const uint8_t>
node_arg_get_const_data_as_u4s(const Graph& graph, const NodeArg& node_arg);
MORPHIZEN_DLL_SPEC gsl::span<const int8_t>
node_arg_get_const_data_as_i4s(const Graph& graph, const NodeArg& node_arg);
MORPHIZEN_DLL_SPEC gsl::span<const uint8_t>
node_arg_get_const_data_as_u8s(const Graph& graph, const NodeArg& node_arg);
MORPHIZEN_DLL_SPEC gsl::span<const int8_t>
node_arg_get_const_data_as_i8s(const Graph& graph, const NodeArg& node_arg);
MORPHIZEN_DLL_SPEC gsl::span<const uint16_t>
node_arg_get_const_data_as_u16s(const Graph& graph, const NodeArg& node_arg);
MORPHIZEN_DLL_SPEC gsl::span<const int16_t>
node_arg_get_const_data_as_i16s(const Graph& graph, const NodeArg& node_arg);
MORPHIZEN_DLL_SPEC gsl::span<const float>
node_arg_get_const_data_as_floats(const Graph& graph, const NodeArg& node_arg);
MORPHIZEN_DLL_SPEC gsl::span<const int32_t>
node_arg_get_const_data_as_i32s(const Graph& graph, const NodeArg& node_arg);
MORPHIZEN_DLL_SPEC gsl::span<const int64_t>
node_arg_get_const_data_as_i64s(const Graph& graph, const NodeArg& node_arg);
MORPHIZEN_DLL_SPEC gsl::span<const int16_t>
node_arg_get_const_data_as_bf16s(const Graph& graph, const NodeArg& node_arg);
MORPHIZEN_DLL_SPEC gsl::span<const int16_t>
node_arg_get_const_data_as_fp16s(const Graph& graph, const NodeArg& node_arg);
MORPHIZEN_DLL_SPEC bool node_arg_is_constant(const Graph& graph,
                                             const NodeArg& node_arg);

/** @brief Create a new node argument in the graph
 *
 * @param graph The graph to add the node argument to
 * @param name Name of the node argument
 * @param shape Shape of the tensor (nullptr for unknown shape)
 * @param element_type Data type of the tensor
 * @return Reference to the created node argument
 */
MORPHIZEN_DLL_SPEC NodeArg& node_arg_new(Graph& graph, const std::string& name,
                                         const std::vector<int64_t>* shape,
                                         int element_type);

// Internal helper function (not exported) - used by C++ wrapper inline methods
std::string node_arg_as_string_internal(const NodeArg& node_arg);

} // namespace morphizen
namespace morphizen_cxx {
// on Linux, friend class declaration is not enough, we need forward
// declaration.

class GraphConstRef;
class NodeConstRef;
class NodeInput;
using bf16_t = int16_t;
using fp16_t = int16_t;
/**
 * @class NodeArgConstRef
 * @brief A reference to a constant NodeArg object.
 *
 * This class provides a read-only reference to a NodeArg object within a Graph.
 * It allows access to various properties of the NodeArg, such as its name,
 * shape, element type, etc. It also provides methods to convert the NodeArg to
 * a string representation and retrieve constant data.
 */
class MORPHIZEN_DLL_SPEC NodeArgConstRef {
  friend class GraphConstRef;
  friend class NodeConstRef;
  friend class NodeInput;

protected:
  NodeArgConstRef(const morphizen::Graph& graph, const morphizen::NodeArg& self)
      : graph_{graph}, self_{self} {}

public:
  static NodeArgConstRef from_node_arg(const morphizen::Graph& graph,
                                       const morphizen::NodeArg& self) {
    return NodeArgConstRef{graph, self};
  }
  operator const morphizen::NodeArg&() const { return self_; }
  const morphizen::NodeArg* ptr() const { return &self_; }
  /**
   * @brief Gets the name of the NodeArg.
   *
   * @return The name of the NodeArg.
   * */
  const std::string& name() const {
    return morphizen::node_arg_get_name(self_);
  }
  /**
   * @brief Overloads the equality operator for comparing two NodeArgConstRef
   * objects.
   *
   * @param other The NodeArgConstRef object to compare with.
   * @return true if the two objects are identical pointers, false otherwise.
   */
  bool operator==(const NodeArgConstRef& other) const {
    return &self_ == &other.self_;
  }
  bool operator<(const NodeArgConstRef& other) const {
    return &self_ < &other.self_;
  }
  /**
   * @brief Converts the NodeArg to a string representation.
   *
   * @return A string representation of the NodeArg.
   * */
  std::string to_string() const {
    return morphizen::node_arg_as_string_internal(self_);
  }
  /**
   * @brief Checks if the node argument is a graph input.
   *
   * @return true if the node argument is a graph input, false otherwise.
   */
  bool is_graph_input() const;
  /**
   * @brief Checks if the node argument is an output of a graph.
   *
   * @return true if the node argument is an output of a graph, false otherwise.
   */
  bool is_graph_output() const;
  /**
   * @brief Gets the shape of the NodeArg.
   *
   * @return A vector of integers representing the shape of the NodeArg.
   * */
  std::unique_ptr<std::vector<int64_t>> shape() const {
    return morphizen::node_arg_get_shape_i64(self_);
  }
  /**
   * @brief Gets the denotation of the NodeArg.
   *
   * @return A vector of strings representing the denotation of the NodeArg.
   * */
  std::unique_ptr<std::vector<std::string>> denotation() const {
    return morphizen::node_arg_get_denotation(self_);
  }
  /**
   * Gets the element type of the node argument.
   *
   * @return The element type of the node argument.
   *
   * here is a list of valid onnx element types:
   * 0: UNDEFINED
   * 1: FLOAT
   * 2: UINT8
   * 3: INT8
   * 4: UINT16
   * 5: INT16
   * 6: INT32
   * 7: INT64
   * 8: STRING
   * 9: BOOL
   * 10: FLOAT16
   * 11: DOUBLE
   * 12: UINT32
   * 13: UINT64
   * 14: COMPLEX64
   * 15: COMPLEX128
   * 16: BFLOAT16
   * 17: FLOAT32
   */
  int element_type() const {
    return morphizen::node_arg_get_element_type(self_);
  }
  /**
   * Checks if the shape of the node argument is unknown.
   *
   * @return True if the shape is unknown, false otherwise.
   */
  bool is_unknown_shape() const {
    return morphizen::node_arg_is_unknown_shape(self_);
  }
  /**
   * Checks if the node argument is a scalar.
   *
   * @return True if the node argument is a scalar, false otherwise.
   */
  bool is_scalar() const { return morphizen::node_arg_is_scalar(self_); }
  /**
   * Checks if the shape of the node argument is zero.
   *
   * @return True if the shape is zero, false otherwise.
   */
  bool is_zero_shape() const {
    return morphizen::node_arg_is_zero_shape(self_);
  }
  /**
   * Checks if the shape of the node argument is dynamic.
   *
   * @return True if the shape is dynamic, false otherwise.
   */
  bool is_dynamic_shape() const {
    return morphizen::node_arg_is_dynamic_shape(self_);
  }
  /**
   * Checks if the node argument is constant.
   *
   * @return True if the node argument is constant, false otherwise.
   */
  bool is_constant() const {
    return morphizen::node_arg_is_constant(graph_, self_);
  }
  /**
   * Finds the consumers of the current node arg.
   *
   * @return A vector of NodeConstRef objects representing the consumers of the
   * current node.
   */
  std::vector<NodeConstRef> find_consumers() const;
  // In the header file (node_arg.hpp)

  /**
   * Retrieves constant data as an int8_t value.
   * @return The constant data converted to int8_t.
   */
  int8_t const_data_as_i8() const;

  /**
   * Retrieves constant data as a uint8_t value.
   * @return The constant data converted to uint8_t.
   */
  uint8_t const_data_as_u8() const;

  /**
   * Retrieves constant data as an int16_t value.
   * @return The constant data converted to int16_t.
   */
  int16_t const_data_as_i16() const;

  /**
   * Retrieves constant data as a uint16_t value.
   * @return The constant data converted to uint16_t.
   */
  uint16_t const_data_as_u16() const;

  /**
   * Retrieves constant data as an int32_t value.
   * @return The constant data converted to int32_t.
   */
  int32_t const_data_as_i32() const;

  /**
   * Retrieves constant data as a uint32_t value.
   * @return The constant data converted to uint32_t.
   */
  uint32_t const_data_as_u32() const;

  /**
   * Retrieves constant data as an int64_t value.
   * @return The constant data converted to int64_t.
   */
  int64_t const_data_as_i64() const;

  /**
   * Retrieves constant data as a uint64_t value.
   * @return The constant data converted to uint64_t.
   */
  uint64_t const_data_as_u64() const;

  /**
   * Retrieves constant data as a float value.
   * @return The constant data converted to float.
   */
  float const_data_as_f32() const;
  /**
   * Retrieves constant data as a double value.
   * @return The constant data converted to double.
   */
  double const_data_as_f64() const;

  /**
   * Retrieves constant data as an bf16 value.
   * @return The constant data converted to int16_t.
   */
  bf16_t const_data_as_bf16() const;
  /**
   * @brief Returns the constant data as a half-precision floating-point value.
   *
   * @return The constant data as a half-precision floating-point value.
   */
  fp16_t const_data_as_fp16() const;
  /**
   * @brief Retrieves the node argument's data as a read-only span of uint8_t.
   *
   * This method provides access to the underlying data of a NodeArg in a
   * type-safe manner, allowing for the data to be viewed as a span of uint8_t
   * without copying it.
   *
   * @return gsl::span<const uint8_t> A span representing a view into the node's
   * data as int8_t.
   */
  gsl::span<const uint8_t> const_data_as_u8_span() const;
  /**
   * @brief Retrieves the node argument's data as a read-only span of int8_t.
   *
   * This method provides access to the underlying data of a NodeArg in a
   * type-safe manner, allowing for the data to be viewed as a span of int8_t
   * without copying it.
   *
   * @return gsl::span<const int8_t> A span representing a view into the node's
   * data as int8_t.
   */
  gsl::span<const int8_t> const_data_as_i8_span() const;

  /**
   * @brief Retrieves the node argument's data as a read-only span of uint16_t.
   *
   * This method provides access to the underlying data of a NodeArg in a
   * type-safe manner, allowing for the data to be viewed as a span of uint16_t
   * without copying it.
   *
   * @return gsl::span<const uint16_t> A span representing a view into the
   * node's data as uint16_t.
   */
  gsl::span<const uint16_t> const_data_as_u16_span() const;

  /**
   * @brief Retrieves the node argument's data as a read-only span of int16_t.
   *
   * This method provides access to the underlying data of a NodeArg in a
   * type-safe manner, allowing for the data to be viewed as a span of int16_t
   * without copying it.
   *
   * @return gsl::span<const int16_t> A span representing a view into the node's
   * data as int16_t.
   */
  gsl::span<const int16_t> const_data_as_i16_span() const;

  /**
   * @brief Retrieves the node argument's data as a read-only span of int32_t.
   *
   * This method provides access to the underlying data of a NodeArg in a
   * type-safe manner, allowing for the data to be viewed as a span of int32_t
   * without copying it.
   *
   * @return gsl::span<const int32_t> A span representing a view into the node's
   * data as int32_t.
   */
  gsl::span<const int32_t> const_data_as_i32_span() const;

  /**
   * @brief Retrieves the node argument's data as a read-only span of uint32_t.
   *
   * This method provides access to the underlying data of a NodeArg in a
   * type-safe manner, allowing for the data to be viewed as a span of uint32_t
   * without copying it.
   *
   * @return gsl::span<const uint32_t> A span representing a view into the
   * node's data as uint32_t.
   */
  gsl::span<const uint32_t> const_data_as_u32_span() const;

  /**
   * @brief Retrieves the node argument's data as a read-only span of int64_t.
   *
   * This method provides access to the underlying data of a NodeArg in a
   * type-safe manner, allowing for the data to be viewed as a span of int64_t
   * without copying it.
   *
   * @return gsl::span<const int64_t> A span representing a view into the node's
   * data as int64_t.
   */
  gsl::span<const int64_t> const_data_as_i64_span() const;

  /**
   * @brief Retrieves the node argument's data as a read-only span of uint64_t.
   *
   * This method provides access to the underlying data of a NodeArg in a
   * type-safe manner, allowing for the data to be viewed as a span of uint64_t
   * without copying it.
   *
   * @return gsl::span<const uint64_t> A span representing a view into the
   * node's data as uint64_t.
   */
  gsl::span<const uint64_t> const_data_as_u64_span() const;

  /**
   * @brief Retrieves the node argument's data as a read-only span of float.
   *
   * This method provides access to the underlying data of a NodeArg in a
   * type-safe manner, allowing for the data to be viewed as a span of float
   * without copying it.
   *
   * @return gsl::span<const float> A span representing a view into the node's
   * data as float.
   */
  gsl::span<const float> const_data_as_f32_span() const;

  /**
   * @brief Retrieves the node argument's data as a read-only span of double.
   *
   * This method provides access to the underlying data of a NodeArg in a
   * type-safe manner, allowing for the data to be viewed as a span of double
   * without copying it.
   *
   * @return gsl::span<const double> A span representing a view into the node's
   * data as double.
   */
  gsl::span<const double> const_data_as_f64_span() const;

  /**
   * @brief Retrieves the node argument's data as a read-only span of bf16.
   *
   * This method provides access to the underlying data of a NodeArg in a
   * type-safe manner, allowing for the data to be viewed as a span of int16_t
   * without copying it.
   *
   * @return gsl::span<const int16_t> A span representing a view into the node's
   * data as int16_t.
   */
  gsl::span<const bf16_t> const_data_as_bf16_span() const;
  /**
   * @brief Returns a read-only span of `fp16_t` elements representing the
   * constant data.
   *
   * This function returns a `gsl::span` object that provides a read-only view
   * of the constant data stored as `fp16_t` elements. The span allows accessing
   * the data without making a copy.
   *
   * @return A read-only span of `fp16_t` elements representing the constant
   * data.
   */
  gsl::span<const fp16_t> const_data_as_fp16_span() const;
  gsl::span<const char> const_data_as_raw() const;
  /**
   * Finds the producer node of the current node argument.
   *
   * @return A reference to the producer node.
   *
   * For graph inputs and constant initializers, they have no
   * producers, so that std::nullopt is returned.
   */
  std::optional<NodeConstRef> find_producer() const;
  /**
   * Overloads the << operator to allow printing the node argument.
   *
   * @param str The output stream.
   * @param self The node argument to be printed.
   * @return The output stream.
   */
  friend std::ostream& operator<<(std::ostream& str,
                                  const NodeArgConstRef& self) {
    return str << self.to_string();
  }

private:
  const morphizen::Graph& graph_;
  const morphizen::NodeArg& self_;
};
class MORPHIZEN_DLL_SPEC NodeArgRef : public NodeArgConstRef {
  friend class GraphRef;

private:
  NodeArgRef(morphizen::Graph& graph, morphizen::NodeArg& self)
      : NodeArgConstRef{graph, self} {}
};
} // namespace morphizen_cxx
