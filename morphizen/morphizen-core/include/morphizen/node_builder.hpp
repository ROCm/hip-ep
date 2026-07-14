/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#pragma once
#include "morphizen/anchor_point.hpp"
#include "morphizen/graph.hpp"
#include "morphizen/pass.hpp"
#include <morphizen/morphizen_gsl.h>

namespace morphizen {

/**
 * @brief The NodeBuilder class is responsible for building nodes in a graph.
 *
 * The NodeBuilder class provides methods for constructing nodes with various
 * properties, such as operation type, input nodes, attributes, shape, and data
 * type. It also supports adding multiple outputs and optional outputs.
 *
 * This is a high-level convenience API that depends on IPass and AnchorPoint,
 * which are morphizen-core concepts. It uses the morphizen-graph wrappers
 * internally but provides a fluent API for node construction.
 */
struct NodeBuilder {
public:
  /**
   * @brief Constructs a NodeBuilder object.
   *
   * @param graph The graph to which the node belongs.
   * @param pass The pass to which the node belongs.
   */
  MORPHIZEN_DLL_SPEC explicit NodeBuilder(Graph &graph, IPass &pass);

  /**
   * @brief Builds and returns the constructed node.
   *
   * @return The constructed node.
   */
  MORPHIZEN_DLL_SPEC const Node &build();

  /**
   * @brief Builds and returns the constructed node as a constant reference.
   *
   * @return The constructed node as a constant reference.
   */
  MORPHIZEN_DLL_SPEC morphizen_cxx::NodeConstRef build_ex();

  /**
   * @brief Clones the given node and sets it as the current node being built.
   *
   * @param node The node to clone.
   * @return A reference to the NodeBuilder object.
   */
  MORPHIZEN_DLL_SPEC NodeBuilder &clone_node(const Node &node);

  /**
   * @brief Clones the operation type of the given node and sets it as the
   * current node being built.
   *
   * @param node The node from which to clone the operation type.
   * @return A reference to the NodeBuilder object.
   */
  MORPHIZEN_DLL_SPEC NodeBuilder &clone_op_type(const Node &node);

  /**
   * @brief Sets the operation type of the current node being built.
   *
   * @param op_type The operation type.
   * @param domain The domain of the operation type (default: "com.xilinx").
   * @return A reference to the NodeBuilder object.
   */
  MORPHIZEN_DLL_SPEC NodeBuilder &
  set_op_type(const std::string &op_type,
              const std::string &domain = "com.xilinx");

  /**
   * @brief Clones the inputs of the given node and sets them as the inputs of
   * the current node being built.
   *
   * @param node The node from which to clone the inputs.
   * @return A reference to the NodeBuilder object.
   */
  MORPHIZEN_DLL_SPEC NodeBuilder &clone_inputs(const Node &node);

  /**
   * @brief Sets the input nodes of the current node being built.
   *
   * @param input_nodes The input nodes.
   * @return A reference to the NodeBuilder object.
   */
  MORPHIZEN_DLL_SPEC NodeBuilder &
  set_input_nodes(const std::vector<const Node *> &input_nodes);

  /**
   * @brief Sets the input node arguments of the current node being built.
   *
   * @param input_args The input node arguments.
   * @return A reference to the NodeBuilder object.
   */
  MORPHIZEN_DLL_SPEC NodeBuilder &
  set_input_node_args(const std::vector<const NodeArg *> &input_args);

  /**
   * @brief Sets the input node arguments for the NodeBuilder.
   *
   * This function sets the input node arguments for the NodeBuilder. The input
   * arguments are provided as a vector of NodeArgConstRef objects.
   *
   * @param input_args The vector of NodeArgConstRef objects representing the
   * input node arguments.
   * @return A reference to the NodeBuilder object.
   */
  MORPHIZEN_DLL_SPEC NodeBuilder &set_input_node_args_ex(
      const std::vector<morphizen_cxx::NodeArgConstRef> &input_args);
  /**
   * @brief Appends the given node as an input to the current node being built.
   *
   * @param node The node to append as an input.
   * @return A reference to the NodeBuilder object.
   */
  MORPHIZEN_DLL_SPEC NodeBuilder &append_input(const Node &node);

  /**
   * @brief Clones the attributes of the given node and sets them as the
   * attributes of the current node being built.
   *
   * @param node The node from which to clone the attributes.
   * @return A reference to the NodeBuilder object.
   */
  MORPHIZEN_DLL_SPEC NodeBuilder &clone_attrs(const Node &node);

  /**
   * @brief Returns the NodeAttributesBuilder object for modifying the
   * attributes of the current node being built.
   *
   * @return The NodeAttributesBuilder object.
   */
  MORPHIZEN_DLL_SPEC NodeAttributesBuilder &get_attrs_builder();

  /**
   * @brief Clones the shape of the given node and sets it as the shape of the
   * current node being built.
   *
   * @param node The node from which to clone the shape.
   * @return A reference to the NodeBuilder object.
   */
  MORPHIZEN_DLL_SPEC NodeBuilder &clone_shape(const Node &node);

  /**
   * @brief Sets the shape of the current node being built.
   *
   * @param shape The shape of the node.
   * @return A reference to the NodeBuilder object.
   */
  MORPHIZEN_DLL_SPEC NodeBuilder &
  set_shape(const gsl::span<const int64_t> &shape);

  /**
   * @brief Clones the shape of the given node argument and sets it as the shape
   * of the current node being built.
   *
   * @param node_arg The node argument from which to clone the shape.
   * @return A reference to the NodeBuilder object.
   */
  MORPHIZEN_DLL_SPEC NodeBuilder &clone_shape(const NodeArg &node_arg);

  /**
   * @brief Clones the data type of the given node and sets it as the data type
   * of the current node being built.
   *
   * @param node The node from which to clone the data type.
   * @return A reference to the NodeBuilder object.
   */
  MORPHIZEN_DLL_SPEC NodeBuilder &clone_data_type(const Node &node);

  /**
   * @brief Clones the data type of the given node argument and sets it as the
   * data type of the current node being built.
   *
   * @param node_arg The node argument from which to clone the data type.
   * @return A reference to the NodeBuilder object.
   */
  MORPHIZEN_DLL_SPEC NodeBuilder &clone_data_type(const NodeArg &node);

  /**
   * @brief Sets the data type of the current node being built.
   *
   * @param data_type The data type.
   * @return A reference to the NodeBuilder object.
   */
  MORPHIZEN_DLL_SPEC NodeBuilder &set_data_type(const std::string &data_type);

  /**
   * @brief Sets the anchor point of the current node being built to the given
   * node.
   *
   * @param node The node to set as the anchor point.
   * @return A reference to the NodeBuilder object.
   */
  MORPHIZEN_DLL_SPEC NodeBuilder &set_anchor_point1(const Node &node);

  /**
   * @brief Sets the anchor point of the current node being built to the given
   * node argument.
   *
   * @param node_arg The node argument to set as the anchor point.
   * @return A reference to the NodeBuilder object.
   */
  MORPHIZEN_DLL_SPEC NodeBuilder &set_anchor_point1(const NodeArg &node);

  /**
   * @brief Sets the anchor point of the current node being built to the given
   * node argument and description.
   *
   * @param node_arg The node argument to set as the anchor point.
   * @param description The description of the anchor point.
   * @return A reference to the NodeBuilder object.
   */
  MORPHIZEN_DLL_SPEC NodeBuilder &
  set_anchor_point2(const NodeArg &node_arg,
                    const AnchorPoint::Description &description);

  /**
   * @brief Sets the anchor point of the current node being built to the given
   * node, description, and shape.
   *
   * @param node_arg The node argument to set as the anchor point.
   * @param description The description of the anchor point.
   * @param shape The shape of the anchor point.
   * @return A reference to the NodeBuilder object.
   */
  MORPHIZEN_DLL_SPEC NodeBuilder &
  set_anchor_point3(const NodeArg &node_arg,
                    const AnchorPoint::Description &description,
                    const std::vector<int64_t> &shape);

  /**
   * @brief Sets the anchor point of the current node being built to the given
   * node, description, shape, and data type.
   *
   * @param node_arg The node argument to set as the anchor point.
   * @param description The description of the anchor point.
   * @param shape The shape of the anchor point.
   * @param data_type The data type of the anchor point.
   * @return A reference to the NodeBuilder object.
   */
  MORPHIZEN_DLL_SPEC NodeBuilder &set_anchor_point4(
      const NodeArg &node_arg, const AnchorPoint::Description &description,
      const std::vector<int64_t> &shape, const std::string &data_type);

  /**
   * @brief Adds an attribute with the given name and value to the current node
   * being built.
   *
   * @tparam T The type of the attribute value.
   * @param name The name of the attribute.
   * @param value The value of the attribute.
   * @return A reference to the NodeBuilder object.
   */
  template <typename T> NodeBuilder &add(const std::string &name, T &&value) {
    if (name == "data_type" || name == "shape") {
      assert(false &&
             "data_type and shape are deprecated, please use set_anchor_point, "
             "clone_shape, set_shape, clone_data_type, set_data_type instead");
    }
    attrs_builder_.add(name, std::forward<T>(value));
    return *this;
  }

  /**
   * @brief Adds an output to the current node being built.
   *
   * @return A reference to the NodeBuilder object.
   */
  MORPHIZEN_DLL_SPEC NodeBuilder &add_output();

  /**
   * @brief Skips the optional output of the current node being built.
   *
   * @return A reference to the NodeBuilder object.
   */
  MORPHIZEN_DLL_SPEC NodeBuilder &skip_optional_output();

private:
  Graph &graph_;
  IPass *pass_;
  std::string op_type_;
  std::string description_;
  std::vector<const NodeArg *> input_args_;
  NodeAttributesPtr attrs_;
  NodeAttributesBuilder attrs_builder_;
  std::string domain_;
  size_t num_of_outputs_ = 1u;
  std::vector<std::vector<int64_t>> shape_;
  std::vector<std::string> data_type_;
  std::vector<std::unique_ptr<AnchorPoint>> anchor_point_;
  std::vector<std::optional<morphizen_cxx::NodeArgConstRef>> anchor_node_arg_;
  // Cache producer nodes
  // After node addition in MLIR, original nodes become inaccessible via
  // graph_producer_node
  std::vector<std::optional<morphizen_cxx::NodeConstRef>> anchor_producer_node_;
};

} // namespace morphizen
