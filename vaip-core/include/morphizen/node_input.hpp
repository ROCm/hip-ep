/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
/**
 * @file node_input.hpp
 * @brief a node input represent a node argument
 * @date 2022-02-24
 *
 * An input could potentially be one of following
 * 1. a node output
 * 2. a constant initializer
 * 3. a graph input
 */
#pragma once
#include "./_sanity_check.hpp"
#include "./graph.hpp"
#include "./node.hpp"
#include "./node_arg.hpp"
#include <optional>
#include <ostream>
#include <string>
namespace vaip_core {
class Binder;
}
/**
 * @brief The namespace vaip_cxx contains classes and functions related to the
 * VAIP library in C++.
 */
namespace vaip_cxx {
/**
 * @brief The NodeInput class represents an input to a node in a graph.
 *
 * It provides methods to access information about the input, such as the
 * associated node argument and the graph it belongs to.
 */
class VAIP_DLL_SPEC NodeInput {
  friend class vaip_core::Binder;

private:
  /**
   * @brief Constructs a NodeInput object.
   *
   * @param graph The graph that the input belongs to.
   * @param node_arg The node argument associated with the input.
   * @param node The node that the input is connected to.
   */
  NodeInput(const GraphConstRef graph, const vaip_core::NodeArg& node_arg,
            const vaip_core::Node* node);

public:
  /**
   * @brief Returns the node argument associated with the input.
   *
   * @return A constant reference to the node argument.
   */
  NodeArgConstRef as_node_arg() const;

  /**
   * @brief Returns the node that the input is connected to, if any.
   *
   * @return An optional reference to the connected node. If the input is not
   * connected to any node, an empty optional is returned.
   */
  std::optional<NodeConstRef> as_node() const;

  /**
   * @brief Returns a string representation of the NodeInput object.
   *
   * @return A string representation of the NodeInput object.
   */
  std::string to_string() const;

  /**
   * @brief Overloads the << operator to allow printing a NodeInput object to an
   * output stream.
   *
   * @param stream The output stream to write to.
   * @param node_input The NodeInput object to be printed.
   * @return The output stream after writing the NodeInput object.
   */
  friend std::ostream& operator<<(std::ostream& stream,
                                  const NodeInput& node_input) {
    return stream << node_input.to_string();
  }

private:
  GraphConstRef graph_;      // The graph that the input belongs to.
  NodeArgConstRef node_arg_; // The node argument associated with the input.
  std::optional<NodeConstRef>
      node_;                 // The node that the input is connected to, if any.
};
} // namespace vaip_cxx
