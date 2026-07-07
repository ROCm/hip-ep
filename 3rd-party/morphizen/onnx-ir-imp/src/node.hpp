/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once
#include "./node-arg-index.hpp"
#include "./node-index.hpp"
#include <memory>
#include <vector>

namespace morphizen {

/**
 * @brief In-memory representation of ONNX Node
 *
 * This class provides operations on ONNX node structure.
 * Currently serves as a placeholder for node operations.
 */
class Node {
private: // Private tag to prevent direct construction
  struct PrivateTag {};
  // Member variables to store self, inputs and outputs
  NodeIndex self_;
  std::vector<NodeArgIndex> inputs_;
  std::vector<NodeArgIndex> outputs_;

public: // Factory method to create a Node instance
  static Node create_node(const NodeIndex &self,
                          const std::vector<NodeArgIndex> &inputs = {},
                          const std::vector<NodeArgIndex> &outputs = {});
  // Constructor is private to enforce use of factory methods
  Node(PrivateTag /*tag*/, const NodeIndex &self,
       const std::vector<NodeArgIndex> &inputs,
       const std::vector<NodeArgIndex> &outputs);

public: // Node API methods
  const NodeIndex &get_self() const;
  const std::vector<NodeArgIndex> &get_inputs() const;
  const std::vector<NodeArgIndex> &get_outputs() const;

  // Empty class for now - placeholder for future node operations
};

} // namespace morphizen
