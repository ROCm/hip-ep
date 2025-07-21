/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "./node.hpp"
#include "./node-arg-index.hpp"
#include "./node-index.hpp"
namespace morphizen {

// Factory method implementation
Node Node::create_node(const NodeIndex& self,
                       const std::vector<NodeArgIndex>& inputs,
                       const std::vector<NodeArgIndex>& outputs) {
  return Node(PrivateTag{}, self, inputs, outputs);
}

// Constructor implementation
Node::Node(PrivateTag /*tag*/, const NodeIndex& self,
           const std::vector<NodeArgIndex>& inputs,
           const std::vector<NodeArgIndex>& outputs)
    : self_(self), inputs_(inputs), outputs_(outputs) {
  // Empty implementation for now - placeholder for future node operations
}

// Node API method implementations

// Accessor method implementations

const NodeIndex& Node::get_self() const { return self_; }

const std::vector<NodeArgIndex>& Node::get_inputs() const { return inputs_; }

const std::vector<NodeArgIndex>& Node::get_outputs() const { return outputs_; }

} // namespace morphizen
