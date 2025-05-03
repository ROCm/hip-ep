/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// env_config.hpp must be included before pass.hpp
// otherwise it is an error, in pass_context.hpp
//
// return ::morphizen::env_config_helper<
//     decltype(env_name::value)>::from_string(p);
//
// we cannot add env_config.hpp into vaip.hpp, otherwise macro
// ENV_PARAM conflicts with vart
#include "morphizen/node_input.hpp"
#include "morphizen/env_config.hpp"
#include <cstdint>

namespace vaip_cxx {
NodeInput::NodeInput(const GraphConstRef graph,
                     const vaip_core::NodeArg& node_arg,
                     const vaip_core::Node* node)
    : graph_(graph), node_arg_(NodeArgConstRef(graph, node_arg)),
      node_(node == nullptr
                ? std::nullopt
                : std::optional<NodeConstRef>(NodeConstRef(graph, *node))) {}

vaip_cxx::NodeArgConstRef NodeInput::as_node_arg() const { return node_arg_; }
std::optional<vaip_cxx::NodeConstRef> NodeInput::as_node() const {
  return node_;
}
std::string NodeInput::to_string() const {
  if (node_.has_value()) {
    return node_.value().to_string();
  }
  return node_arg_.to_string();
}
} // namespace vaip_cxx
