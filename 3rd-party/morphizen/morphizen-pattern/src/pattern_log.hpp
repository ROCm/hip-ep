/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once
#include "morphizen/node.hpp"
#include "morphizen/node_arg.hpp"
#include <glog/logging.h>

// Simple logging for pattern matching (debug level can be controlled via glog
// flags)
#define MY_LOG(n) LOG_IF(INFO, false)
#define MATCH_FAILED MY_LOG(1) << "MATCH FAILED. ID=" << get_id() << ";"
namespace morphizen {
[[maybe_unused]] static std::string node_input_as_string(const Graph& graph,
                                                         const NodeInput& ni) {
  if (ni.node) {
    return morphizen_cxx::NodeConstRef::from_node(graph, *ni.node).to_string();
  } else if (ni.node_arg) {
    return morphizen_cxx::NodeArgConstRef::from_node_arg(graph, *ni.node_arg)
        .to_string();
  }
  return "nil";
}
inline std::string normalize_domain(const std::string& domain) {
  return (domain == "ai.onnx") || (domain == "onnx") ? "" : domain;
}
} // namespace morphizen
