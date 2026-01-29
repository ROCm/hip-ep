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
#define MY_LOG(n) LOG(INFO)
#define MATCH_FAILED MY_LOG(1) << "MATCH FAILED. ID=" << get_id() << ";"
namespace morphizen {
[[maybe_unused]] static std::string node_input_as_string(const NodeInput& ni) {
  if (ni.node) {
    return node_as_string(*ni.node);
  } else if (ni.node_arg) {
    return node_arg_as_string(*ni.node_arg);
  }
  return "nil";
}
inline std::string normalize_domain(const std::string& domain) {
  return (domain == "ai.onnx") || (domain == "onnx") ? "" : domain;
}
} // namespace morphizen
