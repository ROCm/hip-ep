/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "stat.hpp"
#include <set>
#include <string>

namespace vaip_core {

// Minimal stub implementations - statistics feature removed
thread_local StatProto stat_proto;

StatProto& get_stat_proto() { return stat_proto; }

void clean_stat() { stat_proto.Clear(); }

std::set<std::string>& get_vitis_ep_custom_ops() {
  static std::set<std::string> g_vitis_ep_custom_ops;
  return g_vitis_ep_custom_ops;
}

void collect_stat(const onnxruntime::Graph& /* graph */,
                  const ContextProto& /* context_proto */) {
  // Statistics collection disabled - protobuf objects kept readonly
}

} // namespace vaip_core
