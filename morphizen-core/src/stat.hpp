/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once
#include "morphizen/morphizen.hpp"
#include <set>
#include <string>
namespace morphizen {
StatProto& get_stat_proto();
void clean_stat();
void collect_stat(const onnxruntime::Graph& graph, const ContextProto& context);
std::set<std::string>& get_vitis_ep_custom_ops();
} // namespace morphizen
