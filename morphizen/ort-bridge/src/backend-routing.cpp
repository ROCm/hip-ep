/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "./backend-routing.hpp"

#include <algorithm>
#include <array>

namespace morphizen {

std::optional<GpuBackend> parse_gpu_backend(const std::string &value) {
  if (value == "hip")
    return GpuBackend::Hip;
  if (value == "migraphx")
    return GpuBackend::MIGraphX;
  if (value == "auto")
    return GpuBackend::Auto;
  return std::nullopt;
}

const char *to_string(GpuBackend backend) {
  switch (backend) {
  case GpuBackend::Hip:
    return "hip";
  case GpuBackend::MIGraphX:
    return "migraphx";
  case GpuBackend::Auto:
    return "auto";
  }
  return "unknown";
}

Eligibility migraphx_eligibility(const GraphSummary &graph) {
  if (!graph.has_model_path)
    return {false, "the session was created from memory; the MIGraphX "
                   "backend needs the model file path"};
  if (!graph.static_inputs)
    return {false, "a graph input has a dynamic or symbolic dimension"};
  if (!graph.static_outputs)
    return {false, "a graph output shape is only known at run time"};

  // State that must persist across runs stays with one backend (the HIP
  // runtime owns KV caches today).
  static constexpr std::array<const char *, 6> kStatefulOps = {
      "GroupQueryAttention",
      "PagedAttention",
      "SparseAttention",
      "LSTM",
      "GRU",
      "RNN"};
  for (const auto &op : graph.op_types) {
    if (std::find(kStatefulOps.begin(), kStatefulOps.end(), op) !=
        kStatefulOps.end())
      return {false, "graph contains stateful op " + op};
  }
  for (const auto &name : graph.input_names) {
    if (name.rfind("past_", 0) == 0)
      return {false, "graph input '" + name + "' looks like a KV cache"};
  }
  return {};
}

} // namespace morphizen
