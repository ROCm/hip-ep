/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once
// Whole-graph backend selection between the HIP compiler and the MIGraphX
// library (docs/design/migraphx-backend.md). Pure logic with no ORT or HIP
// dependency, so it is unit-tested without a GPU.
#include <optional>
#include <string>
#include <vector>

namespace morphizen {

// Provider option "gpu_backend".
enum class GpuBackend { Hip, MIGraphX, Auto };

inline constexpr const char *kGpuBackendOption = "gpu_backend";

// Bumped whenever the automatic choice changes, so a logged decision can be
// reproduced. v1: use HIP when it compiles the whole graph; otherwise use
// MIGraphX when it is eligible. No performance rules yet.
inline constexpr int kRoutingPolicyVersion = 1;

// Accepts "hip", "migraphx" and "auto"; anything else is nullopt.
std::optional<GpuBackend> parse_gpu_backend(const std::string &value);
const char *to_string(GpuBackend backend);

// What the MIGraphX eligibility check needs to know about the ORT graph.
struct GraphSummary {
  std::vector<std::string> op_types;
  std::vector<std::string> input_names;
  bool has_model_path = false;
  // Every graph input/output is a tensor whose dims are all known.
  bool static_inputs = true;
  bool static_outputs = true;
};

struct Eligibility {
  bool eligible = true;
  std::string reason; // why not, when !eligible
};

// v1 rules for MIGraphX: the model file must be on disk (the plugin parses
// it), every input and output shape must be static (no in-graph output
// allocation), and the graph must not carry KV-cache or recurrent state.
Eligibility migraphx_eligibility(const GraphSummary &graph);

} // namespace morphizen
