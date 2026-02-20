/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "FusionManager.h"

// CRITICAL: morphizen.hpp must be included before any other morphizen headers
#include "morphizen/env_config.hpp"
#include "morphizen/morphizen.hpp"
#include <glog/logging.h>

#include <vector>

// Environment parameters (must be at global scope before namespace)
DEF_ENV_PARAM(MORPHIZEN_DEBUG_MLIR_BACKEND, "0")

#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_MLIR_BACKEND) >= n)

namespace hipdnn {
namespace level1pass {

bool FusionManager::fuseGraph(morphizen::IPass &self,
                              morphizen_cxx::GraphRef &graph,
                              const std::string &metadata,
                              const std::string &unique_id) {

  MY_LOG(1) << "Attempting to fuse entire graph with domain 'MLIR'";

  // Extract graph input names
  std::vector<std::string> input_names;
  for (const auto &input : graph.inputs()) {
    input_names.push_back(input.name());
  }
  MY_LOG(1) << "Graph inputs: " << input_names.size();

  // Extract graph output names
  std::vector<std::string> output_names;
  for (const auto &output : graph.outputs()) {
    output_names.push_back(output.name());
  }
  MY_LOG(1) << "Graph outputs: " << output_names.size();

  auto [meta_def, fuse_error] =
      self.try_fuse(graph, unique_id, input_names, output_names,
                    {},    // constant_initializers (empty for now)
                    "MLIR" // domain name
      );

  if (meta_def == nullptr) {
    MY_LOG(1) << "Fusion failed: " << fuse_error.comments;
    LOG(WARNING) << "Failed to create MLIR fusion: " << fuse_error.comments;
    return false;
  }

  MY_LOG(1) << "Fusion successful, attaching metadata";

  // Attach metadata to MetaDefProto
  self.attach_meta_def_param(*meta_def, metadata.c_str());

  // Fuse the graph (replace nodes with custom op)
  self.fuse(graph, std::move(*meta_def));

  MY_LOG(1) << "MLIR compilation and fusion completed";
  return true;
}

} // namespace level1pass
} // namespace hipdnn
