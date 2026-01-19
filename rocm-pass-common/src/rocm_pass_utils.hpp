// Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <glog/logging.h>
#include <string>
#include "google/protobuf/util/json_util.h"
#include "morphizen/env_config.hpp"
#include "morphizen/vaip.hpp"
#include "morphizen/node_attr.hpp"
#include "rocm.pb.h"

// Common logging macros for ROCm passes
DEF_ENV_PARAM(MORPHIZEN_DEBUG_ROCM, "0")
#define ROCM_LOG(n) LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_ROCM) >= n)

namespace rocm_pass {

/**
 * Generate a unique filename for weight data by sanitizing tensor name.
 * Replaces invalid filename characters with underscores.
 * 
 * @param prefix Prefix for the filename (e.g., "rocm_conv", "rocm_gemm")
 * @param tensor_name Original tensor name from the graph
 * @return Sanitized filename with .bin extension
 */
inline std::string generate_weight_filename(const std::string& prefix,
                                            const std::string& tensor_name) {
  std::string sanitized_name = tensor_name;
  for (auto& c : sanitized_name) {
    if (c == '/' || c == '\\' || c == ':' || c == '*' || c == '?' ||
        c == '"' || c == '<' || c == '>' || c == '|') {
      c = '_';
    }
  }
  return prefix + "_" + sanitized_name + ".bin";
}

/**
 * Finalize a Level-2 fuse operation.
 * 
 * This function handles the common post-fusion logic for all ROCm Level-2 passes:
 * 1. Serializes RocmParamProto to JSON
 * 2. Writes params to cache file for troubleshooting and Level-1 to read
 * 3. Attaches params to meta_def
 * 4. Calls level_2_fuse() to create the fused node
 * 5. Adds rocm_param_file attribute to the fused node
 * 
 * @param self Pointer to the IPass instance
 * @param graph Reference to the Graph being modified
 * @param meta_def Reference to the MetaDefProto from try_fuse()
 * @param rocm_param The RocmParamProto containing operation parameters
 * @param output_name The output name for naming the param file
 * @param log_prefix Logging prefix (e.g., "[ROCm Conv L2]", "[ROCm Gemm L2]")
 * @return true if fusion succeeded, false otherwise
 */
inline bool finalize_level2_fuse(
    vaip_core::IPass* self,
    vaip_core::Graph& graph,
    vaip_core::MetaDefProto& meta_def,
    const rocm::RocmParamProto& rocm_param,
    const std::string& output_name,
    const std::string& log_prefix) {
  
  // Serialize params to JSON
  std::string rocm_param_json;
  auto status = google::protobuf::util::MessageToJsonString(
      rocm_param, &rocm_param_json);
  if (!status.ok()) {
    LOG(ERROR) << log_prefix << " Failed to serialize params: " << status.ToString();
    return false;
  }

  // Write params to cache file (for troubleshooting and Level-1 to read)
  auto pass_context = self->get_context();
  std::string param_filename = "rocm_param_" + output_name + ".json";
  auto param_writer = pass_context->open_file_for_write(param_filename);
  if (param_writer) {
    param_writer->fwrite(rocm_param_json.data(), rocm_param_json.size());
    ROCM_LOG(1) << log_prefix << " Saved params to cache: " << param_filename;
  } else {
    LOG(ERROR) << log_prefix << " Failed to write param file: " << param_filename;
  }

  // Attach params to meta_def (for reference)
  self->attach_meta_def_param(meta_def, rocm_param_json.c_str());

  // Use level_2_fuse() - does NOT update context.json
  // Level-1 will call fuse() after merging subgraphs
  const auto& fused_node = self->level_2_fuse(graph, meta_def);

  // Add node attribute to mark this as ROCm fused node
  // const_cast is needed because level_2_fuse returns const Node&
  vaip_core::Node& mutable_node = const_cast<vaip_core::Node&>(fused_node);
  vaip_core::NodeAttributesBuilder attr_builder;
  attr_builder.add("rocm_param_file", param_filename);
  attr_builder.merge_into(mutable_node);

  ROCM_LOG(1) << log_prefix << " Created fused node via level_2_fuse: " 
              << output_name << " (param_file: " << param_filename << ")";
  return true;
}

/**
 * Save weight tensor data to cache file.
 * 
 * @param pass_context Shared pointer to PassContext for cache file access
 * @param weight_data Float data to write
 * @param filename Target filename in cache
 * @param log_prefix Logging prefix
 * @return true if write succeeded, false otherwise
 */
inline bool save_weight_to_cache(
    const std::shared_ptr<vaip_core::PassContext>& pass_context,
    const gsl::span<const float>& weight_data,
    const std::string& filename,
    const std::string& log_prefix) {
  
  size_t weight_bytes = weight_data.size() * sizeof(float);
  auto writer = pass_context->open_file_for_write(filename);
  if (writer) {
    writer->fwrite(weight_data.data(), weight_bytes);
    ROCM_LOG(1) << log_prefix << " Saved weight to cache: " << filename 
                << " (" << weight_bytes << " bytes)";
    return true;
  } else {
    LOG(ERROR) << log_prefix << " Failed to open weight file for write: " << filename;
    return false;
  }
}

} // namespace rocm_pass
