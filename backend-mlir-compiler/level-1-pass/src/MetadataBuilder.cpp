/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "MetadataBuilder.h"
#include "CompilationConfig.h"

// CRITICAL: morphizen.hpp must be included before any other morphizen headers
#include "morphizen/env_config.hpp"
#include "morphizen/morphizen.hpp"
#include <glog/logging.h>

// Protobuf for metadata serialization
#include "google/protobuf/util/json_util.h"
#include "mlir_compilation.pb.h"

#include <chrono>

// Environment parameters (must be at global scope before namespace)
DEF_ENV_PARAM(MORPHIZEN_DEBUG_MLIR_BACKEND, "0")

#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_MLIR_BACKEND) >= n)

namespace hipdnn {
namespace level1pass {

std::optional<std::string> MetadataBuilder::build(
    const CompilationArtifact &artifact, const CompilationConfig &config,
    int64_t mlir_duration_ms, const std::vector<OutputMetadata> &outputs) {

  auto now = std::chrono::system_clock::now();
  auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                       now.time_since_epoch())
                       .count();

  mlir_compilation::MlirCompilationProto proto;
  proto.set_artifact_format(config.artifactFormat == ArtifactFormat::Native
                                 ? "native"
                                 : "llvm_ir");
  proto.set_compilation_timestamp(timestamp);
  proto.set_compiler_version("mlir-hip-compiler-1.0");
  proto.set_optimization_level(config.optLevel);
  proto.set_artifact_filename(artifact.filename);
  proto.set_artifact_size(artifact.bytes.size());

  // Populate output metadata
  for (const auto &output : outputs) {
    auto *tensor_meta = proto.add_outputs();
    tensor_meta->set_name(output.name);
    tensor_meta->set_rank(output.rank);
    tensor_meta->set_dtype(output.dtype);
    for (int64_t dim : output.shape) {
      tensor_meta->add_shape(dim);
    }
  }
  MY_LOG(1) << "Populated metadata with " << outputs.size() << " outputs";

  // Serialize to JSON
  std::string proto_json;
  auto status = google::protobuf::util::MessageToJsonString(proto, &proto_json);
  if (!status.ok()) {
    LOG(WARNING) << "Failed to serialize metadata to JSON: "
                 << status.ToString();
    return std::nullopt;
  }

  MY_LOG(1) << "Metadata JSON: " << proto_json;
  return proto_json;
}

} // namespace level1pass
} // namespace hipdnn
