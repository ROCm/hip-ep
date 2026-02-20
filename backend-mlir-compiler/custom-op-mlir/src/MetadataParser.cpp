/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "MetadataParser.h"

// CRITICAL: morphizen.hpp must be included before other morphizen headers
#include "google/protobuf/util/json_util.h"
#include "mlir_compilation.pb.h"
#include "morphizen/env_config.hpp"
#include "morphizen/morphizen.hpp"
#include <glog/logging.h>

// Environment parameters (global scope, before namespace)
DEF_ENV_PARAM(MORPHIZEN_DEBUG_MLIR_BACKEND, "0")

#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_MLIR_BACKEND) >= n)

namespace mlir_compilation {
namespace customop {

std::optional<MlirCompilationProto> MetadataParser::parse(
    const std::shared_ptr<const morphizen::PassContext> &context,
    const std::shared_ptr<morphizen::MetaDefProto> &meta_def) {
  MlirCompilationProto proto;

  // Get metadata JSON from MetaDefProto via PassContext
  auto metadata_json = context->get_meta_def_param(*meta_def);

  // Parse JSON to protobuf
  auto status =
      google::protobuf::util::JsonStringToMessage(metadata_json, &proto);
  if (!status.ok()) {
    LOG(WARNING) << "Failed to parse MLIR compilation metadata: "
                 << status.ToString();
    return std::nullopt;
  }

  MY_LOG(1) << "Artifact format: " << proto.artifact_format();
  MY_LOG(1) << "Artifact filename: " << proto.artifact_filename();
  MY_LOG(1) << "Artifact size: " << proto.artifact_size();

  return proto;
}

} // namespace customop
} // namespace mlir_compilation
