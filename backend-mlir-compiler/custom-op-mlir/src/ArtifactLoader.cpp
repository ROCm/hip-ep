/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "ArtifactLoader.h"

// CRITICAL: morphizen.hpp must be included before other morphizen headers
#include "morphizen/env_config.hpp"
#include "morphizen/morphizen.hpp"
#include <glog/logging.h>

// Environment parameters (global scope, before namespace)
DEF_ENV_PARAM(MORPHIZEN_DEBUG_MLIR_BACKEND, "0")

#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_MLIR_BACKEND) >= n)

namespace mlir_compilation {
namespace customop {

std::optional<ArtifactData> ArtifactLoader::load(
    const std::shared_ptr<const morphizen::PassContext> &context,
    const std::string &artifact_filename, int64_t expected_size) {

  ArtifactData data;
  data.filename = artifact_filename;
  data.size = expected_size;
  data.bytes.resize(static_cast<size_t>(expected_size));

  // Open file from EPContext
  auto artifact_stream = context->open_file_for_read(artifact_filename);
  if (!artifact_stream) {
    LOG(WARNING) << "Failed to open artifact from EPContext: "
                 << artifact_filename;
    return std::nullopt;
  }

  // Read artifact bytes
  size_t bytes_read =
      artifact_stream->fread(data.bytes.data(), data.bytes.size());
  if (bytes_read != data.bytes.size()) {
    LOG(WARNING) << "Failed to read complete artifact from EPContext (expected "
                 << data.bytes.size() << " bytes, got " << bytes_read << ")";
    return std::nullopt;
  }

  MY_LOG(1) << "Artifact loaded from EPContext: " << bytes_read << " bytes";

  return data;
}

} // namespace customop
} // namespace mlir_compilation
