/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef METADATA_BUILDER_H
#define METADATA_BUILDER_H

#include "CompilationArtifact.h"
#include "CompilationConfig.h"
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace hipdnn {
namespace level1pass {

// Metadata for output tensors extracted from graph
struct OutputMetadata {
  std::string name;
  int32_t rank;
  std::vector<int64_t> shape;
  std::string dtype;
};

class MetadataBuilder {
public:
  // Build JSON metadata from compilation artifact
  // Returns nullopt on serialization failure
  static std::optional<std::string>
  build(const CompilationArtifact &artifact, const CompilationConfig &config,
        int64_t mlir_duration_ms,
        const std::vector<OutputMetadata> &outputs = {});
};

} // namespace level1pass
} // namespace hipdnn

#endif
