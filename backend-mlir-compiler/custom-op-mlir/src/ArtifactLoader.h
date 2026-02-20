/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef ARTIFACT_LOADER_H
#define ARTIFACT_LOADER_H

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace morphizen {
class PassContext;
}

namespace mlir_compilation {
namespace customop {

struct ArtifactData {
  std::vector<uint8_t> bytes;
  std::string filename;
  int64_t size;
};

class ArtifactLoader {
public:
  // Read artifact bytes from EPContext file
  static std::optional<ArtifactData>
  load(const std::shared_ptr<const morphizen::PassContext> &context,
       const std::string &artifact_filename, int64_t expected_size);
};

} // namespace customop
} // namespace mlir_compilation

#endif
