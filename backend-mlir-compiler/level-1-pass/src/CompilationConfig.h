/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef COMPILATION_CONFIG_H
#define COMPILATION_CONFIG_H

#include <memory>
#include <string>
#include <vector>

namespace morphizen {
class PassContext;
}

namespace hipdnn {
namespace level1pass {

struct CompilationConfig {
  enum class ArtifactFormat { Native, LlvmIr };

  ArtifactFormat artifactFormat;
  int optLevel;
  std::string outputFilename;
  bool useMockRuntime;

  static CompilationConfig
  fromProviderOptions(const std::shared_ptr<morphizen::PassContext> &context);
  static CompilationConfig defaultConfig();
};

} // namespace level1pass
} // namespace hipdnn

#endif
