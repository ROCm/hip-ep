/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef HIPDNN_HIP_COMPILER_H
#define HIPDNN_HIP_COMPILER_H

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace hipdnn::compiler {

enum class CompileMode {
  Standalone, // Raw function exports (for standalone testing)
  Plugin      // init/compute/cleanup exports (for EP integration)
};

struct CompileOptions {
  CompileMode mode = CompileMode::Standalone;
  int optLevel = 2;
  // Directory containing hip_runtime_static.lib; empty = derive from exe path
  std::string runtimeLibDir;
};

struct ModelMetadata {
  std::string entryFunction;
  int inputCount = 0;
  int outputCount = 0;
  std::vector<int> inputRanks;
  std::vector<int> outputRanks;
  std::vector<std::vector<int64_t>> inputShapes;
  std::vector<std::vector<int64_t>> outputShapes;
  int64_t poolSize = 0;
  std::vector<int64_t> bufferOffsets;
};

struct CompileResult {
  std::vector<uint8_t> dllBytes;
  ModelMetadata metadata;
};

class HipCompiler {
public:
  /// Compile a HIP MLIR file to a DLL.
  static std::optional<CompileResult>
  compileFile(const std::string &inputPath, const std::string &outputPath,
              const CompileOptions &options);

  /// Compile HIP MLIR text (string) to a DLL.
  static std::optional<CompileResult>
  compile(const std::string &mlirText, const std::string &outputPath,
          const CompileOptions &options);
};

} // namespace hipdnn::compiler

#endif // HIPDNN_HIP_COMPILER_H
