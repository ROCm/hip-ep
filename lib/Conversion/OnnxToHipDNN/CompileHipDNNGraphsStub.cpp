/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// Stub implementations for mock builds (BUILD_MOCK_RUNTIME=ON).
// The real implementations live in CompileHipDNNGraphs.cpp and require
// the HipDNNGraph library (TheRock SDK).  These stubs satisfy the linker
// for symbols referenced in Pipelines.cpp and CompilerDriver.

#include "hip/Conversion/OnnxToHipDNN/Passes.h"

namespace mlir {
namespace hip {

void GraphDeleter::operator()(void *) const {}

std::unique_ptr<Pass>
createCompileHipDNNGraphsPass(hipdnnHandle_t /*handle*/,
                              CompiledGraphMap /*output_graphs*/) {
  return nullptr;
}

} // namespace hip
} // namespace mlir
