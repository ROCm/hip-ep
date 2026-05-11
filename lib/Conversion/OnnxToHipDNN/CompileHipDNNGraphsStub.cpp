//===- CompileHipDNNGraphsStub.cpp - Mock-build hipDNN stubs ---*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.  All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
//
// Stub implementations of the hipDNN graph compilation entry points used by
// mock builds (`BUILD_MOCK_RUNTIME=ON`).  The real implementations live in
// `CompileHipDNNGraphs.cpp` and pull in the HipDNNGraph library (TheRock
// SDK), which is unavailable in CI / non-GPU developer machines.  These
// stubs satisfy the linker for symbols referenced from `Pipelines.cpp` and
// `CompilerDriver.cpp` while making any attempt to compile a hipDNN graph a
// silent no-op (returns null).
//
// Out-of-tree consumers should never have to look at this file: which TU
// gets linked is selected by `lib/Conversion/OnnxToHipDNN/CMakeLists.txt`
// based on `BUILD_MOCK_RUNTIME`.
//
//===----------------------------------------------------------------------===//

#include "hip/Conversion/OnnxToHipDNN/Passes.h"

#include "mlir/Pass/Pass.h"

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
