/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef MORPHIZEN_MLIR_COMPILER_COMPILER_PASSES_PASSES_H
#define MORPHIZEN_MLIR_COMPILER_COMPILER_PASSES_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace morphizen {
namespace compiler {

/// Creates a pass that generates the C interface for the compiled module.
/// Transforms @main_graph to accept array-of-structs interface for ONNX
/// Runtime.
std::unique_ptr<mlir::Pass> createGenerateInterfacePass();

/// Register all compiler passes.
void registerCompilerPasses();

} // namespace compiler
} // namespace morphizen

#endif // MORPHIZEN_MLIR_COMPILER_COMPILER_PASSES_PASSES_H
