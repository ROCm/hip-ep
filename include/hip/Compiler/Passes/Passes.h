/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef UDNA_COMPILER_COMPILER_PASSES_PASSES_H
#define UDNA_COMPILER_COMPILER_PASSES_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace udna {
namespace compiler {
struct CompilationOptionsT;
} // namespace compiler
} // namespace udna

namespace udna::compiler {
namespace compiler {

/// Creates a pass that generates the C interface for the compiled module.
/// Transforms @main_graph to accept array-of-structs interface for ONNX
/// Runtime. options.constants_file is embedded in the DLL metadata so the
/// runtime knows which file to load.
std::unique_ptr<mlir::Pass>
createGenerateInterfacePass(const udna::compiler::CompilationOptionsT& options);

} // namespace compiler
} // namespace udna::compiler

#endif // UDNA_COMPILER_COMPILER_PASSES_PASSES_H
