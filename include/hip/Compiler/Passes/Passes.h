/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef hip_COMPILER_COMPILER_PASSES_PASSES_H
#define hip_COMPILER_COMPILER_PASSES_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace hip {
namespace compiler {
struct CompilationOptionsT;
} // namespace compiler
} // namespace hip

namespace hip::compiler {
namespace compiler {

/// Creates a pass that generates the C interface for the compiled module.
/// Transforms @main_graph to accept array-of-structs interface for ONNX
/// Runtime. options.constants_file is embedded in the DLL metadata so the
/// runtime knows which file to load.
std::unique_ptr<mlir::Pass>
createGenerateInterfacePass(const hip::compiler::CompilationOptionsT& options);

} // namespace compiler
} // namespace hip::compiler

#endif // hip_COMPILER_COMPILER_PASSES_PASSES_H
