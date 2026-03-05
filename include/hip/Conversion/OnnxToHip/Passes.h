/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef UDNA_COMPILER_CONVERSION_ONNXTOHIP_PASSES_H
#define UDNA_COMPILER_CONVERSION_ONNXTOHIP_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace morphizen {
class FileSystem;
} // namespace morphizen

namespace udna {
namespace compiler {
struct CompilationOptionsT;
} // namespace compiler
} // namespace udna

namespace mlir {
namespace hip {

/// Creates a pass that inserts !hip.context as argument 0 into every
/// func.func in the module, so that OnnxToHip patterns can access the context.
std::unique_ptr<Pass> createHipAddContextArgPass();

/// Creates a pass that converts ONNX operations to HIP dialect.
/// onnx.Constant data is written via fs using options.constants_file as the
/// filename. The module gains hipdnn.constant_sizes and hipdnn.constant_count
/// attributes for use by GenerateInterfacePass.
std::unique_ptr<Pass>
createConvertOnnxToHipPass(morphizen::FileSystem* fs,
                           const udna::compiler::CompilationOptionsT& options);

/// No-arg overload for CLI registration (--convert-onnx-to-hip).
/// Uses default CompilationOptionsT (constants.bin, no FileSystem).
std::unique_ptr<Pass> createConvertOnnxToHipPass();

} // namespace hip
} // namespace mlir

#endif // UDNA_COMPILER_CONVERSION_ONNXTOHIP_PASSES_H
