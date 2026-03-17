/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef HIP_CONVERSION_ONNXTOHIP_PASSES_H
#define HIP_CONVERSION_ONNXTOHIP_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace morphizen {
class FileSystem;
} // namespace morphizen

namespace hip {
namespace compiler {
struct CompilationOptionsT;
} // namespace compiler
} // namespace hip

namespace mlir {
namespace hip {

#define GEN_PASS_DECL
#include "hip/Conversion/OnnxToHip/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "hip/Conversion/OnnxToHip/Passes.h.inc"

/// Creates a pass that converts ONNX operations to HIP dialect.
/// onnx.Constant data is written via fs using options.constants_file as the
/// filename. The module gains hipdnn.constant_sizes and hipdnn.constant_count
/// attributes for use by GenerateInterfacePass.
std::unique_ptr<Pass>
createConvertOnnxToHipPass(morphizen::FileSystem *fs,
                           const ::hip::compiler::CompilationOptionsT &options);

/// No-arg overload for CLI registration (--convert-onnx-to-hip).
/// Uses default CompilationOptionsT (constants.bin, no FileSystem).
std::unique_ptr<Pass> createConvertOnnxToHipPass();

} // namespace hip
} // namespace mlir

#endif // HIP_COMPILER_CONVERSION_ONNXTOHIP_PASSES_H
