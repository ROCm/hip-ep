/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef HIP_CONVERSION_ONNXTOHIP_PASSES_H
#define HIP_CONVERSION_ONNXTOHIP_PASSES_H

#include "hip/Dialect/Transforms/Pipelines.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace morphizen {
class FileSystem;
} // namespace morphizen

namespace mlir {
namespace hip {

/// Creates a pass that converts ONNX operations to HIP dialect.
/// Uses default options (constants.bin, no externalization).
std::unique_ptr<Pass> createConvertOnnxToHipPass();

/// Creates a pass with an external FileSystem for constant storage.
/// When \p fs is non-null, externalized constants are written via \p fs
/// instead of a DiskFileSystem rooted at the output directory. This is
/// the production path used by hip-compiler when called from an ORT EP,
/// where constants must be written into the EPContext archive.
/// \p minNumElements controls the externalization threshold.
std::unique_ptr<Pass> createConvertOnnxToHipPass(
    morphizen::FileSystem *fs,
    int64_t minNumElements = kDefaultExternalizeMinNumElements);

} // namespace hip
} // namespace mlir

#endif // HIP_CONVERSION_ONNXTOHIP_PASSES_H
