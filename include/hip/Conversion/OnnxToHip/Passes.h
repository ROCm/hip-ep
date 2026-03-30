/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef HIP_CONVERSION_ONNXTOHIP_PASSES_H
#define HIP_CONVERSION_ONNXTOHIP_PASSES_H

#include "hip/Dialect/Transforms/Pipelines.h"
#include "mlir/Pass/Pass.h"
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>

namespace morphizen {
class FileSystem;
} // namespace morphizen

namespace mlir {
namespace hip {

using ConstantDataMap =
    std::unordered_map<std::string, std::pair<const void *, size_t>>;

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

/// Creates a pass with zero-copy constant data. Constants referenced by
/// hip.constant_ref attributes in the MLIR are resolved from
/// \p constantDataMap (name -> {ptr, size}) and written to constants.bin.
std::unique_ptr<Pass> createConvertOnnxToHipPass(
    morphizen::FileSystem *fs, int64_t minNumElements,
    const ConstantDataMap *constantDataMap);

} // namespace hip
} // namespace mlir

#endif // HIP_CONVERSION_ONNXTOHIP_PASSES_H
