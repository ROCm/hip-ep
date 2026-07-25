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
/// instead of a DiskFileSystem rooted at the output directory.
std::unique_ptr<Pass> createConvertOnnxToHipPass(
    morphizen::FileSystem *fs,
    int64_t minNumElements = kDefaultExternalizeMinNumElements,
    bool skipConstantData = false);

/// Creates a pass that outlines each onnx.Loop body into a separate
/// func.func and rewrites the loop into a hip.loop op that points at it.
/// Runs BEFORE --convert-onnx-to-hip so the body's onnx.* ops get the
/// same conversion treatment as ops in main_graph.  Requires
/// --hip-add-context-arg to have run first so that !hip.context is
/// available as the parent function's arg 0.
std::unique_ptr<Pass> createOnnxLoopOutlinePass();

/// Creates a pass that outlines each onnx.If branch into separate
/// func.func ops and rewrites the If into a hip.if op.  Runs BEFORE
/// --convert-onnx-to-hip (same pipeline slot as onnx-loop-outline).
std::unique_ptr<Pass> createOnnxIfOutlinePass();

} // namespace hip
} // namespace mlir

#endif // HIP_CONVERSION_ONNXTOHIP_PASSES_H
