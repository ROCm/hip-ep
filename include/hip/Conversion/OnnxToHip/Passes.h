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

/// Creates a pass that converts ONNX operations to HIP dialect. Constants are
/// lowered to neutral hip.constant carriers; externalization to constants.bin
/// is done by the standalone --hip-externalize-constants pass.
std::unique_ptr<Pass> createConvertOnnxToHipPass();

/// Creates a pass that outlines each onnx.Loop body into a separate
/// func.func and rewrites the loop into a hip.loop op that points at it.
/// Runs BEFORE --convert-onnx-to-hip so the body's onnx.* ops get the
/// same conversion treatment as ops in main_graph.  Requires
/// --hip-add-context-arg to have run first so that !hip.context is
/// available as the parent function's arg 0.
std::unique_ptr<Pass> createOnnxLoopOutlinePass();

} // namespace hip
} // namespace mlir

#endif // HIP_CONVERSION_ONNXTOHIP_PASSES_H
