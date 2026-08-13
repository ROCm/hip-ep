/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef HIP_CONVERSION_ONNXTOHIP_PASSES_H
#define HIP_CONVERSION_ONNXTOHIP_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace hip {

/// Creates a pass that converts ONNX operations to HIP dialect. Both constant
/// sweeps lower onnx.Constant only to policy-neutral hip.constant carriers;
/// hip-externalize-constants owns storage policy and artifact I/O.
std::unique_ptr<Pass> createConvertOnnxToHipPass();

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
