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

/// Creates a pass that re-infers `onnx.*` result types inside every
/// outlined `hip.loop` body function from the (already-refined) operand
/// types, then propagates the refined yield operand types into the
/// function signature and the enclosing `hip.loop` op's result types.
///
/// Necessary because `--onnx-loop-outline` rebuilds each body func's
/// signature from the v_init operand types (the source of truth for
/// runtime values flowing in), but the body's cloned `onnx.*` ops still
/// carry the stale result types annotated on the ORIGINAL `onnx.Loop`
/// body block — typically a degenerate rank-0 sentinel type set by some
/// ONNX exporters for accumulator iter_vars. Without this pass,
/// downstream `--convert-onnx-to-hip` rejects the cloned ops because
/// their result types don't match what their (now-refined) operands
/// imply. Runs BETWEEN `--onnx-loop-outline` and `--convert-onnx-to-hip`.
std::unique_ptr<Pass> createRefineLoopBodyTypesPass();

} // namespace hip
} // namespace mlir

#endif // HIP_CONVERSION_ONNXTOHIP_PASSES_H
