/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef HIP_CONVERSION_ONNX_TO_HIPDNN_PASSES_H
#define HIP_CONVERSION_ONNX_TO_HIPDNN_PASSES_H

#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringMap.h"

#include <memory>

struct hipdnnHandle;
typedef hipdnnHandle *hipdnnHandle_t;

namespace mlir {
namespace hip {

/// Custom deleter for type-erased HipDNNGraph pointers.
/// Implemented in CompileHipDNNGraphs.cpp where the real type is visible.
/// Using void* avoids introducing the global ::hip namespace, which collides
/// with ::mlir::hip when unqualified "hip::" is used with "using namespace
/// mlir".
struct GraphDeleter {
  void operator()(void *ptr) const;
};

using OwnedGraph = std::unique_ptr<void, GraphDeleter>;

/// Map of compiled hipDNN graphs, keyed by graph name.
/// Values are type-erased hip::graph::HipDNNGraph pointers.
/// Shared ownership: the pass populates it, CompilerDriver retrieves it.
using CompiledGraphMap = std::shared_ptr<llvm::StringMap<OwnedGraph>>;

/// Create the OutlineOnnxToHipDNN pass (handle-free, CLI-testable).
///
/// Wraps supported ONNX ops in hip.hipdnn_graph_outline regions.
/// Unsupported or dynamic-shape ops are left untouched for ConvertOnnxToHip.
std::unique_ptr<Pass> createOutlineOnnxToHipDNNPass();

/// Create the CompileHipDNNGraphs pass.
///
/// Walks hip.hipdnn_graph_outline ops, compiles their regions via hipDNN,
/// and replaces them with hip.hipdnn_graph.  Un-outlines on failure.
///
/// @param handle        Live hipDNN handle (GPU must be available)
/// @param output_graphs Map to store compiled graphs (populated by the pass)
std::unique_ptr<Pass>
createCompileHipDNNGraphsPass(hipdnnHandle_t handle,
                              CompiledGraphMap output_graphs);

} // namespace hip
} // namespace mlir

#endif // HIP_CONVERSION_ONNX_TO_HIPDNN_PASSES_H
