/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef HIP_PASSES_H
#define HIP_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>
#include <string>

namespace mlir {
namespace hip {

struct CompilationOptionsT;

#define GEN_PASS_DECL
#include "hip/Dialect/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "hip/Dialect/Transforms/Passes.h.inc"

/// Creates a pass that generates the C interface for the compiled module.
/// Transforms @main_graph to produce four C-ABI wrapper functions:
/// inference_init, inference_compute, inference_cleanup,
/// inference_get_metadata_json.
std::unique_ptr<mlir::Pass>
createGenerateInterfacePass(const CompilationOptionsT &options);

/// Creates the allocator-mode counterpart of createGenerateInterfacePass.
/// Emits the same four C-ABI wrappers, but inference_compute and @main_graph
/// use the 2-arg (state, inputs) ABI: graph outputs are allocated in-graph via
/// hip.alloc_output (the hipdnn_ep_alloc_output runtime callback) rather than
/// passed in as out-params. Pairs with the hip-use-output-allocator
/// ONNX-to-HIP pass and the allocator branch of convert-hip-to-llvm.
std::unique_ptr<mlir::Pass>
createGenerateAllocatorInterfacePass(const CompilationOptionsT &options);

} // namespace hip
} // namespace mlir

#endif // HIP_PASSES_H
