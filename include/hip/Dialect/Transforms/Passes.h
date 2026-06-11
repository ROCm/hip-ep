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
/// inference_get_metadata_json. The classic 3-arg (state, inputs, outputs) vs
/// 2-arg allocator (state, inputs) ABI is selected from the module's
/// `hipdnn.use_output_allocator` BoolAttr (set true by
/// hip-use-output-allocator); in allocator mode graph outputs are allocated
/// in-graph via hip.alloc_output (the hipdnn_ep_alloc_output runtime callback)
/// rather than passed as out-params.
std::unique_ptr<mlir::Pass>
createGenerateInterfacePass(const CompilationOptionsT &options);

} // namespace hip
} // namespace mlir

#endif // HIP_PASSES_H
