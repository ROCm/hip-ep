/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef HIP_PASSES_H
#define HIP_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>
#include <string>

namespace morphizen {
class FileSystem;
} // namespace morphizen

namespace mlir {
namespace hip {

struct CompilationOptionsT;

#define GEN_PASS_DECL
#include "hip/Dialect/Transforms/Passes.h.inc"

/// Create hip-externalize-constants with an EP-supplied FileSystem (writes the
/// constants.bin into the EPContext tar) + the fs-mode threshold and
/// skip-constant-data flag. The no-arg / options overloads (from GEN_PASS_DECL)
/// use a DiskFileSystem rooted at externalize-output-dir.
std::unique_ptr<mlir::Pass>
createExternalizeConstantsPass(morphizen::FileSystem *fs,
                               int64_t minNumElements,
                               bool skipConstantData = false);

#define GEN_PASS_REGISTRATION
#include "hip/Dialect/Transforms/Passes.h.inc"

/// Creates a pass that generates the C interface for the compiled module.
/// Transforms @main_graph to produce four C-ABI wrapper functions:
/// inference_init, inference_compute, inference_cleanup,
/// inference_get_metadata_json. inference_compute has the 2-arg
/// (state, inputs) ABI: graph outputs are allocated in-graph via
/// hip.alloc_output (the hipdnn_ep_alloc_output runtime callback) rather than
/// passed as out-params.
std::unique_ptr<mlir::Pass>
createGenerateInterfacePass(const CompilationOptionsT &options);

} // namespace hip
} // namespace mlir

#endif // HIP_PASSES_H
