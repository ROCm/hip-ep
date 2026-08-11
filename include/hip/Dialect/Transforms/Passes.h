/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef HIP_PASSES_H
#define HIP_PASSES_H

#include "mlir/Pass/Pass.h"
#include <cstdint>
#include <memory>
#include <string>

namespace morphizen {
class FileSystem;
} // namespace morphizen

namespace mlir {
namespace hip {

struct CompilationOptionsT;

/// Integer encoding used by the temporary module metadata exchanged between
/// hip-externalize-constants and hip-generate-interface.
enum class ConstantMetadataSourceKind : int32_t {
  Bulk = 0,
  Splat = 1,
  File = 2,
  Memory = 3,
};

#define GEN_PASS_DECL
#include "hip/Dialect/Transforms/Passes.h.inc"

/// Creates hip-externalize-constants for the production EP path. The supplied
/// FileSystem receives constant artifacts, and the production caller normally
/// passes kDefaultExternalizeMinNumElements (1). Supplying the FileSystem also
/// authorizes process-local ORT memory-address carriers while their backing
/// storage is live. The generated no-arg/options overload keeps the direct-pass
/// threshold default at 0, uses DiskFileSystem, and rejects memory addresses.
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
