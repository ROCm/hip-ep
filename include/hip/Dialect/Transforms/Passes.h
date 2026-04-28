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

/// Creates a pass that inserts hip.dump_tensor ops after every HIP compute op.
/// Each dump op copies the output tensor from GPU to host and saves it as a
/// NumPy .npy file under \p dumpTensorsDir.
std::unique_ptr<mlir::Pass> createInsertTensorDumpPass(llvm::StringRef dumpTensorsDir);

} // namespace hip
} // namespace mlir

#endif // HIP_PASSES_H
