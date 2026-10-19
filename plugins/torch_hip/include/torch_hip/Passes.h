/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===- Passes.h - Torch -> HIP conversion passes --------------------------===//
//
// Plugin-agnostic declarations: nothing here knows about the plugin registry,
// so the pass can later move into lib/Conversion/TorchToHip/ unchanged.
//
//===----------------------------------------------------------------------===//

#ifndef TORCH_HIP_PASSES_H
#define TORCH_HIP_PASSES_H

#include <memory>

namespace mlir {
class Pass;
} // namespace mlir

namespace mlir::hip {

/// Command-line name of the pass (used by the pipeline slot request).
inline constexpr const char *kConvertTorchToHipPassName =
    "convert-torch-to-hip";

/// Lowers Torch-MLIR (`torch` dialect) ops to HIP dialect ops.
/// Modules without `torch` ops are left untouched.
std::unique_ptr<mlir::Pass> createConvertTorchToHipPass();

/// Registers the pass with MLIR's global pass registry (by name).
void registerConvertTorchToHipPass();

} // namespace mlir::hip

#endif // TORCH_HIP_PASSES_H
