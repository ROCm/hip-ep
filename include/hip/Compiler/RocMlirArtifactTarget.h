/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// Dependency-free so the level-1 pass and the custom op can both include it
// without pulling in MLIR or rocMLIR headers.

#ifndef HIP_COMPILER_ROCMLIRARTIFACTTARGET_H
#define HIP_COMPILER_ROCMLIRARTIFACTTARGET_H

#include <string>

namespace hip::compiler {

/// What CompilerDriver would embed rocMLIR code objects for right now, or ""
/// when the rocMLIR path is off (HIPDNN_EP_ROCMLIR unset/0, or a build without
/// ENABLE_ROCMLIRTRITON).
///
/// The string is the GPU arch, plus `+autotune=<space>` when benchmarking is
/// on, plus `+lut=<fingerprint>` of the shipped problem-cache bytes. The
/// provider does not read HIP_ROCMLIR_PROBLEM_CACHE, so that fingerprint is
/// the shipped table only. The model cache / EPContext identity is derived
/// from the graph alone, so an artifact that carries an arch-specific HSACO
/// could otherwise be reused on a different GPU, after HIPDNN_EP_ROCMLIR or
/// the autotune space changes, or after the shipped LUT changes. The level-1
/// pass records this value and the custom op refuses a mismatch.
std::string rocMlirArtifactTarget();

} // namespace hip::compiler

#endif // HIP_COMPILER_ROCMLIRARTIFACTTARGET_H
