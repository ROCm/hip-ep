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

/// The GPU arch CompilerDriver would embed rocMLIR code objects for right now,
/// or "" when the rocMLIR path is off (a build without ENABLE_ROCMLIRTRITON,
/// or HIPDNN_EP_PIPELINE / HIPDNN_EP_HIPSR is selected).
///
/// The model cache / EPContext identity is derived from the graph alone, so an
/// artifact that carries an arch-specific HSACO could otherwise be reused on
/// a different GPU. The level-1 pass
/// records this value in the artifact metadata and the custom op refuses to
/// load an artifact whose recorded value differs from the current one.
std::string rocMlirArtifactTarget();

} // namespace hip::compiler

#endif // HIP_COMPILER_ROCMLIRARTIFACTTARGET_H
