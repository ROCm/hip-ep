/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef HIPSR_SCHEME_PASSES_H
#define HIPSR_SCHEME_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace hipsr {

/// Create a pass that prints MLIR operations in generic form using Scheme.
/// Initializes Chez Scheme with embedded boot file (rime included).
std::unique_ptr<Pass> createSchemePrintPass();

/// Register Scheme-related passes
void registerHipsrSchemePasses();

} // namespace hipsr
} // namespace mlir

#endif // HIPSR_SCHEME_PASSES_H
