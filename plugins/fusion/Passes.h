/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef HIP_PLUGIN_FUSION_PASSES_H
#define HIP_PLUGIN_FUSION_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace hip {

#define GEN_PASS_DECL
#include "plugins/fusion/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "plugins/fusion/Passes.h.inc"

} // namespace hip
} // namespace mlir

#endif // HIP_PLUGIN_FUSION_PASSES_H
