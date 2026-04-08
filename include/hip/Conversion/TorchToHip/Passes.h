/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef HIP_CONVERSION_TORCHTOHIP_PASSES_H
#define HIP_CONVERSION_TORCHTOHIP_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace hip {

/// Creates a pass that converts Torch dialect operations to HIP dialect.
std::unique_ptr<Pass> createConvertTorchToHipPass();

} // namespace hip
} // namespace mlir

#endif // HIP_CONVERSION_TORCHTOHIP_PASSES_H
