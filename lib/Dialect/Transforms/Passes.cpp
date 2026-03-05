/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "udna-compiler/Dialect/Hip/Transforms/Passes.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace hip {

void registerHipTransformPasses() {
  registerPass(
      []() -> std::unique_ptr<Pass> { return createMemoryPoolingPass(); });
}

} // namespace hip
} // namespace mlir
