/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "morphizen-mlir-compiler/Dialect/Hip/Transforms/Passes.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace hip {

void registerHipTransformPasses() {
  registerPass(
      []() -> std::unique_ptr<Pass> { return createMemoryPoolingPass(); });
  registerPass([]() -> std::unique_ptr<Pass> {
    return createHipBufferDeallocationPass();
  });
}

} // namespace hip
} // namespace mlir
