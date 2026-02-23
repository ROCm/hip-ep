/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "morphizen-mlir-compiler/Compiler/Passes/Passes.h"
#include "mlir/Pass/Pass.h"

namespace morphizen {
namespace compiler {

void registerCompilerPasses() {
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createGenerateInterfacePass();
  });
}

} // namespace compiler
} // namespace morphizen
