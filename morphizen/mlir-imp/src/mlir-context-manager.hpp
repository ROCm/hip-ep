/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#pragma once

#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/MLIRContext.h"

namespace morphizen {
namespace mlir_impl {

// MLIR Context singleton for the implementation
class MLIRContextManager {
public:
  static MLIRContextManager &getInstance();

  mlir::MLIRContext &getContext();

private:
  MLIRContextManager();

  mlir::MLIRContext context_;
};

} // namespace mlir_impl
} // namespace morphizen
