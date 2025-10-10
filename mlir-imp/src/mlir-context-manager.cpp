/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "mlir-context-manager.hpp"
#include "mlir/Dialect/Arith/IR/Arith.h"  // for arith dialect, ConstOp, AddFOp
#include "mlir/Dialect/Func/IR/FuncOps.h" // for func dialect, FuncOp
namespace morphizen {
namespace mlir_impl {

MLIRContextManager& MLIRContextManager::getInstance() {
  static MLIRContextManager instance;
  return instance;
}

mlir::MLIRContext& MLIRContextManager::getContext() { return context_; }

MLIRContextManager::MLIRContextManager() {
  // Initialize MLIR context with necessary dialects
  context_.getOrLoadDialect<mlir::BuiltinDialect>();
  context_.getOrLoadDialect<mlir::arith::ArithDialect>();
  context_.getOrLoadDialect<mlir::func::FuncDialect>();
  context_.allowUnregisteredDialects();
  // Add other required dialects here as needed
}

} // namespace mlir_impl
} // namespace morphizen
