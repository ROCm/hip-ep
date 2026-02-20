/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "LlvmJitLoader.h"

// CRITICAL: morphizen.hpp must be included before other morphizen headers
#include "NativeDllLoader.h"
#include "morphizen/env_config.hpp"
#include "morphizen/morphizen.hpp"
#include <glog/logging.h>

// Environment parameters (global scope, before namespace)
DEF_ENV_PARAM(MORPHIZEN_DEBUG_MLIR_BACKEND, "0")

#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_MLIR_BACKEND) >= n)

namespace mlir_compilation {
namespace customop {

JitHandle::JitHandle(void *engine, const DllFunctions &functions)
    : engine_(engine), functions_(functions) {}

std::optional<JitHandle>
JitHandle::compile(const std::vector<uint8_t> &llvm_ir_bytes) {
  MY_LOG(1) << "Loading LLVM IR and JIT compiling...";
  LOG(FATAL) << "LLVM IR JIT mode not implemented yet";
  // TODO: Implement LLVM ExecutionEngine JIT compilation
  // This requires linking against LLVM JIT libraries
  return std::nullopt;
}

JitHandle::~JitHandle() {
  // TODO: Cleanup LLVM ExecutionEngine
  engine_ = nullptr;
}

JitHandle::JitHandle(JitHandle &&other) noexcept
    : engine_(other.engine_), functions_(other.functions_) {
  other.engine_ = nullptr;
}

JitHandle &JitHandle::operator=(JitHandle &&other) noexcept {
  if (this != &other) {
    // TODO: Cleanup current engine
    engine_ = other.engine_;
    functions_ = other.functions_;
    other.engine_ = nullptr;
  }
  return *this;
}

const DllFunctions &JitHandle::functions() const { return functions_; }

} // namespace customop
} // namespace mlir_compilation
