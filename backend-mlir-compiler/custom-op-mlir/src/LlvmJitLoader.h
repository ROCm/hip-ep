/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef LLVM_JIT_LOADER_H
#define LLVM_JIT_LOADER_H

#include "NativeDllLoader.h"
#include <cstdint>
#include <optional>
#include <vector>

namespace mlir_compilation {
namespace customop {

class JitHandle {
public:
  // JIT compile LLVM IR to executable code
  static std::optional<JitHandle>
  compile(const std::vector<uint8_t> &llvm_ir_bytes);

  ~JitHandle();

  // Non-copyable, movable
  JitHandle(const JitHandle &) = delete;
  JitHandle &operator=(const JitHandle &) = delete;
  JitHandle(JitHandle &&other) noexcept;
  JitHandle &operator=(JitHandle &&other) noexcept;

  const DllFunctions &functions() const;

private:
  JitHandle(void *engine, const DllFunctions &functions);

  void *engine_; // LLVM ExecutionEngine
  DllFunctions functions_;
};

} // namespace customop
} // namespace mlir_compilation

#endif
