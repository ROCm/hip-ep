/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef LLVM_IR_JIT_H
#define LLVM_IR_JIT_H

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace mlir_compilation::customop {

// In-process loader for the per-model LLVM bitcode artifact carried in
// an EPContext tar (`model_compiled` entry).
//
// `create()` adds two modules into one ORC LLJIT JITDylib:
//   1. `runtime.bc`  -- per-OS EP runtime, embedded in this DLL.
//   2. `bitcode`     -- per-model module supplied by the caller (emitted
//                       with empty triple/datalayout for OS portability).
//
// External symbols (`hip_*` kernel launchers, libamdhip64, MIOpen,
// hipBLASLt, CRT) resolve through search generators installed on the
// JITDylib (process image + per-OS ROCm DLLs); MSVC-only emutls and
// allocation helpers are injected as absolute symbols on Windows.
class LlvmIrJit {
public:
  ~LlvmIrJit();

  LlvmIrJit(const LlvmIrJit &) = delete;
  LlvmIrJit &operator=(const LlvmIrJit &) = delete;
  LlvmIrJit(LlvmIrJit &&) = delete;
  LlvmIrJit &operator=(LlvmIrJit &&) = delete;

  // The bytes are copied internally; `module_name` is a diagnostic tag.
  // Returns nullptr on parse / JIT-init failure (errors logged via glog).
  static std::unique_ptr<LlvmIrJit> create(const std::vector<uint8_t> &bitcode,
                                           const std::string &module_name,
                                           std::string *error = nullptr);

  // Returns nullptr when the symbol is absent (callers use this to probe
  // optional hooks like `hipdnn_ep_runtime_begin_compute`).
  void *lookup_raw(const char *name) const;

  template <typename R, typename... Args>
  auto get_method(const char *name) const -> R (*)(Args...) {
    return reinterpret_cast<R (*)(Args...)>(lookup_raw(name));
  }

private:
  LlvmIrJit();

  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace mlir_compilation::customop

#endif // LLVM_IR_JIT_H
