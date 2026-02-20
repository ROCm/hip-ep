/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef NATIVE_DLL_LOADER_H
#define NATIVE_DLL_LOADER_H

#include "custom_op_mlir.hpp"
#include <cstdint>
#include <optional>
#include <vector>

namespace mlir_compilation {

namespace customop {

struct DllFunctions {
  init_fn init;
  compute_fn compute;
  cleanup_fn cleanup;
};

class DllHandle {
public:
  // Load native DLL from memory
  static std::optional<DllHandle> load(const std::vector<uint8_t> &dll_bytes);

  ~DllHandle();

  // Non-copyable, movable
  DllHandle(const DllHandle &) = delete;
  DllHandle &operator=(const DllHandle &) = delete;
  DllHandle(DllHandle &&other) noexcept;
  DllHandle &operator=(DllHandle &&other) noexcept;

  const DllFunctions &functions() const;

private:
  DllHandle(void *handle, const DllFunctions &functions);

  void *handle_;
  DllFunctions functions_;
};

} // namespace customop
} // namespace mlir_compilation

#endif
