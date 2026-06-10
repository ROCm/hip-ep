/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// Shared plumbing for the ck_dsl-generated drop-in kernels: device-arch
// detection and lazy HSACO loading. The embedded HSACOs are arch-specific
// (gfx1151 today, discrete GPUs to follow), so each op keeps an
// arch -> KernelDef table and selects the entry for the running device:
//
//   auto it = kernelTable().find(ckdsl::deviceArch());
//   if (it == kernelTable().end()) return kRejectFallback;  // -> MIOpen
//   hipFunction_t fn = ckdsl::loadKernel(it->second);
//
// Adding dGPU support is then just another row in that table.

#pragma once

#include <hip/hip_runtime.h>

#include <atomic>
#include <cstdio>
#include <mutex>
#include <string>
#include <unordered_map>

namespace ckdsl {

// Normalized gcnArchName of the active HIP device with any feature suffix
// stripped (e.g. "gfx1151:xnack-" -> "gfx1151"); empty if it can't be
// queried. Cached for the process lifetime.
inline const std::string &deviceArch() {
  static const std::string arch = []() -> std::string {
    int device = 0;
    if (hipGetDevice(&device) != hipSuccess)
      return {};
    hipDeviceProp_t props{};
    if (hipGetDeviceProperties(&props, device) != hipSuccess)
      return {};
    std::string name(props.gcnArchName);
    const auto colon = name.find(':');
    if (colon != std::string::npos)
      name.resize(colon);
    return name;
  }();
  return arch;
}

// One embedded ck_dsl kernel: the HSACO blob, the symbol to launch, a short
// human-readable label for diagnostics, and the launch block size. POD so it
// can live directly as the value of an arch -> kernel table.
struct KernelDef {
  const unsigned char *hsaco;
  const char *symbol;
  const char *label;
  unsigned block_size;
};

// Resolve a KernelDef to a launchable hipFunction_t, loading and caching its
// module on first use (at most once per process, thread-safe). Returns
// nullptr if the HSACO failed to load on this device, in which case the
// caller should fall back to its baseline path.
inline hipFunction_t loadKernel(const KernelDef &def) {
  struct Loaded {
    hipFunction_t function = nullptr;
    std::atomic<bool> ok{false};
    std::once_flag once;
  };
  static std::mutex mutex;
  static std::unordered_map<const char *, Loaded> cache;

  Loaded *entry;
  {
    std::lock_guard<std::mutex> lock(mutex);
    entry = &cache[def.symbol];
  }
  std::call_once(entry->once, [&]() {
    hipModule_t module = nullptr;
    hipError_t err = hipModuleLoadData(&module, def.hsaco);
    if (err != hipSuccess) {
      fprintf(stderr, "ck_dsl: hipModuleLoadData(%s) failed: %s\n", def.label,
              hipGetErrorString(err));
      return;
    }
    hipFunction_t function = nullptr;
    err = hipModuleGetFunction(&function, module, def.symbol);
    if (err != hipSuccess) {
      fprintf(stderr, "ck_dsl: hipModuleGetFunction(%s) failed: %s\n",
              def.symbol, hipGetErrorString(err));
      hipModuleUnload(module);
      return;
    }
    entry->function = function;
    entry->ok.store(true, std::memory_order_release);
  });
  return entry->ok.load(std::memory_order_acquire) ? entry->function : nullptr;
}

} // namespace ckdsl
