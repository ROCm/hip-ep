/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
#include "runtime_types.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <string>
#include <unordered_map>

#include <hip/hip_runtime.h>

// RocMLIR dispatch runtime wrapper (hip.rocmlir).
//
// The generated IR embeds a pre-compiled GPU kernel (ELF/HSACO blob) in
// `kernel_binary`, stages the operand data pointers (inputs first, then output)
// into `kernargs` (a packed buffer of `size` bytes), and passes the launch
// geometry derived from the tuned perfConfig: `block_size` threads per block,
// `grid_size` blocks per grid (rocMLIR's convention, see MIGraphX's
// get_launch_params: global work-items = block_size * grid_size).
//
// The kernel takes its arguments as a single packed buffer, so we hand
// `kernargs` to the driver through the HIP_LAUNCH_PARAM_BUFFER_POINTER "extra"
// config (the module-launch equivalent of MIGraphX gpu::kernel::launch).

namespace {

// Loaded hipModule_t + resolved hipFunction_t for one embedded binary. The
// binary blob is a stable module-level constant, so the (blob pointer, name)
// pair uniquely identifies a kernel across calls; cache to avoid reloading the
// module on every inference.
struct LoadedKernel {
  hipModule_t module = nullptr;
  hipFunction_t func = nullptr;
};

std::mutex g_kernel_cache_mutex;
std::unordered_map<const void *, LoadedKernel> g_kernel_cache;

const LoadedKernel *getOrLoadKernel(const char *kernel_binary,
                                    const char *func_name) {
  std::lock_guard<std::mutex> lock(g_kernel_cache_mutex);
  auto it = g_kernel_cache.find(kernel_binary);
  if (it != g_kernel_cache.end())
    return &it->second;

  LoadedKernel loaded;
  hipError_t err = hipModuleLoadData(&loaded.module, kernel_binary);
  if (err != hipSuccess) {
    fprintf(stderr, "wrap_rocmlir: hipModuleLoadData failed: %s\n",
            hipGetErrorString(err));
    return nullptr;
  }
  err = hipModuleGetFunction(&loaded.func, loaded.module, func_name);
  if (err != hipSuccess) {
    fprintf(stderr, "wrap_rocmlir: hipModuleGetFunction('%s') failed: %s\n",
            func_name, hipGetErrorString(err));
    (void)hipModuleUnload(loaded.module);
    return nullptr;
  }

  auto res = g_kernel_cache.emplace(kernel_binary, loaded);
  return &res.first->second;
}

} // namespace

int wrap_rocmlir(RuntimeState *state, const char *kernel_binary,
                 char *func_name, int64_t block_size, int64_t grid_size,
                 void *kernargs, size_t size) {
  if (!state) {
    fprintf(stderr, "Invalid state in wrap_rocmlir\n");
    return -1;
  }

  OP_PROFILE(
      "rocmlir",
      [&] {
        char b[96];
        snprintf(b, sizeof(b), "%s,grid=%lld,block=%lld",
                 func_name ? func_name : "(null)", (long long)grid_size,
                 (long long)block_size);
        return std::string(b);
      },
      state);

  RUNTIME_DEBUG_LOG("[REAL] wrap_rocmlir(func=%s, block_size=%lld, "
                    "grid_size=%lld, kernargs_size=%zu)\n",
                    func_name ? func_name : "(null)", (long long)block_size,
                    (long long)grid_size, size);

  const LoadedKernel *kernel = getOrLoadKernel(kernel_binary, func_name);
  if (!kernel)
    return -1;

  hipStream_t stream =
      static_cast<hipStream_t>(hipdnn_ep_state_get_stream(state));

  // The kernel reads its arguments from a single packed buffer; pass it via the
  // driver's "extra" config rather than the per-arg kernelParams array.
  void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, kernargs,
                    HIP_LAUNCH_PARAM_BUFFER_SIZE, &size, HIP_LAUNCH_PARAM_END};

  // Clear any stale last-error before launch so the post-launch check reflects
  // only this dispatch.
  (void)hipGetLastError();
  hipError_t err = hipModuleLaunchKernel(
      kernel->func, static_cast<unsigned>(grid_size), 1, 1,
      static_cast<unsigned>(block_size), 1, 1, /*sharedMemBytes=*/0, stream,
      /*kernelParams=*/nullptr, config);
  if (err != hipSuccess) {
    fprintf(stderr, "wrap_rocmlir: hipModuleLaunchKernel failed: %s\n",
            hipGetErrorString(err));
    return -1;
  }
  hipError_t launchErr = hipGetLastError();
  if (launchErr != hipSuccess) {
    fprintf(stderr, "wrap_rocmlir: kernel launch error: %s\n",
            hipGetErrorString(launchErr));
    return -1;
  }
  return 0;
}
