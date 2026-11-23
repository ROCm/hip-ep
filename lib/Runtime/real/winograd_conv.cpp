/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
#include "../op_state.h"
#include "../runtime_state_internal.h"
#include "hip_custom_kernels.h"

#include <cstdio>
#include <string>

#include <hip/hip_runtime.h>

//===----------------------------------------------------------------------===//
// Winograd F(2x2, 3x3) conv (hip.winograd_conv)
//===----------------------------------------------------------------------===//
//
// The compiler builds one kernel per conv shape and epilogue and embeds its
// code object in the op; this entry only prepares the filter and launches it.
// The kernel reads the filter as a [rows, 3, K, C] Winograd store, not as the
// [K, C, 3, 3] model weight. The weight is a model constant, so the store is
// built on the first call and kept in the op state for the session.
//
// Kernel arguments, in order: input, filter store, bias (when present),
// output; each a data pointer. Each call launches the conv kernel and then
// `<name>_fixup` from the same code object, on the session stream.

namespace {

struct WinogradConvState : OpStateT<WinogradConvState> {
  RuntimeState *owner = nullptr;
  void *store = nullptr;
  hipModule_t module = nullptr;
  hipFunction_t main = nullptr;
  hipFunction_t fixup = nullptr;
  ~WinogradConvState() {
    if (store)
      (void)hipdnn_ep_state_device_free(owner, store);
    if (module)
      (void)hipModuleUnload(module);
  }
};

hipError_t launch(hipFunction_t func, int64_t grid, int64_t block,
                  void **kernargs, size_t size, hipStream_t stream) {
  void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, kernargs,
                    HIP_LAUNCH_PARAM_BUFFER_SIZE, &size, HIP_LAUNCH_PARAM_END};
  (void)hipGetLastError();
  hipError_t err = hipModuleLaunchKernel(
      func, static_cast<unsigned>(grid), 1, 1, static_cast<unsigned>(block), 1,
      1, /*sharedMemBytes=*/0, stream, /*kernelParams=*/nullptr, config);
  return err != hipSuccess ? err : hipGetLastError();
}

} // namespace

extern "C" int8_t
hipdnn_ep_op_state_construct_winograd_conv(RuntimeState *state, int32_t slot) {
  hipdnn_ep_op_state_set(state, slot, WinogradConvState::create().release());
  return 0;
}

int wrap_winograd_conv(RuntimeState *state, int32_t op_state_slot,
                       const char *kernel_binary, char *func_name,
                       int64_t block_size, int64_t grid_size, const void *input,
                       const void *weights, const void *bias, void *output,
                       int64_t K, int64_t C, int64_t full_transform) {
  // The generated code drops this status, so a failure must also land in the
  // device error flag that inference_compute reads after the stream sync.
  auto finish = [&](int status) {
    if (status != 0)
      hipdnn_ep_state_set_error_flag(state, status);
    return status;
  };
  OP_PROFILE(
      "winograd_conv",
      [&] {
        char b[96];
        snprintf(b, sizeof(b), "%s,K=%lld,C=%lld",
                 func_name ? func_name : "(null)", (long long)K, (long long)C);
        return std::string(b);
      },
      state);
  if (!state || !kernel_binary || !func_name || !input || !weights || !output ||
      K <= 0 || C <= 0) {
    fprintf(stderr, "[REAL] wrap_winograd_conv: invalid argument\n");
    return finish(-1);
  }
  auto *ws = WinogradConvState::get_op_state(state, op_state_slot);
  if (!ws) {
    fprintf(stderr, "[REAL] wrap_winograd_conv: no op state in slot %d\n",
            op_state_slot);
    return finish(-1);
  }
  if (!ws->store) {
    const size_t bytes =
        static_cast<size_t>(full_transform ? 9 : 12) * K * C * sizeof(uint16_t);
    void *store = nullptr;
    if (hipdnn_ep_state_device_alloc(state, &store, bytes) != hipSuccess) {
      fprintf(stderr, "[REAL] wrap_winograd_conv: filter store allocation "
                      "failed\n");
      return finish(-1);
    }
    int status = hip_winograd_f23_weight_transform_f16(
        hipdnn_ep_state_get_stream(state), weights, store, K, C,
        full_transform ? 1 : 0);
    if (status != 0) {
      (void)hipdnn_ep_state_device_free(state, store);
      fprintf(stderr, "[REAL] wrap_winograd_conv: filter transform failed\n");
      return finish(status);
    }
    ws->owner = state;
    ws->store = store;
  }

  // The code object holds the conv kernel and `<func_name>_fixup`, which
  // recomputes the corner outputs the conv kernel's edge loads miss; see
  // lib/Compiler/WinogradKernel.cpp.
  if (!ws->module) {
    hipModule_t module = nullptr;
    hipError_t err = hipModuleLoadData(&module, kernel_binary);
    if (err != hipSuccess) {
      fprintf(stderr,
              "[REAL] wrap_winograd_conv: hipModuleLoadData failed: %s\n",
              hipGetErrorString(err));
      return finish(-1);
    }
    std::string fixupName = std::string(func_name) + "_fixup";
    if (hipModuleGetFunction(&ws->main, module, func_name) != hipSuccess ||
        hipModuleGetFunction(&ws->fixup, module, fixupName.c_str()) !=
            hipSuccess) {
      fprintf(stderr, "[REAL] wrap_winograd_conv: kernel '%s' not found\n",
              func_name);
      (void)hipModuleUnload(module);
      return finish(-1);
    }
    ws->module = module;
  }

  void *kernargs[4];
  size_t n = 0;
  kernargs[n++] = const_cast<void *>(input);
  kernargs[n++] = ws->store;
  if (bias)
    kernargs[n++] = const_cast<void *>(bias);
  kernargs[n++] = output;
  RUNTIME_DEBUG_LOG("[REAL] wrap_winograd_conv(func=%s, K=%lld, C=%lld, "
                    "bias=%d)\n",
                    func_name, (long long)K, (long long)C, bias ? 1 : 0);
  hipStream_t stream =
      static_cast<hipStream_t>(hipdnn_ep_state_get_stream(state));
  const size_t size = n * sizeof(void *);
  hipError_t err =
      launch(ws->main, grid_size, block_size, kernargs, size, stream);
  if (err == hipSuccess) {
    // One 32-lane group per (corner position, output channel) pair.
    constexpr int64_t kFixupBlock = 256, kGroupsPerBlock = kFixupBlock / 32;
    const int64_t fixupGrid = (12 * K + kGroupsPerBlock - 1) / kGroupsPerBlock;
    err = launch(ws->fixup, fixupGrid, kFixupBlock, kernargs, size, stream);
  }
  if (err != hipSuccess) {
    fprintf(stderr, "[REAL] wrap_winograd_conv: launch of '%s' failed: %s\n",
            func_name, hipGetErrorString(err));
    return finish(-1);
  }
  return 0;
}
