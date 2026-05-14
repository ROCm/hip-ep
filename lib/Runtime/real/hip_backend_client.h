/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef HIPDNN_EP_HIP_BACKEND_CLIENT_H
#define HIPDNN_EP_HIP_BACKEND_CLIENT_H

// Runtime-side RAII client of the per-gfx backend DLL
// (hip-backend-gfx<ARCH>.dll on Windows, libhip-backend-gfx<ARCH>.so on
// Linux). Detects the device's gfx arch via hipGetDeviceProperties at
// construction time, loads the matching backend DLL, validates the ABI
// version and arch reported by the vtable, and calls init/shutdown from
// ctor/dtor.
//
// Lifetime model:
//   - hip::GetBackend() returns std::shared_ptr<hip::Backend>, lazy
//     constructed on first call. An internal std::weak_ptr re-uses an
//     existing instance across concurrent callers; when the last
//     shared_ptr drops, the dtor unloads the backend DLL.
//   - inference_init parks one strong ref on RuntimeState->backend_holder
//     so the backend lives for the session even if all op-site shared_ptrs
//     drop between Compute() calls. Per-op call sites just call
//     hip::GetBackend()->Op(...) without worrying about lifetime.
//
// Op surface:
//   Each op method (today: Conv) null-checks the corresponding vtable slot
//   and throws std::runtime_error if the backend doesn't implement the op
//   or if the op returns a non-zero status. The extern-C wrap_* functions
//   that generated MLIR code targets convert exceptions to return-codes.

#include <cstddef>
#include <memory>

struct HIPBackendVTable;

// Re-declared here (matches the typedef in hipdnn_ep_backend.h) so callers
// of Backend::SetScratchProvider don't have to include the backend header.
extern "C" typedef void *(*hip_backend_scratch_provider_fn)(void *ctx,
                                                            size_t needed_bytes);

namespace hip {

class Backend {
public:
  Backend(const Backend &) = delete;
  Backend &operator=(const Backend &) = delete;
  ~Backend();

  // Register the scratch provider. Throws std::runtime_error if the
  // backend doesn't expose the slot (older backend). Subsequent op calls
  // will pull GPU scratch from the provider; see hipdnn_ep_backend.h for
  // the lifetime + concurrency contract.
  void SetScratchProvider(void *ctx, hip_backend_scratch_provider_fn provider);

  // Conv slot. Throws std::runtime_error on missing slot or non-zero status.
  // Argument list mirrors HIPBackendVTable::conv_fwd_fp16_nchw 1:1.
  void Conv(void *stream,
            const void *input, int N, int C, int H, int W,
            const void *weights, int K, int kernel_h, int kernel_w,
            const void *bias,
            void *output, int Ho, int Wo,
            int stride_h, int stride_w,
            int pad_top, int pad_left, int pad_bottom, int pad_right,
            int dilation_h, int dilation_w,
            int group);

  // Diagnostics.
  const char *Arch() const noexcept;
  unsigned AbiVersion() const noexcept;

private:
  // Constructed only via GetBackend().
  friend std::shared_ptr<Backend> GetBackend();
  Backend();

  void *dll_handle_ = nullptr;
  const HIPBackendVTable *vtable_ = nullptr;
};

// Returns the process-wide backend instance, lazily constructing it on
// first call. Subsequent calls return the same instance via an internal
// weak_ptr cache; when the last shared_ptr drops, the backend DLL is
// unloaded.
//
// Throws std::runtime_error on construction failure (no GPU detected,
// backend DLL not on PATH, ABI mismatch, init() failure). Callers that
// run during inference_init catch and demote to a warning so models that
// don't use the backend keep working.
std::shared_ptr<Backend> GetBackend();

} // namespace hip

#endif // HIPDNN_EP_HIP_BACKEND_CLIENT_H
