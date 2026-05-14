/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef HIPDNN_EP_HIP_BACKEND_CLIENT_MOCK_H
#define HIPDNN_EP_HIP_BACKEND_CLIENT_MOCK_H

// Mock-mode stub of hip::Backend / hip::GetBackend.
//
// The mock runtime has no GPU and no per-gfx backend DLL to load. The
// shared hipdnn_ep_runtime_state.cpp still has to call methods on a
// hip::Backend (e.g. SetScratchProvider) at compile time, so this stub
// provides a COMPLETE class with no-op method bodies -- the calls
// compile cleanly but never run because GetBackend() returns an empty
// shared_ptr in mock mode (and the call site is gated by `if
// (state->backend_holder)`).
//
// Method signatures here MUST stay in sync with the real
// lib/Runtime/real/hip_backend_client.h's class Backend; otherwise the
// shared TU won't compile in mock mode.

#include <cstddef>
#include <memory>

extern "C" typedef void *(*hip_backend_scratch_provider_fn)(
    void *ctx, size_t needed_bytes);

namespace hip {

// Stub. Never instantiated -- mock GetBackend() returns an empty shared_ptr.
class Backend {
public:
  void SetScratchProvider(void * /*ctx*/,
                          hip_backend_scratch_provider_fn /*provider*/) {}
};

inline std::shared_ptr<Backend> GetBackend() {
  // Returning a null shared_ptr signals "no backend available" to callers
  // that can tolerate it. The mock runtime never reaches the CK-conv path
  // (mock_gpu.cpp::wrap_ckConvForward is a no-op stub), so the stub class
  // above is never instantiated and its method bodies never run.
  return std::shared_ptr<Backend>();
}

} // namespace hip

#endif // HIPDNN_EP_HIP_BACKEND_CLIENT_MOCK_H
