/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef HIPDNN_EP_HIP_BACKEND_CLIENT_MOCK_H
#define HIPDNN_EP_HIP_BACKEND_CLIENT_MOCK_H

// Mock-mode stub of hip::Backend / hip::GetBackend.
//
// The mock runtime has no GPU and no per-gfx backend DLL to load. Anything
// that would call into the real backend (today: only the CK conv path,
// which is itself a no-op stub in mock_gpu.cpp) skips the call entirely.
// hipdnn_ep_runtime_state.cpp's init/cleanup paths leave backend_holder
// as nullptr in mock builds; ops that would touch it are gated by the
// mock CK-conv stub returning success without dispatching.
//
// This stub exists solely so the SHARED hipdnn_ep_runtime_state.cpp TU
// can `#include "hip_backend.h"` unconditionally and have it resolve in
// both real and mock builds via the per-mode -I path.

#include <memory>

namespace hip {

class Backend; // incomplete; never instantiated in mock builds

inline std::shared_ptr<Backend> GetBackend() {
  // Returning a null shared_ptr signals "no backend available" to callers
  // that can tolerate it. The mock runtime never reaches the CK-conv path
  // (mock_gpu.cpp::wrap_ckConvForward is a no-op stub), so this is never
  // dereferenced in practice.
  return std::shared_ptr<Backend>();
}

} // namespace hip

#endif // HIPDNN_EP_HIP_BACKEND_CLIENT_MOCK_H
