/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// Single-symbol export for the per-gfx EP backend DLL.
//
// The backend exposes one variable -- `HIPBackendAPI` -- a pointer to a
// static `HIPBackendVTable` instance that holds:
//   - the ABI version baked into this DLL,
//   - the gfx arch baked at build time via -DHIPDNN_EP_BACKEND_ARCH=...,
//   - optional init/shutdown hooks,
//   - one function pointer per op slot (today: conv_fwd_fp16_nchw).
//
// Adding a new op = (a) implement it in lib/Backend/<op>.cpp, (b) declare a
// file-static forward decl here, (c) plug it into the g_vtable initializer
// below. NO change to lib/Backend/hip_backend.def is required -- only the
// vtable variable is exported.

#include "hipdnn_ep_backend.h"

#include <atomic>

#ifndef HIPDNN_EP_BACKEND_ARCH
#error "HIPDNN_EP_BACKEND_ARCH must be defined at compile time (e.g. gfx1151)"
#endif

#define EP_STR_(x) #x
#define EP_STR(x) EP_STR_(x)

namespace {
std::atomic<int> g_init_state{0}; // 0 = not init, 1 = init
}

// Forward decls of slot implementations. Each impl is `static` linkage in
// its own TU; only the vtable variable below is exported from the DLL.
extern int backend_conv_fwd_fp16_nchw_impl(
    void *stream,
    const void *input, int N, int C, int H, int W,
    const void *weights, int K, int kernel_h, int kernel_w,
    const void *bias,
    void *output, int Ho, int Wo,
    int stride_h, int stride_w,
    int pad_top, int pad_left, int pad_bottom, int pad_right,
    int dilation_h, int dilation_w,
    int group);

// conv.cpp owns its caches; release them on shutdown.
extern void backend_conv_shutdown_impl(void);

// Scratch-provider state. model.dll's state_init writes here once;
// conv.cpp (and any other op TU) reads via backend_get_scratch().
// Plain pointers -- the registration write is a single store and reads
// are single loads; in practice a session's init runs before that
// session's compute, so no ordering issue. With concurrent sessions
// the latest setter wins (documented in hipdnn_ep_backend.h). */
namespace backend_scratch {
void *g_ctx = nullptr;
hip_backend_scratch_provider_fn g_provider = nullptr;
} // namespace backend_scratch

// Visible to conv.cpp et al. Returns a GPU pointer to a buffer of at
// least `needed_bytes`, or nullptr if no provider is registered or the
// provider failed to grow.
extern "C" void *backend_get_scratch(size_t needed_bytes) {
  if (!backend_scratch::g_provider)
    return nullptr;
  return backend_scratch::g_provider(backend_scratch::g_ctx, needed_bytes);
}

namespace {

int backend_init(void) {
  // Cheap: any future one-time setup goes here. The conv impl lazily
  // initializes on first call.
  int expected = 0;
  g_init_state.compare_exchange_strong(expected, 1);
  return 0;
}

void backend_shutdown(void) { backend_conv_shutdown_impl(); }

void backend_set_scratch_provider(void *ctx,
                                  hip_backend_scratch_provider_fn provider) {
  backend_scratch::g_ctx = ctx;
  backend_scratch::g_provider = provider;
}

const HIPBackendVTable g_vtable = {
    /* abi_version          */ HIP_BACKEND_API_VERSION,
    /* arch                 */ EP_STR(HIPDNN_EP_BACKEND_ARCH),
    /* init                 */ &backend_init,
    /* shutdown             */ &backend_shutdown,
    /* set_scratch_provider */ &backend_set_scratch_provider,
    /* conv_fwd_fp16_nchw   */ &backend_conv_fwd_fp16_nchw_impl,
};

} // namespace

// Single export. The `DATA` keyword in lib/Backend/hip_backend.def tells
// the MSVC linker this is a variable, not a function -- without it
// GetProcAddress would return a thunk and dereferencing would crash.
#ifdef _WIN32
#define HIP_BACKEND_EXPORT __declspec(dllexport)
#else
#define HIP_BACKEND_EXPORT __attribute__((visibility("default")))
#endif

extern "C" HIP_BACKEND_EXPORT const HIPBackendVTable *const HIPBackendAPI =
    &g_vtable;
