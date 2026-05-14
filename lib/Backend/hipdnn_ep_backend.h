/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef HIPDNN_EP_BACKEND_H
#define HIPDNN_EP_BACKEND_H

/*
 * Stable C ABI exposed by per-gfx EP backend DLLs.
 *
 * Each backend DLL is named `hip-backend-gfx<ARCH>.dll` on Windows or
 * `libhip-backend-gfx<ARCH>.so` on Linux (e.g. `hip-backend-gfx1151.dll`).
 * It encapsulates heavy ROCm libraries (Composable Kernel today; future
 * hipBLASLt-tuned kernels, etc.) so the per-ONNX-graph model.dll never has
 * to link against them directly -- model.dll only loads the backend at
 * runtime and forwards op calls through the vtable below.
 *
 * Single exported symbol: `HIPBackendAPI` -- a pointer to a static
 * `HIPBackendVTable` instance owned by the backend DLL. The runtime client
 * (`hip::Backend` in lib/Runtime/real/hip_backend_client.cpp) resolves this one
 * symbol via dlsym/GetProcAddress, dereferences it once, and dispatches all
 * subsequent ops through the function-pointer slots.
 *
 * ABI evolution rule:
 *   - Appending a new op slot at the END of the vtable is ADDITIVE and
 *     DOES NOT bump HIP_BACKEND_API_VERSION. Older clients walk only the
 *     prefix they know; newer clients null-check the slot before calling
 *     so a backend that hasn't implemented the op yet is gracefully
 *     reported as "not implemented" rather than causing a crash.
 *   - Removing an op, reordering slots, or changing an existing slot's
 *     signature MUST bump HIP_BACKEND_API_VERSION. Mismatched versions
 *     are refused at backend-load time.
 *
 * Adding a new op (recipe):
 *   1. Append a new function-pointer field at the END of HIPBackendVTable.
 *   2. Implement the op in lib/Backend/<op>.cpp; add the source to
 *      BACKEND_HIP_SOURCES in lib/Backend/CMakeLists.txt.
 *   3. Wire the impl into the static vtable instance in lib/Backend/main.cpp.
 *   4. Add a thin `void hip::Backend::<Op>(...)` method in
 *      lib/Runtime/real/hip_backend_client.{h,cpp} that null-checks the slot and
 *      throws std::runtime_error on missing-or-failure.
 *   5. Wire the new wrap in lib/Runtime/real/<op>.cpp + the dispatch shim.
 * No edits to lib/Backend/hip_backend.def are required -- the .def file
 * exports only `HIPBackendAPI` and is permanently a 1-line file.
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Bump only on breaking changes (slot removed / reordered / signature
 * changed). Adding a new slot at the tail is additive. */
#define HIP_BACKEND_API_VERSION 1u

/* Name of the single exported pointer in each backend DLL. Pass this to
 * GetProcAddress / dlsym. */
#define HIP_BACKEND_API_SYMBOL "HIPBackendAPI"

/*
 * Scratch buffer provider.
 *
 * The backend would otherwise need to hipMalloc its own GPU scratch for
 * intermediate buffers (NCHW<->NHWGC transposes, weight materialization,
 * CK's internal workspace). Instead, the model.dll registers a callback
 * at session init that returns a GPU pointer to a buffer of at least
 * `needed_bytes`, growing the underlying allocation if necessary. The
 * callback is expected to be cheap (the model.dll's existing workspace
 * is grow-on-demand, never shrinks); the backend may call it once per
 * op invocation to cover that op's worst-case scratch requirement.
 *
 * Lifetime: the provider pointer + ctx must remain valid for as long as
 * the backend may still dispatch ops. model.dll's state_init registers
 * the provider after loading the backend; state_cleanup leaves the
 * registration in place because the backend DLL might still be live for
 * other sessions (and the next session will overwrite the registration
 * with its own ctx anyway).
 *
 * Concurrency: the backend stores ONE (ctx, provider) globally. With
 * concurrent sessions in the same process, the most recent registration
 * wins -- safe because RuntimeState is documented as not thread-safe and
 * sessions in OGA-style workloads are sequential. If concurrent sessions
 * are needed in the future, route the provider through a per-call arg
 * instead of a setter.
 */
typedef void *(*hip_backend_scratch_provider_fn)(void *ctx,
                                                 size_t needed_bytes);

/*
 * Conv forward, fp16, NCHW activations and weights.
 *
 * Inputs / outputs are HIP device pointers; stream is hipStream_t cast to
 * void*. Layouts:
 *   input    [N, C, H, W]                NCHW row-major, fp16 elements
 *   weights  [K, C/group, kernel_h, kernel_w] fp16
 *   bias     [K] or NULL                 fp16
 *   output   [N, K, Ho, Wo]              NCHW row-major, fp16 elements
 *
 * The backend handles internal NCHW <-> NHWGC transposes for the CK kernel;
 * caller never sees the layout change. Asymmetric padding is permitted iff
 * the underlying kernel supports it; otherwise the backend returns -1.
 *
 * Returns 0 on success, non-zero on failure. The backend MUST NOT call
 * hipStreamSynchronize internally -- the caller orders execution.
 */
typedef int (*hip_backend_conv_fwd_fp16_nchw_fn)(
    void *stream,
    const void *input, int N, int C, int H, int W,
    const void *weights, int K, int kernel_h, int kernel_w,
    const void *bias,
    void *output, int Ho, int Wo,
    int stride_h, int stride_w,
    int pad_top, int pad_left, int pad_bottom, int pad_right,
    int dilation_h, int dilation_w,
    int group);

/*
 * Op vtable. The exported `HIPBackendAPI` is a pointer to a static instance
 * of this struct living in the backend DLL's .rdata.
 *
 * Layout discipline:
 *   - `abi_version` MUST stay first (clients read it before trusting any
 *     other field).
 *   - `arch` is second (also pre-ABI-version-check stable).
 *   - `init` / `shutdown` are optional (NULL => no-op for the client).
 *   - Op slots follow. NULL means the backend does not implement the op;
 *     `hip::Backend::<Op>` will throw std::runtime_error("backend gfxXXXX
 *     does not implement <op>") rather than dispatching.
 *   - New op slots APPEND ONLY. Older clients see the truncated tail as
 *     "not present" and report missing-op accordingly.
 */
typedef struct HIPBackendVTable {
  /* Header. Stable across all ABI versions. */
  uint32_t abi_version; /* == HIP_BACKEND_API_VERSION */
  const char *arch;     /* e.g. "gfx1151"; static storage */

  /* Lifecycle. NULL => no-op. */
  int (*init)(void);
  void (*shutdown)(void);

  /* Set the GPU-scratch provider callback. NULL slot => the backend
   * doesn't support external scratch (older backend). When the slot is
   * present, model.dll calls it once at session init; the backend
   * stores the (ctx, provider) and uses the provider for all subsequent
   * scratch needs. See `hip_backend_scratch_provider_fn` above for the
   * lifetime + concurrency contract. */
  void (*set_scratch_provider)(void *ctx,
                               hip_backend_scratch_provider_fn provider);

  /* Ops. NULL => not implemented. Append-only. */
  hip_backend_conv_fwd_fp16_nchw_fn conv_fwd_fp16_nchw;
} HIPBackendVTable;

#ifdef __cplusplus
}
#endif

#endif /* HIPDNN_EP_BACKEND_H */
