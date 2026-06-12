/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
#include "runtime_types.h"

#include <cstdio>

// GPU D2D memcpy via hipMemcpyAsync.
// Follows opaque RuntimeState pattern - extracts stream internally
int wrap_hipMemcpyAsync(RuntimeState *state, void *dst_ptr, const void *src_ptr,
                        size_t size_bytes) {
  OP_PROFILE_CPU("memcpy_d2d", state);
  if (!state) {
    fprintf(stderr, "wrap_hipMemcpyAsync: null state\n");
    return -1;
  }
  if (!dst_ptr || !src_ptr) {
    fprintf(stderr, "wrap_hipMemcpyAsync: null pointer\n");
    return -1;
  }
  if (size_bytes == 0) {
    return 0; // No-op for zero-sized copy
  }

  // Extract stream from opaque RuntimeState using accessor function
  // (Maintains abstraction barrier - no direct field access)
  hipStream_t stream =
      static_cast<hipStream_t>(hipdnn_ep_state_get_stream(state));

  hipError_t err = hipMemcpyAsync(dst_ptr, src_ptr, size_bytes,
                                  hipMemcpyDeviceToDevice, stream);

  if (err != hipSuccess) {
    fprintf(stderr, "wrap_hipMemcpyAsync: copy failed (%zu bytes): %s\n",
            size_bytes, hipGetErrorString(err));
    return -1;
  }

  return 0;
}

int wrap_hipMemcpy2DAsync(RuntimeState *state, void *dst_ptr, size_t dst_pitch,
                          const void *src_ptr, size_t src_pitch, size_t width,
                          size_t height) {
  if (!state) {
    fprintf(stderr, "wrap_hipMemcpy2DAsync: null state\n");
    return -1;
  }
  if (!dst_ptr || !src_ptr) {
    fprintf(stderr, "wrap_hipMemcpy2DAsync: null pointer\n");
    return -1;
  }
  if (width == 0 || height == 0) {
    return 0;
  }

  hipStream_t stream =
      static_cast<hipStream_t>(hipdnn_ep_state_get_stream(state));

  hipError_t err =
      hipMemcpy2DAsync(dst_ptr, dst_pitch, src_ptr, src_pitch, width, height,
                       hipMemcpyDeviceToDevice, stream);

  if (err != hipSuccess) {
    fprintf(stderr,
            "wrap_hipMemcpy2DAsync: copy failed (width=%zu height=%zu): %s\n",
            width, height, hipGetErrorString(err));
    return -1;
  }

  return 0;
}

// Synchronize the stream, then read back a device-resident i32 scalar (e.g.
// NonZero's non-zero count). The copy is enqueued after the producing kernel
// on the same stream; the synchronize guarantees both have completed before
// the host value is read. Returns 0 on any failure (a zero extent is a safe,
// inert dynamic dim).
int32_t hipdnn_ep_readback_i32(RuntimeState *state, const void *device_scalar) {
  if (!state || !device_scalar) {
    fprintf(stderr, "hipdnn_ep_readback_i32: null argument\n");
    return 0;
  }
  hipStream_t stream =
      static_cast<hipStream_t>(hipdnn_ep_state_get_stream(state));
  int32_t host_val = 0;
  // hipMemcpyDefault (not DeviceToHost): the source may be host-accessible
  // memory (host-mapped scratch / UMA pool), where an explicit D2H fails
  // `invalid argument`. Direction is inferred from the pointer via UVA.
  hipError_t err = hipMemcpyAsync(&host_val, device_scalar, sizeof(int32_t),
                                  hipMemcpyDefault, stream);
  if (err != hipSuccess) {
    fprintf(stderr, "hipdnn_ep_readback_i32: D2H copy failed: %s\n",
            hipGetErrorString(err));
    return 0;
  }
  err = hipStreamSynchronize(stream);
  if (err != hipSuccess) {
    fprintf(stderr, "hipdnn_ep_readback_i32: stream sync failed: %s\n",
            hipGetErrorString(err));
    return 0;
  }
  return host_val;
}

// Synchronize the stream, then copy a small device-resident scalar of arbitrary
// byte width (1/2/4/8) back into the caller-provided host buffer. Generalises
// hipdnn_ep_readback_i32 for scalars whose element type is not i32 — e.g. the
// i64 limit/start/delta and f32/f16 operands of a data-dependent `onnx.Range`,
// whose trip count must be computed on the host. The copy is enqueued after the
// producing kernel on the same stream; the synchronize guarantees the producing
// kernel has finished before the host reads. Without this, generated code that
// does a bare `memref.load` of a GPU-written scalar reads stale memory on
// targets where the pool is true device memory (it accidentally works where the
// pool is UMA-mapped host-accessible memory) — yielding a zero trip count and a
// collapsed dynamic dimension.
void hipdnn_ep_readback_scalar(RuntimeState *state, void *host_dst,
                               const void *device_scalar, int64_t num_bytes) {
  if (!state || !host_dst || !device_scalar || num_bytes <= 0) {
    fprintf(stderr, "hipdnn_ep_readback_scalar: invalid argument\n");
    return;
  }
  hipStream_t stream =
      static_cast<hipStream_t>(hipdnn_ep_state_get_stream(state));
  hipError_t err =
      hipMemcpyAsync(host_dst, device_scalar, static_cast<size_t>(num_bytes),
                     hipMemcpyDeviceToHost, stream);
  if (err != hipSuccess) {
    fprintf(stderr, "hipdnn_ep_readback_scalar: D2H copy failed: %s\n",
            hipGetErrorString(err));
    return;
  }
  err = hipStreamSynchronize(stream);
  if (err != hipSuccess) {
    fprintf(stderr, "hipdnn_ep_readback_scalar: stream sync failed: %s\n",
            hipGetErrorString(err));
  }
}
