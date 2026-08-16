/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
#include "hip_custom_kernels.h"
#include "runtime_types.h"

#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

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
  OP_PROFILE(
      "memcpy_2d",
      [&] {
        char b[64];
        snprintf(b, sizeof(b), "w=%zu h=%zu", width, height);
        return std::string(b);
      },
      state);
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

  RUNTIME_DEBUG_LOG("[REAL] wrap_hipMemcpy2DAsync: dpitch=%zu spitch=%zu "
                    "width=%zu height=%zu (D2D async) begin\n",
                    dst_pitch, src_pitch, width, height);

  hipStream_t stream =
      static_cast<hipStream_t>(hipdnn_ep_state_get_stream(state));

  hipError_t err =
      hipMemcpy2DAsync(dst_ptr, dst_pitch, src_ptr, src_pitch, width, height,
                       hipMemcpyDeviceToDevice, stream);

  RUNTIME_DEBUG_LOG("[REAL] wrap_hipMemcpy2DAsync: end -> %d\n", (int)err);

  if (err != hipSuccess) {
    fprintf(stderr,
            "wrap_hipMemcpy2DAsync: copy failed (width=%zu height=%zu): %s\n",
            width, height, hipGetErrorString(err));
    return -1;
  }

  return 0;
}

// Parallel strided D2D copy via a single hip_strided_copy kernel launch,
// expressed in ELEMENT units (not bytes). This is the fast path for a pitched
// copy whose rows are very thin: hipMemcpy2DAsync processes each row as a
// separate tiny transfer on the copy engine, so a copy of `row_elems` elements
// per row over `height` rows degenerates into `height` micro-transfers (e.g.
// the sinusoidal position-embedding sin/cos interleave produces row_elems=1,
// height=40000 -> ~480ms of copy-engine stall). One kernel launch does the
// whole strided copy with one thread per element instead.
//
// Maps the 2D pitched geometry onto hip_strided_copy's rank-1 outer form:
//   outer_sizes = {height}, src/dst outer strides = {pitch_elems},
//   row_elems   = contiguous inner suffix length (stride 1 on both sides).
// On any kernel failure (unsupported element size, launch error) falls back to
// hipMemcpy2DAsync so correctness never depends on the kernel succeeding.
int wrap_strided_copy(RuntimeState *state, void *dst_ptr, const void *src_ptr,
                      int64_t elem_bytes, int64_t height,
                      int64_t src_pitch_elems, int64_t dst_pitch_elems,
                      int64_t row_elems) {
  OP_PROFILE(
      "strided_copy",
      [&] {
        char b[64];
        snprintf(b, sizeof(b), "h=%lld row=%lld es=%lld", (long long)height,
                 (long long)row_elems, (long long)elem_bytes);
        return std::string(b);
      },
      state);
  if (!state) {
    fprintf(stderr, "wrap_strided_copy: null state\n");
    return -1;
  }
  if (!dst_ptr || !src_ptr) {
    fprintf(stderr, "wrap_strided_copy: null pointer\n");
    return -1;
  }
  if (height == 0 || row_elems == 0)
    return 0;

  RUNTIME_DEBUG_LOG("[REAL] wrap_strided_copy: elem=%lld height=%lld "
                    "spitch_e=%lld dpitch_e=%lld row_e=%lld begin\n",
                    (long long)elem_bytes, (long long)height,
                    (long long)src_pitch_elems, (long long)dst_pitch_elems,
                    (long long)row_elems);

  hipStream_t stream =
      static_cast<hipStream_t>(hipdnn_ep_state_get_stream(state));

  const int64_t outer_sizes[1] = {height};
  const int64_t src_outer_strides[1] = {src_pitch_elems};
  const int64_t dst_outer_strides[1] = {dst_pitch_elems};
  int rc = hip_strided_copy(static_cast<void *>(stream), dst_ptr, src_ptr,
                            elem_bytes, /*outer_rank=*/1, outer_sizes,
                            src_outer_strides, dst_outer_strides, row_elems,
                            /*outer_total=*/height);

  RUNTIME_DEBUG_LOG("[REAL] wrap_strided_copy: end -> %d\n", rc);

  if (rc == 0)
    return 0;

  // Kernel could not handle this copy (e.g. odd element size) -- fall back to
  // the correct (if slower) pitched DMA path.
  RUNTIME_DEBUG_LOG("[REAL] wrap_strided_copy: kernel rc=%d, falling back to "
                    "hipMemcpy2DAsync\n",
                    rc);
  size_t width_bytes = static_cast<size_t>(row_elems) * elem_bytes;
  size_t src_pitch = static_cast<size_t>(src_pitch_elems) * elem_bytes;
  size_t dst_pitch = static_cast<size_t>(dst_pitch_elems) * elem_bytes;
  hipError_t err = hipMemcpy2DAsync(dst_ptr, dst_pitch, src_ptr, src_pitch,
                                    width_bytes, static_cast<size_t>(height),
                                    hipMemcpyDeviceToDevice, stream);
  if (err != hipSuccess) {
    fprintf(stderr, "wrap_strided_copy: fallback 2D copy failed: %s\n",
            hipGetErrorString(err));
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
  // CPU-only profiling: this call issues a hipStreamSynchronize, so its cost is
  // host-side wait time (GPU drain), invisible to GPU per-op events. Surfacing
  // it as a profiled row attributes the dynamic-shape readback sync overhead
  // (Range / Reshape(-1) / readback_dim) that otherwise lands in "outside
  // scopes".
  OP_PROFILE_CPU("readback_i32", state);
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
  // CPU-only profiling: like readback_i32 this issues a hipStreamSynchronize;
  // its cost is host-side GPU-drain time. Profiling it attributes the
  // Range/Expand/Pad/Loop shape-readback sync overhead.
  OP_PROFILE_CPU("readback_scalar", state);
  if (!state || !host_dst || !device_scalar || num_bytes <= 0) {
    fprintf(stderr, "hipdnn_ep_readback_scalar: invalid argument\n");
    return;
  }
  hipStream_t stream =
      static_cast<hipStream_t>(hipdnn_ep_state_get_stream(state));
  // hipMemcpyDefault (not DeviceToHost): the source may be host-accessible
  // memory (host-mapped scratch from the host-scalar materialization pass, or a
  // UMA pool), where an explicit D2H fails `invalid argument`. Direction is
  // inferred from the pointer via UVA. Mirrors hipdnn_ep_readback_i32.
  hipError_t err =
      hipMemcpyAsync(host_dst, device_scalar, static_cast<size_t>(num_bytes),
                     hipMemcpyDefault, stream);
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

void hipdnn_ep_readback_shape_i64(RuntimeState *state, int64_t *host_out,
                                  const void *device_vector, int64_t count) {
  OP_PROFILE_CPU("readback_shape", state);
  if (!state || !host_out || count < 0 || (count > 0 && !device_vector)) {
    fprintf(stderr, "hipdnn_ep_readback_shape_i64: invalid argument\n");
    if (state)
      (void)hipdnn_ep_state_set_error_flag(state);
    return;
  }
  if (count == 0)
    return;

  auto fail = [&]() {
    for (int64_t i = 0; i < count; ++i)
      host_out[i] = 0;
    (void)hipdnn_ep_state_set_error_flag(state);
  };

  hipStream_t stream =
      static_cast<hipStream_t>(hipdnn_ep_state_get_stream(state));
  hipError_t err = hipMemcpyAsync(host_out, device_vector,
                                  static_cast<size_t>(count) * sizeof(int64_t),
                                  hipMemcpyDefault, stream);
  if (err != hipSuccess) {
    fprintf(stderr, "hipdnn_ep_readback_shape_i64: D2H copy failed: %s\n",
            hipGetErrorString(err));
    fail();
    return;
  }
  err = hipStreamSynchronize(stream);
  if (err != hipSuccess) {
    fprintf(stderr, "hipdnn_ep_readback_shape_i64: stream sync failed: %s\n",
            hipGetErrorString(err));
    fail();
    return;
  }

  bool invalid = false;
  for (int64_t i = 0; i < count; ++i) {
    invalid |= host_out[i] < 0;
    if (host_out[i] < 0)
      host_out[i] = 0;
  }
  if (invalid)
    (void)hipdnn_ep_state_set_error_flag(state);
}

int hipdnn_ep_readback_control(RuntimeState *state, int64_t *host_out,
                               const void *const *device_sources,
                               const int64_t *element_counts,
                               const int64_t *element_bytes,
                               int64_t source_count, int64_t total_count) {
  OP_PROFILE_CPU("readback_control", state);

  auto fail = [&]() {
    if (state)
      (void)hipdnn_ep_state_set_error_flag(state);
    return -1;
  };
  if (!state || !host_out || !device_sources || !element_counts ||
      !element_bytes || source_count <= 0 || total_count < 0 ||
      static_cast<uint64_t>(source_count) >
          std::numeric_limits<size_t>::max() / sizeof(size_t) ||
      static_cast<uint64_t>(total_count) >
          std::numeric_limits<size_t>::max() / sizeof(int64_t)) {
    fprintf(stderr, "hipdnn_ep_readback_control: invalid argument\n");
    return fail();
  }

  for (int64_t i = 0; i < total_count; ++i)
    host_out[i] = 0;

  size_t totalBytes = 0;
  int64_t checkedTotal = 0;
  std::vector<size_t> byteOffsets(static_cast<size_t>(source_count));
  for (int64_t i = 0; i < source_count; ++i) {
    int64_t count = element_counts[i];
    int64_t width = element_bytes[i];
    if (count < 0 || (width != 4 && width != 8) ||
        (count > 0 && !device_sources[i]) ||
        count > std::numeric_limits<int64_t>::max() - checkedTotal ||
        static_cast<uint64_t>(count) >
            std::numeric_limits<size_t>::max() / static_cast<size_t>(width)) {
      fprintf(stderr,
              "hipdnn_ep_readback_control: invalid source descriptor %lld\n",
              (long long)i);
      return fail();
    }
    size_t bytes = static_cast<size_t>(count) * static_cast<size_t>(width);
    if (bytes > std::numeric_limits<size_t>::max() - totalBytes) {
      fprintf(stderr, "hipdnn_ep_readback_control: byte count overflow\n");
      return fail();
    }
    byteOffsets[static_cast<size_t>(i)] = totalBytes;
    totalBytes += bytes;
    checkedTotal += count;
  }
  if (checkedTotal != total_count) {
    fprintf(stderr,
            "hipdnn_ep_readback_control: flattened count mismatch "
            "(descriptors=%lld, output=%lld)\n",
            (long long)checkedTotal, (long long)total_count);
    return fail();
  }

  std::vector<unsigned char> staging(totalBytes);
  hipStream_t stream =
      static_cast<hipStream_t>(hipdnn_ep_state_get_stream(state));
  bool enqueueFailed = false;
  for (int64_t i = 0; i < source_count; ++i) {
    int64_t count = element_counts[i];
    if (count == 0)
      continue;
    size_t bytes =
        static_cast<size_t>(count) * static_cast<size_t>(element_bytes[i]);
    hipError_t err =
        hipMemcpyAsync(staging.data() + byteOffsets[static_cast<size_t>(i)],
                       device_sources[i], bytes, hipMemcpyDefault, stream);
    if (err != hipSuccess) {
      fprintf(stderr,
              "hipdnn_ep_readback_control: source %lld D2H enqueue failed: "
              "%s\n",
              (long long)i, hipGetErrorString(err));
      enqueueFailed = true;
    }
  }

  // Exactly one synchronization covers every successfully enqueued source.
  // It is intentionally unconditional after enqueue attempts so a partial
  // enqueue failure cannot leave an earlier copy using `staging` after return.
  hipError_t syncError = hipStreamSynchronize(stream);
  if (syncError != hipSuccess) {
    fprintf(stderr, "hipdnn_ep_readback_control: stream sync failed: %s\n",
            hipGetErrorString(syncError));
  }
  if (enqueueFailed || syncError != hipSuccess)
    return fail();

  int64_t outputIndex = 0;
  for (int64_t i = 0; i < source_count; ++i) {
    if (element_counts[i] == 0)
      continue;
    const unsigned char *source =
        staging.data() + byteOffsets[static_cast<size_t>(i)];
    for (int64_t j = 0; j < element_counts[i]; ++j) {
      if (element_bytes[i] == 4) {
        int32_t value = 0;
        std::memcpy(&value, source + static_cast<size_t>(j) * 4, sizeof(value));
        host_out[outputIndex++] = static_cast<int64_t>(value);
      } else {
        int64_t value = 0;
        std::memcpy(&value, source + static_cast<size_t>(j) * 8, sizeof(value));
        host_out[outputIndex++] = value;
      }
    }
  }
  return 0;
}
