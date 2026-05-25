/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// ONNX NonZero -- canonical Category-C dynamic-output-shape op.
//
// The output shape is `[R, N]` where N is the number of non-zero elements
// in the input. N is only known after the count kernel runs, so the
// wrapper:
//   1. allocates a tiny [int64] counter from the dyn pool, runs
//      hip_nonzero_count over `input`,
//   2. D2H + hipStreamSynchronize to get N (we cannot avoid this --
//      Category C semantics require it),
//   3. publishes N to the metadata slot (`slot_id`) via
//      hipdnn_ep_state_publish_dim, allocates an exact-sized output
//      from the dyn pool, publishes the buffer via
//      hipdnn_ep_state_publish_buffer,
//   4. allocates + zeros a separate atomic counter and runs
//      hip_nonzero_fill, which writes the [R, N] int64 row-major
//      indices into the published buffer.
//
// The EP later reads slot_id and copies (or aliases) the GPU buffer
// into the ORT output OrtValue once the stream is synced.
//
// `input_shape_host` is a stack-resident int64[R] array assembled by
// the HipToLLVM lowering (compile-time constants for static dims,
// MemRefDescriptor `sizes[]` for dynamic dims). We hipMemcpyAsync it
// H2D into the dyn pool so the fill kernel can do per-axis
// coordinate decomposition matching ONNX semantics.

#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
#include "hip_custom_kernels.h"

#include <cstdio>
#include <hip/hip_runtime.h>

namespace {

// HIPDNN_EP_DATATYPE_* -> HIP_DTYPE_* (see hip_custom_kernels.h for the
// supported types of hip_nonzero_count/fill).
int nonzero_hipdnn_to_hip_dtype(int64_t hipdnn_type) {
  switch (hipdnn_type) {
  case HIPDNN_EP_DATATYPE_HALF:
    return HIP_DTYPE_FLOAT16;
  case HIPDNN_EP_DATATYPE_FLOAT:
    return HIP_DTYPE_FLOAT32;
  case HIPDNN_EP_DATATYPE_DOUBLE:
    return HIP_DTYPE_FLOAT64;
  case HIPDNN_EP_DATATYPE_INT32:
    return HIP_DTYPE_INT32;
  case HIPDNN_EP_DATATYPE_INT64:
    return HIP_DTYPE_INT64;
  // ONNX `bool` is marshalled as INT8/UINT8 by the EP; both reuse the
  // byte-wise kernel slot. The predicate is `x != 0`, signedness is
  // irrelevant.
  case HIPDNN_EP_DATATYPE_INT8:
  case HIPDNN_EP_DATATYPE_UINT8:
    return HIP_DTYPE_INT8;
  default:
    return -1;
  }
}

} // namespace

int wrap_nonzero(RuntimeState *state, const void *input,
                 int64_t input_num_elements, int64_t input_rank,
                 int64_t input_data_type, int32_t slot_id,
                 const int64_t *input_shape_host) {
  OP_PROFILE(
      "nonzero",
      [&] {
        char b[64];
        snprintf(b, sizeof(b), "r%lld:n%lld:%s", (long long)input_rank,
                 (long long)input_num_elements,
                 hipdnn_ep_datatype_name(input_data_type));
        return std::string(b);
      },
      state);

  if (!state || !input || !input_shape_host) {
    fprintf(stderr, "[REAL] wrap_nonzero: null state/input/shape\n");
    return -1;
  }
  if (input_rank <= 0) {
    fprintf(stderr, "[REAL] wrap_nonzero: invalid rank=%lld\n",
            (long long)input_rank);
    return -1;
  }
  int hip_dtype = nonzero_hipdnn_to_hip_dtype(input_data_type);
  if (hip_dtype < 0) {
    fprintf(stderr,
            "[REAL] wrap_nonzero: unsupported input_data_type=%s(%lld)\n",
            hipdnn_ep_datatype_name(input_data_type),
            (long long)input_data_type);
    return -1;
  }

  void *stream_v = hipdnn_ep_state_get_stream(state);
  hipStream_t stream = static_cast<hipStream_t>(stream_v);

  // ===== Pass 1: count =====
  // Single int64 counter allocated from the dyn pool. We do NOT use a
  // free hipMalloc here -- the dyn pool is the per-Compute() scratch
  // arena for exactly this kind of one-shot scratch, and it lets
  // wrap_nonzero share the same memory across multiple NonZero ops
  // within one Compute().
  void *count_dev = hipdnn_ep_state_dyn_pool_alloc(state, sizeof(int64_t));
  if (!count_dev) {
    fprintf(stderr, "[REAL] wrap_nonzero: dyn_pool_alloc(count) failed\n");
    return -1;
  }
  // The count kernel writes the final count itself (initialises to 0
  // internally) -- we don't need a hipMemsetAsync here.
  int rc = hip_nonzero_count(stream_v, input, input_num_elements,
                             (int64_t)hip_dtype, count_dev);
  if (rc != 0) {
    fprintf(stderr, "[REAL] wrap_nonzero: hip_nonzero_count rc=%d\n", rc);
    return rc;
  }

  // ===== D2H sync (unavoidable for Category C) =====
  int64_t N = 0;
  hipError_t herr = hipMemcpyAsync(&N, count_dev, sizeof(int64_t),
                                   hipMemcpyDeviceToHost, stream);
  if (herr != hipSuccess) {
    fprintf(stderr, "[REAL] wrap_nonzero: D2H count failed: %s\n",
            hipGetErrorString(herr));
    return -1;
  }
  herr = hipStreamSynchronize(stream);
  if (herr != hipSuccess) {
    fprintf(stderr, "[REAL] wrap_nonzero: stream sync (post-count) failed: %s\n",
            hipGetErrorString(herr));
    return -1;
  }

  // Even when N == 0 we MUST publish the dim and a (possibly empty)
  // buffer so the EP-side dim-spec resolver doesn't trip the
  // "unpublished slot" LOG(FATAL).
  hipdnn_ep_state_publish_dim(state, slot_id, N);

  void *out_dev = nullptr;
  int64_t out_bytes = input_rank * N * (int64_t)sizeof(int64_t);
  // When N==0 we still allocate a 1-byte sentinel from the dyn pool and
  // publish that as the slot buffer. Consumers do null-pointer guarding
  // (transpose / scatter_nd / gather_nd) on their input buffer pointers
  // even when the matching slot-published dim says "0 elements", so
  // publishing a literal nullptr trips a "null required argument" early-
  // exit and we miss the legitimate no-op (scatter_nd should still
  // copy data -> output even when 0 updates land). The sentinel makes
  // the pointer non-null while ensuring 0 reads/writes happen because
  // every consumer is also wired to read the dim from slot `slot_id`.
  int64_t alloc_bytes = out_bytes > 0 ? out_bytes : 1;
  out_dev = hipdnn_ep_state_dyn_pool_alloc(state, alloc_bytes);
  if (!out_dev) {
    fprintf(stderr,
            "[REAL] wrap_nonzero: dyn_pool_alloc(output) failed (R=%lld, "
            "N=%lld, bytes=%lld)\n",
            (long long)input_rank, (long long)N, (long long)alloc_bytes);
    return -1;
  }
  hipdnn_ep_state_publish_buffer(state, slot_id, out_dev);

  if (N == 0) {
    RUNTIME_DEBUG_LOG(
        "[REAL] wrap_nonzero: N=0, skipping fill (slot=%d, R=%lld)\n", slot_id,
        (long long)input_rank);
    return 0;
  }

  // ===== Pass 2: fill =====
  // Stage the per-call int64[R] input_shape into a dyn-pool slab so the
  // GPU fill kernel can read it. Same hipMemcpyAsync the rest of the
  // runtime uses for small staging arrays.
  int64_t shape_bytes = input_rank * (int64_t)sizeof(int64_t);
  void *shape_dev = hipdnn_ep_state_dyn_pool_alloc(state, shape_bytes);
  if (!shape_dev) {
    fprintf(stderr, "[REAL] wrap_nonzero: dyn_pool_alloc(shape) failed\n");
    return -1;
  }
  herr = hipMemcpyAsync(shape_dev, input_shape_host, shape_bytes,
                        hipMemcpyHostToDevice, stream);
  if (herr != hipSuccess) {
    fprintf(stderr, "[REAL] wrap_nonzero: H2D input_shape failed: %s\n",
            hipGetErrorString(herr));
    return -1;
  }

  // Scratch buffers required by the deterministic two-pass scan
  // fill. `atomic_idx_dev` is retained for source compatibility with
  // older bitcode paths (the new fill ignores it) but allocated all
  // the same so the call site doesn't need to special-case nullptr.
  void *atomic_idx_dev =
      hipdnn_ep_state_dyn_pool_alloc(state, sizeof(int64_t));
  // thread_off: 1 int32 per input element. block_sums: 1 int32 per
  // 64-element block. Block size is fixed at kNonZeroScanBS=64 in the
  // kernel; keep this constant in sync if the kernel ever changes.
  constexpr int64_t kBS = 64;
  int64_t thread_off_bytes = input_num_elements * (int64_t)sizeof(int32_t);
  int64_t num_blocks = (input_num_elements + kBS - 1) / kBS;
  if (num_blocks <= 0)
    num_blocks = 1;
  int64_t block_sums_bytes = num_blocks * (int64_t)sizeof(int32_t);
  void *thread_off_dev =
      hipdnn_ep_state_dyn_pool_alloc(state, thread_off_bytes);
  void *block_sums_dev =
      hipdnn_ep_state_dyn_pool_alloc(state, block_sums_bytes);
  if (!atomic_idx_dev || !thread_off_dev || !block_sums_dev) {
    fprintf(stderr,
            "[REAL] wrap_nonzero: dyn_pool_alloc(scratch) failed "
            "(atomic_idx=%p, thread_off=%p[%lld B], block_sums=%p[%lld B])\n",
            atomic_idx_dev, thread_off_dev, (long long)thread_off_bytes,
            block_sums_dev, (long long)block_sums_bytes);
    return -1;
  }

  RUNTIME_DEBUG_LOG(
      "[REAL] wrap_nonzero: slot=%d, R=%lld, N=%lld, dtype=%s -> fill\n",
      slot_id, (long long)input_rank, (long long)N,
      hipdnn_ep_datatype_name(input_data_type));

  rc = hip_nonzero_fill(stream_v, input, input_num_elements,
                        (int64_t)hip_dtype, input_rank, N, shape_dev,
                        atomic_idx_dev, thread_off_dev, block_sums_dev,
                        out_dev);
  if (rc != 0) {
    fprintf(stderr, "[REAL] wrap_nonzero: hip_nonzero_fill rc=%d\n", rc);
    return rc;
  }
  return 0;
}
