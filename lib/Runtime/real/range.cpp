/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "hip_custom_kernels.h"
#include "runtime_types.h"

#include <hip/hip_runtime.h>

int wrap_range(RuntimeState *state, void *start, void *limit, void *delta,
               void *output, int64_t output_num_elements, int64_t hip_dtype) {
  if (!state || !start || !limit || !delta || !output) {
    RUNTIME_DEBUG_LOG("[REAL] wrap_range: null argument\n");
    return -1;
  }

  void *stream = hipdnn_ep_state_get_stream(state);

  RUNTIME_DEBUG_LOG(
      "[REAL] wrap_range: output_num_elements=%lld, hip_dtype=%lld\n",
      (long long)output_num_elements, (long long)hip_dtype);

  void *deviceErrorFlag = hipdnn_ep_state_get_error_flag_device_ptr(state);
  return hip_range(stream, start, limit, delta, output, output_num_elements,
                   hip_dtype, deviceErrorFlag);
}

// Element size for a hip_dtype_t value. Mirrors the table in
// 3rd-party/custom_kernels/hip/range_kernel.hip; kept local to avoid
// pulling kernel-internal headers.
static size_t hipDTypeElementSize(int64_t hip_dtype) {
  switch (hip_dtype) {
  case 0: return 4; // HIP_DTYPE_FLOAT32
  case 2: return 8; // HIP_DTYPE_INT64
  case 3: return 4; // HIP_DTYPE_INT32
  case 4: return 8; // HIP_DTYPE_FLOAT64
  case 6: return 2; // HIP_DTYPE_INT16
  default: return 0;
  }
}

// Category-C Range wrapper.
//
// Called from generated code when at least one of {start, limit, delta} is
// an intermediate GPU value (not a func-arg). The host-side EP cannot
// pre-resolve the output length because the operands aren't host-readable
// without a D2H stage, so the wrapper:
//
//   1. Stages start / limit / delta to host (3 i64 scalars, synchronous D2H).
//   2. Computes the output length on host (CeilDiv((limit-start), delta)
//      following ONNX Range semantics, with empty-range clamp).
//   3. Publishes the resolved length to dyn slot `slot_id`.
//   4. Allocates `length * element_size` bytes from the GPU dyn pool and
//      publishes the buffer pointer to the same slot.
//   5. Launches `hip_range` with the freshly allocated buffer as the
//      output. The EP reads the slot post-compute and D2H-copies the
//      buffer into the actual-sized ORT OrtValue.
//
// Today we only support i64 element type for the Category-C path (the
// only dtype Qwen-style embedding lookups use); extending to int32/i16/
// float32/double is straightforward (just teach `decodeI64Scalar` to
// handle wider/narrower element types). Mixed-type Range falls back to
// the static `wrap_range` (which the compiler refuses to emit a Category
// -C dispatch for, by construction).
extern "C" int wrap_range_dyn(RuntimeState *state, void *start, void *limit,
                              void *delta, int64_t hip_dtype,
                              int32_t slot_id) {
  if (!state || !start || !limit || !delta) {
    RUNTIME_DEBUG_LOG("[REAL] wrap_range_dyn: null argument\n");
    return -1;
  }

  hipStream_t stream = static_cast<hipStream_t>(hipdnn_ep_state_get_stream(state));

  // Currently only i64 element type is supported on the Category-C path.
  // The compiler-side gate in RangeConversion mirrors this constraint.
  if (hip_dtype != /*HIP_DTYPE_INT64=*/2) {
    fprintf(stderr,
            "[REAL] wrap_range_dyn: hip_dtype=%lld not supported on the "
            "Category-C path (only HIP_DTYPE_INT64 = 2 today)\n",
            (long long)hip_dtype);
    return -1;
  }

  int64_t h_start = 0, h_limit = 0, h_delta = 0;
  if (hipMemcpyAsync(&h_start, start, sizeof(int64_t), hipMemcpyDeviceToHost,
                     stream) != hipSuccess) {
    fprintf(stderr, "[REAL] wrap_range_dyn: D2H start failed\n");
    return -1;
  }
  if (hipMemcpyAsync(&h_limit, limit, sizeof(int64_t), hipMemcpyDeviceToHost,
                     stream) != hipSuccess) {
    fprintf(stderr, "[REAL] wrap_range_dyn: D2H limit failed\n");
    return -1;
  }
  if (hipMemcpyAsync(&h_delta, delta, sizeof(int64_t), hipMemcpyDeviceToHost,
                     stream) != hipSuccess) {
    fprintf(stderr, "[REAL] wrap_range_dyn: D2H delta failed\n");
    return -1;
  }
  if (hipStreamSynchronize(stream) != hipSuccess) {
    fprintf(stderr, "[REAL] wrap_range_dyn: stream sync after D2H failed\n");
    return -1;
  }

  // ONNX Range semantics:
  //   delta > 0 and limit > start  -> N = ceil_div(limit - start, delta)
  //   delta < 0 and limit < start  -> N = ceil_div(start - limit, -delta)
  //   otherwise empty -> N = 0
  int64_t N = 0;
  if (h_delta > 0 && h_limit > h_start) {
    int64_t diff = h_limit - h_start;
    N = (diff + h_delta - 1) / h_delta;
  } else if (h_delta < 0 && h_limit < h_start) {
    int64_t diff = h_start - h_limit;
    int64_t neg = -h_delta;
    N = (diff + neg - 1) / neg;
  }

  RUNTIME_DEBUG_LOG("[REAL] wrap_range_dyn: start=%lld limit=%lld delta=%lld "
                    "-> N=%lld slot=%d\n",
                    (long long)h_start, (long long)h_limit, (long long)h_delta,
                    (long long)N, slot_id);

  hipdnn_ep_state_publish_dim(state, slot_id, N);

  const size_t elem_bytes = hipDTypeElementSize(hip_dtype);
  void *buf = nullptr;
  if (N > 0) {
    buf = hipdnn_ep_state_dyn_pool_alloc(
        state, static_cast<int64_t>(N) * static_cast<int64_t>(elem_bytes));
    if (!buf) {
      fprintf(stderr,
              "[REAL] wrap_range_dyn: dyn_pool_alloc failed for N=%lld\n",
              (long long)N);
      return -1;
    }
  }
  hipdnn_ep_state_publish_buffer(state, slot_id, buf);

  if (N <= 0) {
    // Empty range -- nothing to launch; the slot is published with size 0
    // and a null buffer, the EP recognises that and produces a 0-sized
    // OrtValue.
    return 0;
  }

  void *deviceErrorFlag = hipdnn_ep_state_get_error_flag_device_ptr(state);
  return hip_range(stream, start, limit, delta, buf, N, hip_dtype,
                   deviceErrorFlag);
}
