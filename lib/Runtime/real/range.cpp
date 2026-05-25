/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "hip_custom_kernels.h"
#include "runtime_types.h"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>

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
//   1. Stages start / limit / delta to host (3 scalars of `hip_dtype`,
//      synchronous D2H).
//   2. Computes the output length on host using ONNX Range semantics
//      (`ceil((limit-start) / delta)` with empty-range clamping). For
//      integer dtypes the computation is exact in i64. For floating-point
//      dtypes it follows ONNX's `max(ceil((limit - start) / delta), 0)`
//      formulation, evaluated in double precision to avoid f32 rounding
//      bias on borderline inputs (e.g. start=0.0, limit=8.0, delta=1.0
//      must give N=8 rather than N=7 due to a representable-difference
//      underestimate).
//   3. Publishes the resolved length to dyn slot `slot_id`.
//   4. Allocates `length * element_size` bytes from the GPU dyn pool and
//      publishes the buffer pointer to the same slot.
//   5. Launches `hip_range` with the freshly allocated buffer as the
//      output. The EP reads the slot post-compute and D2H-copies the
//      buffer into the actual-sized ORT OrtValue.
//
// Supported element types: i64 (HIP_DTYPE_INT64=2), i32 (=3), f32 (=0).
// These are the only types ONNX Range accepts that the rest of the
// runtime (custom kernel + ORT) also supports. f16 is rejected by ORT
// at graph load time (Range f16 is not in the spec); f64/i16 are rare
// in practice. Mixed-element-type Range is impossible by ONNX Range's
// schema (all three operands share the same T).
extern "C" int wrap_range_dyn(RuntimeState *state, void *start, void *limit,
                              void *delta, int64_t hip_dtype,
                              int32_t slot_id) {
  if (!state || !start || !limit || !delta) {
    RUNTIME_DEBUG_LOG("[REAL] wrap_range_dyn: null argument\n");
    return -1;
  }

  hipStream_t stream = static_cast<hipStream_t>(hipdnn_ep_state_get_stream(state));

  const size_t elem_bytes = hipDTypeElementSize(hip_dtype);
  if (elem_bytes == 0) {
    fprintf(stderr,
            "[REAL] wrap_range_dyn: unsupported hip_dtype=%lld (no element "
            "size known)\n",
            (long long)hip_dtype);
    return -1;
  }

  // Stage the three scalars to host. The dyn buffer is one contiguous
  // {start, limit, delta} layout to issue a single D2H copy instead of
  // three (lower API call count, plus the ranges sit on a single 24B
  // cache line for the host-side compute below).
  alignas(8) unsigned char h_buf[3 * 8] = {0};
  // 8B per slot up to i64/f64; smaller types occupy the low bytes of
  // their slot, so the per-type decode below picks the right prefix.
  if (hipMemcpyAsync(&h_buf[0], start, elem_bytes, hipMemcpyDeviceToHost,
                     stream) != hipSuccess) {
    fprintf(stderr, "[REAL] wrap_range_dyn: D2H start failed\n");
    return -1;
  }
  if (hipMemcpyAsync(&h_buf[8], limit, elem_bytes, hipMemcpyDeviceToHost,
                     stream) != hipSuccess) {
    fprintf(stderr, "[REAL] wrap_range_dyn: D2H limit failed\n");
    return -1;
  }
  if (hipMemcpyAsync(&h_buf[16], delta, elem_bytes, hipMemcpyDeviceToHost,
                     stream) != hipSuccess) {
    fprintf(stderr, "[REAL] wrap_range_dyn: D2H delta failed\n");
    return -1;
  }
  if (hipStreamSynchronize(stream) != hipSuccess) {
    fprintf(stderr, "[REAL] wrap_range_dyn: stream sync after D2H failed\n");
    return -1;
  }

  // ONNX Range semantics for each dtype.
  int64_t N = 0;
  double dbg_start = 0.0, dbg_limit = 0.0, dbg_delta = 0.0;

  if (hip_dtype == /*HIP_DTYPE_INT64=*/2) {
    int64_t s, l, d;
    std::memcpy(&s, &h_buf[0], sizeof(int64_t));
    std::memcpy(&l, &h_buf[8], sizeof(int64_t));
    std::memcpy(&d, &h_buf[16], sizeof(int64_t));
    dbg_start = (double)s; dbg_limit = (double)l; dbg_delta = (double)d;
    if (d > 0 && l > s)        N = (l - s + d - 1) / d;
    else if (d < 0 && l < s)   N = (s - l + (-d) - 1) / (-d);
  } else if (hip_dtype == /*HIP_DTYPE_INT32=*/3) {
    int32_t s, l, d;
    std::memcpy(&s, &h_buf[0], sizeof(int32_t));
    std::memcpy(&l, &h_buf[8], sizeof(int32_t));
    std::memcpy(&d, &h_buf[16], sizeof(int32_t));
    dbg_start = (double)s; dbg_limit = (double)l; dbg_delta = (double)d;
    if (d > 0 && l > s)        N = ((int64_t)l - s + d - 1) / d;
    else if (d < 0 && l < s)   N = ((int64_t)s - l + (-d) - 1) / (-d);
  } else if (hip_dtype == /*HIP_DTYPE_FLOAT32=*/0) {
    float s, l, d;
    std::memcpy(&s, &h_buf[0], sizeof(float));
    std::memcpy(&l, &h_buf[8], sizeof(float));
    std::memcpy(&d, &h_buf[16], sizeof(float));
    dbg_start = (double)s; dbg_limit = (double)l; dbg_delta = (double)d;
    // ONNX f32 Range: N = max(ceil((limit - start) / delta), 0). We
    // evaluate in double to keep the count exact on borderline inputs.
    double diff = (double)l - (double)s;
    double dd = (double)d;
    if (dd > 0.0 && diff > 0.0) {
      double n_f = std::ceil(diff / dd);
      N = (int64_t)n_f;
    } else if (dd < 0.0 && diff < 0.0) {
      double n_f = std::ceil(diff / dd);
      N = (int64_t)n_f;
    }
  } else if (hip_dtype == /*HIP_DTYPE_FLOAT64=*/4) {
    double s, l, d;
    std::memcpy(&s, &h_buf[0], sizeof(double));
    std::memcpy(&l, &h_buf[8], sizeof(double));
    std::memcpy(&d, &h_buf[16], sizeof(double));
    dbg_start = s; dbg_limit = l; dbg_delta = d;
    double diff = l - s;
    if (d > 0.0 && diff > 0.0)        N = (int64_t)std::ceil(diff / d);
    else if (d < 0.0 && diff < 0.0)   N = (int64_t)std::ceil(diff / d);
  } else if (hip_dtype == /*HIP_DTYPE_INT16=*/6) {
    int16_t s, l, d;
    std::memcpy(&s, &h_buf[0], sizeof(int16_t));
    std::memcpy(&l, &h_buf[8], sizeof(int16_t));
    std::memcpy(&d, &h_buf[16], sizeof(int16_t));
    dbg_start = (double)s; dbg_limit = (double)l; dbg_delta = (double)d;
    if (d > 0 && l > s)        N = ((int64_t)l - s + d - 1) / d;
    else if (d < 0 && l < s)   N = ((int64_t)s - l + (-d) - 1) / (-d);
  } else {
    fprintf(stderr,
            "[REAL] wrap_range_dyn: hip_dtype=%lld not supported on the "
            "Category-C path\n",
            (long long)hip_dtype);
    return -1;
  }

  RUNTIME_DEBUG_LOG("[REAL] wrap_range_dyn: dtype=%lld start=%g limit=%g "
                    "delta=%g -> N=%lld slot=%d\n",
                    (long long)hip_dtype, dbg_start, dbg_limit, dbg_delta,
                    (long long)N, slot_id);

  hipdnn_ep_state_publish_dim(state, slot_id, N);

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
