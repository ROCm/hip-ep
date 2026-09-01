/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// Runtime wrappers for ONNX QuantizeLinear / DequantizeLinear.
//
// This layer is the glue between the lowering ABI (all-i64 scalars plus bare
// shape pointers) and the GPU kernels (hip_quantize_linear /
// hip_dequantize_linear in lib/Runtime/Kernels). It handles:
//   * pulling the stream out of the opaque RuntimeState
//   * axis normalization (the lowering forwards the ONNX attribute verbatim,
//     so it may be negative)
//   * dtype mapping, HIPDNN_EP_DATATYPE_* -> HIP_DTYPE_*
//   * granularity self-check, i.e. whether per-tensor / per-axis / blocked is
//     consistent with scale_shape
//

#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
#include "hip_custom_kernels.h"
#include "runtime_types.h"

#include <cstdio>

namespace {

int map_to_hip_dtype(int64_t hipdnn_dtype) {
  switch (hipdnn_dtype) {
  case HIPDNN_EP_DATATYPE_FLOAT:
    return HIP_DTYPE_FLOAT32;
  case HIPDNN_EP_DATATYPE_HALF:
    return HIP_DTYPE_FLOAT16;
  case HIPDNN_EP_DATATYPE_BFLOAT16:
    return HIP_DTYPE_BFLOAT16;
  case HIPDNN_EP_DATATYPE_INT32:
    return HIP_DTYPE_INT32;
  case HIPDNN_EP_DATATYPE_INT64:
    return HIP_DTYPE_INT64;
  case HIPDNN_EP_DATATYPE_INT8:
    return HIP_DTYPE_INT8;
  case HIPDNN_EP_DATATYPE_UINT8:
    return HIP_DTYPE_UINT8;
  case HIPDNN_EP_DATATYPE_INT16:
    return HIP_DTYPE_INT16;
  case HIPDNN_EP_DATATYPE_UINT16:
    return HIP_DTYPE_UINT16;
  default:
    return -1;
  }
}

int64_t total_elements(const int64_t *shape, int64_t rank) {
  int64_t total = 1;
  for (int64_t i = 0; i < rank; ++i)
    total *= shape[i];
  return total;
}

struct QdqParams {
  void *stream;
  int axis;
  int in_dtype;
  int scale_dtype;
  int out_dtype;
  int64_t total;
};

int prepare_qdq(const char *name, RuntimeState *state, const void *input,
                const void *scale, const void *output,
                const int64_t *input_shape, int64_t input_rank,
                const int64_t *scale_shape, int64_t scale_rank, int64_t axis,
                int64_t block_size, int64_t input_dtype, int64_t scale_dtype,
                int64_t output_dtype, QdqParams *out) {
  if (!state || !input || !scale || !output || !input_shape || !scale_shape) {
    fprintf(stderr, "[REAL] %s: null required argument\n", name);
    return -1;
  }

  int axis_n = static_cast<int>(axis);
  if (axis_n < 0)
    axis_n += static_cast<int>(input_rank);
  if (scale_rank > 0 && (axis_n < 0 || axis_n >= input_rank)) {
    fprintf(stderr,
            "[REAL] %s: axis out of range (input_rank=%lld axis=%lld)\n", name,
            (long long)input_rank, (long long)axis);
    return -1;
  }

  if (block_size > 0) {
    if (scale_rank != input_rank || input_rank == 0) {
      fprintf(stderr,
              "[REAL] %s: blocked quantization needs scale_rank == "
              "input_rank (got %lld vs %lld)\n",
              name, (long long)scale_rank, (long long)input_rank);
      return -1;
    }
    int64_t blocks = scale_shape[axis_n];
    int64_t covered = blocks * block_size;
    if (covered < input_shape[axis_n] ||
        (blocks - 1) * block_size >= input_shape[axis_n]) {
      fprintf(stderr,
              "[REAL] %s: scale extent %lld with block_size %lld does not "
              "cover input extent %lld on axis %d\n",
              name, (long long)blocks, (long long)block_size,
              (long long)input_shape[axis_n], axis_n);
      return -1;
    }
  } else if (scale_rank == 1 && scale_shape[0] != 1 &&
             scale_shape[0] != input_shape[axis_n]) {
    fprintf(stderr,
            "[REAL] %s: per-axis scale length %lld does not match input "
            "extent %lld on axis %d\n",
            name, (long long)scale_shape[0], (long long)input_shape[axis_n],
            axis_n);
    return -1;
  }

  int in_dt = map_to_hip_dtype(input_dtype);
  int scale_dt = map_to_hip_dtype(scale_dtype);
  int out_dt = map_to_hip_dtype(output_dtype);
  if (in_dt < 0 || scale_dt < 0 || out_dt < 0) {
    fprintf(stderr,
            "[REAL] %s: unsupported dtype (input=%s scale=%s output=%s)\n",
            name, hipdnn_ep_datatype_name(input_dtype),
            hipdnn_ep_datatype_name(scale_dtype),
            hipdnn_ep_datatype_name(output_dtype));
    return -1;
  }

  int64_t total = total_elements(input_shape, input_rank);
  if (total <= 0)
    return 1;

  out->stream = hipdnn_ep_state_get_stream(state);
  out->axis = axis_n;
  out->in_dtype = in_dt;
  out->scale_dtype = scale_dt;
  out->out_dtype = out_dt;
  out->total = total;
  return 0;
}

} // namespace

extern "C" int wrap_quantize_linear(
    RuntimeState *state, const void *input, const void *scale,
    const void *zero_point, void *output, const int64_t *input_shape,
    int64_t input_rank, const int64_t *scale_shape, int64_t scale_rank,
    int64_t axis, int64_t block_size, int64_t precision, int64_t saturate,
    int64_t input_dtype, int64_t scale_dtype, int64_t output_dtype) {
  OP_PROFILE(
      "quantize_linear",
      [&] {
        char b[128];
        snprintf(b, sizeof(b), "axis=%lld bs=%lld ir=%lld sr=%lld",
                 (long long)axis, (long long)block_size, (long long)input_rank,
                 (long long)scale_rank);
        return std::string(b);
      },
      state);

  QdqParams p;
  int rc = prepare_qdq("wrap_quantize_linear", state, input, scale, output,
                       input_shape, input_rank, scale_shape, scale_rank, axis,
                       block_size, input_dtype, scale_dtype, output_dtype, &p);
  if (rc != 0)
    return rc < 0 ? rc : 0;

  RUNTIME_DEBUG_LOG(
      "[REAL] wrap_quantize_linear: input_dtype=%s scale_dtype=%s "
      "output_dtype=%s axis=%d block_size=%lld precision=%lld saturate=%lld "
      "has_zp=%d input_rank=%lld scale_rank=%lld total=%lld\n",
      hipdnn_ep_datatype_name(input_dtype),
      hipdnn_ep_datatype_name(scale_dtype),
      hipdnn_ep_datatype_name(output_dtype), p.axis, (long long)block_size,
      (long long)precision, (long long)saturate, zero_point ? 1 : 0,
      (long long)input_rank, (long long)scale_rank, (long long)p.total);

  // `saturate` stops here. Per the ONNX spec it only affects float8 targets,
  // and every quantized type the kernel supports is an integer whose range
  // clamp is unconditional, so forwarding it would only invite the kernel to
  // branch on something that cannot change its result.
  return hip_quantize_linear(
      p.stream, input, scale, zero_point, output, input_shape,
      static_cast<int>(input_rank), scale_shape, static_cast<int>(scale_rank),
      p.axis, static_cast<int>(block_size), static_cast<int>(precision),
      p.in_dtype, p.scale_dtype, p.out_dtype);
}

extern "C" int
wrap_dequantize_linear(RuntimeState *state, const void *input,
                       const void *scale, const void *zero_point, void *output,
                       const int64_t *input_shape, int64_t input_rank,
                       const int64_t *scale_shape, int64_t scale_rank,
                       int64_t axis, int64_t block_size, int64_t input_dtype,
                       int64_t scale_dtype, int64_t output_dtype) {
  OP_PROFILE(
      "dequantize_linear",
      [&] {
        char b[128];
        snprintf(b, sizeof(b), "axis=%lld bs=%lld ir=%lld sr=%lld",
                 (long long)axis, (long long)block_size, (long long)input_rank,
                 (long long)scale_rank);
        return std::string(b);
      },
      state);

  QdqParams p;
  int rc = prepare_qdq("wrap_dequantize_linear", state, input, scale, output,
                       input_shape, input_rank, scale_shape, scale_rank, axis,
                       block_size, input_dtype, scale_dtype, output_dtype, &p);
  if (rc != 0)
    return rc < 0 ? rc : 0;

  RUNTIME_DEBUG_LOG(
      "[REAL] wrap_dequantize_linear: input_dtype=%s scale_dtype=%s "
      "output_dtype=%s axis=%d block_size=%lld has_zp=%d input_rank=%lld "
      "scale_rank=%lld total=%lld\n",
      hipdnn_ep_datatype_name(input_dtype),
      hipdnn_ep_datatype_name(scale_dtype),
      hipdnn_ep_datatype_name(output_dtype), p.axis, (long long)block_size,
      zero_point ? 1 : 0, (long long)input_rank, (long long)scale_rank,
      (long long)p.total);

  return hip_dequantize_linear(p.stream, input, scale, zero_point, output,
                               input_shape, static_cast<int>(input_rank),
                               scale_shape, static_cast<int>(scale_rank),
                               p.axis, static_cast<int>(block_size), p.in_dtype,
                               p.scale_dtype, p.out_dtype);
}
