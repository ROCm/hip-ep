/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//
// Runtime wrappers for standard ONNX QuantizeLinear / DequantizeLinear.

#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "hip_custom_kernels.h"
#include "nan_check.h"
#include "runtime_types.h"

#include <cstdio>

static int hipdnn_ep_to_hip_dtype_qdq(int64_t data_type) {
  switch (data_type) {
  case HIPDNN_EP_DATATYPE_FLOAT:
    return HIP_DTYPE_FLOAT32;
  case HIPDNN_EP_DATATYPE_HALF:
    return HIP_DTYPE_FLOAT16;
  case HIPDNN_EP_DATATYPE_BFLOAT16:
    return HIP_DTYPE_BFLOAT16;
  case HIPDNN_EP_DATATYPE_INT8:
    return HIP_DTYPE_INT8;
  case HIPDNN_EP_DATATYPE_UINT8:
    return HIP_DTYPE_UINT8;
  default:
    return -1;
  }
}

extern "C" int wrap_dequantize_linear(
    RuntimeState *state, void *input, void *scale, void *zero_point,
    void *output, int64_t num_elements, int64_t input_data_type,
    int64_t scale_data_type, int64_t output_data_type,
    int64_t scale_num_elements, int64_t inner_size) {
  if (!state || !input || !scale || !output) {
    fprintf(stderr, "wrap_dequantize_linear: null required argument\n");
    return -1;
  }
  int in_dtype = hipdnn_ep_to_hip_dtype_qdq(input_data_type);
  int scale_dtype = hipdnn_ep_to_hip_dtype_qdq(scale_data_type);
  int out_dtype = hipdnn_ep_to_hip_dtype_qdq(output_data_type);
  if (in_dtype < 0 || scale_dtype < 0 || out_dtype < 0) {
    fprintf(stderr,
            "wrap_dequantize_linear: unsupported dtypes in=%lld scale=%lld "
            "out=%lld\n",
            (long long)input_data_type, (long long)scale_data_type,
            (long long)output_data_type);
    return -1;
  }
  void *stream = hipdnn_ep_state_get_stream(state);
  RUNTIME_DEBUG_LOG(
      "[REAL] wrap_dequantize_linear: n=%lld in=%s scale=%s out=%s "
      "scale_n=%lld inner=%lld\n",
      (long long)num_elements, hipdnn_ep_datatype_name(input_data_type),
      hipdnn_ep_datatype_name(scale_data_type),
      hipdnn_ep_datatype_name(output_data_type),
      (long long)scale_num_elements, (long long)inner_size);
  int rc = hip_dequantize_linear(
      stream, input, scale, zero_point, output, num_elements, in_dtype,
      scale_dtype, out_dtype, scale_num_elements, inner_size);
  if (rc == 0)
    nan_trace_check("dequantize_linear", output, num_elements,
                    hipdnn_ep_datatype_size(output_data_type));
  return rc;
}

extern "C" int wrap_quantize_linear(
    RuntimeState *state, void *input, void *scale, void *zero_point,
    void *output, int64_t num_elements, int64_t input_data_type,
    int64_t scale_data_type, int64_t output_data_type,
    int64_t scale_num_elements, int64_t inner_size) {
  if (!state || !input || !scale || !output) {
    fprintf(stderr, "wrap_quantize_linear: null required argument\n");
    return -1;
  }
  int in_dtype = hipdnn_ep_to_hip_dtype_qdq(input_data_type);
  int scale_dtype = hipdnn_ep_to_hip_dtype_qdq(scale_data_type);
  int out_dtype = hipdnn_ep_to_hip_dtype_qdq(output_data_type);
  if (in_dtype < 0 || scale_dtype < 0 || out_dtype < 0) {
    fprintf(stderr,
            "wrap_quantize_linear: unsupported dtypes in=%lld scale=%lld "
            "out=%lld\n",
            (long long)input_data_type, (long long)scale_data_type,
            (long long)output_data_type);
    return -1;
  }
  void *stream = hipdnn_ep_state_get_stream(state);
  RUNTIME_DEBUG_LOG(
      "[REAL] wrap_quantize_linear: n=%lld in=%s scale=%s out=%s "
      "scale_n=%lld inner=%lld\n",
      (long long)num_elements, hipdnn_ep_datatype_name(input_data_type),
      hipdnn_ep_datatype_name(scale_data_type),
      hipdnn_ep_datatype_name(output_data_type),
      (long long)scale_num_elements, (long long)inner_size);
  int rc = hip_quantize_linear(stream, input, scale, zero_point, output,
                               num_elements, in_dtype, scale_dtype, out_dtype,
                               scale_num_elements, inner_size);
  if (rc == 0)
    nan_trace_check("quantize_linear", output, num_elements,
                    hipdnn_ep_datatype_size(output_data_type));
  return rc;
}
