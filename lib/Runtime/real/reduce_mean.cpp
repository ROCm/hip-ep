/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//
// Runtime wrapper for hip_reduce_mean.  The compiler emits a single call per
// onnx.ReduceMean op with the input/output element counts already computed
// at lowering time.  inner_size > 1 indicates a strided (non-tail-axis)
// reduction.

#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "hip_custom_kernels.h"
#include "nan_check.h"
#include "runtime_types.h"

#include <cstdio>
#include <cstdint>
#include <vector>

static int hipdnn_ep_to_hip_dtype_reduce(int64_t data_type) {
  switch (data_type) {
  case HIPDNN_EP_DATATYPE_FLOAT:
    return HIP_DTYPE_FLOAT32;
  case HIPDNN_EP_DATATYPE_HALF:
    return HIP_DTYPE_FLOAT16;
  case HIPDNN_EP_DATATYPE_BFLOAT16:
    return HIP_DTYPE_BFLOAT16;
  default:
    return -1;
  }
}

extern "C" int wrap_reduce_mean(RuntimeState *state, void *input, void *output,
                                int64_t num_input_elements,
                                int64_t num_output_elements,
                                int64_t data_type,
                                int64_t inner_size) {
  if (!state || !input || !output) {
    fprintf(stderr, "wrap_reduce_mean: null argument\n");
    return -1;
  }

  int hip_dtype = hipdnn_ep_to_hip_dtype_reduce(data_type);
  if (hip_dtype < 0) {
    fprintf(stderr, "wrap_reduce_mean: unsupported data_type %lld (%s)\n",
            (long long)data_type, hipdnn_ep_datatype_name(data_type));
    return -1;
  }

  void *stream = hipdnn_ep_state_get_stream(state);
  int64_t reduce_size = num_input_elements / num_output_elements;

  RUNTIME_DEBUG_LOG(
      "[REAL] wrap_reduce_mean: dtype=%s, in=%lld, out=%lld, inner=%lld\n",
      hipdnn_ep_datatype_name(data_type), (long long)num_input_elements,
      (long long)num_output_elements, (long long)inner_size);

  if (inner_size > 1 && data_type == HIPDNN_EP_DATATYPE_FLOAT &&
      num_input_elements > 0 && num_output_elements > 0) {
    hipDeviceSynchronize();
    std::vector<float> h_in(num_input_elements),
        h_out(num_output_elements, 0.0f);
    hipMemcpy(h_in.data(), input, num_input_elements * sizeof(float),
              hipMemcpyDeviceToHost);

    int64_t outer_size = num_output_elements / inner_size;
    for (int64_t o = 0; o < outer_size; o++) {
      for (int64_t i = 0; i < inner_size; i++) {
        double sum = 0.0;
        for (int64_t r = 0; r < reduce_size; r++)
          sum += h_in[(o * reduce_size + r) * inner_size + i];
        h_out[o * inner_size + i] = static_cast<float>(sum / reduce_size);
      }
    }

    hipMemcpy(output, h_out.data(), num_output_elements * sizeof(float),
              hipMemcpyHostToDevice);
    hipDeviceSynchronize();
    nan_trace_check("reduce_mean", output, num_output_elements);
    return 0;
  }

  int rc = hip_reduce_mean(stream, input, output, num_input_elements,
                           num_output_elements, hip_dtype);
  if (rc != 0 && data_type == HIPDNN_EP_DATATYPE_FLOAT &&
      num_input_elements > 0 && num_output_elements > 0) {
    fprintf(stderr,
            "[reduce_mean] hip_reduce_mean FAILED rc=%d in=%lld out=%lld "
            "-- host fallback\n",
            rc, (long long)num_input_elements, (long long)num_output_elements);
    fflush(stderr);
    hipDeviceSynchronize();
    (void)hipGetLastError();
    std::vector<float> h_in(num_input_elements),
        h_out(num_output_elements, 0.0f);
    hipMemcpy(h_in.data(), input, num_input_elements * sizeof(float),
              hipMemcpyDeviceToHost);
    for (int64_t o = 0; o < num_output_elements; o++) {
      double sum = 0.0;
      for (int64_t r = 0; r < reduce_size; r++)
        sum += h_in[o * reduce_size + r];
      h_out[o] = static_cast<float>(sum / reduce_size);
    }
    hipMemcpy(output, h_out.data(), num_output_elements * sizeof(float),
              hipMemcpyHostToDevice);
    hipDeviceSynchronize();
    rc = 0;
  }
  nan_trace_check("reduce_mean", output, num_output_elements);
  return rc;
}
