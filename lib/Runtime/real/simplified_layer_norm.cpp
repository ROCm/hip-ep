/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// SimplifiedLayerNormalization / RMSNormalization (RMS Norm):
//
//   rms    = sqrt(mean(input^2, axis) + epsilon)
//   output = (input / rms) * scale
//
// One block per row HIP kernel (see rms_norm_kernel.hip). Internal math is
// FP32 regardless of I/O dtype, and fp16 rows of even width take a packed
// __half2 path.
//
// Tensor layout (row-major): input is viewed as [num_rows, hidden_dim], where
// hidden_dim = scale_num_elements and num_rows = input_num_elements /
// hidden_dim. `scale` is [hidden_dim], broadcast across rows.

#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
#include "hip_custom_kernels.h"
#include "runtime_types.h"

#include <cstdio>

int wrap_rms_norm(RuntimeState *state, void *input, void *scale, void *output,
                  int64_t input_num_elements, int64_t scale_num_elements,
                  int64_t element_size_bytes, int64_t axis, float epsilon,
                  int64_t stash_type) {
  OP_PROFILE(
      "layernorm",
      [&] {
        char b[64];
        snprintf(b, sizeof(b), "%lldx%lld",
                 (long long)(scale_num_elements > 0
                                 ? input_num_elements / scale_num_elements
                                 : 0),
                 (long long)scale_num_elements);
        return std::string(b);
      },
      state);

  // The kernel derives the normalized extent from scale_num_elements, so the
  // ONNX `axis` and `stash_type` attributes are not consumed here.
  (void)axis;
  (void)stash_type;

  if (!state || !input || !scale || !output) {
    fprintf(stderr, "Invalid arguments to wrap_rms_norm\n");
    return -1;
  }
  if (scale_num_elements <= 0) {
    fprintf(stderr, "wrap_rms_norm: scale_num_elements=%lld\n",
            (long long)scale_num_elements);
    return -1;
  }

  int hip_dtype;
  if (element_size_bytes == 2)
    hip_dtype = HIP_DTYPE_FLOAT16;
  else if (element_size_bytes == 4)
    hip_dtype = HIP_DTYPE_FLOAT32;
  else {
    fprintf(stderr,
            "wrap_rms_norm: unsupported element_size %lld "
            "(supported: 2=fp16, 4=fp32)\n",
            (long long)element_size_bytes);
    return -1;
  }

  int64_t hidden_dim = scale_num_elements;
  int64_t num_rows = input_num_elements / hidden_dim;

  RUNTIME_DEBUG_LOG("[REAL] wrap_rms_norm: num_rows=%lld, hidden_dim=%lld, "
                    "data_type=%s, epsilon=%e, total_bytes=%lld\n",
                    (long long)num_rows, (long long)hidden_dim,
                    (element_size_bytes == 2) ? "f16" : "f32", (double)epsilon,
                    (long long)(input_num_elements * element_size_bytes));

  return hip_rms_norm(hipdnn_ep_state_get_stream(state), input, scale, output,
                      num_rows, hidden_dim, epsilon, hip_dtype);
}
