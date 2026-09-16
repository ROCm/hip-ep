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
// Tensor layout (row-major): input is viewed as [num_rows, norm_size], where
// norm_size is the reduction width the lowering derived from the ONNX `axis`
// (the product of the input dims from `axis` on) and num_rows =
// input_num_elements / norm_size. `scale` carries scale_num_elements /
// norm_size gain vectors of that width, applied per row group -- one vector
// shared by all rows in the ordinary case, one per group for a grouped norm.

#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
#include "hip_custom_kernels.h"
#include "runtime_types.h"

#include <cstdio>

int wrap_rms_norm(RuntimeState *state, void *input, void *scale, void *output,
                  int64_t input_num_elements, int64_t scale_num_elements,
                  int64_t norm_num_elements, int64_t element_size_bytes,
                  int64_t axis, float epsilon, int64_t stash_type) {
  OP_PROFILE(
      "layernorm",
      [&] {
        char b[64];
        snprintf(b, sizeof(b), "%lldx%lld",
                 (long long)(norm_num_elements > 0
                                 ? input_num_elements / norm_num_elements
                                 : 0),
                 (long long)norm_num_elements);
        return std::string(b);
      },
      state);

  // `axis` arrives already resolved as norm_num_elements (the lowering has the
  // input shape and folds the two together). stash_type needs no handling: the
  // kernel always accumulates in FP32, which is what stash_type = 1 asks for.
  (void)axis;
  (void)stash_type;

  if (!state || !input || !scale || !output) {
    fprintf(stderr, "Invalid arguments to wrap_rms_norm\n");
    return -1;
  }
  if (scale_num_elements <= 0 || norm_num_elements <= 0) {
    fprintf(stderr,
            "wrap_rms_norm: scale_num_elements=%lld norm_num_elements=%lld\n",
            (long long)scale_num_elements, (long long)norm_num_elements);
    return -1;
  }

  // ONNX reduces over the axes `axis` selects and only then multiplies by
  // scale, so scale may span whole rows instead of widening the reduction.
  // Row tiling covers every scale whose shape is a suffix of the input's; a
  // scale carrying an interior broadcast dim (e.g. [G, 1] against [N, G, D])
  // cannot be expressed that way, so refuse it rather than pair rows with the
  // wrong gains.
  if (scale_num_elements % norm_num_elements != 0) {
    fprintf(stderr,
            "wrap_rms_norm: scale_num_elements=%lld is not a multiple of the "
            "normalized width %lld; a scale that broadcasts along the reduced "
            "axes is unsupported\n",
            (long long)scale_num_elements, (long long)norm_num_elements);
    return -1;
  }

  // ONNX only broadcasts Scale onto X, never the reverse. For that
  // unidirectional broadcast, |X| must be a multiple of |Scale|. Together
  // with the check above this also implies |X| is a multiple of the
  // normalized width, so the integer row split below does not truncate.
  if (input_num_elements % scale_num_elements != 0) {
    fprintf(stderr,
            "wrap_rms_norm: input_num_elements=%lld is not a multiple of "
            "scale_num_elements=%lld; Scale is not unidirectional-"
            "broadcastable onto X under row-tiled packing\n",
            (long long)input_num_elements, (long long)scale_num_elements);
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

  const int64_t norm_size = norm_num_elements;
  const int64_t num_rows = input_num_elements / norm_size;
  const int64_t scale_rows = scale_num_elements / norm_size;

  RUNTIME_DEBUG_LOG("[REAL] wrap_rms_norm: num_rows=%lld, norm_size=%lld, "
                    "scale_rows=%lld, data_type=%s, epsilon=%e, "
                    "total_bytes=%lld\n",
                    (long long)num_rows, (long long)norm_size,
                    (long long)scale_rows,
                    (element_size_bytes == 2) ? "f16" : "f32", (double)epsilon,
                    (long long)(input_num_elements * element_size_bytes));

  return hip_rms_norm(hipdnn_ep_state_get_stream(state), input, scale, output,
                      num_rows, norm_size, scale_rows, epsilon, hip_dtype);
}
