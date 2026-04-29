/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//
// Runtime wrapper for hip_expand (ONNX Expand: numpy-style broadcast).
//
// The compiler-side lowering hands us:
//   - in_shape         : the input's dense shape   (length = in_rank)
//   - in_strides_elems : the input's row-major element strides (length = in_rank)
//   - out_shape        : the output shape          (length = out_rank)
//
// We right-align the input to the output rank and produce the
// `effective_in_strides` array the kernel actually consumes (length =
// out_rank, with 0 on broadcast axes).  Doing it here keeps the kernel
// itself oblivious to rank-diff bookkeeping.

#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "hip_custom_kernels.h"
#include "runtime_types.h"

#include <cstdint>
#include <cstdio>

#define HIP_EXPAND_RT_MAX_RANK 8

static int hipdnn_ep_to_hip_dtype_expand(int64_t data_type) {
  switch (data_type) {
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
  default:
    return -1;
  }
}

extern "C" int wrap_expand(RuntimeState *state, void *input, void *output,
                           const int64_t *in_shape,
                           const int64_t *in_strides_elems,
                           const int64_t *out_shape, int64_t in_rank,
                           int64_t out_rank, int64_t data_type) {
  if (!state || !input || !output || !in_shape || !in_strides_elems ||
      !out_shape) {
    fprintf(stderr, "wrap_expand: null argument\n");
    return -1;
  }
  if (in_rank < 0 || out_rank <= 0 || out_rank > HIP_EXPAND_RT_MAX_RANK ||
      in_rank > out_rank) {
    fprintf(stderr,
            "wrap_expand: invalid ranks in_rank=%lld out_rank=%lld\n",
            (long long)in_rank, (long long)out_rank);
    return -1;
  }

  int hip_dtype = hipdnn_ep_to_hip_dtype_expand(data_type);
  if (hip_dtype < 0) {
    fprintf(stderr, "wrap_expand: unsupported data_type %lld (%s)\n",
            (long long)data_type, hipdnn_ep_datatype_name(data_type));
    return -1;
  }

  // Right-align: build an out_rank-long view of the input's shape and
  // its row-major strides, prepending 1s / 0s as needed.  Strides are
  // forced to zero whenever the matching aligned input dim is 1 (i.e.
  // the broadcast case, including the implicit prepended dims).
  int64_t aligned_in_shape[HIP_EXPAND_RT_MAX_RANK];
  int64_t effective_in_strides[HIP_EXPAND_RT_MAX_RANK];
  int64_t rank_diff = out_rank - in_rank;
  for (int64_t d = 0; d < out_rank; ++d) {
    int64_t src = d - rank_diff;
    if (src < 0) {
      aligned_in_shape[d] = 1;
      effective_in_strides[d] = 0;
    } else {
      aligned_in_shape[d] = in_shape[src];
      effective_in_strides[d] =
          (in_shape[src] == 1) ? 0 : in_strides_elems[src];
    }
  }

  void *stream = hipdnn_ep_state_get_stream(state);
  RUNTIME_DEBUG_LOG(
      "[REAL] wrap_expand: dtype=%s, in_rank=%lld, out_rank=%lld\n",
      hipdnn_ep_datatype_name(data_type), (long long)in_rank,
      (long long)out_rank);

  int64_t num_out = 1;
  for (int64_t d = 0; d < out_rank; ++d)
    num_out *= out_shape[d];
  int rc = hip_expand(stream, input, output, aligned_in_shape,
                      effective_in_strides, out_shape, out_rank, hip_dtype);
  return rc;
}
