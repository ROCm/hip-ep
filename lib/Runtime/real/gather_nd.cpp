/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// GatherND: gather slices of `data` along the first
// K = indices.shape[-1] dims (after `batch_dims`), one slice per row of
// `indices`. ONNX-13+ semantics with the `batch_dims` attribute.
//
// Indices must be INT64 (the most common case for ONNX vision models). The
// `data` element type is dispatched via `data_type`. The host wrapper
// forwards both shapes plus batch_dims to the kernel; no scratch buffer is
// needed because the kernel reads K indices inline per output element.
//
// Source: onnxruntime/core/providers/cuda/tensor/gather_nd_impl.cu @
//         v1.22.2 (ComputeSliceOffsetsImpl + _GatherNDKernel fused).
#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
#include "hip_custom_kernels.h"

#include <cstdio>

static int gather_nd_hipdnn_to_hip_dtype(int64_t hipdnn_type) {
  switch (hipdnn_type) {
  case HIPDNN_EP_DATATYPE_HALF:
    return HIP_DTYPE_FLOAT16;
  case HIPDNN_EP_DATATYPE_FLOAT:
    return HIP_DTYPE_FLOAT32;
  case HIPDNN_EP_DATATYPE_INT32:
    return HIP_DTYPE_INT32;
  case HIPDNN_EP_DATATYPE_INT64:
    return HIP_DTYPE_INT64;
  default:
    return -1;
  }
}

int wrap_gather_nd(RuntimeState *state, void *data, void *indices, void *output,
                   const int64_t *data_shape, int64_t data_rank,
                   const int64_t *indices_shape, int64_t indices_rank,
                   const int64_t *output_shape, int64_t output_rank,
                   int64_t batch_dims, int64_t data_type) {
  OP_PROFILE(
      "gather_nd",
      [&] {
        char b[64];
        snprintf(b, sizeof(b), "dr%lld:ir%lld:bd%lld:%s", (long long)data_rank,
                 (long long)indices_rank, (long long)batch_dims,
                 hipdnn_ep_datatype_name(data_type));
        return std::string(b);
      },
      state);

  (void)output_shape;
  (void)output_rank;

  if (!state || !data || !indices || !output || !data_shape || !indices_shape) {
    RUNTIME_DEBUG_LOG("[REAL] wrap_gather_nd: null argument\n");
    return -1;
  }
  if (data_rank <= 0 || indices_rank <= 0) {
    hipdnn_ep_log_emit("[REAL] wrap_gather_nd: zero-rank input "
                       "(data_rank=%lld, indices_rank=%lld)\n",
                       (long long)data_rank, (long long)indices_rank);
    return -1;
  }

  int hip_dtype = gather_nd_hipdnn_to_hip_dtype(data_type);
  if (hip_dtype < 0) {
    hipdnn_ep_log_emit("[REAL] wrap_gather_nd: unsupported data_type=%s(%lld) "
                       "(supported: f16, f32, i32, i64)\n",
                       hipdnn_ep_datatype_name(data_type),
                       (long long)data_type);
    return -1;
  }

  void *stream = hipdnn_ep_state_get_stream(state);
  RUNTIME_DEBUG_LOG("[REAL] wrap_gather_nd: data_rank=%lld, indices_rank=%lld, "
                    "batch_dims=%lld, data_type=%s -> hip_gather_nd\n",
                    (long long)data_rank, (long long)indices_rank,
                    (long long)batch_dims, hipdnn_ep_datatype_name(data_type));

  return hip_gather_nd(stream, data, indices, output, data_shape,
                       static_cast<int>(data_rank), indices_shape,
                       static_cast<int>(indices_rank),
                       static_cast<int>(batch_dims), hip_dtype);
}
