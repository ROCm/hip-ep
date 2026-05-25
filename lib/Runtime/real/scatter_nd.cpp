/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// ScatterND: produce `output` with the shape of `data` whose values are
// `data` copied, then `updates` overwritten / reduced into at positions
// specified by `indices` (INT64 only). Five reduction modes per ONNX-13+:
//
//   id  reduction    semantics
//   --  ---------    ----------------------------------------------
//    0  "none"       output[idx] = updates[slice]      (last-writer-wins)
//    1  "add"        output[idx] += updates[slice]     (atomic)
//    2  "mul"        output[idx] *= updates[slice]     (atomic CAS)
//    3  "min"        output[idx]  = min(.,.)            (atomic CAS)
//    4  "max"        output[idx]  = max(.,.)            (atomic CAS)
//
// Source: onnxruntime/core/providers/cuda/tensor/scatter_nd_impl.cu @
//         v1.22.2 (`_ScatterNDKernel` -- single fused kernel with native
//         atomics where available, CAS emulation otherwise).
//
// All host-side shape arrays come from the MLIR lowering -- no GPU shape
// D2H is required. The only D-side work is one `output := data` D2D copy
// and one launch of `hip_scatter_nd_kernel`.

#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
#include "hip_custom_kernels.h"

#include <cstdio>

static int scatter_nd_hipdnn_to_hip_dtype(int64_t hipdnn_type) {
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

static const char *reduction_name(int64_t id) {
  switch (id) {
  case 0:
    return "none";
  case 1:
    return "add";
  case 2:
    return "mul";
  case 3:
    return "min";
  case 4:
    return "max";
  default:
    return "<unknown>";
  }
}

int wrap_scatter_nd(RuntimeState *state, void *data, void *indices,
                    void *updates, void *output, const int32_t *count_ptr,
                    const int64_t *data_shape, int64_t data_rank,
                    const int64_t *indices_shape, int64_t indices_rank,
                    const int64_t *updates_shape, int64_t updates_rank,
                    const int64_t *output_shape, int64_t output_rank,
                    int64_t reduction_id, int64_t data_type) {
  OP_PROFILE(
      "scatter_nd",
      [&] {
        char b[64];
        snprintf(b, sizeof(b), "dr%lld:ir%lld:%s:%s", (long long)data_rank,
                 (long long)indices_rank, reduction_name(reduction_id),
                 hipdnn_ep_datatype_name(data_type));
        return std::string(b);
      },
      state);

  (void)updates_shape;
  (void)updates_rank;
  (void)output_shape;
  (void)output_rank;

  if (!state || !data || !indices || !updates || !output || !data_shape ||
      !indices_shape) {
    RUNTIME_DEBUG_LOG("[REAL] wrap_scatter_nd: null required argument\n");
    return -1;
  }
  if (data_rank <= 0 || indices_rank <= 0) {
    fprintf(stderr,
            "[REAL] wrap_scatter_nd: zero-rank input "
            "(data_rank=%lld, indices_rank=%lld)\n",
            (long long)data_rank, (long long)indices_rank);
    return -1;
  }
  if (reduction_id < 0 || reduction_id > 4) {
    fprintf(stderr, "[REAL] wrap_scatter_nd: invalid reduction_id=%lld\n",
            (long long)reduction_id);
    return -1;
  }

  int hip_dtype = scatter_nd_hipdnn_to_hip_dtype(data_type);
  if (hip_dtype < 0) {
    fprintf(stderr,
            "[REAL] wrap_scatter_nd: unsupported data_type=%s(%lld) "
            "(supported: f16, f32, i32, i64)\n",
            hipdnn_ep_datatype_name(data_type), (long long)data_type);
    return -1;
  }

  void *stream = hipdnn_ep_state_get_stream(state);
  RUNTIME_DEBUG_LOG(
      "[REAL] wrap_scatter_nd: data_rank=%lld, indices_rank=%lld, "
      "reduction=%s, data_type=%s, count_ptr=%p -> hip_scatter_nd\n",
      (long long)data_rank, (long long)indices_rank,
      reduction_name(reduction_id), hipdnn_ep_datatype_name(data_type),
      (const void *)count_ptr);

  return hip_scatter_nd(stream, data, indices, updates, output, count_ptr,
                        data_shape, static_cast<int>(data_rank), indices_shape,
                        static_cast<int>(indices_rank),
                        static_cast<int>(reduction_id), hip_dtype);
}
