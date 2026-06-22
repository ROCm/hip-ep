/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// Tile: copy `input` over `output`, with `output_shape[d] =
// input_shape[d] * repeats[d]`.
//
// Source: onnxruntime/core/providers/cuda/tensor/tile.cu @ v1.22.2.
//
// The GPU `repeats` tensor pointer is unused -- the lowering passes us host
// input_shape + output_shape directly, which encodes the same information
// (repeats[d] = output_shape[d] / input_shape[d]). Avoids a per-call D2H
// round-trip of the repeats input.
#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
#include "hip_custom_kernels.h"

#include <cstdio>

static int tile_hipdnn_to_hip_dtype(int64_t hipdnn_type) {
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

int wrap_tile(RuntimeState *state, void *input, void *repeats, void *output,
              const int64_t *input_shape, int64_t input_rank,
              const int64_t *output_shape, int64_t output_rank,
              int64_t data_type) {
  OP_PROFILE(
      "tile",
      [&] {
        char b[64];
        snprintf(b, sizeof(b), "r%lld:%s", (long long)output_rank,
                 hipdnn_ep_datatype_name(data_type));
        return std::string(b);
      },
      state);

  (void)repeats;

  if (!state || !input || !output || !input_shape || !output_shape) {
    RUNTIME_DEBUG_LOG("[REAL] wrap_tile: null argument\n");
    return -1;
  }
  if (input_rank != output_rank) {
    hipdnn_ep_log_emit(
        "[REAL] wrap_tile: input_rank(%lld) != output_rank(%lld)\n",
        (long long)input_rank, (long long)output_rank);
    return -1;
  }
  if (input_rank == 0) {
    return 0;
  }

  int hip_dtype = tile_hipdnn_to_hip_dtype(data_type);
  if (hip_dtype < 0) {
    hipdnn_ep_log_emit("[REAL] wrap_tile: unsupported data_type=%s(%lld) "
                       "(supported: f16, f32, i32, i64)\n",
                       hipdnn_ep_datatype_name(data_type),
                       (long long)data_type);
    return -1;
  }

  void *stream = hipdnn_ep_state_get_stream(state);
  RUNTIME_DEBUG_LOG("[REAL] wrap_tile: rank=%lld, data_type=%s -> hip_tile\n",
                    (long long)input_rank, hipdnn_ep_datatype_name(data_type));

  return hip_tile(stream, input, output, input_shape, output_shape,
                  static_cast<int>(input_rank), hip_dtype);
}
