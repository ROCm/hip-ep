/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//
// Runtime wrapper for hip_scatter_nd (ONNX ScatterND opset 13).
//
// We follow the standard DPS contract: `output` is an uninitialised buffer
// the same size as `data`.  The wrapper first copies `data` -> `output`
// (skipping the copy when the pointers alias), then dispatches the kernel
// to apply the per-index updates.

#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "hip_custom_kernels.h"
#include "runtime_types.h"

#include <cstdint>
#include <cstdio>

static int hipdnn_ep_to_hip_dtype_data(int64_t data_type) {
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

static int hipdnn_ep_to_hip_dtype_indices(int64_t data_type) {
  switch (data_type) {
  case HIPDNN_EP_DATATYPE_INT32:
    return HIP_DTYPE_INT32;
  case HIPDNN_EP_DATATYPE_INT64:
    return HIP_DTYPE_INT64;
  default:
    return -1;
  }
}

static size_t element_size_for(int hip_dtype) {
  switch (hip_dtype) {
  case HIP_DTYPE_FLOAT32:
  case HIP_DTYPE_INT32:
    return 4;
  case HIP_DTYPE_FLOAT16:
  case HIP_DTYPE_BFLOAT16:
    return 2;
  case HIP_DTYPE_INT64:
  case HIP_DTYPE_FLOAT64:
    return 8;
  default:
    return 0;
  }
}

extern "C" int wrap_scatter_nd(RuntimeState *state, void *data, void *indices,
                               void *updates, void *output,
                               const int64_t *data_shape, int64_t data_rank,
                               const int64_t *indices_shape,
                               int64_t indices_rank, int64_t data_type,
                               int64_t indices_type, int64_t reduction) {
  if (!state || !data || !indices || !updates || !output || !data_shape ||
      !indices_shape) {
    fprintf(stderr, "wrap_scatter_nd: null argument\n");
    return -1;
  }
  if (data_rank <= 0 || indices_rank < 1) {
    fprintf(stderr,
            "wrap_scatter_nd: invalid ranks data_rank=%lld indices_rank=%lld\n",
            (long long)data_rank, (long long)indices_rank);
    return -1;
  }

  int data_dtype = hipdnn_ep_to_hip_dtype_data(data_type);
  if (data_dtype < 0) {
    fprintf(stderr, "wrap_scatter_nd: unsupported data dtype %lld (%s)\n",
            (long long)data_type, hipdnn_ep_datatype_name(data_type));
    return -1;
  }
  int indices_dtype = hipdnn_ep_to_hip_dtype_indices(indices_type);
  if (indices_dtype < 0) {
    fprintf(stderr, "wrap_scatter_nd: unsupported indices dtype %lld (%s)\n",
            (long long)indices_type, hipdnn_ep_datatype_name(indices_type));
    return -1;
  }

  hipStream_t stream =
      static_cast<hipStream_t>(hipdnn_ep_state_get_stream(state));

  // Copy data -> output unless the buffers already alias.  When they alias
  // we trust the caller-provided contract that data has been pre-copied.
  if (data != output) {
    int64_t total = 1;
    for (int64_t d = 0; d < data_rank; ++d)
      total *= data_shape[d];
    size_t elem_size = element_size_for(data_dtype);
    if (elem_size == 0) {
      fprintf(stderr,
              "wrap_scatter_nd: cannot determine element size for dtype %d\n",
              data_dtype);
      return -1;
    }
    hipError_t err =
        hipMemcpyAsync(output, data, total * elem_size,
                       hipMemcpyDeviceToDevice, stream);
    if (err != hipSuccess) {
      fprintf(stderr, "wrap_scatter_nd: memcpy failed: %s\n",
              hipGetErrorString(err));
      return static_cast<int>(err);
    }
  }

  RUNTIME_DEBUG_LOG(
      "[REAL] wrap_scatter_nd: data_dtype=%s, indices_dtype=%s, "
      "data_rank=%lld, indices_rank=%lld, reduction=%lld\n",
      hipdnn_ep_datatype_name(data_type), hipdnn_ep_datatype_name(indices_type),
      (long long)data_rank, (long long)indices_rank, (long long)reduction);

  int64_t num_out = 1;
  for (int64_t d = 0; d < data_rank; ++d)
    num_out *= data_shape[d];
  int rc = hip_scatter_nd(stream, output, indices, updates, data_shape, data_rank,
                          indices_shape, indices_rank, data_dtype, indices_dtype,
                          static_cast<int>(reduction));
  return rc;
}
