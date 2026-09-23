/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// Pad: ONNX-18 Pad with optional `axes` input. Four modes (constant,
// reflect, edge, wrap) handled by a single HIP kernel branched on the
// `pad_mode` arg.
//
// Source: onnxruntime/core/providers/cuda/tensor/pad_impl.cu @ v1.22.2
//         (_PadKernel; ONNX `wrap` mode added on top).
//
// pads / axes / constant_value stay on the device. wrap_pad checks
// lengths (host-known) and launches; hip_pad maps ONNX-18 pads onto
// per-dim lower_pads in shared memory. Invalid axes set the session
// device-error flag instead of stalling the inference stream.
#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
#include "hip_custom_kernels.h"

#include <cstdio>

static int pad_hipdnn_to_hip_dtype(int64_t hipdnn_type) {
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

int wrap_pad(RuntimeState *state, void *data, void *pads, void *constant_value,
             void *axes, void *output, const int64_t *data_shape,
             int64_t data_rank, const int64_t *output_shape,
             int64_t output_rank, int64_t pads_num_elements,
             int64_t axes_num_elements, int64_t data_type, int64_t mode_id) {
  OP_PROFILE(
      "pad",
      [&] {
        char b[64];
        const char *mn = (mode_id == 0)   ? "const"
                         : (mode_id == 1) ? "refl"
                         : (mode_id == 2) ? "edge"
                                          : "wrap";
        snprintf(b, sizeof(b), "r%lld:%s:%s", (long long)data_rank,
                 hipdnn_ep_datatype_name(data_type), mn);
        return std::string(b);
      },
      state);

  if (!state || !data || !pads || !output || !data_shape || !output_shape) {
    RUNTIME_DEBUG_LOG("[REAL] wrap_pad: null required argument\n");
    return -1;
  }
  if (data_rank <= 0) {
    fprintf(stderr, "[REAL] wrap_pad: invalid data_rank=%lld\n",
            (long long)data_rank);
    return -1;
  }
  if (data_rank != output_rank) {
    fprintf(stderr, "[REAL] wrap_pad: data_rank(%lld) != output_rank(%lld)\n",
            (long long)data_rank, (long long)output_rank);
    return -1;
  }
  if (pads_num_elements <= 0) {
    fprintf(stderr, "[REAL] wrap_pad: pads_num_elements=%lld must be > 0\n",
            (long long)pads_num_elements);
    return -1;
  }

  int hip_dtype = pad_hipdnn_to_hip_dtype(data_type);
  if (hip_dtype < 0) {
    fprintf(stderr,
            "[REAL] wrap_pad: unsupported data_type=%s(%lld) "
            "(supported: f16, f32, i32, i64)\n",
            hipdnn_ep_datatype_name(data_type), (long long)data_type);
    return -1;
  }

  const int64_t *axes_dev = nullptr;
  int64_t n_axes = data_rank;
  if (axes && axes_num_elements > 0) {
    if (axes_num_elements > data_rank) {
      fprintf(stderr,
              "[REAL] wrap_pad: axes_num_elements(%lld) > data_rank(%lld)\n",
              (long long)axes_num_elements, (long long)data_rank);
      return -1;
    }
    axes_dev = static_cast<const int64_t *>(axes);
    n_axes = axes_num_elements;
  }
  if (pads_num_elements != 2 * n_axes) {
    fprintf(stderr,
            "[REAL] wrap_pad: pads length(%lld) != 2 * num_axes(%lld)\n",
            (long long)pads_num_elements, (long long)n_axes);
    return -1;
  }

  const void *cval_dev = nullptr;
  if (constant_value && mode_id == 0)
    cval_dev = constant_value;

  RUNTIME_DEBUG_LOG("[REAL] wrap_pad: rank=%lld, data_type=%s, mode=%lld, "
                    "num_axes_padded=%lld -> hip_pad\n",
                    (long long)data_rank, hipdnn_ep_datatype_name(data_type),
                    (long long)mode_id, (long long)n_axes);

  return hip_pad(
      hipdnn_ep_state_get_stream(state), data, output, data_shape, output_shape,
      static_cast<const int64_t *>(pads), axes_dev, pads_num_elements, n_axes,
      static_cast<int>(data_rank), hip_dtype, static_cast<int>(mode_id),
      cval_dev, hipdnn_ep_state_get_error_flag_device_ptr(state));
}
