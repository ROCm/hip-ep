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
// Compile-time controls arrive through host pointers materialized by lowering.
// Runtime-dynamic controls are mapped to per-axis lower pads on the GPU.
#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
#include "hip_custom_kernels.h"

#include <cstdio>
#include <hip/hip_runtime.h>
#include <vector>

static constexpr int kPadMaxRank = 8;

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

int wrap_pad(RuntimeState *state, void *data, void *pads_device,
             const int64_t *pads_host, void *constant_value_device,
             const void *constant_value_host, void *axes_device,
             const int64_t *axes_host, void *output, const int64_t *data_shape,
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

  (void)output_shape;
  (void)output_rank;

  if (!state || !data || (!pads_device && !pads_host) || !output ||
      !data_shape) {
    RUNTIME_DEBUG_LOG("[REAL] wrap_pad: null required argument\n");
    return -1;
  }
  if (data_rank <= 0) {
    fprintf(stderr, "[REAL] wrap_pad: invalid data_rank=%lld\n",
            (long long)data_rank);
    return -1;
  }
  if (data_rank > kPadMaxRank) {
    fprintf(stderr, "[REAL] wrap_pad: data_rank=%lld exceeds max %d\n",
            (long long)data_rank, kPadMaxRank);
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
  int element_size = static_cast<int>(hipdnn_ep_datatype_size(data_type));
  if (element_size <= 0) {
    fprintf(stderr, "[REAL] wrap_pad: bad element size for data_type=%lld\n",
            (long long)data_type);
    return -1;
  }

  // ONNX-18 pads layout: [begin_0, begin_1, ..., begin_K-1, end_0, ...,
  // end_K-1] where K = num_axes_padded (= data_rank if axes is omitted).
  const bool axes_present = axes_num_elements > 0;
  if (axes_present && !axes_host && !axes_device) {
    fprintf(stderr,
            "[REAL] wrap_pad: axes count is nonzero but axes is null\n");
    return -1;
  }
  int64_t num_axes_padded = axes_present ? axes_num_elements : data_rank;
  if (num_axes_padded > kPadMaxRank) {
    fprintf(stderr, "[REAL] wrap_pad: num_axes=%lld exceeds max %d\n",
            (long long)num_axes_padded, kPadMaxRank);
    return -1;
  }
  if (pads_num_elements != 2 * num_axes_padded) {
    fprintf(stderr,
            "[REAL] wrap_pad: pads length(%lld) != 2 * num_axes(%lld)\n",
            (long long)pads_num_elements, (long long)num_axes_padded);
    return -1;
  }

  // Host-known pads/axes keep the by-value PadParams fast path. If either
  // control is device-only, a one-thread prepass builds lower_pads in the
  // RuntimeState workspace. Invalid axes are skipped and duplicates are
  // deterministic last-writer-wins in that device path.
  std::vector<int64_t> lower_pads;
  const int64_t *lower_pads_arg = nullptr;
  bool lower_pads_on_device = !pads_host || (axes_present && !axes_host);
  if (!lower_pads_on_device) {
    lower_pads.assign(data_rank, 0);
    for (int64_t i = 0; i < num_axes_padded; ++i) {
      int64_t axis = axes_present ? axes_host[i] : i;
      if (axis < 0)
        axis += data_rank;
      if (axis < 0 || axis >= data_rank) {
        fprintf(stderr, "[REAL] wrap_pad: axis=%lld out of range [0, %lld)\n",
                (long long)axis, (long long)data_rank);
        return -1;
      }
      lower_pads[axis] = pads_host[i];
    }
    lower_pads_arg = lower_pads.data();
  } else {
    const size_t workspace_bytes = kPadMaxRank * sizeof(int64_t);
    if (hipdnn_ep_state_ensure_workspace(state, workspace_bytes) != 0) {
      fprintf(stderr, "[REAL] wrap_pad: failed to allocate lower-pads table\n");
      return -1;
    }
    auto *lower_pads_device =
        static_cast<int64_t *>(hipdnn_ep_state_get_workspace(state));
    int status = hip_pad_resolve(
        hipdnn_ep_state_get_stream(state),
        pads_host ? nullptr : static_cast<const int64_t *>(pads_device),
        pads_host,
        axes_present && !axes_host ? static_cast<const int64_t *>(axes_device)
                                   : nullptr,
        axes_present ? axes_host : nullptr, static_cast<int>(num_axes_padded),
        static_cast<int>(data_rank), lower_pads_device);
    if (status != 0)
      return status;
    lower_pads_arg = lower_pads_device;
  }

  const void *pad_value = nullptr;
  bool pad_value_on_device = false;
  if (mode_id == 0) {
    pad_value =
        constant_value_host ? constant_value_host : constant_value_device;
    pad_value_on_device = !constant_value_host && constant_value_device;
  }

  RUNTIME_DEBUG_LOG("[REAL] wrap_pad: rank=%lld, data_type=%s, mode=%lld, "
                    "num_axes_padded=%lld -> hip_pad\n",
                    (long long)data_rank, hipdnn_ep_datatype_name(data_type),
                    (long long)mode_id, (long long)num_axes_padded);

  return hip_pad(hipdnn_ep_state_get_stream(state), data, output, data_shape,
                 output_shape, lower_pads_arg, lower_pads_on_device,
                 static_cast<int>(data_rank), hip_dtype,
                 static_cast<int>(mode_id), pad_value, pad_value_on_device);
}
