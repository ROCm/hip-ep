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
// `pads` and `axes` arrive as HOST pointers: PadConversion wraps them with
// `hip.transfer_to_host` (bufferizes to a host buffer + async D2H + stream
// sync), so the bytes are already host-resident and synced by the time wrap_pad
// runs -- it just reads them with a plain memcpy (no D2H, no sync here). Doing
// the crossing in the IR instead of hiding it here is the point of the transfer
// pilot; the total sync count is the same as the old implicit-D2H version.
//
// `constant_value` is also not in GPU memory: the compiler passes the scalar
// fill value BY VALUE through a host stack slot (see PadLowering /
// PadConversion), so it too is read with a plain memcpy -- no device alloc, no
// D2H.
#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
#include "hip_custom_kernels.h"

#include <cstdio>
#include <cstring>
#include <hip/hip_runtime.h>
#include <vector>

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

  (void)output_shape;
  (void)output_rank;

  if (!state || !data || !pads || !output || !data_shape) {
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
  int element_size = static_cast<int>(hipdnn_ep_datatype_size(data_type));
  if (element_size <= 0) {
    fprintf(stderr, "[REAL] wrap_pad: bad element size for data_type=%lld\n",
            (long long)data_type);
    return -1;
  }

  // pads/axes are HOST-resident (brought over by hip.transfer_to_host -> a
  // preceding hip.memcpy_d2h_async + hip.stream_sync in the IR), so read with a
  // plain memcpy -- no internal D2H, no stream sync here.
  // ONNX-18 pads layout: [begin_0, begin_1, ..., begin_K-1, end_0, ...,
  // end_K-1] where K = num_axes_padded (= data_rank if axes is omitted).
  std::vector<int64_t> pads_host(pads_num_elements);
  memcpy(pads_host.data(), pads, pads_num_elements * sizeof(int64_t));

  std::vector<int64_t> axes_host;
  if (axes && axes_num_elements > 0) {
    axes_host.resize(axes_num_elements);
    memcpy(axes_host.data(), axes, axes_num_elements * sizeof(int64_t));
  }

  // constant_value: a by-value scalar of `data_type` passed by the generated
  // code through a HOST stack slot (not a device buffer), so read it directly.
  // Copy into a local 8-byte buffer so we can hand a typed pointer to the
  // kernel launcher.
  alignas(8) unsigned char cv_buf[8] = {};
  bool have_cv = false;
  if (constant_value && mode_id == 0) {
    memcpy(cv_buf, constant_value, element_size);
    have_cv = true;
  }

  // Build per-axis lower_pads[data_rank], defaulting to 0. If `axes` is
  // omitted, pads is laid out per-axis (length 2*data_rank). If `axes`
  // is given, pads has length 2 * len(axes), keyed by the axes vector.
  std::vector<int64_t> lower_pads(data_rank, 0);
  int64_t num_axes_padded =
      axes_host.empty() ? data_rank : static_cast<int64_t>(axes_host.size());
  if (pads_num_elements != 2 * num_axes_padded) {
    fprintf(stderr,
            "[REAL] wrap_pad: pads length(%lld) != 2 * num_axes(%lld)\n",
            (long long)pads_num_elements, (long long)num_axes_padded);
    return -1;
  }
  for (int64_t i = 0; i < num_axes_padded; ++i) {
    int64_t axis = axes_host.empty() ? i : axes_host[i];
    if (axis < 0)
      axis += data_rank;
    if (axis < 0 || axis >= data_rank) {
      fprintf(stderr, "[REAL] wrap_pad: axis=%lld out of range [0, %lld)\n",
              (long long)axis, (long long)data_rank);
      return -1;
    }
    lower_pads[axis] = pads_host[i];
    // The kernel doesn't need upper pads (it uses output_shape - input
    // - lower implicitly via the out_coord >= lower + in_dim check).
  }

  RUNTIME_DEBUG_LOG("[REAL] wrap_pad: rank=%lld, data_type=%s, mode=%lld, "
                    "num_axes_padded=%lld -> hip_pad\n",
                    (long long)data_rank, hipdnn_ep_datatype_name(data_type),
                    (long long)mode_id, (long long)num_axes_padded);

  return hip_pad(hipdnn_ep_state_get_stream(state), data, output, data_shape,
                 output_shape, lower_pads.data(), static_cast<int>(data_rank),
                 hip_dtype, static_cast<int>(mode_id),
                 have_cv ? static_cast<const void *>(cv_buf) : nullptr);
}
