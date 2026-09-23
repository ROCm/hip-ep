/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// Slice (ONNX-13+) -- non-constant indices / negative-step fallback.
//
// The compile-time-constant + positive-stride case is folded upstream to
// `tensor.extract_slice` (see lib/Conversion/OnnxToHip/SliceConversion.cpp
// ::SliceDecompose), so this entry point only fires for slices whose
// `starts` / `ends` / `axes` / `steps` are not graph-constant (or that
// use negative steps).
//
// Per ONNX-13+ Slice spec:
//
//   * `starts` and `ends` are 1-D int64 (or int32) tensors with one entry
//     per axis listed in `axes`.
//   * `axes` defaults to [0, ..., rank-1] when absent.
//   * `steps` defaults to all-ones; non-zero negative steps are allowed.
//   * Per-axis negative indices: idx<0 -> idx += dim.
//   * Clamping:
//       step > 0: start in [0, dim], end in [0, dim].
//       step < 0: start in [0, dim-1], end in [-1, dim-1].
//
// Output shape is statically known (the SliceToHip lowering enforces
// this), so the only work we do at runtime is:
//
//   1. Resolve compile-time controls on the host, or runtime controls on GPU.
//   2. Store per-input-axis (start, step, logical extent) in device scratch.
//   3. Launch `hip_slice` -- one thread per output element.
//
// We assume the four index tensors are INT64 -- the standard ONNX form
// and what every test in the LIT suite uses. If a model produces INT32
// index tensors we will need to dispatch on stride (the ABI omits the
// dtype for these operands today).

#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
#include "hip_custom_kernels.h"

#include <algorithm>
#include <cstdio>
#include <hip/hip_runtime.h>

static constexpr int kSliceRuntimeMaxRank = 8;

static int slice_hipdnn_to_hip_dtype(int64_t hipdnn_type) {
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

int wrap_slice(RuntimeState *state, void *data, void *starts_device,
               const int64_t *starts_attr, void *ends_device,
               const int64_t *ends_attr, void *axes_device,
               const int64_t *axes_attr, void *steps_device,
               const int64_t *steps_attr, void *output,
               const int64_t *data_shape, int64_t data_rank,
               const int64_t *output_shape, int64_t output_rank,
               int64_t starts_num_elements, int64_t axes_num_elements,
               int64_t steps_num_elements, int64_t data_type) {
  // (Slice runtime entry trace removed; was used for the slice-empty-buffer
  // root-cause investigation. Re-add with a HIPDNN_EP_DEBUG gate if needed.)
  OP_PROFILE(
      "slice",
      [&] {
        char b[64];
        snprintf(b, sizeof(b), "r%lld:K%lld:%s", (long long)data_rank,
                 (long long)starts_num_elements,
                 hipdnn_ep_datatype_name(data_type));
        return std::string(b);
      },
      state);

  if (!state || (!starts_device && !starts_attr) ||
      (!ends_device && !ends_attr) || !output || !data_shape || !output_shape) {
    RUNTIME_DEBUG_LOG("[REAL] wrap_slice: null required argument\n");
    return -1;
  }
  if (data_rank <= 0 || data_rank != output_rank) {
    fprintf(
        stderr,
        "[REAL] wrap_slice: invalid ranks (data_rank=%lld, output_rank=%lld)\n",
        (long long)data_rank, (long long)output_rank);
    return -1;
  }
  // Empty input fast path: ORT passes a null `data` pointer for a tensor
  // with zero elements (e.g. `image_features` with shape[0]==0 in the VLM
  // embedding sub-model on a text-only call). Slicing an empty input
  // necessarily produces an empty logical output; the physical output buffer
  // is over-allocated by SliceToHip to the IR static dim and must be
  // zero-filled so downstream consumers (e.g. ScatterND's updates buffer)
  // see the zero tail the existing logical-extent==0 kernel path would have
  // written. Returning -1 here would leave the buffer at its pool-init
  // value and silently feed downstream ops garbage / zero, masking the
  // missing-write as "EP produced zero output".
  int64_t data_num_elements = 1;
  for (int d = 0; d < data_rank; ++d)
    data_num_elements *= data_shape[d];
  if (!data || data_num_elements == 0) {
    int64_t out_num_elements = 1;
    for (int d = 0; d < output_rank; ++d)
      out_num_elements *= output_shape[d];
    int64_t elem_size = hipdnn_ep_datatype_size(data_type);
    if (elem_size <= 0) {
      fprintf(stderr,
              "[REAL] wrap_slice: empty-input path -- unsupported "
              "data_type=%s(%lld)\n",
              hipdnn_ep_datatype_name(data_type), (long long)data_type);
      return -1;
    }
    if (output && out_num_elements > 0) {
      hipStream_t s =
          static_cast<hipStream_t>(hipdnn_ep_state_get_stream(state));
      hipError_t err = hipMemsetAsync(output, 0,
                                      static_cast<size_t>(out_num_elements) *
                                          static_cast<size_t>(elem_size),
                                      s);
      if (err != hipSuccess) {
        fprintf(stderr,
                "[REAL] wrap_slice: empty-input hipMemsetAsync failed: %s\n",
                hipGetErrorString(err));
        return -1;
      }
    }
    RUNTIME_DEBUG_LOG(
        "[REAL] wrap_slice: empty input (data=%p data_num_elements=%lld) "
        "-- zeroed output and returning success\n",
        data, (long long)data_num_elements);
    return 0;
  }
  if (data_rank > kSliceRuntimeMaxRank) {
    fprintf(stderr, "[REAL] wrap_slice: data_rank=%lld exceeds max %d\n",
            (long long)data_rank, kSliceRuntimeMaxRank);
    return -1;
  }
  if (starts_num_elements <= 0) {
    fprintf(stderr, "[REAL] wrap_slice: starts_num_elements=%lld must be > 0\n",
            (long long)starts_num_elements);
    return -1;
  }
  if (starts_num_elements > kSliceRuntimeMaxRank) {
    fprintf(stderr,
            "[REAL] wrap_slice: starts_num_elements=%lld exceeds max %d\n",
            (long long)starts_num_elements, kSliceRuntimeMaxRank);
    return -1;
  }

  int hip_dtype = slice_hipdnn_to_hip_dtype(data_type);
  if (hip_dtype < 0) {
    fprintf(stderr,
            "[REAL] wrap_slice: unsupported data_type=%s(%lld) "
            "(supported: f16, f32, i32, i64)\n",
            hipdnn_ep_datatype_name(data_type), (long long)data_type);
    return -1;
  }

  hipStream_t hip_stream =
      static_cast<hipStream_t>(hipdnn_ep_state_get_stream(state));

  const int64_t K = starts_num_elements;
  const bool axes_present = (axes_device || axes_attr) && axes_num_elements > 0;
  const bool steps_present =
      (steps_device || steps_attr) && steps_num_elements > 0;
  if (axes_present) {
    if (axes_num_elements != K) {
      fprintf(stderr,
              "[REAL] wrap_slice: axes_num_elements(%lld) != "
              "starts_num_elements(%lld)\n",
              (long long)axes_num_elements, (long long)K);
      return -1;
    }
  }
  if (steps_present) {
    if (steps_num_elements != K) {
      fprintf(stderr,
              "[REAL] wrap_slice: steps_num_elements(%lld) != "
              "starts_num_elements(%lld)\n",
              (long long)steps_num_elements, (long long)K);
      return -1;
    }
  }

  constexpr size_t kTableBytes = kSliceRuntimeMaxRank * sizeof(int64_t);
  constexpr size_t kWorkspaceBytes = 3 * kTableBytes;
  if (hipdnn_ep_state_ensure_workspace(state, kWorkspaceBytes) != 0) {
    fprintf(stderr, "[REAL] wrap_slice: failed to allocate device tables\n");
    return -1;
  }
  auto *workspace =
      static_cast<unsigned char *>(hipdnn_ep_state_get_workspace(state));
  auto *start_device = reinterpret_cast<int64_t *>(workspace);
  auto *step_device = reinterpret_cast<int64_t *>(workspace + kTableBytes);
  auto *logical_device =
      reinterpret_cast<int64_t *>(workspace + 2 * kTableBytes);

  const bool all_controls_on_host = starts_attr && ends_attr &&
                                    (!axes_present || axes_attr) &&
                                    (!steps_present || steps_attr);
  if (all_controls_on_host) {
    int64_t start_per_axis[kSliceRuntimeMaxRank] = {};
    int64_t step_per_axis[kSliceRuntimeMaxRank];
    int64_t logical_extent[kSliceRuntimeMaxRank];
    bool axis_set[kSliceRuntimeMaxRank] = {};
    for (int d = 0; d < data_rank; ++d) {
      step_per_axis[d] = 1;
      logical_extent[d] = output_shape[d];
    }

    for (int64_t k = 0; k < K; ++k) {
      int64_t axis = axes_present ? axes_attr[k] : k;
      if (axis < 0)
        axis += data_rank;
      if (axis < 0 || axis >= data_rank || axis_set[axis]) {
        fprintf(stderr, "[REAL] wrap_slice: invalid or duplicate axis %lld\n",
                (long long)axis);
        return -1;
      }
      axis_set[axis] = true;
      const int64_t dim = data_shape[axis];
      int64_t start = starts_attr[k];
      int64_t end = ends_attr[k];
      int64_t step = steps_present ? steps_attr[k] : 1;
      if (step == 0) {
        fprintf(stderr, "[REAL] wrap_slice: zero step on axis %lld\n",
                (long long)axis);
        return -1;
      }
      if (start < 0)
        start += dim;
      if (end < 0)
        end += dim;
      if (step > 0) {
        start = std::clamp<int64_t>(start, 0, dim);
        end = std::clamp<int64_t>(end, 0, dim);
      } else {
        start = std::clamp<int64_t>(start, 0, dim - 1);
        end = std::clamp<int64_t>(end, -1, dim - 1);
      }
      int64_t expected = step > 0 ? (end - start + step - 1) / step
                                  : (end - start + step + 1) / step;
      expected = std::max<int64_t>(expected, 0);
      logical_extent[axis] = std::min<int64_t>(expected, output_shape[axis]);
      start_per_axis[axis] = start;
      step_per_axis[axis] = step;
    }

    hipError_t err = hipMemcpyAsync(start_device, start_per_axis,
                                    data_rank * sizeof(int64_t),
                                    hipMemcpyHostToDevice, hip_stream);
    if (err == hipSuccess)
      err = hipMemcpyAsync(step_device, step_per_axis,
                           data_rank * sizeof(int64_t), hipMemcpyHostToDevice,
                           hip_stream);
    if (err == hipSuccess)
      err = hipMemcpyAsync(logical_device, logical_extent,
                           data_rank * sizeof(int64_t), hipMemcpyHostToDevice,
                           hip_stream);
    if (err != hipSuccess) {
      fprintf(stderr, "[REAL] wrap_slice: table H2D failed: %s\n",
              hipGetErrorString(err));
      return -1;
    }
  } else {
    int status = hip_slice_resolve(
        hipdnn_ep_state_get_stream(state),
        starts_attr ? nullptr : static_cast<const int64_t *>(starts_device),
        starts_attr,
        ends_attr ? nullptr : static_cast<const int64_t *>(ends_device),
        ends_attr,
        axes_present && !axes_attr ? static_cast<const int64_t *>(axes_device)
                                   : nullptr,
        axes_present ? axes_attr : nullptr,
        steps_present && !steps_attr
            ? static_cast<const int64_t *>(steps_device)
            : nullptr,
        steps_present ? steps_attr : nullptr, data_shape, output_shape,
        static_cast<int>(K), static_cast<int>(data_rank), start_device,
        step_device, logical_device);
    if (status != 0)
      return status;
  }

  RUNTIME_DEBUG_LOG("[REAL] wrap_slice: rank=%lld, K=%lld, data_type=%s "
                    "-> hip_slice\n",
                    (long long)data_rank, (long long)K,
                    hipdnn_ep_datatype_name(data_type));

  return hip_slice(hipdnn_ep_state_get_stream(state), data, output, data_shape,
                   output_shape, logical_device, start_device, step_device,
                   static_cast<int>(data_rank), hip_dtype);
}
