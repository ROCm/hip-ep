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
//   1. D2H of starts / ends / (optional) axes / (optional) steps.
//   2. Resolve to per-input-axis (start, step) per ONNX rules.
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
#include <vector>

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

int wrap_slice(RuntimeState *state, void *data, void *starts, void *ends,
               void *axes, void *steps, void *output, const int64_t *data_shape,
               int64_t data_rank, const int64_t *output_shape,
               int64_t output_rank, int64_t starts_num_elements,
               int64_t axes_num_elements, int64_t steps_num_elements,
               int64_t data_type) {
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

  if (!state || !data || !starts || !ends || !output || !data_shape ||
      !output_shape) {
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

  // D2H the (typically tiny) index tensors. Treat them as INT64 -- the
  // standard form for ONNX Slice indices and what every model in scope
  // emits. If a future model uses INT32 we'd need a dtype param in the
  // ABI; for now bail explicitly rather than silently mis-read.
  const int64_t K = starts_num_elements;
  std::vector<int64_t> starts_host(K);
  std::vector<int64_t> ends_host(K);
  std::vector<int64_t> axes_host;
  std::vector<int64_t> steps_host;

  hipError_t err =
      hipMemcpyAsync(starts_host.data(), starts, K * sizeof(int64_t),
                     hipMemcpyDeviceToHost, hip_stream);
  if (err != hipSuccess) {
    fprintf(stderr, "[REAL] wrap_slice: starts D2H failed: %s\n",
            hipGetErrorString(err));
    return -1;
  }
  err = hipMemcpyAsync(ends_host.data(), ends, K * sizeof(int64_t),
                       hipMemcpyDeviceToHost, hip_stream);
  if (err != hipSuccess) {
    fprintf(stderr, "[REAL] wrap_slice: ends D2H failed: %s\n",
            hipGetErrorString(err));
    return -1;
  }

  if (axes && axes_num_elements > 0) {
    if (axes_num_elements != K) {
      fprintf(stderr,
              "[REAL] wrap_slice: axes_num_elements(%lld) != "
              "starts_num_elements(%lld)\n",
              (long long)axes_num_elements, (long long)K);
      return -1;
    }
    axes_host.resize(K);
    err = hipMemcpyAsync(axes_host.data(), axes, K * sizeof(int64_t),
                         hipMemcpyDeviceToHost, hip_stream);
    if (err != hipSuccess) {
      fprintf(stderr, "[REAL] wrap_slice: axes D2H failed: %s\n",
              hipGetErrorString(err));
      return -1;
    }
  }

  if (steps && steps_num_elements > 0) {
    if (steps_num_elements != K) {
      fprintf(stderr,
              "[REAL] wrap_slice: steps_num_elements(%lld) != "
              "starts_num_elements(%lld)\n",
              (long long)steps_num_elements, (long long)K);
      return -1;
    }
    steps_host.resize(K);
    err = hipMemcpyAsync(steps_host.data(), steps, K * sizeof(int64_t),
                         hipMemcpyDeviceToHost, hip_stream);
    if (err != hipSuccess) {
      fprintf(stderr, "[REAL] wrap_slice: steps D2H failed: %s\n",
              hipGetErrorString(err));
      return -1;
    }
  }

  err = hipStreamSynchronize(hip_stream);
  if (err != hipSuccess) {
    fprintf(stderr, "[REAL] wrap_slice: stream sync after D2H failed: %s\n",
            hipGetErrorString(err));
    return -1;
  }

  // Build per-input-axis (start, step) arrays. Axes not listed in `axes`
  // default to full-range, unit-stride: start=0, step=1.
  int64_t start_per_axis[kSliceRuntimeMaxRank] = {};
  int64_t step_per_axis[kSliceRuntimeMaxRank];
  for (int d = 0; d < data_rank; ++d) {
    step_per_axis[d] = 1;
  }

  // Validation: axes must not repeat and must be in range.
  bool axis_set[kSliceRuntimeMaxRank] = {};
  for (int64_t k = 0; k < K; ++k) {
    int64_t axis = axes_host.empty() ? k : axes_host[k];
    if (axis < 0)
      axis += data_rank;
    if (axis < 0 || axis >= data_rank) {
      fprintf(stderr, "[REAL] wrap_slice: axis=%lld out of range [0, %lld)\n",
              (long long)axis, (long long)data_rank);
      return -1;
    }
    if (axis_set[axis]) {
      fprintf(stderr, "[REAL] wrap_slice: duplicate axis %lld\n",
              (long long)axis);
      return -1;
    }
    axis_set[axis] = true;

    int64_t dim = data_shape[axis];
    int64_t start = starts_host[k];
    int64_t end = ends_host[k];
    int64_t step = steps_host.empty() ? 1 : steps_host[k];

    if (step == 0) {
      fprintf(stderr, "[REAL] wrap_slice: zero step on axis %lld\n",
              (long long)axis);
      return -1;
    }

    // Per ONNX-13+ Slice negative-index + clamping rules.
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

    start_per_axis[axis] = start;
    step_per_axis[axis] = step;
  }

  // Sanity check: derived output extents must fit inside the IR-supplied
  // output shape (the IR-derived shape is the static upper bound; the
  // runtime extent is the actual count coming from a dynamic producer like
  // NonZero). `derived > output_shape` would write out-of-bounds and is a
  // genuine lowering bug; `derived < output_shape` is a legitimate partial
  // fill (downstream ops such as ScatterND consume a `count` and ignore
  // the unused tail). When we partial-fill, we *narrow* output_shape_eff
  // to `derived` so hip_slice's element count and stride math match the
  // actual data written.
  int64_t output_shape_eff[kSliceRuntimeMaxRank];
  for (int d = 0; d < data_rank; ++d) {
    output_shape_eff[d] = output_shape[d];
  }
  for (int d = 0; d < data_rank; ++d) {
    if (!axis_set[d])
      continue; // unmodified axis: output extent must == data extent.
    int64_t dim = data_shape[d];
    int64_t start = start_per_axis[d];
    int64_t step = step_per_axis[d];
    // The actual `end` was already clamp-resolved; we recompute the
    // expected output size from start/step against ends_host[k]. Walk
    // the K array to find the matching k for this axis.
    int64_t end = 0;
    for (int64_t k = 0; k < K; ++k) {
      int64_t ax = axes_host.empty() ? k : axes_host[k];
      if (ax < 0)
        ax += data_rank;
      if (ax == d) {
        end = ends_host[k];
        break;
      }
    }
    if (end < 0)
      end += dim;
    if (step > 0)
      end = std::clamp<int64_t>(end, 0, dim);
    else
      end = std::clamp<int64_t>(end, -1, dim - 1);
    int64_t expected;
    if (step > 0)
      expected = (end - start + step - 1) / step;
    else
      expected = (end - start + step + 1) / step;
    if (expected < 0)
      expected = 0;
    if (expected > output_shape[d]) {
      fprintf(stderr,
              "[REAL] wrap_slice: derived output extent on axis %d "
              "(%lld) > IR output_shape (%lld) -- aborting to avoid "
              "writing out-of-bounds\n",
              d, (long long)expected, (long long)output_shape[d]);
      return -1;
    }
    // Legitimate partial fill: dynamic producer upstream chose a smaller
    // count than the static upper bound. Narrow the effective output
    // extent so the kernel only writes the actual data.
    output_shape_eff[d] = expected;
  }

  for (int d = 0; d < data_rank; ++d) {
    if (!axis_set[d]) {
      start_per_axis[d] = 0;
      step_per_axis[d] = 1;
    }
  }

  RUNTIME_DEBUG_LOG("[REAL] wrap_slice: rank=%lld, K=%lld, data_type=%s "
                    "-> hip_slice\n",
                    (long long)data_rank, (long long)K,
                    hipdnn_ep_datatype_name(data_type));

  return hip_slice(hipdnn_ep_state_get_stream(state), data, output, data_shape,
                   output_shape_eff, start_per_axis, step_per_axis,
                   static_cast<int>(data_rank), hip_dtype);
}
