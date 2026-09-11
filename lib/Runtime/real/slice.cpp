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
// Those rules are applied on the GPU, in `hip_slice`. The index tensors are
// device-resident and the output shape is statically known (the SliceToHip
// lowering enforces this), so the launch geometry does not depend on their
// values -- reading them back would buy nothing but a full stream drain on
// every call. This wrapper therefore only validates what it can see from
// host-side metadata (ranks, element counts, dtype) and forwards the device
// pointers. Malformed index values surface through the device error flag.
//
// We assume the four index tensors are INT64 -- the standard ONNX form
// and what every test in the LIT suite uses. If a model produces INT32
// index tensors we will need to dispatch on stride (the ABI omits the
// dtype for these operands today).

#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
#include "hip_custom_kernels.h"

#include <atomic>
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

  if (!state || !starts || !ends || !output || !data_shape || !output_shape) {
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

  // `axes` and `steps` are optional, but when present must be keyed 1:1 to
  // `starts`. This is a metadata check, so it stays on the host.
  const int64_t K = starts_num_elements;
  if (axes && axes_num_elements > 0 && axes_num_elements != K) {
    fprintf(stderr,
            "[REAL] wrap_slice: axes_num_elements(%lld) != "
            "starts_num_elements(%lld)\n",
            (long long)axes_num_elements, (long long)K);
    return -1;
  }
  if (steps && steps_num_elements > 0 && steps_num_elements != K) {
    fprintf(stderr,
            "[REAL] wrap_slice: steps_num_elements(%lld) != "
            "starts_num_elements(%lld)\n",
            (long long)steps_num_elements, (long long)K);
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

  // A zero-capacity output on an axis whose input is non-empty is legal ONNX
  // but is, in every graph seen so far, the `Slice(x, k, k, axis)` accumulator
  // seed having been sized at its exact (zero) extent instead of the data dim.
  // Nothing here can fail: the damage lands later, when the loop appends into
  // the empty buffer and the strided copy gets a zero pitch. So warn once
  // rather than abort, and name the shape that has to be fixed upstream in
  // SliceToHip. See lib/Conversion/OnnxToHip/SliceConversion.cpp.
  for (int d = 0; d < data_rank; ++d) {
    if (output_shape[d] == 0 && data_shape[d] > 0) {
      static std::atomic<bool> warned{false};
      if (!warned.exchange(true)) {
        fprintf(stderr,
                "[REAL] wrap_slice: zero-capacity output on axis %d "
                "with a non-empty input dim (%lld) -- the compile-time extent "
                "collapsed to 0; a consumer that appends into this buffer "
                "(e.g. a hip.loop Concat accumulator) will fail\n",
                d, (long long)data_shape[d]);
      }
      break;
    }
  }

  RUNTIME_DEBUG_LOG("[REAL] wrap_slice: rank=%lld, K=%lld, data_type=%s "
                    "-> hip_slice\n",
                    (long long)data_rank, (long long)K,
                    hipdnn_ep_datatype_name(data_type));

  return hip_slice(hipdnn_ep_state_get_stream(state), data, output, data_shape,
                   output_shape, starts, ends,
                   (axes && axes_num_elements > 0) ? axes : nullptr,
                   (steps && steps_num_elements > 0) ? steps : nullptr,
                   static_cast<int>(K), static_cast<int>(data_rank), hip_dtype,
                   hipdnn_ep_state_get_error_flag_device_ptr(state));
}
