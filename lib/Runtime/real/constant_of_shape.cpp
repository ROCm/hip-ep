/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// ONNX ConstantOfShape — scalar splat over a runtime-resolved shape.
//
// Two wrappers, matching the operand-provenance dispatch in
// ConstantOfShapeConversion:
//
//   wrap_constant_of_shape (Category B):
//     The shape tensor traces to a func-arg (or any other host-resolvable
//     value). The EP has already used `output_dim_specs` to allocate the
//     ORT OrtValue at the right size and passed its GPU buffer + element
//     count. The wrapper just dispatches the fill kernel.
//
//   wrap_constant_of_shape_dyn (Category C):
//     The shape tensor is GPU-resident (e.g. the result of a hip.range on
//     intermediate operands). The wrapper:
//       1. D2H + sync the shape tensor (length = output rank).
//       2. Compute num_elements = prod(shape).
//       3. Publish dim sizes (one slot per output dim) so the EP-side
//          post-compute resolver can build the actual OrtValue shape.
//       4. Allocate `num_elements * elem_size` bytes from the dyn pool and
//          publish the buffer pointer to slot[0] (the first dim's slot
//          serves as the buffer-publish key — there is one buffer per
//          tensor regardless of rank).
//       5. Dispatch the fill kernel.
//
// The Category-C path is the one that makes this operator dynamic-aware;
// the Category-B path exists so the same kernel + wrapper can also serve
// non-dynamic outputs that simply benefit from running on the EP stream.

#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
#include "hip_custom_kernels.h"

#include <cstdio>
#include <hip/hip_runtime.h>

namespace {

// Per-dtype size (bytes). Mirrors `onnx_elem_type_size` on the EP side
// without pulling in ORT headers. Hot path; kept local.
size_t constant_of_shape_elem_size(int hip_dtype) {
  switch (hip_dtype) {
  case HIP_DTYPE_FLOAT32:
    return 4;
  case HIP_DTYPE_FLOAT16:
    return 2;
  case HIP_DTYPE_BFLOAT16:
    return 2;
  case HIP_DTYPE_FLOAT64:
    return 8;
  case HIP_DTYPE_INT8:
    return 1;
  case HIP_DTYPE_INT16:
    return 2;
  case HIP_DTYPE_INT32:
    return 4;
  case HIP_DTYPE_INT64:
    return 8;
  default:
    return 0;
  }
}

} // namespace

// Category-B: output buffer + element count already known to the caller
// (the EP has resolved the shape via output_dim_specs and allocated the
// OrtValue). Just dispatch the fill kernel.
extern "C" int wrap_constant_of_shape(RuntimeState *state, void *output,
                                      int64_t output_num_elements,
                                      int64_t value_bits, int64_t hip_dtype) {
  OP_PROFILE(
      "constant_of_shape",
      [&] {
        char b[64];
        snprintf(b, sizeof(b), "n%lld:dt%lld", (long long)output_num_elements,
                 (long long)hip_dtype);
        return std::string(b);
      },
      state);

  if (!state || !output) {
    fprintf(stderr, "[REAL] wrap_constant_of_shape: null state/output\n");
    return -1;
  }
  if (output_num_elements <= 0) {
    return 0;
  }
  void *stream = hipdnn_ep_state_get_stream(state);
  return hip_constant_of_shape(stream, output, output_num_elements, value_bits,
                               static_cast<int>(hip_dtype));
}

// Category-C: shape tensor is GPU-resident, so the wrapper owns the
// shape D2H, dim publish, buffer publish, and kernel dispatch. One slot
// per output dim. The buffer is associated with `slot_ids[0]` -- the EP
// post-compute resolver reads the buffer from slot_ids[0] and the dim
// sizes from each slot in order.
extern "C" int
wrap_constant_of_shape_dyn(RuntimeState *state, const void *shape_dev,
                           int64_t shape_dtype, int64_t output_rank,
                           const int32_t *slot_ids, int64_t value_bits,
                           int64_t output_dtype) {
  OP_PROFILE(
      "constant_of_shape_dyn",
      [&] {
        char b[64];
        snprintf(b, sizeof(b), "r%lld:dt%lld", (long long)output_rank,
                 (long long)output_dtype);
        return std::string(b);
      },
      state);

  if (!state || !shape_dev || !slot_ids) {
    fprintf(stderr,
            "[REAL] wrap_constant_of_shape_dyn: null state/shape/slot_ids\n");
    return -1;
  }
  if (output_rank <= 0 || output_rank > 8) {
    fprintf(stderr,
            "[REAL] wrap_constant_of_shape_dyn: invalid rank=%lld (1..8)\n",
            (long long)output_rank);
    return -1;
  }
  int hip_dtype = static_cast<int>(output_dtype);
  size_t elem_size = constant_of_shape_elem_size(hip_dtype);
  if (elem_size == 0) {
    fprintf(stderr, "[REAL] wrap_constant_of_shape_dyn: unsupported dtype=%d\n",
            hip_dtype);
    return -1;
  }

  void *stream_v = hipdnn_ep_state_get_stream(state);
  hipStream_t stream = static_cast<hipStream_t>(stream_v);

  // ===== Stage shape D2H =====
  // Shape tensor is rank-1 of length output_rank. Element type may be
  // INT32 or INT64 depending on the upstream ONNX op (ONNX spec allows
  // either). We materialise to int64 for the dim publish + product.
  int64_t shape_host[8] = {0};
  if (shape_dtype == HIP_DTYPE_INT64) {
    hipError_t herr =
        hipMemcpyAsync(shape_host, shape_dev, sizeof(int64_t) * output_rank,
                       hipMemcpyDeviceToHost, stream);
    if (herr != hipSuccess) {
      fprintf(stderr,
              "[REAL] wrap_constant_of_shape_dyn: D2H i64 shape failed: %s\n",
              hipGetErrorString(herr));
      return -1;
    }
  } else if (shape_dtype == HIP_DTYPE_INT32) {
    int32_t shape_host_i32[8] = {0};
    hipError_t herr =
        hipMemcpyAsync(shape_host_i32, shape_dev, sizeof(int32_t) * output_rank,
                       hipMemcpyDeviceToHost, stream);
    if (herr != hipSuccess) {
      fprintf(stderr,
              "[REAL] wrap_constant_of_shape_dyn: D2H i32 shape failed: %s\n",
              hipGetErrorString(herr));
      return -1;
    }
    if (hipStreamSynchronize(stream) != hipSuccess) {
      fprintf(stderr,
              "[REAL] wrap_constant_of_shape_dyn: sync after i32 shape D2H "
              "failed\n");
      return -1;
    }
    for (int64_t i = 0; i < output_rank; ++i)
      shape_host[i] = static_cast<int64_t>(shape_host_i32[i]);
  } else {
    fprintf(stderr,
            "[REAL] wrap_constant_of_shape_dyn: unsupported shape dtype=%lld "
            "(only INT32 / INT64)\n",
            (long long)shape_dtype);
    return -1;
  }
  if (shape_dtype == HIP_DTYPE_INT64) {
    if (hipStreamSynchronize(stream) != hipSuccess) {
      fprintf(stderr,
              "[REAL] wrap_constant_of_shape_dyn: sync after i64 shape D2H "
              "failed\n");
      return -1;
    }
  }

  // ===== Compute num_elements + publish dims =====
  int64_t num_elements = 1;
  for (int64_t i = 0; i < output_rank; ++i) {
    int64_t d = shape_host[i];
    if (d < 0) {
      fprintf(stderr,
              "[REAL] wrap_constant_of_shape_dyn: negative dim %lld at idx "
              "%lld\n",
              (long long)d, (long long)i);
      return -1;
    }
    num_elements *= d;
    hipdnn_ep_state_publish_dim(state, slot_ids[i], d);
  }

  RUNTIME_DEBUG_LOG(
      "[REAL] wrap_constant_of_shape_dyn: rank=%lld num_elements=%lld "
      "slot[0]=%d dtype=%d\n",
      (long long)output_rank, (long long)num_elements, slot_ids[0], hip_dtype);

  // ===== Allocate output + publish buffer =====
  // The buffer is associated with slot_ids[0] -- the EP post-compute
  // resolver reads the buffer pointer from there. Empty outputs publish a
  // null buffer (the EP recognises this and produces a 0-numel OrtValue).
  void *out_dev = nullptr;
  if (num_elements > 0) {
    int64_t out_bytes = num_elements * static_cast<int64_t>(elem_size);
    out_dev = hipdnn_ep_state_dyn_pool_alloc(state, out_bytes);
    if (!out_dev) {
      fprintf(stderr,
              "[REAL] wrap_constant_of_shape_dyn: dyn_pool_alloc(%lld) "
              "failed\n",
              (long long)out_bytes);
      return -1;
    }
  }
  hipdnn_ep_state_publish_buffer(state, slot_ids[0], out_dev);

  if (num_elements == 0)
    return 0;

  // ===== Fill =====
  return hip_constant_of_shape(stream_v, out_dev, num_elements, value_bits,
                               hip_dtype);
}
