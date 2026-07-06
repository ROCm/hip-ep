/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "../cpu_fallback_invoke.h"
#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
#include "hip_custom_kernels.h"
#include "runtime_types.h"

#include <cstdio>

static int64_t transpose_elem_size_to_hip_dtype(int64_t element_size_bytes) {
  switch (element_size_bytes) {
  case 1:
    return HIPDNN_EP_DATATYPE_UINT8;
  case 2:
    return HIPDNN_EP_DATATYPE_HALF;
  case 4:
    return HIPDNN_EP_DATATYPE_FLOAT;
  case 8:
    return HIPDNN_EP_DATATYPE_INT64;
  default:
    return -1;
  }
}

int wrap_transpose(RuntimeState *state, const void *input, void *output,
                   int64_t rank, const int64_t *input_shape,
                   const int64_t *perm, int64_t num_elements,
                   int64_t element_size_bytes) {
  OP_PROFILE(
      "transpose",
      [&] {
        char b[64];
        snprintf(b, sizeof(b), "n=%lld rank=%lld", (long long)num_elements,
                 (long long)rank);
        return std::string(b);
      },
      state);
  // Empty transpose is a no-op (NonZero count=0 → [3,0] indices in embedding).
  if (num_elements <= 0)
    return 0;

  if (!state || !input || !output || !input_shape || !perm) {
    RUNTIME_DEBUG_LOG(
        "[REAL] wrap_transpose: null argument (state=%p input=%p output=%p "
        "input_shape=%p perm=%p rank=%lld num=%lld elem=%lld)\n",
        (void *)state, input, output, (const void *)input_shape,
        (const void *)perm, (long long)rank, (long long)num_elements,
        (long long)element_size_bytes);
    if (input_shape && rank > 0 && rank <= 8) {
      fprintf(stderr, "  input_shape=[");
      for (int64_t i = 0; i < rank; ++i)
        fprintf(stderr, "%lld%s", (long long)input_shape[i],
                i + 1 == rank ? "" : ",");
      fprintf(stderr, "]");
      if (perm) {
        fprintf(stderr, "  perm=[");
        for (int64_t i = 0; i < rank; ++i)
          fprintf(stderr, "%lld%s", (long long)perm[i],
                  i + 1 == rank ? "" : ",");
        fprintf(stderr, "]");
      }
      fprintf(stderr, "\n");
    }
    return -1;
  }
  if (rank <= 0) {
    RUNTIME_DEBUG_LOG("[REAL] wrap_transpose: invalid rank=%lld\n",
                      (long long)rank);
    return -1;
  }

  void *stream = hipdnn_ep_state_get_stream(state);

  RUNTIME_DEBUG_LOG(
      "[REAL] wrap_transpose: rank=%lld, num_elements=%lld, elem_size=%lld\n",
      (long long)rank, (long long)num_elements, (long long)element_size_bytes);

  {
    const int64_t hip_dtype = transpose_elem_size_to_hip_dtype(element_size_bytes);
    if (hip_dtype < 0) {
      fprintf(stderr,
              "[REAL] wrap_transpose: unsupported element_size_bytes=%lld\n",
              (long long)element_size_bytes);
      return -1;
    }
    int64_t out_shape[8];
    for (int64_t i = 0; i < rank; ++i)
      out_shape[i] = input_shape[perm[i]];
    HipdnnCpuFbGenericDesc fb{};
    fb.op_name = "Transpose";
    fb.opset = 13;
    fb.num_inputs = 1;
    fb.num_outputs = 1;
    fb.inputs[0] = {const_cast<void *>(input), rank, input_shape,
                    num_elements, hip_dtype};
    fb.outputs[0] = {output, rank, out_shape, num_elements, hip_dtype};
    fb.num_attrs = 1;
    fb.attrs[0] = {"perm", HIPDNN_CPU_FB_ATTR_INTS, 0, perm,
                   static_cast<int32_t>(rank), 0.f};
    const int fb_rc =
        hipdnn_cpu_fallback_try_generic(state, stream, "Transpose", &fb);
    if (fb_rc == 0)
      return 0;
    if (fb_rc < 0)
      return -1;
  }

  return hip_transpose(stream, input, output, rank, input_shape, perm,
                       num_elements, static_cast<int>(element_size_bytes));
}
