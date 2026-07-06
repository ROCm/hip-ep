/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// And: y = a & b  (bool / 1-byte tensors).
//
// Source:
// onnxruntime/core/providers/cuda/math/binary_elementwise_ops_impl.{h,cu}
//         @ v1.22.2 (BINARY_OP_NAME_EXPR(And, (a & b)),
//                    SPECIALIZED_BINARY_ELEMENTWISE_IMPL(And, bool))
//
// ONNX `And` is specified on bool tensors only. Bool is marshalled as a
// 1-byte (uint8) stream on the GPU side, matching the convention used by
// wrap_not / wrap_equal / wrap_less. The data_type argument is logged
// only -- the kernel always treats lhs/rhs/output as 1-byte streams.
#include "../cpu_fallback_invoke.h"
#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
#include "hip_custom_kernels.h"

#include <cstdio>

int wrap_and(RuntimeState *state, void *a, void *b, void *output,
             int64_t num_elements, int64_t data_type) {
  OP_PROFILE(
      "and",
      [&] {
        char buf[48];
        snprintf(buf, sizeof(buf), "%lld:%s", (long long)num_elements,
                 hipdnn_ep_datatype_name(data_type));
        return std::string(buf);
      },
      state);

  if (!state || !a || !b || !output) {
    RUNTIME_DEBUG_LOG("[REAL] wrap_and: null argument\n");
    return -1;
  }
  if (num_elements <= 0) {
    return 0;
  }

  // hip-to-llvm passes data_type = 0 for tensor<i1> (no enum slot for i1);
  // log for diagnostics but do not enforce -- the wire format is always
  // 1 byte per element when the graph type was tensor<i1>.
  (void)data_type;

  void *stream = hipdnn_ep_state_get_stream(state);
  {
    const int64_t bool_dtype = HIPDNN_EP_DATATYPE_UINT8;
    const int fb_rc = hipdnn_cpu_fb_try_binary_1d(
        state, stream, "And", a, b, output, num_elements, bool_dtype,
        bool_dtype);
    if (fb_rc == 0)
      return 0;
    if (fb_rc < 0)
      return -1;
  }
  RUNTIME_DEBUG_LOG("[REAL] wrap_and: num=%lld, data_type=%s "
                    "(treated as 1-byte bool) -> hip_elementwise_and\n",
                    (long long)num_elements,
                    hipdnn_ep_datatype_name(data_type));
  return hip_elementwise_and(stream, a, b, output, num_elements);
}
