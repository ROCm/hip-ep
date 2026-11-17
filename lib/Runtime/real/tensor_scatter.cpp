/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// TensorScatter (ONNX opset 24): row writes along one axis. When the caller
// binds past and output to the same buffer the two pointers are equal and
// only the update rows are written; otherwise past is copied into output
// first. Write positions stay on the device.

#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
#include "hip_custom_kernels.h"

#include <cstdio>
#include <hip/hip_runtime.h>

int wrap_tensor_scatter(RuntimeState *state, void *past, void *update,
                        void *write_indices, void *output, int64_t elem_bytes,
                        int64_t batch, int64_t outer, int64_t max_seq,
                        int64_t seq, int64_t inner, int64_t circular) {
  OP_PROFILE(
      "tensor_scatter",
      [&] {
        char b[96];
        snprintf(b, sizeof(b), "b%lld:o%lld:m%lld:s%lld:i%lld:%s",
                 (long long)batch, (long long)outer, (long long)max_seq,
                 (long long)seq, (long long)inner,
                 past == output ? "inplace" : "copy");
        return std::string(b);
      },
      state);

  if (!state || elem_bytes <= 0 || batch < 0 || outer < 0 || max_seq < 0 ||
      seq < 0 || inner < 0) {
    fprintf(stderr, "[REAL] wrap_tensor_scatter: invalid arguments\n");
    return -1;
  }
  const size_t total_bytes = static_cast<size_t>(batch) * outer * max_seq *
                             inner * static_cast<size_t>(elem_bytes);
  if (total_bytes == 0)
    return 0;
  if (!past || !output) {
    fprintf(stderr, "[REAL] wrap_tensor_scatter: null past or output\n");
    return -1;
  }
  if (!circular && seq > max_seq) {
    fprintf(stderr,
            "[REAL] wrap_tensor_scatter: %lld update rows exceed the %lld-row "
            "cache\n",
            (long long)seq, (long long)max_seq);
    return -1;
  }

  hipStream_t stream =
      static_cast<hipStream_t>(hipdnn_ep_state_get_stream(state));
  if (past != output &&
      hipMemcpyAsync(output, past, total_bytes, hipMemcpyDeviceToDevice,
                     stream) != hipSuccess) {
    fprintf(stderr, "[REAL] wrap_tensor_scatter: past copy failed\n");
    return -1;
  }
  if (seq == 0 || !update)
    return 0;

  RUNTIME_DEBUG_LOG("[REAL] wrap_tensor_scatter: b=%lld outer=%lld "
                    "max_seq=%lld seq=%lld inner=%lld in_place=%d\n",
                    (long long)batch, (long long)outer, (long long)max_seq,
                    (long long)seq, (long long)inner, past == output);
  return hip_tensor_scatter(stream, output, update,
                            static_cast<const int64_t *>(write_indices), batch,
                            outer, max_seq, seq, inner * elem_bytes,
                            static_cast<int>(circular));
}
