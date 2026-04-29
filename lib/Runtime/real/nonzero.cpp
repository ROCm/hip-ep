/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//
// Runtime wrapper for hip_nonzero (ONNX NonZero opset 13).
//
// Allocates a single int64 device counter via the runtime's shared
// workspace, zero-fills both the counter and the output buffer (so unused
// slots read back as the zero coord), then dispatches the kernel.  We
// intentionally stay synchronous-on-stream here -- callers that need the
// final K written to host can read `output[0,0]` (or arrange a separate
// length output) since the counter itself is a scratch device value.
//
// k_max is the worst-case bound baked into the output type at conversion
// time; the kernel silently drops slots beyond that.

#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "hip_custom_kernels.h"
#include "runtime_types.h"

#include <cstdint>
#include <cstdio>

// Map HIPDNN_EP_DATATYPE_* to the hip_dtype_t enum used by the kernel.
// Unrecognised types fall through to a sentinel that the kernel treats as
// "1-byte / bool"; this covers the (compiler-side) i8 case used for ONNX
// bool / ui8 buffers.
static int hipdnn_ep_to_hip_dtype_nonzero(int64_t data_type) {
  switch (data_type) {
  case HIPDNN_EP_DATATYPE_FLOAT:
    return HIP_DTYPE_FLOAT32;
  case HIPDNN_EP_DATATYPE_HALF:
    return HIP_DTYPE_FLOAT16;
  case HIPDNN_EP_DATATYPE_BFLOAT16:
    return HIP_DTYPE_BFLOAT16;
  case HIPDNN_EP_DATATYPE_INT32:
    return HIP_DTYPE_INT32;
  case HIPDNN_EP_DATATYPE_INT64:
    return HIP_DTYPE_INT64;
  default:
    // Sentinel handled by the kernel's `default:` (uint8_t) branch.
    return -1;
  }
}

extern "C" int wrap_nonzero(RuntimeState *state, void *input, void *output,
                            const int64_t *in_shape, int64_t rank,
                            int64_t total_elements, int64_t k_max,
                            int64_t data_type) {
  if (!state || !input || !output || !in_shape) {
    fprintf(stderr, "wrap_nonzero: null argument\n");
    return -1;
  }
  if (rank <= 0) {
    fprintf(stderr, "wrap_nonzero: invalid rank=%lld\n", (long long)rank);
    return -1;
  }
  if (k_max < 0) {
    fprintf(stderr, "wrap_nonzero: invalid k_max=%lld\n", (long long)k_max);
    return -1;
  }
  if (k_max == 0 || total_elements <= 0)
    return 0;

  hipStream_t stream =
      static_cast<hipStream_t>(hipdnn_ep_state_get_stream(state));

  // Carve a single int64 counter out of the shared runtime workspace.  This
  // keeps wrap_nonzero allocation-free at steady state -- the workspace is
  // pre-grown at session init and re-used by every op that needs scratch.
  if (hipdnn_ep_state_ensure_workspace(state, sizeof(int64_t)) != 0) {
    fprintf(stderr, "wrap_nonzero: failed to ensure scratch workspace\n");
    return -1;
  }
  void *counter = hipdnn_ep_state_get_workspace(state);
  if (!counter) {
    fprintf(stderr, "wrap_nonzero: workspace pointer is null\n");
    return -1;
  }

  // Zero the counter and the output buffer so trailing (unused) slots read
  // back as the zero coord.  Output is `rank * k_max` int64 elements.
  hipError_t err = hipMemsetAsync(counter, 0, sizeof(int64_t), stream);
  if (err != hipSuccess) {
    fprintf(stderr, "wrap_nonzero: counter memset failed: %s\n",
            hipGetErrorString(err));
    return static_cast<int>(err);
  }
  err = hipMemsetAsync(output, 0, sizeof(int64_t) * rank * k_max, stream);
  if (err != hipSuccess) {
    fprintf(stderr, "wrap_nonzero: output memset failed: %s\n",
            hipGetErrorString(err));
    return static_cast<int>(err);
  }

  int hip_dtype = hipdnn_ep_to_hip_dtype_nonzero(data_type);
  RUNTIME_DEBUG_LOG("[REAL] wrap_nonzero: dtype=%s, rank=%lld, total=%lld, "
                    "k_max=%lld\n",
                    hipdnn_ep_datatype_name(data_type), (long long)rank,
                    (long long)total_elements, (long long)k_max);

  int rc = hip_nonzero(stream, input, output, counter, in_shape, rank,
                       total_elements, k_max, hip_dtype);
  return rc;
}
