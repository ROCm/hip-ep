/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//
// Stub implementations for runtime symbols whose dialect lowerings exist
// but whose real GPU kernels haven't been implemented yet.  Without
// these stubs the model DLL fails to link with "undefined symbol", even
// though the surviving dataflow paths through these ops are
// dead/unreachable for the unit-test inputs we care about.
//
// Each stub:
//   - logs a one-time warning (so we know if a model actually needs the
//     missing path),
//   - touches the output buffer in a benign way (zero / pass-through),
//   - returns 0 to satisfy the runtime-driver contract.
//
// Replace each stub with a real implementation as it becomes a
// correctness blocker for a model we want to ship.

#include "../hipdnn_ep_runtime.h"
#include "runtime_types.h"

#include <cstdint>
#include <cstdio>
#include <cstring>

#include <hip/hip_runtime.h>

extern "C" {

// MIOpen softmax (used by ActivationLowering for hip.miopen_softmax).
// Signature per ActivationLowering: (handle, input, output, rows, cols).
// Wraps miopenSoftmaxForward_V2 with MIOPEN_SOFTMAX_ACCURATE +
// MIOPEN_SOFTMAX_MODE_INSTANCE (per-row softmax over the last
// dimension, which is what onnx.Softmax with axis=-1 expands to).
__declspec(dllexport) int hip_miopen_softmax(void *handle_v, const void *input,
                                              void *output, int64_t rows,
                                              int64_t cols) {
  if (!handle_v || !input || !output || rows <= 0 || cols <= 0) {
    fprintf(stderr,
            "[hipdnn_ep] hip_miopen_softmax: bad args (handle=%p in=%p "
            "out=%p rows=%lld cols=%lld)\n",
            handle_v, input, output, (long long)rows, (long long)cols);
    return -1;
  }
  RuntimeState *state = static_cast<RuntimeState *>(handle_v);
  miopenHandle_t miopen_handle =
      static_cast<miopenHandle_t>(hipdnn_ep_state_get_miopen_handle(state));
  miopenTensorDescriptor_t inout_desc = nullptr;
  if (miopenCreateTensorDescriptor(&inout_desc) != miopenStatusSuccess)
    return -1;
  // MIOpen wants 4-D NCHW.  We treat (rows, cols) as (rows, cols, 1, 1)
  // -- equivalent for SOFTMAX_MODE_INSTANCE which reduces along
  // C*H*W per N.
  if (miopenSet4dTensorDescriptor(inout_desc, miopenFloat,
                                   static_cast<int>(rows),
                                   static_cast<int>(cols), 1, 1) !=
      miopenStatusSuccess) {
    miopenDestroyTensorDescriptor(inout_desc);
    return -1;
  }
  float alpha = 1.0f, beta = 0.0f;
  miopenStatus_t st = miopenSoftmaxForward_V2(
      miopen_handle, &alpha, inout_desc, input, &beta, inout_desc, output,
      MIOPEN_SOFTMAX_ACCURATE, MIOPEN_SOFTMAX_MODE_INSTANCE);
  miopenDestroyTensorDescriptor(inout_desc);
  if (st != miopenStatusSuccess) {
    fprintf(stderr, "[hipdnn_ep] miopenSoftmaxForward_V2 returned %d\n",
            (int)st);
    return -1;
  }
  return 0;
}

// hip_transpose: legacy stub removed.  TransposeLowering now routes all
// ranks to hip_transpose_nd.  Keeping a dllexport C function for
// link-time compatibility in case an old model DLL references it, but
// it's a no-op that doesn't touch any HIP stream (to avoid poisoning
// the stream with errors).
__declspec(dllexport) void hip_transpose(void *handle, const void *input,
                                          void *output, intptr_t rank,
                                          intptr_t dim0, intptr_t dim1,
                                          intptr_t s0, intptr_t s1,
                                          intptr_t s2) {
  (void)handle;
  (void)input;
  (void)output;
  (void)rank;
  (void)dim0;
  (void)dim1;
  (void)s0;
  (void)s1;
  (void)s2;
}

} // extern "C"
