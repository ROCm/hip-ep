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
#include "hip_custom_kernels.h"
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
                                              int64_t cols,
                                              int64_t data_type) {
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
  hipStream_t stream =
      static_cast<hipStream_t>(hipdnn_ep_state_get_stream(state));
  if (data_type == HIPDNN_EP_DATATYPE_HALF) {
    int64_t elems = rows * cols;
    void *in32 = nullptr;
    void *out32 = nullptr;
    if (hipMalloc(&in32, static_cast<size_t>(elems) * sizeof(float)) !=
            hipSuccess ||
        hipMalloc(&out32, static_cast<size_t>(elems) * sizeof(float)) !=
            hipSuccess) {
      fprintf(stderr,
              "[hipdnn_ep] hip_miopen_softmax: f16 upcast alloc failed\n");
      if (in32) hipFree(in32);
      if (out32) hipFree(out32);
      return -1;
    }
    int rc = hip_cast(stream, input, in32, elems, HIP_DTYPE_FLOAT16,
                      HIP_DTYPE_FLOAT32);
    if (rc == 0) {
      rc = hip_miopen_softmax(handle_v, in32, out32, rows, cols,
                              HIPDNN_EP_DATATYPE_FLOAT);
    }
    if (rc == 0) {
      rc = hip_cast(stream, out32, output, elems, HIP_DTYPE_FLOAT32,
                    HIP_DTYPE_FLOAT16);
    }
    hipFree(in32);
    hipFree(out32);
    return rc;
  }
  miopenDataType_t mio_dtype;
  switch (data_type) {
  case HIPDNN_EP_DATATYPE_FLOAT:
    mio_dtype = miopenFloat;
    break;
  case HIPDNN_EP_DATATYPE_HALF:
    mio_dtype = miopenHalf;
    break;
  case HIPDNN_EP_DATATYPE_BFLOAT16:
    mio_dtype = miopenBFloat16;
    break;
  default:
    fprintf(stderr, "[hipdnn_ep] hip_miopen_softmax: unsupported dtype %lld\n",
            (long long)data_type);
    return -1;
  }
  miopenTensorDescriptor_t inout_desc = nullptr;
  if (miopenCreateTensorDescriptor(&inout_desc) != miopenStatusSuccess)
    return -1;
  // MIOpen wants 4-D NCHW.  We treat (rows, cols) as (rows, cols, 1, 1)
  // -- equivalent for SOFTMAX_MODE_INSTANCE which reduces along
  // C*H*W per N.
  if (miopenSet4dTensorDescriptor(inout_desc, mio_dtype,
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

} // extern "C"
