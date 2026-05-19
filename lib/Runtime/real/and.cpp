/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "../hipdnn_ep_runtime.h"

// ONNX `And` element-wise logical AND on boolean tensors.
//
// Today this is a stub: it accepts the runtime ABI for an element-wise
// binary op (lhs, rhs, output, num_elements, data_type) so models that
// reference onnx.And still lower end-to-end through the OnnxToHip /
// HipToLLVM pipeline and link against the runtime bitcode. The actual
// kernel call (e.g. a bitwise-AND kernel over uint8/bool buffers) is
// pending and will be added once a model in the CI set exercises this op.
//
// ONNX bool tensors are marshalled as 1-byte elements in this EP, so a
// future implementation should expect data_type == HIPDNN_EP_DATATYPE_INT8
// here.
int wrap_and(RuntimeState *state, void *a, void *b, void *output,
             int64_t num_elements, int64_t data_type) {
  (void)state;
  (void)a;
  (void)b;
  (void)output;
  (void)num_elements;
  (void)data_type;
  // Stub: pretend success. Output buffer is left uninitialized.
  // TODO: implement element-wise logical AND on GPU.
  return 0;
}
