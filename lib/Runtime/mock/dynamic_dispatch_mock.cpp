/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===----------------------------------------------------------------------===//
// Mock DynamicDispatch Backend
//===----------------------------------------------------------------------===//
//
// Stub implementations of DynamicDispatch wrappers for mock (GPU-free) builds.
// Returns errors indicating XRT/DD not available.
//===----------------------------------------------------------------------===//

#include "../hipdnn_ep_runtime.h"
#include <cstdio>

extern "C" {

int wrap_dd_matmul(RuntimeState *state, int32_t op_state_slot,
                   const void *input_a, const void *input_b, const void *bias,
                   void *output, int64_t M, int64_t N, int64_t K, double alpha,
                   double beta, int64_t transA, int64_t transB,
                   int64_t data_type) {
  (void)state;
  (void)op_state_slot;
  (void)input_a;
  (void)input_b;
  (void)bias;
  (void)output;
  (void)M;
  (void)N;
  (void)K;
  (void)alpha;
  (void)beta;
  (void)transA;
  (void)transB;
  (void)data_type;

  fprintf(stderr,
          "wrap_dd_matmul: DynamicDispatch not available (mock runtime)\n");
  return HIPDNN_STATUS_NOT_SUPPORTED;
}

int wrap_dd_conv2d(RuntimeState *state, int32_t op_state_slot,
                   const void *input, int64_t n, int64_t c, int64_t h,
                   int64_t w, const void *weights, int64_t k, const void *bias,
                   void *output, int64_t out_h, int64_t out_w,
                   int64_t kernel_h, int64_t kernel_w, int64_t stride_h,
                   int64_t stride_w, int64_t pad_top, int64_t pad_left,
                   int64_t pad_bottom, int64_t pad_right, int64_t dilation_h,
                   int64_t dilation_w, int64_t group, int64_t data_type) {
  (void)state;
  (void)op_state_slot;
  (void)input;
  (void)n;
  (void)c;
  (void)h;
  (void)w;
  (void)weights;
  (void)k;
  (void)bias;
  (void)output;
  (void)out_h;
  (void)out_w;
  (void)kernel_h;
  (void)kernel_w;
  (void)stride_h;
  (void)stride_w;
  (void)pad_top;
  (void)pad_left;
  (void)pad_bottom;
  (void)pad_right;
  (void)dilation_h;
  (void)dilation_w;
  (void)group;
  (void)data_type;

  fprintf(stderr,
          "wrap_dd_conv2d: DynamicDispatch not available (mock runtime)\n");
  return HIPDNN_STATUS_NOT_SUPPORTED;
}

// XRT context accessors (return NULL in mock)
void *hipdnn_ep_state_get_xrt_device(RuntimeState *state) {
  (void)state;
  return nullptr;
}

void *hipdnn_ep_state_get_xrt_context(RuntimeState *state) {
  (void)state;
  return nullptr;
}

} // extern "C"
