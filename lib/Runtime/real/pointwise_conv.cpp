/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
#include "hip_custom_kernels.h"
#include "runtime_types.h"

#include <cstdio>

//===----------------------------------------------------------------------===//
// Pointwise (1x1) convolution -- fused GEMM + bias custom kernel
//===----------------------------------------------------------------------===//
//
// Lowering signature:
//   wrap_pointwise_conv(state, input, weights, bias, output,
//                       N, Cin, Cout, HW, data_type)
//
// The HIP->LLVM lowering (ConvOpLowering) selects this wrapper instead of
// wrap_miopenConvolutionForward for a 1x1 / stride-1 / no-pad / no-dilation /
// group-1 conv with small Cin and fully static shapes. It collapses the conv
// to a batched GEMM W[Cout,Cin] @ X[Cin,HW] -> Y[Cout,HW] and folds the
// per-channel bias add into the same kernel launch. See the kernel header
// (hip_pointwise_conv) for the data layout and the routing rationale.
//
// `bias` may be null (Conv without a bias operand); the kernel skips the add.

static int pwc_to_hip_dtype(int64_t data_type) {
  switch (data_type) {
  case HIPDNN_EP_DATATYPE_FLOAT:
    return HIP_DTYPE_FLOAT32;
  case HIPDNN_EP_DATATYPE_HALF:
    return HIP_DTYPE_FLOAT16;
  case HIPDNN_EP_DATATYPE_BFLOAT16:
    return HIP_DTYPE_BFLOAT16;
  default:
    return -1;
  }
}

int wrap_pointwise_conv(RuntimeState *state, const void *input,
                        const void *weights, const void *bias, void *output,
                        int64_t N, int64_t Cin, int64_t Cout, int64_t HW,
                        int64_t data_type) {
  OP_PROFILE(
      "pointwise_conv",
      [&] {
        char b[80];
        const char *dt = (data_type == HIPDNN_EP_DATATYPE_HALF)       ? "f16"
                         : (data_type == HIPDNN_EP_DATATYPE_BFLOAT16) ? "bf16"
                                                                      : "f32";
        snprintf(b, sizeof(b), "%lldx%lldx%lld,Cout=%lld,%s", (long long)N,
                 (long long)Cin, (long long)HW, (long long)Cout, dt);
        return std::string(b);
      },
      state);

  if (!state || !input || !weights || !output) {
    fprintf(stderr, "[REAL] wrap_pointwise_conv: null argument\n");
    return -1;
  }

  int hip_dtype = pwc_to_hip_dtype(data_type);
  if (hip_dtype < 0) {
    fprintf(stderr, "[REAL] wrap_pointwise_conv: unsupported data_type %lld\n",
            (long long)data_type);
    return -1;
  }

  RUNTIME_DEBUG_LOG(
      "[REAL] wrap_pointwise_conv N=%lld Cin=%lld Cout=%lld HW=%lld bias=%s "
      "dtype=%lld\n",
      (long long)N, (long long)Cin, (long long)Cout, (long long)HW,
      bias ? "yes" : "null", (long long)data_type);

  void *stream = hipdnn_ep_state_get_stream(state);

  int result = hip_pointwise_conv(stream, input, weights, bias, output, N, Cin,
                                  Cout, HW, hip_dtype);
  if (result != 0) {
    fprintf(stderr, "[REAL] wrap_pointwise_conv: kernel launch failed (%d)\n",
            result);
    return -1;
  }

  RUNTIME_DEBUG_LOG("[REAL] wrap_pointwise_conv: completed successfully\n");
  return 0;
}
