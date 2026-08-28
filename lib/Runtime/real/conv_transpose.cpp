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
#include <string>

//===----------------------------------------------------------------------===//
// ConvTranspose — transposed convolution via the custom HIP kernel
//===----------------------------------------------------------------------===//
//
// Lowering signature (matches ConvTransposeLowering.cpp):
//   wrap_conv_transpose(state, slot, input, input_n, input_c, input_h, input_w,
//                       weights, bias, output, output_c, output_h, output_w,
//                       kernel_h, kernel_w, stride_h, stride_w,
//                       pad_top, pad_left, pad_bottom, pad_right,
//                       dilation_h, dilation_w,
//                       output_padding_h, output_padding_w,
//                       group, data_type)
//
// This replaced MIOpen on the transpose path, which was the last thing calling
// it from either convolution direction, and took the descriptor/solution cache
// (ConvTable in the former real/miopen.cpp) with it. What changes for a caller:
//
//   * `bias` is fused. MIOpen could not fuse it here -- and
//     miopenConvolutionForwardBias was observed to double the deconvolution
//     result outright -- so the old path followed every deconvolution with a
//     separate miopenOpTensor add over the whole output. That pass is gone.
//   * There is no op state, no solution cache and no workspace. MIOpen needed a
//     per-shape Find(); this kernel derives everything from its arguments, so
//     `op_state_slot` is accepted for ABI stability and ignored.
//
// Four arguments are accepted and not used, all for the same reason: they
// affect only how many output positions exist, and the caller already passes
// that as output_h / output_w.
//
//   * pad_bottom / pad_right. Padding crops a transposed convolution's output,
//     and cropping at the far end just removes trailing positions.
//   * output_padding_h / output_padding_w (ONNX "adjs"). These disambiguate the
//     output size when a stride > 1 leaves several input sizes mapping onto
//     overlapping output ranges. The cells they add are ones no input reaches,
//     and the kernel's divisibility and range tests leave those at just the
//     bias, which is what ONNX asks for.
//===----------------------------------------------------------------------===//

static int hipdnn_ep_to_hip_dtype(int64_t data_type) {
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

int wrap_conv_transpose(RuntimeState *state, int32_t op_state_slot,
                        const void *input, int64_t input_n, int64_t input_c,
                        int64_t input_h, int64_t input_w, const void *weights,
                        const void *bias, void *output, int64_t output_c,
                        int64_t output_h, int64_t output_w, int64_t kernel_h,
                        int64_t kernel_w, int64_t stride_h, int64_t stride_w,
                        int64_t pad_top, int64_t pad_left, int64_t pad_bottom,
                        int64_t pad_right, int64_t dilation_h,
                        int64_t dilation_w, int64_t output_padding_h,
                        int64_t output_padding_w, int64_t group,
                        int64_t data_type) {
  OP_PROFILE(
      "conv_transpose",
      [&] {
        char b[80];
        snprintf(b, sizeof(b), "%lldx%lldx%lldx%lld,m=%lld,k=%lldx%lld,s=%lld",
                 (long long)input_n, (long long)input_c, (long long)input_h,
                 (long long)input_w, (long long)output_c, (long long)kernel_h,
                 (long long)kernel_w, (long long)stride_h);
        return std::string(b);
      },
      state);

  // Folded into output_h / output_w by the caller; see the header comment.
  (void)pad_bottom;
  (void)pad_right;
  (void)output_padding_h;
  (void)output_padding_w;
  // No op state: this kernel needs no cached solution or workspace.
  (void)op_state_slot;

  if (!state || !input || !weights || !output) {
    fprintf(stderr, "[REAL] wrap_conv_transpose: null argument\n");
    return -1;
  }
  if (group < 1 || input_c % group != 0 || output_c % group != 0) {
    fprintf(stderr,
            "[REAL] wrap_conv_transpose: group %lld does not divide Cin %lld / "
            "Cout %lld\n",
            (long long)group, (long long)input_c, (long long)output_c);
    return -1;
  }

  int hip_dtype = hipdnn_ep_to_hip_dtype(data_type);
  if (hip_dtype < 0) {
    fprintf(stderr, "[REAL] wrap_conv_transpose: unsupported data_type %lld\n",
            (long long)data_type);
    return -1;
  }

  void *stream = hipdnn_ep_state_get_stream(state);

  RUNTIME_DEBUG_LOG(
      "[REAL] wrap_conv_transpose: dtype=%s(%lld) N=%lld Cin=%lld Cout=%lld "
      "group=%lld in=[%lld,%lld] out=[%lld,%lld] k=[%lld,%lld] s=[%lld,%lld] "
      "pbegin=[%lld,%lld] dil=[%lld,%lld] bias=%s\n",
      hipdnn_ep_datatype_name(data_type), (long long)data_type,
      (long long)input_n, (long long)input_c, (long long)output_c,
      (long long)group, (long long)input_h, (long long)input_w,
      (long long)output_h, (long long)output_w, (long long)kernel_h,
      (long long)kernel_w, (long long)stride_h, (long long)stride_w,
      (long long)pad_top, (long long)pad_left, (long long)dilation_h,
      (long long)dilation_w, bias ? "yes" : "null");

  int rc = hip_conv_transpose(
      stream, input, weights, bias, output, hip_dtype, input_n, input_c,
      output_c, input_h, input_w, output_h, output_w, kernel_h, kernel_w,
      stride_h, stride_w, pad_top, pad_left, dilation_h, dilation_w, group);
  if (rc != 0) {
    fprintf(stderr, "[REAL] wrap_conv_transpose: kernel launch failed (%d)\n",
            rc);
    return -1;
  }

  RUNTIME_DEBUG_LOG("[REAL] wrap_conv_transpose: completed successfully\n");
  return 0;
}
