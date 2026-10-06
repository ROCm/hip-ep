/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
#include "hip_custom_kernels.h"

#include <cstdio>
#include <string>

// onnx.QLinearConv. 8-bit activations and weights, optional int32 bias,
// grouped NCHW convolution. Not wrap_qconv: that entry point is the W4A16
// 1x1 fusion and rejects this shape.

static int hipdnn_to_hip_dtype_qlinear(int64_t hipdnn_type) {
  switch (hipdnn_type) {
  case HIPDNN_EP_DATATYPE_INT8:
    return HIP_DTYPE_INT8;
  case HIPDNN_EP_DATATYPE_UINT8:
    return HIP_DTYPE_UINT8;
  default:
    return -1;
  }
}

static int check_count(const char *name, int64_t count, int64_t channels,
                       bool allow_per_channel) {
  if (count == 1)
    return 0;
  if (allow_per_channel && count == channels)
    return 0;
  fprintf(stderr,
          "[REAL] wrap_qlinear_conv: %s count %lld is not per-tensor%s\n", name,
          (long long)count, allow_per_channel ? " or per output channel" : "");
  return -1;
}

int wrap_qlinear_conv(
    RuntimeState *state, const void *input, const void *input_scale,
    const void *input_zero_point, const void *weights, const void *weight_scale,
    const void *weight_zero_point, const void *output_scale,
    const void *output_zero_point, const void *bias, void *output,
    int64_t batch, int64_t in_channels, int64_t out_channels, int64_t height_in,
    int64_t width_in, int64_t height_out, int64_t width_out, int64_t kernel_h,
    int64_t kernel_w, int64_t stride_h, int64_t stride_w, int64_t pad_h,
    int64_t pad_w, int64_t dilation_h, int64_t dilation_w, int64_t group,
    int64_t input_dtype, int64_t weight_dtype, int64_t output_dtype,
    int64_t bias_dtype, int64_t input_scale_count, int64_t weight_scale_count,
    int64_t output_scale_count, int64_t input_zp_count, int64_t weight_zp_count,
    int64_t output_zp_count) {
  OP_PROFILE(
      "qlinear_conv",
      [&] {
        char b[160];
        snprintf(
            b, sizeof(b), "%lldx%lldx%lldx%lld,cin=%lld,k=%lldx%lld,g=%lld,%s",
            (long long)batch, (long long)out_channels, (long long)height_out,
            (long long)width_out, (long long)in_channels, (long long)kernel_h,
            (long long)kernel_w, (long long)group,
            hipdnn_ep_datatype_name(input_dtype));
        return std::string(b);
      },
      state);

  if (!state || !input || !input_scale || !input_zero_point || !weights ||
      !weight_scale || !weight_zero_point || !output_scale ||
      !output_zero_point || !output) {
    fprintf(stderr, "[REAL] wrap_qlinear_conv: null argument\n");
    return -1;
  }
  if (batch < 1 || in_channels < 1 || out_channels < 1 || height_in < 1 ||
      width_in < 1 || height_out < 1 || width_out < 1 || kernel_h < 1 ||
      kernel_w < 1 || stride_h < 1 || stride_w < 1 || dilation_h < 1 ||
      dilation_w < 1 || group < 1) {
    fprintf(stderr, "[REAL] wrap_qlinear_conv: non-positive geometry\n");
    return -1;
  }
  if (in_channels % group != 0 || out_channels % group != 0) {
    fprintf(stderr,
            "[REAL] wrap_qlinear_conv: group %lld does not divide channels "
            "(cin=%lld, cout=%lld)\n",
            (long long)group, (long long)in_channels, (long long)out_channels);
    return -1;
  }
  if (bias && bias_dtype != HIPDNN_EP_DATATYPE_INT32) {
    fprintf(stderr, "[REAL] wrap_qlinear_conv: bias must be i32, got %lld\n",
            (long long)bias_dtype);
    return -1;
  }

  int hip_input = hipdnn_to_hip_dtype_qlinear(input_dtype);
  int hip_weight = hipdnn_to_hip_dtype_qlinear(weight_dtype);
  int hip_output = hipdnn_to_hip_dtype_qlinear(output_dtype);
  if (hip_input < 0 || hip_weight < 0 || hip_output < 0) {
    fprintf(stderr,
            "[REAL] wrap_qlinear_conv: unsupported dtypes x=%lld w=%lld "
            "y=%lld\n",
            (long long)input_dtype, (long long)weight_dtype,
            (long long)output_dtype);
    return -1;
  }
  if (check_count("input scale", input_scale_count, out_channels, false) ||
      check_count("output scale", output_scale_count, out_channels, false) ||
      check_count("input zero point", input_zp_count, out_channels, false) ||
      check_count("output zero point", output_zp_count, out_channels, false) ||
      check_count("weight scale", weight_scale_count, out_channels, true) ||
      check_count("weight zero point", weight_zp_count, out_channels, true))
    return -1;

  void *stream = hipdnn_ep_state_get_stream(state);
  RUNTIME_DEBUG_LOG(
      "[REAL] wrap_qlinear_conv: N=%lld Cin=%lld Cout=%lld Hin=%lld Win=%lld "
      "Hout=%lld Wout=%lld k=%lldx%lld s=%lldx%lld g=%lld bias=%s\n",
      (long long)batch, (long long)in_channels, (long long)out_channels,
      (long long)height_in, (long long)width_in, (long long)height_out,
      (long long)width_out, (long long)kernel_h, (long long)kernel_w,
      (long long)stride_h, (long long)stride_w, (long long)group,
      bias ? "yes" : "null");

  int rc = hip_qlinear_conv(
      stream, input, input_scale, input_zero_point, weights, weight_scale,
      weight_zero_point, output_scale, output_zero_point, bias, output, batch,
      in_channels, out_channels, height_in, width_in, height_out, width_out,
      kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dilation_h,
      dilation_w, group, hip_input, hip_weight, hip_output, weight_scale_count,
      weight_zp_count);
  if (rc != 0) {
    fprintf(stderr, "[REAL] wrap_qlinear_conv: kernel launch failed (%d)\n",
            rc);
    return -1;
  }
  return 0;
}
