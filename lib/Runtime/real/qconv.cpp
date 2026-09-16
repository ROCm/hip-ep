/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "../debug_log.h"
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
#include "hip_custom_kernels.h"

#include <cstdio>

//===----------------------------------------------------------------------===//
// QConv — fused quantized 1x1 convolution (W4A16)
//===----------------------------------------------------------------------===//
//
// Lowering signature (matches QConvLowering.cpp):
//   wrap_qconv(state, input, weights, weight_scales, weight_zero_points,
//              bias, output,
//              batch, in_channels, out_channels, spatial_size,
//              activation_dtype, weight_dtype, weight_bits, bias_dtype,
//              input_scale, input_zp, output_scale, output_zp)
//
// Replaces the DQ(x) + DQ(w) -> Conv -> Q chain a W4A16 export produces. What
// the fusion buys is not the arithmetic -- the unfused path already accumulated
// in float -- but the traffic: the weights stay at half a byte per value all
// the way into the kernel instead of being expanded to f32 in global memory
// first, and no full-precision activation or output is ever materialised.
//
// There is no op state. Like wrap_conv's kernel this derives everything from
// its arguments, and unlike wrap_conv there is not even a slot to accept, since
// hip.qconv carries no OpStateOpInterface.
//
// Scale folding happens HERE rather than in the lowering (where qadd/qmul fold
// theirs), because the dominant factor cannot be folded at compile time at all:
// weight_scales is a device array indexed by output channel. Keeping the true
// scales in the ABI costs one multiply and keeps the profile and debug output
// readable.

static int hipdnn_to_hip_dtype_qconv(int64_t hipdnn_type) {
  switch (hipdnn_type) {
  case HIPDNN_EP_DATATYPE_INT8:
    return HIP_DTYPE_INT8;
  case HIPDNN_EP_DATATYPE_UINT8:
    return HIP_DTYPE_UINT8;
  case HIPDNN_EP_DATATYPE_UINT16:
    return HIP_DTYPE_UINT16;
  default:
    return -1;
  }
}

int wrap_qconv(RuntimeState *state, const void *input, const void *weights,
               const void *weight_scales, const void *weight_zero_points,
               const void *bias, void *output, int64_t batch,
               int64_t in_channels, int64_t out_channels, int64_t spatial_size,
               int64_t activation_dtype, int64_t weight_dtype,
               int64_t weight_bits, int64_t bias_dtype, float input_scale,
               int64_t input_zp, float output_scale, int64_t output_zp) {
  OP_PROFILE(
      "qconv",
      [&] {
        char b[128];
        snprintf(b, sizeof(b), "%lldx%lld,K=%lld,P=%lld,w%lldb,%s",
                 (long long)batch, (long long)out_channels,
                 (long long)in_channels, (long long)spatial_size,
                 (long long)weight_bits,
                 hipdnn_ep_datatype_name(activation_dtype));
        return std::string(b);
      },
      state);

  // weight_zero_points is required, not optional: the fusion only matches a
  // three-operand DequantizeLinear, so an absent zero point cannot reach here.
  if (!state || !input || !weights || !weight_scales || !weight_zero_points ||
      !output) {
    fprintf(stderr, "[REAL] wrap_qconv: null argument\n");
    return -1;
  }
  if (batch < 1 || in_channels < 1 || out_channels < 1 || spatial_size < 1) {
    fprintf(stderr,
            "[REAL] wrap_qconv: non-positive extent (N=%lld, K=%lld, M=%lld, "
            "P=%lld)\n",
            (long long)batch, (long long)in_channels, (long long)out_channels,
            (long long)spatial_size);
    return -1;
  }

  // Only W4A16 is implemented. The fusion pattern already refuses anything
  // else, so reaching this is a pattern/kernel disagreement rather than an
  // unsupported model.
  if (activation_dtype != HIPDNN_EP_DATATYPE_UINT16) {
    fprintf(stderr,
            "[REAL] wrap_qconv: unsupported activation_dtype %lld (%s), "
            "expected UINT16\n",
            (long long)activation_dtype,
            hipdnn_ep_datatype_name(activation_dtype));
    return -1;
  }
  int hip_act_dtype = hipdnn_to_hip_dtype_qconv(activation_dtype);
  int hip_weight_dtype = hipdnn_to_hip_dtype_qconv(weight_dtype);
  if (hip_act_dtype < 0 || hip_weight_dtype < 0 ||
      (weight_dtype != HIPDNN_EP_DATATYPE_INT8 &&
       weight_dtype != HIPDNN_EP_DATATYPE_UINT8)) {
    fprintf(stderr,
            "[REAL] wrap_qconv: unsupported weight_dtype %lld (%s), expected "
            "INT8 or UINT8 storage\n",
            (long long)weight_dtype, hipdnn_ep_datatype_name(weight_dtype));
    return -1;
  }
  // 4 means two values per byte; 8 means full width. Anything else would hand
  // the kernel a stride nothing agreed on.
  if (weight_bits != 4 && weight_bits != 8) {
    fprintf(stderr, "[REAL] wrap_qconv: unsupported weight_bits %lld\n",
            (long long)weight_bits);
    return -1;
  }
  if (bias && bias_dtype != HIPDNN_EP_DATATYPE_FLOAT) {
    fprintf(stderr, "[REAL] wrap_qconv: bias must be f32, got %lld (%s)\n",
            (long long)bias_dtype, hipdnn_ep_datatype_name(bias_dtype));
    return -1;
  }
  if (output_scale == 0.0f) {
    fprintf(stderr, "[REAL] wrap_qconv: output_scale must be non-zero\n");
    return -1;
  }

  // Fold what can be folded. The per-channel weight scale stays on the device;
  // the kernel forms M[co] = act_out_scale_ratio * weight_scales[co], so it
  // never divides. inv_output_scale only rescales the bias.
  const float act_out_scale_ratio = input_scale / output_scale;
  const float inv_output_scale = 1.0f / output_scale;

  void *stream = hipdnn_ep_state_get_stream(state);

  RUNTIME_DEBUG_LOG(
      "[REAL] wrap_qconv: act=%s(%lld) w=%s(%lld) w_bits=%lld N=%lld K=%lld "
      "M=%lld P=%lld s_x=%g z_x=%lld s_out=%g z_out=%lld ratio=%g bias=%s\n",
      hipdnn_ep_datatype_name(activation_dtype), (long long)activation_dtype,
      hipdnn_ep_datatype_name(weight_dtype), (long long)weight_dtype,
      (long long)weight_bits, (long long)batch, (long long)in_channels,
      (long long)out_channels, (long long)spatial_size, (double)input_scale,
      (long long)input_zp, (double)output_scale, (long long)output_zp,
      (double)act_out_scale_ratio, bias ? "yes" : "null");

  int rc =
      hip_qconv(stream, input, weights, weight_scales, weight_zero_points, bias,
                output, batch, in_channels, out_channels, spatial_size,
                hip_act_dtype, hip_weight_dtype, static_cast<int>(weight_bits),
                act_out_scale_ratio, input_zp, inv_output_scale, output_zp);
  if (rc != 0) {
    fprintf(stderr, "[REAL] wrap_qconv: kernel launch failed (%d)\n", rc);
    return -1;
  }

  RUNTIME_DEBUG_LOG("[REAL] wrap_qconv: completed successfully\n");
  return 0;
}
