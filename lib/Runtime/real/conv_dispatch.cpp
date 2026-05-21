/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// Conv forward dispatch shim.
//
// hip.conv lowers to wrap_conv_forward_dispatch (HipToLLVM/ConvLowering.cpp).
// This file picks the actual backend based on HIPDNN_EP_CONV (read once
// per process into a static):
//   HIPDNN_EP_CONV unset / "auto"   -> default: ck for fp16, miopen for fp32
//   HIPDNN_EP_CONV=ck               -> always wrap_ckConvForward (fp16-only)
//   HIPDNN_EP_CONV=miopen           -> always wrap_miopenConvolutionForward
//
// The Auto default reflects bench data: CK wins per-shape on fp16 nearly
// everywhere on RDNA3+ (~1.27x-1.49x on VL patch convs, see CLAUDE.md
// "Verified perf snapshot -- Conv backends"). It only loses on the rn50
// 1x512x7x7 stage and on shapes that don't have CK-supported instances.
// CK is fp16-only in v1 (TheRock's CK build has CK_ENABLE_DL_KERNELS undef'd
// so fp32 NHWGC has no precompiled instances), so Auto routes fp32 to
// MIOpen automatically.
//
// Env-var read uses std::getenv -- matches the existing pattern in
// lib/Runtime/real/gqa.cpp. Read once into a static so subsequent calls
// are a single load.

#include "hipdnn_ep_runtime.h"

#include <cstdlib>
#include <cstring>

namespace {

enum class ConvBackend { Auto, Miopen, Ck };

ConvBackend selected_backend() {
  static const ConvBackend choice = [] {
    const char *v = std::getenv("HIPDNN_EP_CONV");
    if (!v || !*v)
      return ConvBackend::Auto;
    if (std::strcmp(v, "ck") == 0 || std::strcmp(v, "CK") == 0)
      return ConvBackend::Ck;
    if (std::strcmp(v, "miopen") == 0 || std::strcmp(v, "MIOpen") == 0 ||
        std::strcmp(v, "MIOPEN") == 0)
      return ConvBackend::Miopen;
    // Unknown value (e.g. "auto") -> Auto.
    return ConvBackend::Auto;
  }();
  return choice;
}

} // namespace

extern "C" int wrap_conv_forward_dispatch(
    RuntimeState *state, const void *input, int64_t input_n, int64_t input_c,
    int64_t input_h, int64_t input_w, const void *weights, int64_t weights_k,
    const void *bias, void *output, int64_t output_h, int64_t output_w,
    int64_t kernel_h, int64_t kernel_w, int64_t stride_h, int64_t stride_w,
    int64_t pad_top, int64_t pad_left, int64_t pad_bottom, int64_t pad_right,
    int64_t dilation_h, int64_t dilation_w, int64_t group,
    int64_t element_size_bytes) {
  ConvBackend choice = selected_backend();

  // Auto: CK wherever it can run (fp16 only); fall back to MIOpen for
  // fp32 (and any future dtype CK doesn't ship instances for).
  if (choice == ConvBackend::Auto) {
    choice = (element_size_bytes == 2) ? ConvBackend::Ck : ConvBackend::Miopen;
  }

  if (choice == ConvBackend::Ck) {
    return wrap_ckConvForward(
        state, input, input_n, input_c, input_h, input_w, weights, weights_k,
        bias, output, output_h, output_w, kernel_h, kernel_w, stride_h,
        stride_w, pad_top, pad_left, pad_bottom, pad_right, dilation_h,
        dilation_w, group, element_size_bytes);
  }
  return wrap_miopenConvolutionForward(
      state, input, input_n, input_c, input_h, input_w, weights, weights_k,
      bias, output, output_h, output_w, kernel_h, kernel_w, stride_h, stride_w,
      pad_top, pad_left, pad_bottom, pad_right, dilation_h, dilation_w, group,
      element_size_bytes);
}
