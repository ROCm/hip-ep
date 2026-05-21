/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// Thin extern-C shim from generated MLIR code into hip::Backend::Conv.
// All DLL discovery, ABI checking, and lifetime are handled by
// lib/Runtime/real/hip_backend_client.cpp; this TU just translates
// exceptions (which generated LLVM-IR code can't catch) back into the
// int return code the wrap_* C ABI promises.

#include "hip_backend_client.h"
#include "hipdnn_ep_runtime.h"
#include "op_profile.h"

#include <cstdio>
#include <exception>
#include <string>

extern "C" int wrap_ckConvForward(
    RuntimeState *state, const void *input, int64_t input_n, int64_t input_c,
    int64_t input_h, int64_t input_w, const void *weights, int64_t weights_k,
    const void *bias, void *output, int64_t output_h, int64_t output_w,
    int64_t kernel_h, int64_t kernel_w, int64_t stride_h, int64_t stride_w,
    int64_t pad_top, int64_t pad_left, int64_t pad_bottom, int64_t pad_right,
    int64_t dilation_h, int64_t dilation_w, int64_t group,
    int64_t element_size_bytes) {
  if (element_size_bytes != 2) {
    fprintf(stderr,
            "[ck_conv] element_size_bytes=%lld unsupported (CK backend is "
            "fp16-only in v1)\n",
            static_cast<long long>(element_size_bytes));
    return -1;
  }

  OP_PROFILE(
      "ck_conv",
      [&] {
        char b[80];
        snprintf(b, sizeof(b), "%lldx%lldx%lldx%lld,k=%lldx%lldx%lld,g=%lld",
                 (long long)input_n, (long long)input_c, (long long)input_h,
                 (long long)input_w, (long long)weights_k, (long long)kernel_h,
                 (long long)kernel_w, (long long)group);
        return std::string(b);
      },
      state);

  try {
    auto backend = hip::GetBackend();
    backend->Conv(hipdnn_ep_state_get_stream(state), input,
                  static_cast<int>(input_n), static_cast<int>(input_c),
                  static_cast<int>(input_h), static_cast<int>(input_w), weights,
                  static_cast<int>(weights_k), static_cast<int>(kernel_h),
                  static_cast<int>(kernel_w), bias, output,
                  static_cast<int>(output_h), static_cast<int>(output_w),
                  static_cast<int>(stride_h), static_cast<int>(stride_w),
                  static_cast<int>(pad_top), static_cast<int>(pad_left),
                  static_cast<int>(pad_bottom), static_cast<int>(pad_right),
                  static_cast<int>(dilation_h), static_cast<int>(dilation_w),
                  static_cast<int>(group));
    return 0;
  } catch (const std::exception &e) {
    fprintf(stderr, "[ck_conv] %s\n", e.what());
    return -1;
  }
}
