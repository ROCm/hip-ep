/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once

#include "../hipdnn_ep_runtime.h"

#ifdef __cplusplus
extern "C" {
#endif

// Attempt AMDMLSS HIP conv forward.
//
// Returns:
//   0  — MLSS conv completed successfully
//   1  — not attempted (backend disabled / unsupported shape); caller should
//        use MIOpen
//  -1  — MLSS was required but failed (HIPDNN_EP_CONV_BACKEND=mlss only)
int try_mlss_conv_forward(
    RuntimeState *state, const void *input, int64_t input_n, int64_t input_c,
    int64_t input_h, int64_t input_w, const void *weights, int64_t weights_k,
    const void *bias, void *output, int64_t output_h, int64_t output_w,
    int64_t kernel_h, int64_t kernel_w, int64_t stride_h, int64_t stride_w,
    int64_t pad_top, int64_t pad_left, int64_t pad_bottom, int64_t pad_right,
    int64_t dilation_h, int64_t dilation_w, int64_t group, int64_t data_type);

#ifdef __cplusplus
}
#endif
