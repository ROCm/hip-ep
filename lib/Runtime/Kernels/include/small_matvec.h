// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include "hip_custom_kernels.h"

// FP32 [1,2688] times row-major [2688,128]. No scratch is needed.
// Use sequential FMA order, with register prefetching.
extern "C" HIP_KERNEL_API int hip_sequential_router_f32(
    void *stream, const void *input, const void *weights, void *output,
    int n, int k);
