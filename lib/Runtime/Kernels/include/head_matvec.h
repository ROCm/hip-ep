// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include "hip_custom_kernels.h"

// Experimental FP16 [1,2688] times [2688,131072]. Scratch holds 4*N floats.
// FP32 accumulation uses a different reduction order from BLAS.
extern "C" HIP_KERNEL_API int hip_head_matvec_f16(
    void *stream, const void *input, const void *weights, void *output,
    void *scratch, int n, int k);
