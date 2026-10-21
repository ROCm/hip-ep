// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include "hip_custom_kernels.h"

// Shapes have four axes, padded with leading ones. Storage is contiguous.
// FP32 inputs must not alias the FP16 destination.
extern "C" HIP_KERNEL_API int hip_rounded_scaled_add_f16(
    void *stream, const void *base, const void *x, const void *y, void *output,
    const int64_t *baseShape, const int64_t *xShape, const int64_t *yShape,
    const int64_t *outShape, int64_t productOnLhs);
