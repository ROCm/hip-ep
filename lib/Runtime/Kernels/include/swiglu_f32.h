// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include "hip_custom_kernels.h"

// Shapes have four axes, padded with leading ones. Storage is contiguous.
// Output may alias an input only when that input has the full output shape.
extern "C" HIP_KERNEL_API int
hip_swiglu_f16_f32_bcast(void *stream, const void *gate, const void *up,
                         void *output, const int64_t *gateShape,
                         const int64_t *upShape, const int64_t *outShape);
