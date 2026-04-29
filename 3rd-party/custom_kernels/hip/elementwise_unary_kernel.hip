/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//
// Generic element-wise unary kernel for the ONNX ops that don't have a
// hipDNN/MIOpen primitive worth using on Kokoro-sized tensors:
//
//   sin, cos, exp, tanh, floor, round, atan, leaky_relu(alpha), clip(min,max)
//
// All kernels promote half/bf16 inputs to f32 for the math, then narrow the
// result back to the original element type.  This matches the precision of
// the existing reciprocal/sqrt kernels and keeps us within IEEE semantics
// for the half formats.
//
// Dispatch is by hip_unary_kind_t; see hip_custom_kernels.h.

#include "hip_custom_kernels.h"

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>

namespace {

template <hip_unary_kind_t Kind>
__device__ __forceinline__ float apply(float x, float alpha, float beta) {
    if constexpr (Kind == HIP_UNARY_SIN) {
        float scale = (alpha == 0.0f) ? 1.0f : alpha;
        return sinf(x * scale);
    }
    else if constexpr (Kind == HIP_UNARY_COS)
        return cosf(x);
    else if constexpr (Kind == HIP_UNARY_EXP)
        return expf(x);
    else if constexpr (Kind == HIP_UNARY_TANH)
        return tanhf(x);
    else if constexpr (Kind == HIP_UNARY_FLOOR)
        return floorf(x);
    else if constexpr (Kind == HIP_UNARY_ROUND)
        // ONNX Round is round-half-to-even (banker's rounding).
        // rintf() with the default rounding mode gives that semantic.
        return rintf(x);
    else if constexpr (Kind == HIP_UNARY_ATAN)
        return atanf(x);
    else if constexpr (Kind == HIP_UNARY_LEAKY_RELU)
        return x >= 0.0f ? x : alpha * x;
    else if constexpr (Kind == HIP_UNARY_CLIP)
        return fminf(fmaxf(x, alpha), beta);
    else if constexpr (Kind == HIP_UNARY_SIGMOID)
        return 1.0f / (1.0f + expf(-x));
    else
        return x;
}

template <hip_unary_kind_t Kind>
__global__ void unary_f32_kernel(const float *__restrict__ input,
                                 float *__restrict__ output,
                                 int64_t num_elements, float alpha,
                                 float beta) {
    int64_t idx =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < num_elements)
        output[idx] = apply<Kind>(input[idx], alpha, beta);
}

template <hip_unary_kind_t Kind>
__global__ void unary_f16_kernel(const __half *__restrict__ input,
                                 __half *__restrict__ output,
                                 int64_t num_elements, float alpha,
                                 float beta) {
    int64_t idx =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < num_elements)
        output[idx] =
            __float2half(apply<Kind>(__half2float(input[idx]), alpha, beta));
}

template <hip_unary_kind_t Kind>
__global__ void unary_bf16_kernel(const __hip_bfloat16 *__restrict__ input,
                                  __hip_bfloat16 *__restrict__ output,
                                  int64_t num_elements, float alpha,
                                  float beta) {
    int64_t idx =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < num_elements)
        output[idx] = __float2bfloat16(
            apply<Kind>(__bfloat162float(input[idx]), alpha, beta));
}

template <hip_unary_kind_t Kind>
int launch_unary(hipStream_t stream, const void *input, void *output,
                 int64_t num_elements, int hip_dtype, float alpha, float beta) {
    constexpr int kBlockSize = 256;
    int grid_size =
        static_cast<int>((num_elements + kBlockSize - 1) / kBlockSize);
    switch (hip_dtype) {
    case HIP_DTYPE_FLOAT32:
        hipLaunchKernelGGL(unary_f32_kernel<Kind>, dim3(grid_size),
                           dim3(kBlockSize), 0, stream,
                           static_cast<const float *>(input),
                           static_cast<float *>(output), num_elements, alpha,
                           beta);
        break;
    case HIP_DTYPE_FLOAT16:
        hipLaunchKernelGGL(unary_f16_kernel<Kind>, dim3(grid_size),
                           dim3(kBlockSize), 0, stream,
                           static_cast<const __half *>(input),
                           static_cast<__half *>(output), num_elements, alpha,
                           beta);
        break;
    case HIP_DTYPE_BFLOAT16:
        hipLaunchKernelGGL(unary_bf16_kernel<Kind>, dim3(grid_size),
                           dim3(kBlockSize), 0, stream,
                           static_cast<const __hip_bfloat16 *>(input),
                           static_cast<__hip_bfloat16 *>(output), num_elements,
                           alpha, beta);
        break;
    default:
        fprintf(stderr,
                "[custom_kernels] hip_elementwise_unary: unsupported dtype=%d\n",
                hip_dtype);
        return -1;
    }
    return static_cast<int>(hipGetLastError());
}

} // namespace

extern "C" int hip_elementwise_unary(void *stream, const void *input,
                                     void *output, int64_t num_elements,
                                     int hip_dtype, int kind, float alpha,
                                     float beta) {
    if (num_elements <= 0)
        return 0;
    if (!input || !output) {
        fprintf(stderr, "[custom_kernels] hip_elementwise_unary: null ptr\n");
        return -1;
    }

    hipStream_t hip_stream = static_cast<hipStream_t>(stream);


    int rc = -1;
    switch (kind) {
    case HIP_UNARY_SIN:
        rc = launch_unary<HIP_UNARY_SIN>(hip_stream, input, output,
                                         num_elements, hip_dtype, alpha, beta);
        break;
    case HIP_UNARY_COS:
        rc = launch_unary<HIP_UNARY_COS>(hip_stream, input, output,
                                         num_elements, hip_dtype, alpha, beta);
        break;
    case HIP_UNARY_EXP:
        rc = launch_unary<HIP_UNARY_EXP>(hip_stream, input, output,
                                         num_elements, hip_dtype, alpha, beta);
        break;
    case HIP_UNARY_TANH:
        rc = launch_unary<HIP_UNARY_TANH>(hip_stream, input, output,
                                          num_elements, hip_dtype, alpha, beta);
        break;
    case HIP_UNARY_FLOOR:
        rc = launch_unary<HIP_UNARY_FLOOR>(hip_stream, input, output,
                                           num_elements, hip_dtype, alpha,
                                           beta);
        break;
    case HIP_UNARY_ROUND:
        rc = launch_unary<HIP_UNARY_ROUND>(hip_stream, input, output,
                                           num_elements, hip_dtype, alpha,
                                           beta);
        break;
    case HIP_UNARY_ATAN:
        rc = launch_unary<HIP_UNARY_ATAN>(hip_stream, input, output,
                                          num_elements, hip_dtype, alpha, beta);
        break;
    case HIP_UNARY_LEAKY_RELU:
        rc = launch_unary<HIP_UNARY_LEAKY_RELU>(hip_stream, input, output,
                                                num_elements, hip_dtype, alpha,
                                                beta);
        break;
    case HIP_UNARY_CLIP:
        rc = launch_unary<HIP_UNARY_CLIP>(hip_stream, input, output,
                                          num_elements, hip_dtype, alpha, beta);
        break;
    case HIP_UNARY_SIGMOID:
        rc = launch_unary<HIP_UNARY_SIGMOID>(hip_stream, input, output,
                                             num_elements, hip_dtype, alpha,
                                             beta);
        break;
    default:
        fprintf(stderr,
                "[custom_kernels] hip_elementwise_unary: unsupported kind=%d\n",
                kind);
        return -1;
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr,
                "[custom_kernels] hip_elementwise_unary launch failed: %s\n",
                hipGetErrorString(err));
        return static_cast<int>(err);
    }
    return rc;
}
