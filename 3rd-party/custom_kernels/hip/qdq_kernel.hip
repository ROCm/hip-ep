/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//
// Standard ONNX QuantizeLinear / DequantizeLinear kernels.
//
// Supports per-tensor and per-axis scale/zero-point for int8/uint8.  Math is
// performed in fp32, matching ONNX Q/DQ semantics rather than GGUF block
// quantization.

#include "debug_log.h"
#include "hip_custom_kernels.h"

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>

namespace {

__device__ __forceinline__ float read_real(const void *ptr, int dtype,
                                           int64_t idx) {
    if (dtype == HIP_DTYPE_FLOAT32)
        return static_cast<const float *>(ptr)[idx];
    if (dtype == HIP_DTYPE_FLOAT16)
        return __half2float(static_cast<const __half *>(ptr)[idx]);
    if (dtype == HIP_DTYPE_BFLOAT16)
        return __bfloat162float(static_cast<const __hip_bfloat16 *>(ptr)[idx]);
    return 0.0f;
}

__device__ __forceinline__ void write_real(void *ptr, int dtype, int64_t idx,
                                           float v) {
    if (dtype == HIP_DTYPE_FLOAT32)
        static_cast<float *>(ptr)[idx] = v;
    else if (dtype == HIP_DTYPE_FLOAT16)
        static_cast<__half *>(ptr)[idx] = __float2half(v);
    else if (dtype == HIP_DTYPE_BFLOAT16)
        static_cast<__hip_bfloat16 *>(ptr)[idx] = __float2bfloat16(v);
}

__device__ __forceinline__ int read_quant(const void *ptr, int dtype,
                                          int64_t idx, int default_zero) {
    if (!ptr)
        return default_zero;
    if (dtype == HIP_DTYPE_UINT8)
        return static_cast<int>(static_cast<const uint8_t *>(ptr)[idx]);
    if (dtype == HIP_DTYPE_INT8)
        return static_cast<int>(static_cast<const int8_t *>(ptr)[idx]);
    return default_zero;
}

__device__ __forceinline__ void write_quant(void *ptr, int dtype, int64_t idx,
                                            int v) {
    if (dtype == HIP_DTYPE_UINT8) {
        v = max(0, min(255, v));
        static_cast<uint8_t *>(ptr)[idx] = static_cast<uint8_t>(v);
    } else if (dtype == HIP_DTYPE_INT8) {
        v = max(-128, min(127, v));
        static_cast<int8_t *>(ptr)[idx] = static_cast<int8_t>(v);
    }
}

__device__ __forceinline__ int64_t scale_index(int64_t flat,
                                               int64_t scale_num_elements,
                                               int64_t inner_size) {
    if (scale_num_elements <= 1)
        return 0;
    return (flat / inner_size) % scale_num_elements;
}

__global__ void dequantize_linear_kernel(
    const void *__restrict__ input, const void *__restrict__ scale,
    const void *__restrict__ zero_point, void *__restrict__ output,
    int64_t num_elements, int input_dtype, int scale_dtype, int output_dtype,
    int64_t scale_num_elements, int64_t inner_size) {
    int64_t idx =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= num_elements)
        return;
    int64_t qidx = scale_index(idx, scale_num_elements, inner_size);
    int zp = read_quant(zero_point, input_dtype, qidx, 0);
    int x = read_quant(input, input_dtype, idx, 0);
    float s = read_real(scale, scale_dtype, qidx);
    write_real(output, output_dtype, idx, (static_cast<float>(x - zp) * s));
}

__global__ void quantize_linear_kernel(
    const void *__restrict__ input, const void *__restrict__ scale,
    const void *__restrict__ zero_point, void *__restrict__ output,
    int64_t num_elements, int input_dtype, int scale_dtype, int output_dtype,
    int64_t scale_num_elements, int64_t inner_size) {
    int64_t idx =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= num_elements)
        return;
    int64_t qidx = scale_index(idx, scale_num_elements, inner_size);
    int zp = read_quant(zero_point, output_dtype, qidx, 0);
    float x = read_real(input, input_dtype, idx);
    float s = read_real(scale, scale_dtype, qidx);
    int q = static_cast<int>(nearbyintf(x / s)) + zp;
    write_quant(output, output_dtype, idx, q);
}

static int launch_1d(hipStream_t stream, void (*)(void), const void *input,
                     const void *scale, const void *zero_point, void *output,
                     int64_t num_elements, int input_dtype, int scale_dtype,
                     int output_dtype, int64_t scale_num_elements,
                     int64_t inner_size, bool quantize) {
    constexpr int kBlockSize = 256;
    int grid = static_cast<int>((num_elements + kBlockSize - 1) / kBlockSize);
    if (grid <= 0)
        return 0;
    if (quantize) {
        hipLaunchKernelGGL(quantize_linear_kernel, dim3(grid),
                           dim3(kBlockSize), 0, stream, input, scale,
                           zero_point, output, num_elements, input_dtype,
                           scale_dtype, output_dtype, scale_num_elements,
                           inner_size);
    } else {
        hipLaunchKernelGGL(dequantize_linear_kernel, dim3(grid),
                           dim3(kBlockSize), 0, stream, input, scale,
                           zero_point, output, num_elements, input_dtype,
                           scale_dtype, output_dtype, scale_num_elements,
                           inner_size);
    }
    return static_cast<int>(hipGetLastError());
}

static bool valid_quant_dtype(int dtype) {
    return dtype == HIP_DTYPE_INT8 || dtype == HIP_DTYPE_UINT8;
}

static bool valid_real_dtype(int dtype) {
    return dtype == HIP_DTYPE_FLOAT32 || dtype == HIP_DTYPE_FLOAT16 ||
           dtype == HIP_DTYPE_BFLOAT16;
}

} // namespace

extern "C" int hip_dequantize_linear(
    void *stream, const void *input, const void *scale, const void *zero_point,
    void *output, int64_t num_elements, int input_dtype, int scale_dtype,
    int output_dtype, int64_t scale_num_elements, int64_t inner_size) {
    if (num_elements <= 0)
        return 0;
    if (!input || !scale || !output) {
        fprintf(stderr, "[custom_kernels] hip_dequantize_linear: null ptr\n");
        return -1;
    }
    if (!valid_quant_dtype(input_dtype) || !valid_real_dtype(scale_dtype) ||
        !valid_real_dtype(output_dtype)) {
        fprintf(stderr,
                "[custom_kernels] hip_dequantize_linear: bad dtype in=%d "
                "scale=%d out=%d\n",
                input_dtype, scale_dtype, output_dtype);
        return -1;
    }
    CUSTOM_KERNELS_DEBUG_LOG(
        "[custom_kernels] hip_dequantize_linear: n=%lld in=%d scale=%d "
        "out=%d scale_n=%lld inner=%lld\n",
        (long long)num_elements, input_dtype, scale_dtype, output_dtype,
        (long long)scale_num_elements, (long long)inner_size);
    return launch_1d(static_cast<hipStream_t>(stream), nullptr, input, scale,
                     zero_point, output, num_elements, input_dtype, scale_dtype,
                     output_dtype, scale_num_elements, inner_size, false);
}

extern "C" int hip_quantize_linear(
    void *stream, const void *input, const void *scale, const void *zero_point,
    void *output, int64_t num_elements, int input_dtype, int scale_dtype,
    int output_dtype, int64_t scale_num_elements, int64_t inner_size) {
    if (num_elements <= 0)
        return 0;
    if (!input || !scale || !output) {
        fprintf(stderr, "[custom_kernels] hip_quantize_linear: null ptr\n");
        return -1;
    }
    if (!valid_real_dtype(input_dtype) || !valid_real_dtype(scale_dtype) ||
        !valid_quant_dtype(output_dtype)) {
        fprintf(stderr,
                "[custom_kernels] hip_quantize_linear: bad dtype in=%d "
                "scale=%d out=%d\n",
                input_dtype, scale_dtype, output_dtype);
        return -1;
    }
    CUSTOM_KERNELS_DEBUG_LOG(
        "[custom_kernels] hip_quantize_linear: n=%lld in=%d scale=%d out=%d "
        "scale_n=%lld inner=%lld\n",
        (long long)num_elements, input_dtype, scale_dtype, output_dtype,
        (long long)scale_num_elements, (long long)inner_size);
    return launch_1d(static_cast<hipStream_t>(stream), nullptr, input, scale,
                     zero_point, output, num_elements, input_dtype, scale_dtype,
                     output_dtype, scale_num_elements, inner_size, true);
}
