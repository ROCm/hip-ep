/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//
// ONNX LayerNormalization (opset 17): subtract per-row mean, normalize by
// rms with epsilon, then scale by gamma and add beta.
//
//   y[i, j] = ((x[i, j] - mean[i]) / sqrt(var[i] + eps)) * gamma[j] + beta[j]
//
// Inputs are flattened by the host into:
//   x : (outer * norm_size,)  -- norm axes form a contiguous tail
//   gamma, beta : (norm_size,)
//
// One block per outer row.  Math in fp32; half/bf16 storage formats are
// promoted on load and narrowed on store.

#include "hip_custom_kernels.h"

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>

namespace {

// Two-pass LayerNorm for numerical stability (matches onnxruntime/GGUF):
//   Pass 1: compute mean via Kahan-compensated sum
//   Pass 2: compute variance as mean((x - mean)^2) via Kahan-compensated sum
//   Pass 3: normalize, scale, shift
// The single-pass E[X^2]-E[X]^2 formula suffers from catastrophic
// cancellation when values have a non-trivial mean. The two-pass approach
// avoids this entirely.
template <typename T, typename ToF, typename FromF>
__global__ void layer_norm_kernel(const T *__restrict__ x,
                                  const T *__restrict__ gamma,
                                  const T *__restrict__ beta,
                                  T *__restrict__ y, int64_t outer,
                                  int64_t norm_size, float epsilon,
                                  ToF to_float, FromF from_float) {
    extern __shared__ float smem[];
    float *sdata = smem;
    float *comp  = smem + blockDim.x;

    int64_t row = blockIdx.x;
    if (row >= outer)
        return;

    int tid = threadIdx.x;
    int stride = blockDim.x;

    const T *row_in = x + row * norm_size;
    T *row_out = y + row * norm_size;

    // --- Pass 1: compute mean ---
    float local_sum = 0.0f;
    float cs = 0.0f;
    for (int64_t i = tid; i < norm_size; i += stride) {
        float v = to_float(row_in[i]);
        float ys = v - cs;
        float ts = local_sum + ys;
        cs = (ts - local_sum) - ys;
        local_sum = ts;
    }
    sdata[tid] = local_sum;
    comp[tid] = cs;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (static_cast<unsigned int>(tid) < s) {
            float ys = sdata[tid + s] - (comp[tid] + comp[tid + s]);
            float ts = sdata[tid] + ys;
            comp[tid] = (ts - sdata[tid]) - ys;
            sdata[tid] = ts;
        }
        __syncthreads();
    }

    float mean = sdata[0] / static_cast<float>(norm_size);
    __syncthreads();

    // --- Pass 2: compute variance = mean((x - mean)^2) ---
    float local_var = 0.0f;
    float cv = 0.0f;
    for (int64_t i = tid; i < norm_size; i += stride) {
        float d = to_float(row_in[i]) - mean;
        float sq = d * d;
        float yv = sq - cv;
        float tv = local_var + yv;
        cv = (tv - local_var) - yv;
        local_var = tv;
    }
    sdata[tid] = local_var;
    comp[tid] = cv;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (static_cast<unsigned int>(tid) < s) {
            float yv = sdata[tid + s] - (comp[tid] + comp[tid + s]);
            float tv = sdata[tid] + yv;
            comp[tid] = (tv - sdata[tid]) - yv;
            sdata[tid] = tv;
        }
        __syncthreads();
    }

    float var = sdata[0] / static_cast<float>(norm_size);
    float inv_std = rsqrtf(var + epsilon);

    // --- Pass 3: normalize, scale, shift ---
    for (int64_t i = tid; i < norm_size; i += stride) {
        float v = to_float(row_in[i]);
        float n = (v - mean) * inv_std;
        float g = to_float(gamma[i]);
        float b = beta ? to_float(beta[i]) : 0.0f;
        row_out[i] = from_float(n * g + b);
    }
}

template <typename T, typename ToF, typename FromF>
int launch_layer_norm(hipStream_t stream, const void *x, const void *gamma,
                      const void *beta, void *y, int64_t outer,
                      int64_t norm_size, float epsilon, ToF to_float,
                      FromF from_float) {
    int block_size = 256;
    if (norm_size < 256) {
        block_size = 1;
        while (block_size < norm_size)
            block_size <<= 1;
        if (block_size < 1)
            block_size = 1;
    }
    size_t shared_mem = 2 * block_size * sizeof(float);
    hipLaunchKernelGGL((layer_norm_kernel<T, ToF, FromF>),
                       dim3(static_cast<unsigned>(outer)), dim3(block_size),
                       shared_mem, stream, static_cast<const T *>(x),
                       static_cast<const T *>(gamma),
                       static_cast<const T *>(beta), static_cast<T *>(y),
                       outer, norm_size, epsilon, to_float, from_float);
    return static_cast<int>(hipGetLastError());
}

} // namespace

extern "C" int hip_layer_norm(void *stream, const void *x, const void *gamma,
                              const void *beta, void *y, int64_t outer,
                              int64_t norm_size, float epsilon,
                              int hip_dtype) {
    if (outer <= 0 || norm_size <= 0)
        return 0;
    if (!x || !gamma || !y) {
        fprintf(stderr, "[custom_kernels] hip_layer_norm: null pointer\n");
        return -1;
    }

    hipStream_t hip_stream = static_cast<hipStream_t>(stream);

    int rc = -1;
    switch (hip_dtype) {
    case HIP_DTYPE_FLOAT32: {
        auto id = [] __device__(float v) { return v; };
        rc = launch_layer_norm<float, decltype(id), decltype(id)>(
            hip_stream, x, gamma, beta, y, outer, norm_size, epsilon, id, id);
        break;
    }
    case HIP_DTYPE_FLOAT16: {
        auto to_f = [] __device__(__half v) { return __half2float(v); };
        auto from_f = [] __device__(float v) { return __float2half(v); };
        rc = launch_layer_norm<__half, decltype(to_f), decltype(from_f)>(
            hip_stream, x, gamma, beta, y, outer, norm_size, epsilon, to_f,
            from_f);
        break;
    }
    case HIP_DTYPE_BFLOAT16: {
        auto to_f =
            [] __device__(__hip_bfloat16 v) { return __bfloat162float(v); };
        auto from_f = [] __device__(float v) { return __float2bfloat16(v); };
        rc = launch_layer_norm<__hip_bfloat16, decltype(to_f), decltype(from_f)>(
            hip_stream, x, gamma, beta, y, outer, norm_size, epsilon, to_f,
            from_f);
        break;
    }
    default:
        fprintf(stderr,
                "[custom_kernels] hip_layer_norm: unsupported dtype=%d\n",
                hip_dtype);
        return -1;
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "[custom_kernels] hip_layer_norm launch failed: %s\n",
                hipGetErrorString(err));
        return static_cast<int>(err);
    }
    return rc;
}
