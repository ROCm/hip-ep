/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//
// Generic ReduceMean kernel for fp16/bf16/fp32 inputs.  The host-side wrapper
// flattens the input to (num_output_elements * reduce_size) and asks the
// kernel to compute the mean of every reduce_size-element slab.  All math is
// done in fp32; half/bf16 inputs are promoted then narrowed back, matching
// the precision of the existing reciprocal/sqrt kernels.

#include "debug_log.h"
#include "hip_custom_kernels.h"

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>

namespace {

template <typename T, typename ToF, typename FromF>
__global__ void reduce_mean_kernel(const T *__restrict__ data,
                                   T *__restrict__ output,
                                   int64_t reduce_size,
                                   int64_t num_output_elements, ToF to_float,
                                   FromF from_float) {
    extern __shared__ float smem[];

    int64_t out_idx = blockIdx.x;
    if (out_idx >= num_output_elements)
        return;

    const T *slice = data + out_idx * reduce_size;
    int tid = threadIdx.x;
    int stride = blockDim.x;

    float partial = 0.0f;
    for (int64_t i = tid; i < reduce_size; i += stride)
        partial += to_float(slice[i]);

    smem[tid] = partial;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (static_cast<unsigned int>(tid) < s)
            smem[tid] += smem[tid + s];
        __syncthreads();
    }

    if (tid == 0)
        output[out_idx] =
            from_float(smem[0] / static_cast<float>(reduce_size));
}

} // namespace

extern "C" int hip_reduce_mean(void *stream, const void *data, void *output,
                               int64_t num_input_elements,
                               int64_t num_output_elements, int hip_dtype) {
    if (num_output_elements <= 0)
        return 0;
    if (num_input_elements % num_output_elements != 0) {
        fprintf(stderr,
                "[custom_kernels] hip_reduce_mean: input(%lld) not divisible "
                "by output(%lld)\n",
                (long long)num_input_elements, (long long)num_output_elements);
        return -1;
    }

    int64_t reduce_size = num_input_elements / num_output_elements;
    hipStream_t hip_stream = static_cast<hipStream_t>(stream);

    int block_size = 256;
    if (reduce_size < 256) {
        block_size = 1;
        while (block_size < reduce_size)
            block_size <<= 1;
        if (block_size < 1)
            block_size = 1;
    }
    size_t shared_mem = block_size * sizeof(float);

    CUSTOM_KERNELS_DEBUG_LOG(
        "[custom_kernels] hip_reduce_mean: dtype=%d, in=%lld, out=%lld, "
        "reduce=%lld, block=%d, smem=%zu\n",
        hip_dtype, (long long)num_input_elements,
        (long long)num_output_elements, (long long)reduce_size, block_size,
        shared_mem);

    switch (hip_dtype) {
    case HIP_DTYPE_FLOAT32: {
        auto id = [] __device__(float v) { return v; };
        hipLaunchKernelGGL(
            (reduce_mean_kernel<float, decltype(id), decltype(id)>),
            dim3(static_cast<unsigned>(num_output_elements)),
            dim3(block_size), shared_mem, hip_stream,
            static_cast<const float *>(data), static_cast<float *>(output),
            reduce_size, num_output_elements, id, id);
        break;
    }
    case HIP_DTYPE_FLOAT16: {
        auto to_f = [] __device__(__half v) { return __half2float(v); };
        auto from_f = [] __device__(float v) { return __float2half(v); };
        hipLaunchKernelGGL(
            (reduce_mean_kernel<__half, decltype(to_f), decltype(from_f)>),
            dim3(static_cast<unsigned>(num_output_elements)),
            dim3(block_size), shared_mem, hip_stream,
            static_cast<const __half *>(data), static_cast<__half *>(output),
            reduce_size, num_output_elements, to_f, from_f);
        break;
    }
    case HIP_DTYPE_BFLOAT16: {
        auto to_f =
            [] __device__(__hip_bfloat16 v) { return __bfloat162float(v); };
        auto from_f = [] __device__(float v) { return __float2bfloat16(v); };
        hipLaunchKernelGGL((reduce_mean_kernel<__hip_bfloat16, decltype(to_f),
                                               decltype(from_f)>),
                           dim3(static_cast<unsigned>(num_output_elements)),
                           dim3(block_size), shared_mem, hip_stream,
                           static_cast<const __hip_bfloat16 *>(data),
                           static_cast<__hip_bfloat16 *>(output), reduce_size,
                           num_output_elements, to_f, from_f);
        break;
    }
    default:
        fprintf(stderr,
                "[custom_kernels] hip_reduce_mean: unsupported dtype=%d\n",
                hip_dtype);
        return -1;
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr,
                "[custom_kernels] hip_reduce_mean launch failed: %s\n",
                hipGetErrorString(err));
        return static_cast<int>(err);
    }
    return 0;
}
