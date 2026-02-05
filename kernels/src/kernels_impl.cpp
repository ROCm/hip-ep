// Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// This file provides C++ implementations that use HIP runtime API + HIPRTC
// On Windows with MSVC, we cannot directly compile .hip files
// Instead, we use hiprtc (runtime compilation)

#include "rocm_kernels.h"
#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <mutex>
#include <iostream>

//============================================================================
// Kernel source code (from .hip files, adapted for HIPRTC)
//============================================================================

// Simple transpose_0213 kernel: [N, A, B, C] -> [N, B, A, C]  
static const char* transpose_0213_src = R"(
extern "C" __global__ void transpose_0213_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    long long n, long long a, long long b, long long c)
{
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long total = n * a * b * c;
    
    if (idx >= total) return;
    
    // Decode output index [n_i, b_i, a_i, c_i]
    long long c_i = idx % c;
    long long temp = idx / c;
    long long a_i = temp % a;
    temp = temp / a;
    long long b_i = temp % b;
    long long n_i = temp / b;
    
    // Input index for [n_i, a_i, b_i, c_i]
    long long in_idx = n_i * (a * b * c) + a_i * (b * c) + b_i * c + c_i;
    
    output[idx] = input[in_idx];
}
)";

// General 4D transpose kernel
static const char* transpose_4d_src = R"(
extern "C" __global__ void transpose_4d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    long long d0, long long d1, long long d2, long long d3,
    int p0, int p1, int p2, int p3)
{
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long total = d0 * d1 * d2 * d3;
    
    if (idx >= total) return;
    
    // Decompose input linear index to [i0, i1, i2, i3]
    long long i3 = idx % d3;
    long long tmp = idx / d3;
    long long i2 = tmp % d2;
    tmp = tmp / d2;
    long long i1 = tmp % d1;
    long long i0 = tmp / d1;
    
    // Input indices array
    long long in_idx[4] = {i0, i1, i2, i3};
    long long in_dims[4] = {d0, d1, d2, d3};
    
    // Compute output dimensions and indices
    long long out_dims[4] = {in_dims[p0], in_dims[p1], in_dims[p2], in_dims[p3]};
    long long out_idx[4] = {in_idx[p0], in_idx[p1], in_idx[p2], in_idx[p3]};
    
    // Compute linear output index
    long long out_linear = out_idx[0] * (out_dims[1] * out_dims[2] * out_dims[3]) +
                           out_idx[1] * (out_dims[2] * out_dims[3]) +
                           out_idx[2] * out_dims[3] +
                           out_idx[3];
    
    output[out_linear] = input[idx];
}
)";

// Tile kernel for 4D tensors
static const char* tile_4d_src = R"(
extern "C" __global__ void tile_4d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    long long in_d0, long long in_d1, long long in_d2, long long in_d3,
    long long r0, long long r1, long long r2, long long r3,
    long long out_size)
{
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= out_size) return;
    
    long long out_d0 = in_d0 * r0;
    long long out_d1 = in_d1 * r1;
    long long out_d2 = in_d2 * r2;
    long long out_d3 = in_d3 * r3;
    
    // Decompose output index
    long long o3 = idx % out_d3;
    long long tmp = idx / out_d3;
    long long o2 = tmp % out_d2;
    tmp = tmp / out_d2;
    long long o1 = tmp % out_d1;
    long long o0 = tmp / out_d1;
    
    // Map to input index (wrap around for tiling)
    long long i0 = o0 % in_d0;
    long long i1 = o1 % in_d1;
    long long i2 = o2 % in_d2;
    long long i3 = o3 % in_d3;
    
    long long in_idx = i0 * (in_d1 * in_d2 * in_d3) + i1 * (in_d2 * in_d3) + i2 * in_d3 + i3;
    output[idx] = input[in_idx];
}
)";

// Tile kernel for 5D tensors (needed for GQA attention)
static const char* tile_5d_src = R"(
extern "C" __global__ void tile_5d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    long long in_d0, long long in_d1, long long in_d2, long long in_d3, long long in_d4,
    long long r0, long long r1, long long r2, long long r3, long long r4,
    long long out_size)
{
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= out_size) return;
    
    long long out_d0 = in_d0 * r0;
    long long out_d1 = in_d1 * r1;
    long long out_d2 = in_d2 * r2;
    long long out_d3 = in_d3 * r3;
    long long out_d4 = in_d4 * r4;
    
    // Decompose output index
    long long o4 = idx % out_d4;
    long long tmp = idx / out_d4;
    long long o3 = tmp % out_d3;
    tmp = tmp / out_d3;
    long long o2 = tmp % out_d2;
    tmp = tmp / out_d2;
    long long o1 = tmp % out_d1;
    long long o0 = tmp / out_d1;
    
    // Map to input index (wrap around for tiling)
    long long i0 = o0 % in_d0;
    long long i1 = o1 % in_d1;
    long long i2 = o2 % in_d2;
    long long i3 = o3 % in_d3;
    long long i4 = o4 % in_d4;
    
    long long in_idx = i0 * (in_d1 * in_d2 * in_d3 * in_d4) + 
                       i1 * (in_d2 * in_d3 * in_d4) + 
                       i2 * (in_d3 * in_d4) + 
                       i3 * in_d4 + i4;
    output[idx] = input[in_idx];
}
)";

// Mul kernels
static const char* mul_src = R"(
extern "C" __global__ void mul_elementwise_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ output,
    long long total_size, long long b_size)
{
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;
    
    // Support broadcasting: b_size can be smaller than total_size
    long long b_idx = idx % b_size;
    output[idx] = a[idx] * b[b_idx];
}

extern "C" __global__ void mul_scalar_kernel(
    const float* __restrict__ a,
    float scalar,
    float* __restrict__ output,
    long long size)
{
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    output[idx] = a[idx] * scalar;
}
)";

// Add bias kernel for Conv (NCHW format)
static const char* add_bias_src = R"(
extern "C" __global__ void add_bias_nchw_kernel(
    float* data,
    const float* __restrict__ bias,
    long long channels, long long spatial_size, long long total_size)
{
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;
    
    // For NCHW: idx = n * (C * H * W) + c * (H * W) + hw
    // We need to extract c from idx
    long long chw = channels * spatial_size;
    long long c = (idx % chw) / spatial_size;
    
    data[idx] += bias[c];
}
)";

// Softmax kernel (numerically stable, single block per batch)
static const char* softmax_src = R"(
extern "C" __global__ void softmax_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    long long batch, long long dim)
{
    long long batch_idx = blockIdx.x;
    if (batch_idx >= batch) return;
    
    const float* in_row = input + batch_idx * dim;
    float* out_row = output + batch_idx * dim;
    
    // Find max for numerical stability
    float max_val = in_row[0];
    for (long long i = 1; i < dim; ++i) {
        if (in_row[i] > max_val) max_val = in_row[i];
    }
    
    // Compute exp(x - max) and sum
    float sum = 0.0f;
    for (long long i = 0; i < dim; ++i) {
        out_row[i] = expf(in_row[i] - max_val);
        sum += out_row[i];
    }
    
    // Normalize
    float inv_sum = 1.0f / sum;
    for (long long i = 0; i < dim; ++i) {
        out_row[i] *= inv_sum;
    }
}
)";

//============================================================================
// Kernel cache and compilation helpers
//============================================================================
static std::mutex kernel_cache_mutex;

struct KernelCache {
    hipModule_t module = nullptr;
    hipFunction_t func = nullptr;
};

static KernelCache transpose_0213_cache;
static KernelCache transpose_4d_cache;
static KernelCache tile_4d_cache;
static KernelCache tile_5d_cache;
static KernelCache mul_elementwise_cache;
static KernelCache mul_scalar_cache;
static KernelCache softmax_cache;
static KernelCache add_bias_cache;

static bool compile_kernel(const char* src, const char* kernel_name, KernelCache& cache) {
    std::lock_guard<std::mutex> lock(kernel_cache_mutex);
    
    if (cache.func != nullptr) return true;
    
    hiprtcProgram prog;
    if (hiprtcCreateProgram(&prog, src, "kernel.hip", 0, nullptr, nullptr) != HIPRTC_SUCCESS) {
        std::cerr << "HIPRTC: Failed to create program for " << kernel_name << std::endl;
        return false;
    }
    
    const char* options[] = {"--std=c++14"};
    hiprtcResult compileResult = hiprtcCompileProgram(prog, 1, options);
    
    if (compileResult != HIPRTC_SUCCESS) {
        size_t logSize;
        hiprtcGetProgramLogSize(prog, &logSize);
        std::string log(logSize, '\0');
        hiprtcGetProgramLog(prog, &log[0]);
        std::cerr << "HIPRTC compile error for " << kernel_name << ": " << log << std::endl;
        hiprtcDestroyProgram(&prog);
        return false;
    }
    
    size_t codeSize;
    hiprtcGetCodeSize(prog, &codeSize);
    std::string code(codeSize, '\0');
    hiprtcGetCode(prog, &code[0]);
    hiprtcDestroyProgram(&prog);
    
    if (hipModuleLoadData(&cache.module, code.data()) != hipSuccess) {
        std::cerr << "HIP: Failed to load module for " << kernel_name << std::endl;
        return false;
    }
    if (hipModuleGetFunction(&cache.func, cache.module, kernel_name) != hipSuccess) {
        std::cerr << "HIP: Failed to get function " << kernel_name << std::endl;
        return false;
    }
    
    std::cerr << "HIPRTC: Successfully compiled " << kernel_name << std::endl;
    return true;
}

namespace rocm_kernels {

//============================================================================
// Add Bias Implementation (for Conv)
//============================================================================

void add_bias_nchw(float* data, const float* bias,
                   int64_t batch, int64_t channels, int64_t spatial_size,
                   hipStream_t stream) {
    if (!compile_kernel(add_bias_src, "add_bias_nchw_kernel", add_bias_cache)) {
        // Fallback: do nothing (bias not added)
        return;
    }
    
    int64_t total_size = batch * channels * spatial_size;
    int blockSize = 256;
    int gridSize = (total_size + blockSize - 1) / blockSize;
    
    void* args[] = {(void*)&data, (void*)&bias, (void*)&channels, (void*)&spatial_size, (void*)&total_size};
    hipModuleLaunchKernel(add_bias_cache.func, gridSize, 1, 1, blockSize, 1, 1,
                          0, stream, args, nullptr);
}

//============================================================================
// Mul Implementation
//============================================================================

void mul_elementwise(const float* a, const float* b, float* output,
                     int64_t total_size, int64_t b_size, hipStream_t stream) {
    if (!compile_kernel(mul_src, "mul_elementwise_kernel", mul_elementwise_cache)) {
        // Fallback: copy a to output
        hipMemcpyAsync(output, a, total_size * sizeof(float), hipMemcpyDeviceToDevice, stream);
        return;
    }
    
    int blockSize = 256;
    int gridSize = (total_size + blockSize - 1) / blockSize;
    
    void* args[] = {(void*)&a, (void*)&b, (void*)&output, (void*)&total_size, (void*)&b_size};
    hipModuleLaunchKernel(mul_elementwise_cache.func, gridSize, 1, 1, blockSize, 1, 1, 
                          0, stream, args, nullptr);
}

void mul_scalar(const float* a, float scalar, float* output,
                int64_t size, hipStream_t stream) {
    if (!compile_kernel(mul_src, "mul_scalar_kernel", mul_scalar_cache)) {
        hipMemcpyAsync(output, a, size * sizeof(float), hipMemcpyDeviceToDevice, stream);
        return;
    }
    
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    
    void* args[] = {(void*)&a, (void*)&scalar, (void*)&output, (void*)&size};
    hipModuleLaunchKernel(mul_scalar_cache.func, gridSize, 1, 1, blockSize, 1, 1,
                          0, stream, args, nullptr);
}

//============================================================================
// Softmax Implementation
//============================================================================

void softmax(const float* input, float* output,
             int64_t batch, int64_t dim, hipStream_t stream) {
    if (!compile_kernel(softmax_src, "softmax_kernel", softmax_cache)) {
        hipMemcpyAsync(output, input, batch * dim * sizeof(float), hipMemcpyDeviceToDevice, stream);
        return;
    }
    
    // One block per batch element
    void* args[] = {(void*)&input, (void*)&output, (void*)&batch, (void*)&dim};
    hipModuleLaunchKernel(softmax_cache.func, batch, 1, 1, 1, 1, 1,
                          0, stream, args, nullptr);
}

//============================================================================
// Reshape Implementation (just a memory copy)
//============================================================================

void reshape_copy(const float* input, float* output,
                  int64_t size, hipStream_t stream) {
    if (input != output) {
        hipMemcpyAsync(output, input, size * sizeof(float), hipMemcpyDeviceToDevice, stream);
    }
}

//============================================================================
// Transpose Implementation
//============================================================================

void transpose(const float* input, float* output,
               const int64_t* in_shape, const int64_t* out_shape,
               const int32_t* perm, int32_t ndim,
               int64_t total_size, hipStream_t stream) {
    if (ndim == 4) {
        if (!compile_kernel(transpose_4d_src, "transpose_4d_kernel", transpose_4d_cache)) {
            hipMemcpyAsync(output, input, total_size * sizeof(float), hipMemcpyDeviceToDevice, stream);
            return;
        }
        
        int64_t d0 = in_shape[0], d1 = in_shape[1], d2 = in_shape[2], d3 = in_shape[3];
        int p0 = perm[0], p1 = perm[1], p2 = perm[2], p3 = perm[3];
        
        int blockSize = 256;
        int gridSize = (total_size + blockSize - 1) / blockSize;
        
        void* args[] = {(void*)&input, (void*)&output, 
                        (void*)&d0, (void*)&d1, (void*)&d2, (void*)&d3,
                        (void*)&p0, (void*)&p1, (void*)&p2, (void*)&p3};
        hipModuleLaunchKernel(transpose_4d_cache.func, gridSize, 1, 1, blockSize, 1, 1,
                              0, stream, args, nullptr);
    } else {
        // Fallback for non-4D
        hipMemcpyAsync(output, input, total_size * sizeof(float), hipMemcpyDeviceToDevice, stream);
    }
}

void transpose_0213(const float* input, float* output,
                    int64_t n, int64_t a, int64_t b, int64_t c,
                    hipStream_t stream) {
    if (!compile_kernel(transpose_0213_src, "transpose_0213_kernel", transpose_0213_cache)) {
        int64_t total = n * a * b * c;
        hipMemcpyAsync(output, input, total * sizeof(float), hipMemcpyDeviceToDevice, stream);
        return;
    }
    
    int64_t total = n * a * b * c;
    int blockSize = 256;
    int gridSize = (total + blockSize - 1) / blockSize;
    
    void* args[] = {(void*)&input, (void*)&output, (void*)&n, (void*)&a, (void*)&b, (void*)&c};
    hipModuleLaunchKernel(transpose_0213_cache.func, gridSize, 1, 1, blockSize, 1, 1,
                          0, stream, args, nullptr);
}

//============================================================================
// Tile Implementation
//============================================================================

void tile(const float* input, float* output,
          const int64_t* in_shape, const int64_t* repeats,
          int32_t ndim, int64_t in_size, int64_t out_size,
          hipStream_t stream) {
    if (in_size == out_size) {
        // No tiling needed
        hipMemcpyAsync(output, input, in_size * sizeof(float), hipMemcpyDeviceToDevice, stream);
        return;
    }
    
    int blockSize = 256;
    int gridSize = (out_size + blockSize - 1) / blockSize;
    
    if (ndim == 4) {
        if (!compile_kernel(tile_4d_src, "tile_4d_kernel", tile_4d_cache)) {
            hipMemsetAsync(output, 0, out_size * sizeof(float), stream);
            return;
        }
        
        int64_t d0 = in_shape[0], d1 = in_shape[1], d2 = in_shape[2], d3 = in_shape[3];
        int64_t r0 = repeats[0], r1 = repeats[1], r2 = repeats[2], r3 = repeats[3];
        
        void* args[] = {(void*)&input, (void*)&output,
                        (void*)&d0, (void*)&d1, (void*)&d2, (void*)&d3,
                        (void*)&r0, (void*)&r1, (void*)&r2, (void*)&r3,
                        (void*)&out_size};
        hipModuleLaunchKernel(tile_4d_cache.func, gridSize, 1, 1, blockSize, 1, 1,
                              0, stream, args, nullptr);
    } else if (ndim == 5) {
        if (!compile_kernel(tile_5d_src, "tile_5d_kernel", tile_5d_cache)) {
            hipMemsetAsync(output, 0, out_size * sizeof(float), stream);
            return;
        }
        
        int64_t d0 = in_shape[0], d1 = in_shape[1], d2 = in_shape[2], d3 = in_shape[3], d4 = in_shape[4];
        int64_t r0 = repeats[0], r1 = repeats[1], r2 = repeats[2], r3 = repeats[3], r4 = repeats[4];
        
        void* args[] = {(void*)&input, (void*)&output,
                        (void*)&d0, (void*)&d1, (void*)&d2, (void*)&d3, (void*)&d4,
                        (void*)&r0, (void*)&r1, (void*)&r2, (void*)&r3, (void*)&r4,
                        (void*)&out_size};
        hipModuleLaunchKernel(tile_5d_cache.func, gridSize, 1, 1, blockSize, 1, 1,
                              0, stream, args, nullptr);
    } else {
        // Fallback for other dimensions: just zero the output
        std::cerr << "Tile: unsupported ndim=" << ndim << ", falling back to zero" << std::endl;
        hipMemsetAsync(output, 0, out_size * sizeof(float), stream);
    }
}

} // namespace rocm_kernels
