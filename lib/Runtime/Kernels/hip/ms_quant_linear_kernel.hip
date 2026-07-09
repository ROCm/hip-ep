/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// HIP kernels for com.microsoft.DequantizeLinear / QuantizeLinear.
// These handle activation quantization in W2A8 models (e.g. ORCA 2-bit).
//
// Layout: per-tensor scale/zp (n_channels == 1) is the common case for
// activation quantization. Per-channel is handled by indexing
// scale[element_index / (n_elements / n_channels)].

#include "hip_custom_kernels.h"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cmath>

/* -----------------------------------------------------------------------
 * DequantizeLinear kernel
 * out[i] = (float(in[i]) - float(zp)) * scale
 * Supports per-tensor (n_channels=1) and per-channel scale/zp.
 * ----------------------------------------------------------------------- */

__global__ void ms_dequant_i8_to_f16_kernel(
    const int8_t*  __restrict__ in,
    const __half*  __restrict__ scale,
    const int8_t*  __restrict__ zp,
    __half*        __restrict__ out,
    int64_t n_elements,
    int64_t n_channels)
{
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n_elements) return;

    // Per-tensor: ch_idx = 0; per-channel: ch_idx = i / stride
    int64_t stride   = (n_channels > 1) ? (n_elements / n_channels) : n_elements;
    int64_t ch_idx   = (n_channels > 1) ? (i / stride) : 0;
    float   sc       = static_cast<float>(scale[ch_idx]);
    float   zp_val   = zp ? static_cast<float>(zp[ch_idx]) : 0.0f;
    float   q        = static_cast<float>(in[i]);
    out[i] = static_cast<__half>((q - zp_val) * sc);
}

__global__ void ms_dequant_u8_to_f16_kernel(
    const uint8_t* __restrict__ in,
    const __half*  __restrict__ scale,
    const uint8_t* __restrict__ zp,
    __half*        __restrict__ out,
    int64_t n_elements,
    int64_t n_channels)
{
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n_elements) return;

    int64_t stride  = (n_channels > 1) ? (n_elements / n_channels) : n_elements;
    int64_t ch_idx  = (n_channels > 1) ? (i / stride) : 0;
    float   sc      = static_cast<float>(scale[ch_idx]);
    float   zp_val  = zp ? static_cast<float>(zp[ch_idx]) : 0.0f;
    float   q       = static_cast<float>(in[i]);
    out[i] = static_cast<__half>((q - zp_val) * sc);
}

__global__ void ms_dequant_i8_to_f32_kernel(
    const int8_t*  __restrict__ in,
    const float*   __restrict__ scale,
    const int8_t*  __restrict__ zp,
    float*         __restrict__ out,
    int64_t n_elements,
    int64_t n_channels)
{
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n_elements) return;

    int64_t stride  = (n_channels > 1) ? (n_elements / n_channels) : n_elements;
    int64_t ch_idx  = (n_channels > 1) ? (i / stride) : 0;
    float   sc      = scale[ch_idx];
    float   zp_val  = zp ? static_cast<float>(zp[ch_idx]) : 0.0f;
    out[i] = (static_cast<float>(in[i]) - zp_val) * sc;
}

/* -----------------------------------------------------------------------
 * QuantizeLinear kernel
 * out[i] = clamp(round(in[i] / scale) + zp, dtype_min, dtype_max)
 * ----------------------------------------------------------------------- */

__global__ void ms_quant_f16_to_i8_kernel(
    const __half*  __restrict__ in,
    const __half*  __restrict__ scale,
    const int8_t*  __restrict__ zp,
    int8_t*        __restrict__ out,
    int64_t n_elements,
    int64_t n_channels)
{
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n_elements) return;

    int64_t stride  = (n_channels > 1) ? (n_elements / n_channels) : n_elements;
    int64_t ch_idx  = (n_channels > 1) ? (i / stride) : 0;
    float   sc      = static_cast<float>(scale[ch_idx]);
    float   zp_val  = zp ? static_cast<float>(zp[ch_idx]) : 0.0f;
    float   v       = static_cast<float>(in[i]);
    float   q       = __builtin_rintf(v / sc) + zp_val;
    q = (q < -128.0f) ? -128.0f : (q > 127.0f) ? 127.0f : q;
    out[i] = static_cast<int8_t>(q);
}

__global__ void ms_quant_f16_to_u8_kernel(
    const __half*  __restrict__ in,
    const __half*  __restrict__ scale,
    const uint8_t* __restrict__ zp,
    uint8_t*       __restrict__ out,
    int64_t n_elements,
    int64_t n_channels)
{
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n_elements) return;

    int64_t stride  = (n_channels > 1) ? (n_elements / n_channels) : n_elements;
    int64_t ch_idx  = (n_channels > 1) ? (i / stride) : 0;
    float   sc      = static_cast<float>(scale[ch_idx]);
    float   zp_val  = zp ? static_cast<float>(zp[ch_idx]) : 0.0f;
    float   v       = static_cast<float>(in[i]);
    float   q       = __builtin_rintf(v / sc) + zp_val;
    q = (q < 0.0f) ? 0.0f : (q > 255.0f) ? 255.0f : q;
    out[i] = static_cast<uint8_t>(q);
}

__global__ void ms_quant_f32_to_i8_kernel(
    const float*   __restrict__ in,
    const float*   __restrict__ scale,
    const int8_t*  __restrict__ zp,
    int8_t*        __restrict__ out,
    int64_t n_elements,
    int64_t n_channels)
{
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n_elements) return;

    int64_t stride  = (n_channels > 1) ? (n_elements / n_channels) : n_elements;
    int64_t ch_idx  = (n_channels > 1) ? (i / stride) : 0;
    float   sc      = scale[ch_idx];
    float   zp_val  = zp ? static_cast<float>(zp[ch_idx]) : 0.0f;
    float   q       = __builtin_rintf(in[i] / sc) + zp_val;
    q = (q < -128.0f) ? -128.0f : (q > 127.0f) ? 127.0f : q;
    out[i] = static_cast<int8_t>(q);
}

// Dequant UINT16 quantized activations (2-byte storage) with fp32 scale → fp32
__global__ void ms_dequant_u16_to_f32_kernel(
    const uint16_t* __restrict__ in,
    const float*    __restrict__ scale,
    const uint16_t* __restrict__ zp,
    float*          __restrict__ out,
    int64_t n_elements,
    int64_t n_channels)
{
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n_elements) return;

    int64_t stride  = (n_channels > 1) ? (n_elements / n_channels) : n_elements;
    int64_t ch_idx  = (n_channels > 1) ? (i / stride) : 0;
    float   sc      = scale[ch_idx];
    float   zp_val  = zp ? static_cast<float>(zp[ch_idx]) : 0.0f;
    out[i] = (static_cast<float>(in[i]) - zp_val) * sc;
}

// Dequant uint16 → fp16 with fp16 scale (ORCA W2A8 dequant path after MatMul)
__global__ void ms_dequant_u16_to_f16_kernel(
    const uint16_t* __restrict__ in,
    const __half*   __restrict__ scale,
    const uint16_t* __restrict__ zp,
    __half*         __restrict__ out,
    int64_t n_elements,
    int64_t n_channels)
{
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n_elements) return;

    int64_t stride  = (n_channels > 1) ? (n_elements / n_channels) : n_elements;
    int64_t ch_idx  = (n_channels > 1) ? (i / stride) : 0;
    float   sc      = static_cast<float>(scale[ch_idx]);
    float   zp_val  = zp ? static_cast<float>(zp[ch_idx]) : 0.0f;
    out[i] = static_cast<__half>((static_cast<float>(in[i]) - zp_val) * sc);
}

// Dequant fp16 values with fp32 scale → fp32 (fp16 intermediate activation path)
__global__ void ms_dequant_f16_to_f32_kernel(
    const __half* __restrict__ in,
    const float*  __restrict__ scale,
    const __half* __restrict__ zp,
    float*        __restrict__ out,
    int64_t n_elements,
    int64_t n_channels)
{
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n_elements) return;

    int64_t stride  = (n_channels > 1) ? (n_elements / n_channels) : n_elements;
    int64_t ch_idx  = (n_channels > 1) ? (i / stride) : 0;
    float   sc      = scale[ch_idx];
    float   zp_val  = zp ? static_cast<float>(zp[ch_idx]) : 0.0f;
    out[i] = (static_cast<float>(in[i]) - zp_val) * sc;
}

// Quantize fp16 → uint16 with fp16 scale (ORCA W2A8 activation quant path)
__global__ void ms_quant_f16_to_u16_kernel(
    const __half*   __restrict__ in,
    const __half*   __restrict__ scale,
    const uint16_t* __restrict__ zp,
    uint16_t*       __restrict__ out,
    int64_t n_elements,
    int64_t n_channels)
{
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n_elements) return;

    int64_t stride  = (n_channels > 1) ? (n_elements / n_channels) : n_elements;
    int64_t ch_idx  = (n_channels > 1) ? (i / stride) : 0;
    float   sc      = static_cast<float>(scale[ch_idx]);
    float   zp_val  = zp ? static_cast<float>(zp[ch_idx]) : 0.0f;
    float   v       = static_cast<float>(in[i]);
    float   q       = __builtin_rintf(v / sc) + zp_val;
    q = (q < 0.0f) ? 0.0f : (q > 65535.0f) ? 65535.0f : q;
    out[i] = static_cast<uint16_t>(q);
}

// Quantize fp32 → uint16 with fp32 scale
__global__ void ms_quant_f32_to_u16_kernel(
    const float*    __restrict__ in,
    const float*    __restrict__ scale,
    const uint16_t* __restrict__ zp,
    uint16_t*       __restrict__ out,
    int64_t n_elements,
    int64_t n_channels)
{
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n_elements) return;

    int64_t stride  = (n_channels > 1) ? (n_elements / n_channels) : n_elements;
    int64_t ch_idx  = (n_channels > 1) ? (i / stride) : 0;
    float   sc      = scale[ch_idx];
    float   zp_val  = zp ? static_cast<float>(zp[ch_idx]) : 0.0f;
    float   q       = __builtin_rintf(in[i] / sc) + zp_val;
    q = (q < 0.0f) ? 0.0f : (q > 65535.0f) ? 65535.0f : q;
    out[i] = static_cast<uint16_t>(q);
}

// Quantize fp32 → fp16 with fp32 scale
__global__ void ms_quant_f32_to_f16_kernel(
    const float*  __restrict__ in,
    const float*  __restrict__ scale,
    const __half* __restrict__ zp,
    __half*       __restrict__ out,
    int64_t n_elements,
    int64_t n_channels)
{
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n_elements) return;

    int64_t stride  = (n_channels > 1) ? (n_elements / n_channels) : n_elements;
    int64_t ch_idx  = (n_channels > 1) ? (i / stride) : 0;
    float   sc      = scale[ch_idx];
    float   zp_val  = zp ? static_cast<float>(zp[ch_idx]) : 0.0f;
    out[i] = static_cast<__half>((in[i] / sc) + zp_val);
}

/* -----------------------------------------------------------------------
 * Public C API
 * ----------------------------------------------------------------------- */

static constexpr int kBlockSize = 256;

extern "C" int hip_ms_dequantize_linear(
    void *stream,
    const void *input,
    const void *scale,
    const void *zero_point,
    void *output,
    int64_t n_elements,
    int64_t n_channels,
    int64_t input_elem_size,
    int64_t scale_elem_size)
{
    if (n_elements <= 0) return 0;

    hipStream_t s = static_cast<hipStream_t>(stream);
    dim3 blk(kBlockSize);
    dim3 grd(static_cast<unsigned>((n_elements + kBlockSize - 1) / kBlockSize));

    (void)hipGetLastError();

    if (input_elem_size == 1 && scale_elem_size == 2) {
        // i8 → f16 (scale f16)
        hipLaunchKernelGGL(ms_dequant_i8_to_f16_kernel, grd, blk, 0, s,
            static_cast<const int8_t*>(input),
            static_cast<const __half*>(scale),
            static_cast<const int8_t*>(zero_point),
            static_cast<__half*>(output),
            n_elements, n_channels);
    } else if (input_elem_size == 1 && scale_elem_size == 4) {
        // i8 → f32 (scale f32)
        hipLaunchKernelGGL(ms_dequant_i8_to_f32_kernel, grd, blk, 0, s,
            static_cast<const int8_t*>(input),
            static_cast<const float*>(scale),
            static_cast<const int8_t*>(zero_point),
            static_cast<float*>(output),
            n_elements, n_channels);
    } else if (input_elem_size == 2 && scale_elem_size == 4) {
        // u16/f16 → f32 (scale f32); ORCA W2A8 uses UINT16 activations
        hipLaunchKernelGGL(ms_dequant_u16_to_f32_kernel, grd, blk, 0, s,
            static_cast<const uint16_t*>(input),
            static_cast<const float*>(scale),
            static_cast<const uint16_t*>(zero_point),
            static_cast<float*>(output),
            n_elements, n_channels);
    } else if (input_elem_size == 2 && scale_elem_size == 2) {
        // uint16 → f16 (ORCA W2A8: dequantize uint16 activation with fp16 scale)
        hipLaunchKernelGGL(ms_dequant_u16_to_f16_kernel, grd, blk, 0, s,
            static_cast<const uint16_t*>(input),
            static_cast<const __half*>(scale),
            static_cast<const uint16_t*>(zero_point),
            static_cast<__half*>(output),
            n_elements, n_channels);
    } else {
        fprintf(stderr,
                "[REAL] hip_ms_dequantize_linear: unsupported in=%lld sc=%lld\n",
                (long long)input_elem_size, (long long)scale_elem_size);
        return -1;
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "[REAL] hip_ms_dequantize_linear launch failed: %s\n",
                hipGetErrorString(err));
    }
    return static_cast<int>(err);
}

extern "C" int hip_ms_quantize_linear(
    void *stream,
    const void *input,
    const void *scale,
    const void *zero_point,
    void *output,
    int64_t n_elements,
    int64_t n_channels,
    int64_t input_elem_size,
    int64_t output_elem_size)
{
    if (n_elements <= 0) return 0;

    hipStream_t s = static_cast<hipStream_t>(stream);
    dim3 blk(kBlockSize);
    dim3 grd(static_cast<unsigned>((n_elements + kBlockSize - 1) / kBlockSize));

    (void)hipGetLastError();

    if (input_elem_size == 2 && output_elem_size == 1) {
        // fp16 → i8 (scale fp16)
        hipLaunchKernelGGL(ms_quant_f16_to_i8_kernel, grd, blk, 0, s,
            static_cast<const __half*>(input),
            static_cast<const __half*>(scale),
            static_cast<const int8_t*>(zero_point),
            static_cast<int8_t*>(output),
            n_elements, n_channels);
    } else if (input_elem_size == 4 && output_elem_size == 1) {
        // fp32 → i8 (scale fp32)
        hipLaunchKernelGGL(ms_quant_f32_to_i8_kernel, grd, blk, 0, s,
            static_cast<const float*>(input),
            static_cast<const float*>(scale),
            static_cast<const int8_t*>(zero_point),
            static_cast<int8_t*>(output),
            n_elements, n_channels);
    } else if (input_elem_size == 4 && output_elem_size == 2) {
        // fp32 → u16 (ORCA W2A8 activation quant, scale fp32)
        hipLaunchKernelGGL(ms_quant_f32_to_u16_kernel, grd, blk, 0, s,
            static_cast<const float*>(input),
            static_cast<const float*>(scale),
            static_cast<const uint16_t*>(zero_point),
            static_cast<uint16_t*>(output),
            n_elements, n_channels);
    } else if (input_elem_size == 2 && output_elem_size == 2) {
        // fp16 → uint16 (ORCA W2A8 activation quantization, scale fp16)
        hipLaunchKernelGGL(ms_quant_f16_to_u16_kernel, grd, blk, 0, s,
            static_cast<const __half*>(input),
            static_cast<const __half*>(scale),
            static_cast<const uint16_t*>(zero_point),
            static_cast<uint16_t*>(output),
            n_elements, n_channels);
    } else {
        fprintf(stderr,
                "[REAL] hip_ms_quantize_linear: unsupported in=%lld out=%lld\n",
                (long long)input_elem_size, (long long)output_elem_size);
        return -1;
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "[REAL] hip_ms_quantize_linear launch failed: %s\n",
                hipGetErrorString(err));
    }
    return static_cast<int>(err);
}
