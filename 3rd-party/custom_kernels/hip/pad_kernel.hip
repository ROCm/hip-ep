/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//
// ONNX Pad (opset 18): pad each axis with `pads_begin[axis]` elements before
// and `pads_end[axis]` elements after.
//
// Mode is one of:
//   0 = constant : pad with `value` cast to T
//   1 = reflect  : mirror without repeating the boundary sample
//   2 = edge     : replicate the boundary sample (clamp)
//
// One thread per output element.  The flat output index is decoded via the
// row-major output shape into per-axis output coords; per-axis the input
// coord is computed (or the pad value substituted for constant mode), then
// the input linear offset is built from `in_strides_elems` (in elements).
//
// Grid tiling: HIP / RDNA caps grid.x at 65535.  When the total number of
// output elements exceeds 65535 * blockDim.x we tile across grid.y and
// recover the global flat index inside the kernel as
//   flat = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x
// This is essential for Kokoro -- long audio inflates the innermost dim past
// the single-axis grid cap.

#include "hip_custom_kernels.h"

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>

#define HIP_PAD_MAX_RANK 8

namespace {

struct PadDesc {
    int rank;
    int mode;
    int64_t out_shape[HIP_PAD_MAX_RANK];
    int64_t in_shape[HIP_PAD_MAX_RANK];
    int64_t in_strides[HIP_PAD_MAX_RANK]; // elements
    int64_t pads_begin[HIP_PAD_MAX_RANK];
};

// Reflect index for `relative` against [0, in_dim).  Bounces off both edges
// without repeating the boundary sample (matches numpy.pad mode='reflect').
//   period = max(2 * (in_dim - 1), 1)
//   idx    = ((relative % period) + period) % period
//   if idx >= in_dim: idx = period - idx
__device__ __forceinline__ int64_t reflect_index(int64_t relative,
                                                 int64_t in_dim) {
    if (in_dim <= 1)
        return 0;
    int64_t period = 2 * (in_dim - 1);
    int64_t idx = relative % period;
    if (idx < 0)
        idx += period;
    if (idx >= in_dim)
        idx = period - idx;
    return idx;
}

__device__ __forceinline__ int64_t edge_index(int64_t relative,
                                              int64_t in_dim) {
    if (relative < 0)
        return 0;
    if (relative >= in_dim)
        return in_dim - 1;
    return relative;
}

template <typename T>
__device__ __forceinline__ T cast_pad_value(float v);

template <> __device__ __forceinline__ float cast_pad_value<float>(float v) {
    return v;
}
template <> __device__ __forceinline__ __half cast_pad_value<__half>(float v) {
    return __float2half(v);
}
template <>
__device__ __forceinline__ __hip_bfloat16
cast_pad_value<__hip_bfloat16>(float v) {
    return __float2bfloat16(v);
}

template <typename T>
__global__ void pad_kernel(const T *__restrict__ in, T *__restrict__ out,
                           int64_t total, PadDesc desc, float pad_value) {
    int64_t bx = static_cast<int64_t>(blockIdx.y) * gridDim.x + blockIdx.x;
    int64_t flat = bx * blockDim.x + threadIdx.x;
    if (flat >= total)
        return;

    int64_t in_off = 0;
    int64_t rem = flat;
    bool use_pad = false;

    // Decode flat index into per-axis output coords (row-major).  Walk axes
    // from innermost to outermost so we can divide by the per-axis size.
    for (int d = desc.rank - 1; d >= 0; --d) {
        int64_t out_dim = desc.out_shape[d];
        int64_t out_coord = rem % out_dim;
        rem /= out_dim;

        int64_t in_dim = desc.in_shape[d];
        int64_t relative = out_coord - desc.pads_begin[d];
        int64_t src_coord;

        if (relative >= 0 && relative < in_dim) {
            src_coord = relative;
        } else if (desc.mode == 1) {
            src_coord = reflect_index(relative, in_dim);
        } else if (desc.mode == 2) {
            src_coord = edge_index(relative, in_dim);
        } else {
            // mode == 0 (constant)
            use_pad = true;
            src_coord = 0; // unused
        }

        in_off += src_coord * desc.in_strides[d];
    }

    if (use_pad)
        out[flat] = cast_pad_value<T>(pad_value);
    else
        out[flat] = in[in_off];
}

template <typename T>
int launch_pad(hipStream_t stream, const T *in, T *out, int64_t total,
               const PadDesc &desc, float pad_value) {
    if (total <= 0)
        return 0;
    constexpr int kBlockSize = 256;
    int64_t blocks = (total + kBlockSize - 1) / kBlockSize;

    // HIP / RDNA caps grid.x at 65535.  Tile across grid.y for big workloads
    // (Kokoro-scale audio inflates the innermost dim well past this).
    constexpr int64_t kMaxGridX = 65535;
    int64_t grid_x = blocks;
    int64_t grid_y = 1;
    if (blocks > kMaxGridX) {
        grid_y = (blocks + kMaxGridX - 1) / kMaxGridX;
        grid_x = (blocks + grid_y - 1) / grid_y;
    }

    dim3 grid(static_cast<unsigned>(grid_x), static_cast<unsigned>(grid_y), 1);
    dim3 block(kBlockSize, 1, 1);
    hipLaunchKernelGGL(pad_kernel<T>, grid, block, 0, stream, in, out, total,
                       desc, pad_value);
    return static_cast<int>(hipGetLastError());
}

} // namespace

extern "C" int hip_pad(void *stream, const void *input, void *output,
                       const int64_t *in_shape, const int64_t *in_strides_elems,
                       const int64_t *out_shape,
                       const int64_t * /*out_strides_elems*/, int64_t rank,
                       const int64_t *pads_begin, int64_t pads_begin_len,
                       int hip_dtype, int mode, float value) {
    if (!input || !output || !in_shape || !in_strides_elems || !out_shape ||
        !pads_begin) {
        fprintf(stderr, "[custom_kernels] hip_pad: null pointer\n");
        return -1;
    }
    if (rank <= 0 || rank > HIP_PAD_MAX_RANK) {
        fprintf(stderr, "[custom_kernels] hip_pad: rank %lld out of range "
                        "[1, %d]\n",
                (long long)rank, HIP_PAD_MAX_RANK);
        return -1;
    }
    if (pads_begin_len != rank) {
        fprintf(stderr,
                "[custom_kernels] hip_pad: pads_begin_len %lld != rank %lld\n",
                (long long)pads_begin_len, (long long)rank);
        return -1;
    }
    if (mode < 0 || mode > 2) {
        fprintf(stderr, "[custom_kernels] hip_pad: unsupported mode %d\n",
                mode);
        return -1;
    }

    PadDesc desc{};
    desc.rank = static_cast<int>(rank);
    desc.mode = mode;
    int64_t total = 1;
    for (int64_t i = 0; i < rank; ++i) {
        desc.out_shape[i] = out_shape[i];
        desc.in_shape[i] = in_shape[i];
        desc.in_strides[i] = in_strides_elems[i];
        desc.pads_begin[i] = pads_begin[i];
        total *= out_shape[i];
    }

    if (total == 0)
        return 0;

    hipStream_t hip_stream = static_cast<hipStream_t>(stream);

    int rc = -1;
    switch (hip_dtype) {
    case HIP_DTYPE_FLOAT32:
        rc = launch_pad<float>(hip_stream, static_cast<const float *>(input),
                               static_cast<float *>(output), total, desc,
                               value);
        break;
    case HIP_DTYPE_FLOAT16:
        rc = launch_pad<__half>(hip_stream, static_cast<const __half *>(input),
                                static_cast<__half *>(output), total, desc,
                                value);
        break;
    case HIP_DTYPE_BFLOAT16:
        rc = launch_pad<__hip_bfloat16>(
            hip_stream, static_cast<const __hip_bfloat16 *>(input),
            static_cast<__hip_bfloat16 *>(output), total, desc, value);
        break;
    default:
        fprintf(stderr, "[custom_kernels] hip_pad: unsupported dtype=%d\n",
                hip_dtype);
        return -1;
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "[custom_kernels] hip_pad launch failed: %s\n",
                hipGetErrorString(err));
        return static_cast<int>(err);
    }
    return rc;
}
