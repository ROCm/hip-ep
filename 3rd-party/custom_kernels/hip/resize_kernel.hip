/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//
// ONNX Resize (opset 18): resize the spatial dims (last 2) of an N-D
// input tensor to a new size, preserving batch / channel and any leading
// dims.  Modes:
//   0 = nearest (round_prefer_floor per ONNX 18 default)
//   1 = linear  (bilinear over the last 2 dims)
//   2 = cubic   (bicubic over the last 2 dims, parameterised by
//                cubic_coeff_a -- defaults to -0.75 = Mitchell-Netravali
//                / Catmull-Rom-ish)
//
// Coordinate transform modes (output-coord -> input-coord):
//   0 = half_pixel:        src = (out + 0.5) * (in/out) - 0.5
//   1 = pytorch_half_pixel: out_size > 1 ? half_pixel : 0
//   2 = align_corners:     src = out * (in - 1) / (out - 1)  (out_size==1: 0)
//   3 = asymmetric:        src = out * in / out_size
//   4 = tf_crop_and_resize: rejected at the lowering level.
//
// One thread per output element.  The flat output index is decoded into
// per-axis output coords (row-major).  Spatial coords (last 2 axes) feed
// the interpolation; leading coords (batch / channel / etc.) just
// propagate to the input via row-major strides.
//
// Grid tiling: HIP / RDNA caps grid.x at 65535, so we tile across grid.y
// when the total number of output elements exceeds 65535 * blockDim.x:
//   bx   = blockIdx.y * gridDim.x + blockIdx.x
//   flat = bx * blockDim.x + threadIdx.x
// This matches concat / pad and is essential for Kokoro because the
// vocoder upsamples 1-D audio frames where the resized spatial dim ends
// up well past 65535 * 256.

#include "hip_custom_kernels.h"

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>

#define HIP_RESIZE_MAX_RANK 8

// Local fallback definitions of the public enums.  These mirror the
// canonical values in `hip_custom_kernels.h`; we redeclare them here
// so this TU keeps compiling even if the shared header gets reverted
// by sibling-agent merges in flight.  Symbols reach the runtime via
// the wrap_resize wrapper which translates HIPDNN_EP_DATATYPE_* values
// directly, so this is purely for the kernel-internal switch below.
#ifndef HIP_RESIZE_MODE_NEAREST
#define HIP_RESIZE_MODE_NEAREST 0
#define HIP_RESIZE_MODE_LINEAR 1
#define HIP_RESIZE_MODE_CUBIC 2
#endif
#ifndef HIP_RESIZE_COORD_HALF_PIXEL
#define HIP_RESIZE_COORD_HALF_PIXEL 0
#define HIP_RESIZE_COORD_PT_HALF_PIXEL 1
#define HIP_RESIZE_COORD_ALIGN_CORNERS 2
#define HIP_RESIZE_COORD_ASYMMETRIC 3
#define HIP_RESIZE_COORD_TF_CROP_RESIZE 4
#endif

namespace {

struct ResizeDesc {
    int rank;
    int mode;
    int coord_xform;
    float cubic_coeff_a;
    int64_t in_shape[HIP_RESIZE_MAX_RANK];
    int64_t in_strides[HIP_RESIZE_MAX_RANK]; // elements
    int64_t out_shape[HIP_RESIZE_MAX_RANK];
    // out_strides not needed -- we walk the flat output index in
    // row-major order via repeated div/mod.
};

// Map an output coordinate to the corresponding input coordinate per the
// requested coord-transform.  Returns a float that subsequent code will
// snap to the nearest neighbour (nearest mode) or linearly / cubically
// interpolate around (linear / cubic).
__device__ __forceinline__ float coord_xform(int coord_xform_mode,
                                             int out_idx, int in_size,
                                             int out_size) {
    if (in_size <= 0 || out_size <= 0)
        return 0.0f;
    float fout = static_cast<float>(out_idx);
    float fin = static_cast<float>(in_size);
    float fout_size = static_cast<float>(out_size);

    switch (coord_xform_mode) {
    case HIP_RESIZE_COORD_HALF_PIXEL:
        return (fout + 0.5f) * (fin / fout_size) - 0.5f;
    case HIP_RESIZE_COORD_PT_HALF_PIXEL:
        if (out_size > 1)
            return (fout + 0.5f) * (fin / fout_size) - 0.5f;
        return 0.0f;
    case HIP_RESIZE_COORD_ALIGN_CORNERS:
        if (out_size == 1)
            return 0.0f;
        return fout * (fin - 1.0f) / (fout_size - 1.0f);
    case HIP_RESIZE_COORD_ASYMMETRIC:
        return fout * fin / fout_size;
    default:
        // tf_crop_and_resize is filtered out at lowering time; fall back
        // to half_pixel so the kernel still produces something sensible
        // if it ever gets here.
        return (fout + 0.5f) * (fin / fout_size) - 0.5f;
    }
}

// ONNX 18 nearest_mode default = "round_prefer_floor":
//   Round to nearest integer; break ties by choosing floor.
//   idx = floor(src + 0.5) when frac != 0.5
//   idx = floor(src)       when frac == 0.5  (tie → prefer floor)
__device__ __forceinline__ int round_prefer_floor(float src, int in_size) {
    int idx = static_cast<int>(floorf(src));
    float frac = src - static_cast<float>(idx);
    // Only round up if frac is STRICTLY greater than 0.5.
    // At exactly 0.5, prefer floor (don't increment).
    if (frac > 0.5f) {
        idx++;
    }
    if (idx < 0)
        idx = 0;
    if (idx > in_size - 1)
        idx = in_size - 1;
    return idx;
}

__device__ __forceinline__ int clamp_idx(int idx, int in_size) {
    if (idx < 0)
        return 0;
    if (idx > in_size - 1)
        return in_size - 1;
    return idx;
}

// Standard cubic Hermite weight, a.k.a. Mitchell-Netravali / Keys filter
// parameterised by `a` (default -0.75 in PyTorch / OpenCV / ONNX).
//   |t| < 1: ((a + 2)|t|^3 - (a + 3)|t|^2 + 1)
//   1 <= |t| < 2: (a|t|^3 - 5a|t|^2 + 8a|t| - 4a)
//   |t| >= 2: 0
__device__ __forceinline__ float cubic_weight(float t, float a) {
    t = fabsf(t);
    if (t < 1.0f) {
        return ((a + 2.0f) * t - (a + 3.0f)) * t * t + 1.0f;
    } else if (t < 2.0f) {
        return ((a * t - 5.0f * a) * t + 8.0f * a) * t - 4.0f * a;
    }
    return 0.0f;
}

// Convert an in-element value to fp32 for the math (keeps half / bf16
// math identical to the f32 path apart from the input/output cast).
template <typename T>
__device__ __forceinline__ float to_float(T v);

template <> __device__ __forceinline__ float to_float<float>(float v) {
    return v;
}
template <> __device__ __forceinline__ float to_float<__half>(__half v) {
    return __half2float(v);
}
template <>
__device__ __forceinline__ float to_float<__hip_bfloat16>(__hip_bfloat16 v) {
    return __bfloat162float(v);
}

template <typename T> __device__ __forceinline__ T from_float(float v);

template <> __device__ __forceinline__ float from_float<float>(float v) {
    return v;
}
template <> __device__ __forceinline__ __half from_float<__half>(float v) {
    return __float2half(v);
}
template <>
__device__ __forceinline__ __hip_bfloat16
from_float<__hip_bfloat16>(float v) {
    return __float2bfloat16(v);
}

// Compute the row-major linear offset for the leading (rank-2) dims --
// the spatial offset is added separately by the per-mode code below.
// `lead_off` is in elements.
template <typename T>
__device__ __forceinline__ T fetch(const T *base, int64_t lead_off,
                                   int h_idx, int w_idx, int64_t h_stride,
                                   int64_t w_stride) {
    return base[lead_off + static_cast<int64_t>(h_idx) * h_stride +
                static_cast<int64_t>(w_idx) * w_stride];
}

template <typename T>
__global__ void resize_kernel(const T *__restrict__ in, T *__restrict__ out,
                              int64_t total, ResizeDesc desc) {
    int64_t bx = static_cast<int64_t>(blockIdx.y) * gridDim.x + blockIdx.x;
    int64_t flat = bx * blockDim.x + threadIdx.x;
    if (flat >= total)
        return;

    // Decode the flat output index into per-axis output coords (row-major).
    int64_t out_coord[HIP_RESIZE_MAX_RANK];
    int64_t rem = flat;
    for (int d = desc.rank - 1; d >= 0; --d) {
        int64_t s = desc.out_shape[d];
        out_coord[d] = rem % s;
        rem /= s;
    }

    // Leading offset: contributions from all axes except the last 2.
    // The H/W contribution is added per interpolation tap below.
    int64_t lead_off = 0;
    for (int d = 0; d < desc.rank - 2; ++d) {
        lead_off += out_coord[d] * desc.in_strides[d];
    }

    int h_axis = desc.rank - 2;
    int w_axis = desc.rank - 1;
    int in_h = static_cast<int>(desc.in_shape[h_axis]);
    int in_w = static_cast<int>(desc.in_shape[w_axis]);
    int out_h = static_cast<int>(desc.out_shape[h_axis]);
    int out_w = static_cast<int>(desc.out_shape[w_axis]);
    int64_t h_stride = desc.in_strides[h_axis];
    int64_t w_stride = desc.in_strides[w_axis];

    int oh = static_cast<int>(out_coord[h_axis]);
    int ow = static_cast<int>(out_coord[w_axis]);

    float src_h = coord_xform(desc.coord_xform, oh, in_h, out_h);
    float src_w = coord_xform(desc.coord_xform, ow, in_w, out_w);

    float result = 0.0f;

    if (desc.mode == HIP_RESIZE_MODE_NEAREST) {
        int ih = round_prefer_floor(src_h, in_h);
        int iw = round_prefer_floor(src_w, in_w);
        result = to_float(fetch(in, lead_off, ih, iw, h_stride, w_stride));
    } else if (desc.mode == HIP_RESIZE_MODE_LINEAR) {
        // Bilinear: 4 taps, fractional-bilerp in fp32.
        int h0 = static_cast<int>(floorf(src_h));
        int w0 = static_cast<int>(floorf(src_w));
        float dh = src_h - static_cast<float>(h0);
        float dw = src_w - static_cast<float>(w0);
        int h1 = h0 + 1;
        int w1 = w0 + 1;
        h0 = clamp_idx(h0, in_h);
        h1 = clamp_idx(h1, in_h);
        w0 = clamp_idx(w0, in_w);
        w1 = clamp_idx(w1, in_w);
        float v00 =
            to_float(fetch(in, lead_off, h0, w0, h_stride, w_stride));
        float v01 =
            to_float(fetch(in, lead_off, h0, w1, h_stride, w_stride));
        float v10 =
            to_float(fetch(in, lead_off, h1, w0, h_stride, w_stride));
        float v11 =
            to_float(fetch(in, lead_off, h1, w1, h_stride, w_stride));
        float vh0 = v00 + (v01 - v00) * dw;
        float vh1 = v10 + (v11 - v10) * dw;
        result = vh0 + (vh1 - vh0) * dh;
    } else if (desc.mode == HIP_RESIZE_MODE_CUBIC) {
        // Bicubic: 4x4 neighbourhood centred on (floor(src_h), floor(src_w)).
        int h_base = static_cast<int>(floorf(src_h));
        int w_base = static_cast<int>(floorf(src_w));
        float dh = src_h - static_cast<float>(h_base);
        float dw = src_w - static_cast<float>(w_base);
        float wh[4];
        float ww[4];
        // Taps at h_base + (-1, 0, 1, 2): t = (offset) - dh
        for (int k = 0; k < 4; ++k) {
            float t_h = static_cast<float>(k - 1) - dh;
            float t_w = static_cast<float>(k - 1) - dw;
            wh[k] = cubic_weight(t_h, desc.cubic_coeff_a);
            ww[k] = cubic_weight(t_w, desc.cubic_coeff_a);
        }
        float acc = 0.0f;
        for (int kh = 0; kh < 4; ++kh) {
            int ih = clamp_idx(h_base + (kh - 1), in_h);
            float row = 0.0f;
            for (int kw = 0; kw < 4; ++kw) {
                int iw = clamp_idx(w_base + (kw - 1), in_w);
                float v = to_float(
                    fetch(in, lead_off, ih, iw, h_stride, w_stride));
                row += v * ww[kw];
            }
            acc += row * wh[kh];
        }
        result = acc;
    } else {
        // Unknown mode -- match input directly to keep the kernel
        // deterministic (lowering should already have rejected this).
        int ih = round_prefer_floor(src_h, in_h);
        int iw = round_prefer_floor(src_w, in_w);
        result = to_float(fetch(in, lead_off, ih, iw, h_stride, w_stride));
    }

    out[flat] = from_float<T>(result);
}

template <typename T>
int launch_resize(hipStream_t stream, const T *in, T *out, int64_t total,
                  const ResizeDesc &desc) {
    if (total <= 0)
        return 0;
    constexpr int kBlockSize = 256;
    int64_t blocks = (total + kBlockSize - 1) / kBlockSize;

    // HIP / RDNA caps grid.x at 65535.  Tile across grid.y for big
    // workloads (long Kokoro audio frames easily exceed this).
    constexpr int64_t kMaxGridX = 65535;
    int64_t grid_x = blocks;
    int64_t grid_y = 1;
    if (blocks > kMaxGridX) {
        grid_y = (blocks + kMaxGridX - 1) / kMaxGridX;
        grid_x = (blocks + grid_y - 1) / grid_y;
    }

    dim3 grid(static_cast<unsigned>(grid_x), static_cast<unsigned>(grid_y), 1);
    dim3 block(kBlockSize, 1, 1);
    hipLaunchKernelGGL(resize_kernel<T>, grid, block, 0, stream, in, out,
                       total, desc);
    return static_cast<int>(hipGetLastError());
}

} // namespace

extern "C" int hip_resize(void *stream, const void *input, void *output,
                          const int64_t *in_shape,
                          const int64_t *in_strides_elems,
                          const int64_t *out_shape,
                          const int64_t * /*out_strides_elems*/, int64_t rank,
                          int hip_dtype, int mode, int coord_xform,
                          float cubic_coeff_a) {
    if (!input || !output || !in_shape || !in_strides_elems || !out_shape) {
        fprintf(stderr, "[custom_kernels] hip_resize: null pointer\n");
        return -1;
    }
    if (rank < 2 || rank > HIP_RESIZE_MAX_RANK) {
        fprintf(stderr,
                "[custom_kernels] hip_resize: rank %lld out of range [2, %d]\n",
                (long long)rank, HIP_RESIZE_MAX_RANK);
        return -1;
    }
    if (mode < 0 || mode > 2) {
        fprintf(stderr, "[custom_kernels] hip_resize: unsupported mode %d\n",
                mode);
        return -1;
    }
    if (coord_xform < 0 || coord_xform > 4) {
        fprintf(stderr,
                "[custom_kernels] hip_resize: unsupported coord_xform %d\n",
                coord_xform);
        return -1;
    }

    ResizeDesc desc{};
    desc.rank = static_cast<int>(rank);
    desc.mode = mode;
    desc.coord_xform = coord_xform;
    desc.cubic_coeff_a = cubic_coeff_a;
    int64_t total = 1;
    for (int64_t i = 0; i < rank; ++i) {
        desc.in_shape[i] = in_shape[i];
        desc.in_strides[i] = in_strides_elems[i];
        desc.out_shape[i] = out_shape[i];
        total *= out_shape[i];
    }

    if (total == 0)
        return 0;

    hipStream_t hip_stream = static_cast<hipStream_t>(stream);

    int rc = -1;
    switch (hip_dtype) {
    case HIP_DTYPE_FLOAT32:
        rc = launch_resize<float>(hip_stream,
                                  static_cast<const float *>(input),
                                  static_cast<float *>(output), total, desc);
        break;
    case HIP_DTYPE_FLOAT16:
        rc = launch_resize<__half>(hip_stream,
                                   static_cast<const __half *>(input),
                                   static_cast<__half *>(output), total, desc);
        break;
    case HIP_DTYPE_BFLOAT16:
        rc = launch_resize<__hip_bfloat16>(
            hip_stream, static_cast<const __hip_bfloat16 *>(input),
            static_cast<__hip_bfloat16 *>(output), total, desc);
        break;
    default:
        fprintf(stderr, "[custom_kernels] hip_resize: unsupported dtype=%d\n",
                hip_dtype);
        return -1;
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "[custom_kernels] hip_resize launch failed: %s\n",
                hipGetErrorString(err));
        return static_cast<int>(err);
    }
    return rc;
}
