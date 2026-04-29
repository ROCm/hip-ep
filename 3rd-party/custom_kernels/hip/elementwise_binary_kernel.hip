/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//
// Generic element-wise binary kernel for the ONNX ops that miopenOpTensor
// can't (or shouldn't) handle: Div and Pow.  Both kernels promote half/bf16
// to f32 for the math and narrow back, matching the precision of the
// existing reciprocal/sqrt/sub kernels.
//
// Broadcasting:  the host-side wrapper materialises the lhs/rhs/out shapes
// into 8-dim padded sizes and per-dim strides (in elements).  A stride of 0
// in either input represents a broadcast over that axis, including
// scalar->tensor.  Up to 8 dimensions are supported, which is enough for
// every Kokoro tensor (max rank ~5).

#include "hip_custom_kernels.h"

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>

#define HIP_BINARY_MAX_RANK 8

namespace {

template <hip_binary_kind_t Kind>
__device__ __forceinline__ float apply(float a, float b) {
    if constexpr (Kind == HIP_BINARY_DIV)
        return (a == 0.0f && b == 0.0f) ? 0.0f : (a / b);
    else if constexpr (Kind == HIP_BINARY_POW) {
        // Fast-path small integer exponents: more precise than powf()
        // (which uses exp(b*log(a)) and fails for negative a with
        // non-integer b) and matches what CUDA libdevice does internally.
        // Kokoro uses Pow(x,2) for Snake activation (sin²) and Pow(x,3)
        // for GELU, both of which have negative inputs.
        if (b == 2.0f) return a * a;
        if (b == 3.0f) return a * a * a;
        if (b == 0.5f) return sqrtf(a);
        if (b == 0.0f) return 1.0f;
        if (b == 1.0f) return a;
        return powf(a, b);
    }
    else if constexpr (Kind == HIP_BINARY_MUL)
        return a * b;
    else if constexpr (Kind == HIP_BINARY_ADD)
        return a + b;
    else if constexpr (Kind == HIP_BINARY_SUB)
        return a - b;
    else
        return a;
}

struct BroadcastDesc {
    int rank;
    int64_t out_shape[HIP_BINARY_MAX_RANK];
    int64_t lhs_strides[HIP_BINARY_MAX_RANK]; // elements; 0 = broadcast
    int64_t rhs_strides[HIP_BINARY_MAX_RANK];
};

__device__ __forceinline__ void unflatten(int64_t flat, const int64_t *shape,
                                          int rank, int64_t *out_idx) {
    for (int d = rank - 1; d >= 0; --d) {
        int64_t s = shape[d];
        out_idx[d] = flat % s;
        flat /= s;
    }
}

__device__ __forceinline__ int64_t dot_strides(const int64_t *idx,
                                               const int64_t *strides,
                                               int rank) {
    int64_t off = 0;
    for (int d = 0; d < rank; ++d)
        off += idx[d] * strides[d];
    return off;
}

template <hip_binary_kind_t Kind, typename T, typename ToF, typename FromF>
__global__ void
binary_kernel(const T *__restrict__ lhs, const T *__restrict__ rhs,
              T *__restrict__ out, int64_t num_elements, BroadcastDesc desc,
              ToF to_float, FromF from_float) {
    int64_t bx =
        static_cast<int64_t>(blockIdx.y) * gridDim.x + blockIdx.x;
    int64_t flat = bx * blockDim.x + threadIdx.x;
    if (flat >= num_elements)
        return;
    int64_t idx[HIP_BINARY_MAX_RANK];
    unflatten(flat, desc.out_shape, desc.rank, idx);
    int64_t lhs_off = dot_strides(idx, desc.lhs_strides, desc.rank);
    int64_t rhs_off = dot_strides(idx, desc.rhs_strides, desc.rank);
    float a = to_float(lhs[lhs_off]);
    float b = to_float(rhs[rhs_off]);
    out[flat] = from_float(apply<Kind>(a, b));
}

template <hip_binary_kind_t Kind>
int launch_binary(hipStream_t stream, const void *lhs, const void *rhs,
                  void *out, int64_t num_elements, int hip_dtype,
                  const BroadcastDesc &desc) {
    constexpr int kBlockSize = 256;
    int64_t blocks = (num_elements + kBlockSize - 1) / kBlockSize;
    constexpr int64_t kMaxGridX = 65535;
    int64_t grid_x = blocks;
    int64_t grid_y = 1;
    if (blocks > kMaxGridX) {
        grid_y = (blocks + kMaxGridX - 1) / kMaxGridX;
        grid_x = (blocks + grid_y - 1) / grid_y;
    }
    dim3 grid(static_cast<unsigned>(grid_x), static_cast<unsigned>(grid_y), 1);
    dim3 block(kBlockSize, 1, 1);
    switch (hip_dtype) {
    case HIP_DTYPE_FLOAT32: {
        auto id = [] __device__(float v) { return v; };
        hipLaunchKernelGGL(
            (binary_kernel<Kind, float, decltype(id), decltype(id)>),
            grid, block, 0, stream,
            static_cast<const float *>(lhs), static_cast<const float *>(rhs),
            static_cast<float *>(out), num_elements, desc, id, id);
        break;
    }
    case HIP_DTYPE_FLOAT16: {
        auto to_f = [] __device__(__half v) { return __half2float(v); };
        auto from_f = [] __device__(float v) { return __float2half(v); };
        hipLaunchKernelGGL(
            (binary_kernel<Kind, __half, decltype(to_f), decltype(from_f)>),
            grid, block, 0, stream,
            static_cast<const __half *>(lhs), static_cast<const __half *>(rhs),
            static_cast<__half *>(out), num_elements, desc, to_f, from_f);
        break;
    }
    case HIP_DTYPE_BFLOAT16: {
        auto to_f =
            [] __device__(__hip_bfloat16 v) { return __bfloat162float(v); };
        auto from_f = [] __device__(float v) { return __float2bfloat16(v); };
        hipLaunchKernelGGL((binary_kernel<Kind, __hip_bfloat16, decltype(to_f),
                                          decltype(from_f)>),
                           grid, block, 0, stream,
                           static_cast<const __hip_bfloat16 *>(lhs),
                           static_cast<const __hip_bfloat16 *>(rhs),
                           static_cast<__hip_bfloat16 *>(out), num_elements,
                           desc, to_f, from_f);
        break;
    }
    case HIP_DTYPE_INT64: {
        auto to_f = [] __device__(int64_t v) { return static_cast<float>(v); };
        auto from_f = [] __device__(float v) { return static_cast<int64_t>(v); };
        hipLaunchKernelGGL(
            (binary_kernel<Kind, int64_t, decltype(to_f), decltype(from_f)>),
            grid, block, 0, stream,
            static_cast<const int64_t *>(lhs), static_cast<const int64_t *>(rhs),
            static_cast<int64_t *>(out), num_elements, desc, to_f, from_f);
        break;
    }
    default:
        fprintf(stderr,
                "[custom_kernels] hip_elementwise_binary: unsupported dtype=%d\n",
                hip_dtype);
        return -1;
    }
    return static_cast<int>(hipGetLastError());
}

} // namespace

extern "C" int hip_elementwise_binary(void *stream, const void *lhs,
                                      const void *rhs, void *out,
                                      int64_t num_elements, int hip_dtype,
                                      int kind, int rank,
                                      const int64_t *out_shape,
                                      const int64_t *lhs_strides_elems,
                                      const int64_t *rhs_strides_elems) {
    if (num_elements <= 0)
        return 0;
    if (!lhs || !rhs || !out || !out_shape || !lhs_strides_elems ||
        !rhs_strides_elems) {
        fprintf(stderr,
                "[custom_kernels] hip_elementwise_binary: null pointer\n");
        return -1;
    }
    if (rank <= 0 || rank > HIP_BINARY_MAX_RANK) {
        fprintf(stderr,
                "[custom_kernels] hip_elementwise_binary: rank %d out of "
                "range [1, %d]\n",
                rank, HIP_BINARY_MAX_RANK);
        return -1;
    }

    BroadcastDesc desc{};
    desc.rank = rank;
    for (int i = 0; i < rank; ++i) {
        desc.out_shape[i] = out_shape[i];
        desc.lhs_strides[i] = lhs_strides_elems[i];
        desc.rhs_strides[i] = rhs_strides_elems[i];
    }

    hipStream_t hip_stream = static_cast<hipStream_t>(stream);


    int rc = -1;
    switch (kind) {
    case HIP_BINARY_DIV:
        rc = launch_binary<HIP_BINARY_DIV>(hip_stream, lhs, rhs, out,
                                           num_elements, hip_dtype, desc);
        break;
    case HIP_BINARY_POW:
        rc = launch_binary<HIP_BINARY_POW>(hip_stream, lhs, rhs, out,
                                           num_elements, hip_dtype, desc);
        break;
    case HIP_BINARY_MUL:
        rc = launch_binary<HIP_BINARY_MUL>(hip_stream, lhs, rhs, out,
                                           num_elements, hip_dtype, desc);
        break;
    case HIP_BINARY_ADD:
        rc = launch_binary<HIP_BINARY_ADD>(hip_stream, lhs, rhs, out,
                                           num_elements, hip_dtype, desc);
        break;
    case HIP_BINARY_SUB:
        rc = launch_binary<HIP_BINARY_SUB>(hip_stream, lhs, rhs, out,
                                           num_elements, hip_dtype, desc);
        break;
    default:
        fprintf(stderr,
                "[custom_kernels] hip_elementwise_binary: unsupported kind=%d\n",
                kind);
        return -1;
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr,
                "[custom_kernels] hip_elementwise_binary launch failed: %s\n",
                hipGetErrorString(err));
        return static_cast<int>(err);
    }
    return rc;
}
