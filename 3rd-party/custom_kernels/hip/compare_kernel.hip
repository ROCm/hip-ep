/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//
// Element-wise comparison / logical kernel for the ONNX bool-output ops:
//
//   Equal, Greater, Less, GreaterOrEqual, And
//
// Output is always 1 byte per element (the ONNX bool storage convention).
// Inputs may be any of fp16, bf16, fp32, int32, int64; And accepts only
// boolean (i8) inputs.  Broadcasting is computed exactly like the existing
// generic binary kernel.

#include "debug_log.h"
#include "hip_custom_kernels.h"

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>

#define HIP_CMP_MAX_RANK 8

namespace {

template <hip_compare_kind_t Kind>
__device__ __forceinline__ uint8_t apply_float(float a, float b) {
    if constexpr (Kind == HIP_COMPARE_EQ)
        return a == b;
    else if constexpr (Kind == HIP_COMPARE_GT)
        return a > b;
    else if constexpr (Kind == HIP_COMPARE_LT)
        return a < b;
    else if constexpr (Kind == HIP_COMPARE_GE)
        return a >= b;
    else
        return 0;
}

template <hip_compare_kind_t Kind>
__device__ __forceinline__ uint8_t apply_int(int64_t a, int64_t b) {
    if constexpr (Kind == HIP_COMPARE_EQ)
        return a == b;
    else if constexpr (Kind == HIP_COMPARE_GT)
        return a > b;
    else if constexpr (Kind == HIP_COMPARE_LT)
        return a < b;
    else if constexpr (Kind == HIP_COMPARE_GE)
        return a >= b;
    else
        return 0;
}

struct CmpDesc {
    int rank;
    int64_t out_shape[HIP_CMP_MAX_RANK];
    int64_t lhs_strides[HIP_CMP_MAX_RANK];
    int64_t rhs_strides[HIP_CMP_MAX_RANK];
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

template <hip_compare_kind_t Kind, typename T, typename ToScalar>
__global__ void cmp_kernel_float(const T *__restrict__ lhs,
                                 const T *__restrict__ rhs,
                                 uint8_t *__restrict__ out,
                                 int64_t num_elements, CmpDesc desc,
                                 ToScalar to_scalar) {
    int64_t flat =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (flat >= num_elements)
        return;
    int64_t idx[HIP_CMP_MAX_RANK];
    unflatten(flat, desc.out_shape, desc.rank, idx);
    int64_t lhs_off = dot_strides(idx, desc.lhs_strides, desc.rank);
    int64_t rhs_off = dot_strides(idx, desc.rhs_strides, desc.rank);
    out[flat] = apply_float<Kind>(to_scalar(lhs[lhs_off]),
                                  to_scalar(rhs[rhs_off]));
}

template <hip_compare_kind_t Kind, typename T>
__global__ void cmp_kernel_int(const T *__restrict__ lhs,
                               const T *__restrict__ rhs,
                               uint8_t *__restrict__ out,
                               int64_t num_elements, CmpDesc desc) {
    int64_t flat =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (flat >= num_elements)
        return;
    int64_t idx[HIP_CMP_MAX_RANK];
    unflatten(flat, desc.out_shape, desc.rank, idx);
    int64_t lhs_off = dot_strides(idx, desc.lhs_strides, desc.rank);
    int64_t rhs_off = dot_strides(idx, desc.rhs_strides, desc.rank);
    out[flat] = apply_int<Kind>(static_cast<int64_t>(lhs[lhs_off]),
                                static_cast<int64_t>(rhs[rhs_off]));
}

template <hip_compare_kind_t Kind>
__global__ void cmp_kernel_and(const uint8_t *__restrict__ lhs,
                               const uint8_t *__restrict__ rhs,
                               uint8_t *__restrict__ out,
                               int64_t num_elements, CmpDesc desc) {
    int64_t flat =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (flat >= num_elements)
        return;
    int64_t idx[HIP_CMP_MAX_RANK];
    unflatten(flat, desc.out_shape, desc.rank, idx);
    int64_t lhs_off = dot_strides(idx, desc.lhs_strides, desc.rank);
    int64_t rhs_off = dot_strides(idx, desc.rhs_strides, desc.rank);
    out[flat] = (lhs[lhs_off] != 0) && (rhs[rhs_off] != 0);
}

template <hip_compare_kind_t Kind>
int dispatch_compare(hipStream_t stream, const void *lhs, const void *rhs,
                     void *out, int64_t num_elements, int hip_dtype,
                     const CmpDesc &desc) {
    constexpr int kBlockSize = 256;
    int grid = static_cast<int>((num_elements + kBlockSize - 1) / kBlockSize);
    switch (hip_dtype) {
    case HIP_DTYPE_FLOAT32: {
        auto id = [] __device__(float v) { return v; };
        hipLaunchKernelGGL((cmp_kernel_float<Kind, float, decltype(id)>),
                           dim3(grid), dim3(kBlockSize), 0, stream,
                           static_cast<const float *>(lhs),
                           static_cast<const float *>(rhs),
                           static_cast<uint8_t *>(out), num_elements, desc, id);
        break;
    }
    case HIP_DTYPE_FLOAT16: {
        auto to = [] __device__(__half v) { return __half2float(v); };
        hipLaunchKernelGGL((cmp_kernel_float<Kind, __half, decltype(to)>),
                           dim3(grid), dim3(kBlockSize), 0, stream,
                           static_cast<const __half *>(lhs),
                           static_cast<const __half *>(rhs),
                           static_cast<uint8_t *>(out), num_elements, desc, to);
        break;
    }
    case HIP_DTYPE_BFLOAT16: {
        auto to =
            [] __device__(__hip_bfloat16 v) { return __bfloat162float(v); };
        hipLaunchKernelGGL((cmp_kernel_float<Kind, __hip_bfloat16, decltype(to)>),
                           dim3(grid), dim3(kBlockSize), 0, stream,
                           static_cast<const __hip_bfloat16 *>(lhs),
                           static_cast<const __hip_bfloat16 *>(rhs),
                           static_cast<uint8_t *>(out), num_elements, desc, to);
        break;
    }
    case HIP_DTYPE_INT32:
        hipLaunchKernelGGL(
            (cmp_kernel_int<Kind, int32_t>), dim3(grid), dim3(kBlockSize), 0,
            stream, static_cast<const int32_t *>(lhs),
            static_cast<const int32_t *>(rhs),
            static_cast<uint8_t *>(out), num_elements, desc);
        break;
    case HIP_DTYPE_INT64:
        hipLaunchKernelGGL(
            (cmp_kernel_int<Kind, int64_t>), dim3(grid), dim3(kBlockSize), 0,
            stream, static_cast<const int64_t *>(lhs),
            static_cast<const int64_t *>(rhs),
            static_cast<uint8_t *>(out), num_elements, desc);
        break;
    default:
        fprintf(stderr,
                "[custom_kernels] hip_compare: unsupported dtype=%d\n",
                hip_dtype);
        return -1;
    }
    return static_cast<int>(hipGetLastError());
}

} // namespace

extern "C" int hip_compare(void *stream, const void *lhs, const void *rhs,
                           void *out, int64_t num_elements, int hip_dtype,
                           int kind, int rank, const int64_t *out_shape,
                           const int64_t *lhs_strides_elems,
                           const int64_t *rhs_strides_elems) {
    if (num_elements <= 0)
        return 0;
    if (!lhs || !rhs || !out || !out_shape || !lhs_strides_elems ||
        !rhs_strides_elems) {
        fprintf(stderr, "[custom_kernels] hip_compare: null pointer\n");
        return -1;
    }
    if (rank <= 0 || rank > HIP_CMP_MAX_RANK) {
        fprintf(stderr, "[custom_kernels] hip_compare: rank %d out of range\n",
                rank);
        return -1;
    }

    CmpDesc desc{};
    desc.rank = rank;
    for (int i = 0; i < rank; ++i) {
        desc.out_shape[i] = out_shape[i];
        desc.lhs_strides[i] = lhs_strides_elems[i];
        desc.rhs_strides[i] = rhs_strides_elems[i];
    }

    hipStream_t hip_stream = static_cast<hipStream_t>(stream);
    CUSTOM_KERNELS_DEBUG_LOG(
        "[custom_kernels] hip_compare: kind=%d, dtype=%d, n=%lld, rank=%d\n",
        kind, hip_dtype, (long long)num_elements, rank);

    int rc = -1;
    switch (kind) {
    case HIP_COMPARE_EQ:
        rc = dispatch_compare<HIP_COMPARE_EQ>(hip_stream, lhs, rhs, out,
                                              num_elements, hip_dtype, desc);
        break;
    case HIP_COMPARE_GT:
        rc = dispatch_compare<HIP_COMPARE_GT>(hip_stream, lhs, rhs, out,
                                              num_elements, hip_dtype, desc);
        break;
    case HIP_COMPARE_LT:
        rc = dispatch_compare<HIP_COMPARE_LT>(hip_stream, lhs, rhs, out,
                                              num_elements, hip_dtype, desc);
        break;
    case HIP_COMPARE_GE:
        rc = dispatch_compare<HIP_COMPARE_GE>(hip_stream, lhs, rhs, out,
                                              num_elements, hip_dtype, desc);
        break;
    case HIP_COMPARE_AND: {
        // And only takes bool (i8) inputs; ignore hip_dtype.
        constexpr int kBlockSize = 256;
        int grid =
            static_cast<int>((num_elements + kBlockSize - 1) / kBlockSize);
        hipLaunchKernelGGL((cmp_kernel_and<HIP_COMPARE_AND>), dim3(grid),
                           dim3(kBlockSize), 0, hip_stream,
                           static_cast<const uint8_t *>(lhs),
                           static_cast<const uint8_t *>(rhs),
                           static_cast<uint8_t *>(out), num_elements, desc);
        rc = static_cast<int>(hipGetLastError());
        break;
    }
    default:
        fprintf(stderr, "[custom_kernels] hip_compare: unsupported kind=%d\n",
                kind);
        return -1;
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "[custom_kernels] hip_compare launch failed: %s\n",
                hipGetErrorString(err));
        return static_cast<int>(err);
    }
    return rc;
}
