/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//
// ONNX Where(cond, x, y) = cond ? x : y
//
// Inputs:
//   - cond: tensor of bool (i8) values, broadcastable to output
//   - x, y: tensors of any element type, broadcastable to output
//
// We dispatch by element size of x/y so the same kernel covers fp16, bf16,
// fp32, and int64 inputs.

#include "hip_custom_kernels.h"

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>

#define HIP_WHERE_MAX_RANK 8

namespace {

struct WhereDesc {
    int rank;
    int64_t out_shape[HIP_WHERE_MAX_RANK];
    int64_t cond_strides[HIP_WHERE_MAX_RANK];
    int64_t x_strides[HIP_WHERE_MAX_RANK];
    int64_t y_strides[HIP_WHERE_MAX_RANK];
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

template <typename Word>
__global__ void where_kernel(const uint8_t *__restrict__ cond,
                             const Word *__restrict__ x,
                             const Word *__restrict__ y,
                             Word *__restrict__ out, int64_t num_elements,
                             WhereDesc desc) {
    int64_t flat =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (flat >= num_elements)
        return;
    int64_t idx[HIP_WHERE_MAX_RANK];
    unflatten(flat, desc.out_shape, desc.rank, idx);
    int64_t c_off = dot_strides(idx, desc.cond_strides, desc.rank);
    int64_t x_off = dot_strides(idx, desc.x_strides, desc.rank);
    int64_t y_off = dot_strides(idx, desc.y_strides, desc.rank);
    out[flat] = cond[c_off] ? x[x_off] : y[y_off];
}

template <typename Word>
int launch_where(hipStream_t stream, const void *cond, const void *x,
                 const void *y, void *out, int64_t num_elements,
                 const WhereDesc &desc) {
    constexpr int kBlockSize = 256;
    int grid = static_cast<int>((num_elements + kBlockSize - 1) / kBlockSize);
    hipLaunchKernelGGL(where_kernel<Word>, dim3(grid), dim3(kBlockSize), 0,
                       stream, static_cast<const uint8_t *>(cond),
                       static_cast<const Word *>(x),
                       static_cast<const Word *>(y),
                       static_cast<Word *>(out), num_elements, desc);
    return static_cast<int>(hipGetLastError());
}

} // namespace

extern "C" int hip_where(void *stream, const void *cond, const void *x,
                         const void *y, void *out, int64_t num_elements,
                         int element_size_bytes, int rank,
                         const int64_t *out_shape,
                         const int64_t *cond_strides_elems,
                         const int64_t *x_strides_elems,
                         const int64_t *y_strides_elems) {
    if (num_elements <= 0)
        return 0;
    if (!cond || !x || !y || !out || !out_shape || !cond_strides_elems ||
        !x_strides_elems || !y_strides_elems) {
        fprintf(stderr, "[custom_kernels] hip_where: null pointer\n");
        return -1;
    }
    if (rank <= 0 || rank > HIP_WHERE_MAX_RANK) {
        fprintf(stderr, "[custom_kernels] hip_where: rank %d out of range\n",
                rank);
        return -1;
    }

    WhereDesc desc{};
    desc.rank = rank;
    for (int i = 0; i < rank; ++i) {
        desc.out_shape[i] = out_shape[i];
        desc.cond_strides[i] = cond_strides_elems[i];
        desc.x_strides[i] = x_strides_elems[i];
        desc.y_strides[i] = y_strides_elems[i];
    }

    hipStream_t hip_stream = static_cast<hipStream_t>(stream);

    int rc = -1;
    switch (element_size_bytes) {
    case 1:
        rc = launch_where<uint8_t>(hip_stream, cond, x, y, out, num_elements,
                                    desc);
        break;
    case 2:
        rc = launch_where<uint16_t>(hip_stream, cond, x, y, out, num_elements,
                                     desc);
        break;
    case 4:
        rc = launch_where<uint32_t>(hip_stream, cond, x, y, out, num_elements,
                                     desc);
        break;
    case 8:
        rc = launch_where<uint64_t>(hip_stream, cond, x, y, out, num_elements,
                                     desc);
        break;
    default:
        fprintf(stderr,
                "[custom_kernels] hip_where: unsupported element_size=%d\n",
                element_size_bytes);
        return -1;
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "[custom_kernels] hip_where launch failed: %s\n",
                hipGetErrorString(err));
        return static_cast<int>(err);
    }
    return rc;
}
