/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//
// Generic N-D transpose kernel.  Swaps two dimensions of a contiguous
// row-major tensor.  One thread per output element.

#include "hip_custom_kernels.h"

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>

#define HIP_TRANSPOSE_MAX_RANK 8

namespace {

struct TPArgs {
    int64_t in_shape[HIP_TRANSPOSE_MAX_RANK];
    int64_t out_strides[HIP_TRANSPOSE_MAX_RANK];
    int64_t in_strides[HIP_TRANSPOSE_MAX_RANK];
    int rank;
    int dim0;
    int dim1;
    int64_t total;
};

template <typename T>
__global__ void transpose_kernel(const T *__restrict__ in, T *__restrict__ out,
                                 TPArgs args) {
    int64_t flat_y = (int64_t)blockIdx.y * gridDim.x * blockDim.x;
    int64_t flat_x = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t flat = flat_y + flat_x;
    if (flat >= args.total)
        return;

    int64_t coord[HIP_TRANSPOSE_MAX_RANK];
    int64_t rem = flat;
    for (int i = 0; i < args.rank; ++i) {
        coord[i] = rem / args.out_strides[i];
        rem -= coord[i] * args.out_strides[i];
    }
    int64_t tmp = coord[args.dim0];
    coord[args.dim0] = coord[args.dim1];
    coord[args.dim1] = tmp;

    int64_t in_off = 0;
    for (int i = 0; i < args.rank; ++i)
        in_off += coord[i] * args.in_strides[i];

    out[flat] = in[in_off];
}

template <typename T>
int launch_transpose(hipStream_t stream, const void *in, void *out,
                     const TPArgs &args) {
    if (args.total <= 0)
        return 0;
    (void)hipGetLastError();
    constexpr int kBlock = 256;
    int64_t blocks = (args.total + kBlock - 1) / kBlock;
    int64_t grid_x = (blocks > 65535) ? 65535 : blocks;
    int64_t grid_y = (blocks + grid_x - 1) / grid_x;

    transpose_kernel<T><<<dim3((unsigned)grid_x, (unsigned)grid_y, 1),
                          dim3(kBlock, 1, 1), 0, stream>>>(
        static_cast<const T *>(in), static_cast<T *>(out), args);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr,
                "[custom_kernels] hip_transpose_nd launch failed: %s\n",
                hipGetErrorString(err));
        return static_cast<int>(err);
    }
    return 0;
}

} // namespace

extern "C" int hip_transpose_nd(void *stream, const void *input, void *output,
                                int rank, const int64_t *in_shape, int dim0,
                                int dim1, int hip_dtype) {
    if (!input || !output || !in_shape) {
        fprintf(stderr, "[custom_kernels] hip_transpose_nd: null pointer\n");
        return -1;
    }
    if (rank < 1 || rank > HIP_TRANSPOSE_MAX_RANK) {
        fprintf(stderr,
                "[custom_kernels] hip_transpose_nd: rank %d out of range\n",
                rank);
        return -1;
    }
    if (dim0 < 0 || dim0 >= rank || dim1 < 0 || dim1 >= rank) {
        fprintf(stderr,
                "[custom_kernels] hip_transpose_nd: bad dims (%d,%d)\n",
                dim0, dim1);
        return -1;
    }

    int64_t total = 1;
    for (int i = 0; i < rank; ++i)
        total *= in_shape[i];

    if (dim0 == dim1 || total <= 0) {
        if (total > 0) {
            size_t elem_size = 0;
            switch (hip_dtype) {
            case HIP_DTYPE_FLOAT32: case HIP_DTYPE_INT32: elem_size = 4; break;
            case HIP_DTYPE_FLOAT16: case HIP_DTYPE_BFLOAT16: elem_size = 2; break;
            case HIP_DTYPE_INT64: case HIP_DTYPE_FLOAT64: elem_size = 8; break;
            default: return -1;
            }
            return static_cast<int>(hipMemcpyAsync(
                output, input, total * elem_size, hipMemcpyDeviceToDevice,
                static_cast<hipStream_t>(stream)));
        }
        return 0;
    }

    TPArgs args{};
    for (int i = 0; i < rank; ++i)
        args.in_shape[i] = in_shape[i];
    args.rank = rank;
    args.dim0 = dim0;
    args.dim1 = dim1;
    args.total = total;

    int64_t out_shape[HIP_TRANSPOSE_MAX_RANK];
    for (int i = 0; i < rank; ++i) out_shape[i] = in_shape[i];
    { int64_t t = out_shape[dim0]; out_shape[dim0] = out_shape[dim1];
      out_shape[dim1] = t; }

    args.in_strides[rank - 1] = 1;
    args.out_strides[rank - 1] = 1;
    for (int i = rank - 2; i >= 0; --i) {
        args.in_strides[i] = args.in_strides[i + 1] * in_shape[i + 1];
        args.out_strides[i] = args.out_strides[i + 1] * out_shape[i + 1];
    }

    hipStream_t hip_stream = static_cast<hipStream_t>(stream);

    switch (hip_dtype) {
    case HIP_DTYPE_FLOAT32:
    case HIP_DTYPE_INT32:
        return launch_transpose<int32_t>(hip_stream, input, output, args);
    case HIP_DTYPE_FLOAT16:
    case HIP_DTYPE_BFLOAT16:
        return launch_transpose<int16_t>(hip_stream, input, output, args);
    case HIP_DTYPE_INT64:
    case HIP_DTYPE_FLOAT64:
        return launch_transpose<int64_t>(hip_stream, input, output, args);
    default:
        fprintf(stderr,
                "[custom_kernels] hip_transpose_nd: unsupported dtype %d\n",
                hip_dtype);
        return -1;
    }
}
