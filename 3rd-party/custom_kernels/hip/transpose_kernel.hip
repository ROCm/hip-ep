/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//
// Generic N-D transpose kernel.  Swaps two dimensions of a contiguous
// row-major tensor.  One thread per output element decodes its flat
// index against the OUTPUT shape (which is the input shape with dim0
// and dim1 swapped) and computes the source linear offset by mapping
// the per-axis output coord back through the swap.
//
// The host-side launcher takes:
//   - input / output device pointers
//   - rank (1..HIP_TRANSPOSE_MAX_RANK)
//   - in_shape[rank]
//   - dim0, dim1 (the two axes to swap; if dim0 == dim1 the kernel is
//     a no-op memcpy)
//   - hip_dtype (HIP_DTYPE_*).  Implementation copies sizeof(elem)
//     bytes per thread, so any POD element type works as long as it's
//     listed.
//
// Grid.x is capped at 65535; we tile across grid.y to support outputs
// > 65535 * blockDim.x elements.

#include "debug_log.h"
#include "hip_custom_kernels.h"

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>

#define HIP_TRANSPOSE_MAX_RANK 8

namespace {

template <typename T>
__global__ void transpose_kernel(const T *__restrict__ in, T *__restrict__ out,
                                 int rank,
                                 int64_t in_shape0, int64_t in_shape1,
                                 int64_t in_shape2, int64_t in_shape3,
                                 int64_t in_shape4, int64_t in_shape5,
                                 int64_t in_shape6, int64_t in_shape7,
                                 int dim0, int dim1, int64_t total) {
    int64_t in_shape_arr[HIP_TRANSPOSE_MAX_RANK] = {
        in_shape0, in_shape1, in_shape2, in_shape3,
        in_shape4, in_shape5, in_shape6, in_shape7};
    int64_t out_shape_arr[HIP_TRANSPOSE_MAX_RANK];
    for (int i = 0; i < rank; ++i)
        out_shape_arr[i] = in_shape_arr[i];
    int64_t tmp = out_shape_arr[dim0];
    out_shape_arr[dim0] = out_shape_arr[dim1];
    out_shape_arr[dim1] = tmp;

    int64_t in_strides[HIP_TRANSPOSE_MAX_RANK];
    int64_t s = 1;
    for (int i = rank - 1; i >= 0; --i) {
        in_strides[i] = s;
        s *= in_shape_arr[i];
    }
    int64_t out_strides[HIP_TRANSPOSE_MAX_RANK];
    s = 1;
    for (int i = rank - 1; i >= 0; --i) {
        out_strides[i] = s;
        s *= out_shape_arr[i];
    }

    int64_t flat_y =
        (int64_t)blockIdx.y * gridDim.x * blockDim.x;
    int64_t flat_x = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t flat = flat_y + flat_x;
    if (flat >= total)
        return;

    // Decode flat into out_coord.
    int64_t coord[HIP_TRANSPOSE_MAX_RANK];
    int64_t rem = flat;
    for (int i = 0; i < rank; ++i) {
        coord[i] = rem / out_strides[i];
        rem -= coord[i] * out_strides[i];
    }
    // Swap to in_coord by undoing the dim0/dim1 swap.
    int64_t tmpc = coord[dim0];
    coord[dim0] = coord[dim1];
    coord[dim1] = tmpc;
    int64_t in_off = 0;
    for (int i = 0; i < rank; ++i)
        in_off += coord[i] * in_strides[i];

    out[flat] = in[in_off];
}

template <typename T>
int launch_transpose(hipStream_t stream, const void *in, void *out, int rank,
                     const int64_t *in_shape, int dim0, int dim1) {
    int64_t total = 1;
    for (int i = 0; i < rank; ++i)
        total *= in_shape[i];
    if (total <= 0)
        return 0;
    constexpr int kBlock = 128;
    int64_t blocks = (total + kBlock - 1) / kBlock;
    int64_t grid_x = (blocks > 65535) ? 65535 : blocks;
    int64_t grid_y = (blocks + 65535 - 1) / 65535;

    int64_t shape[HIP_TRANSPOSE_MAX_RANK] = {1, 1, 1, 1, 1, 1, 1, 1};
    for (int i = 0; i < rank; ++i)
        shape[i] = in_shape[i];

    hipLaunchKernelGGL(
        (transpose_kernel<T>), dim3((unsigned)grid_x, (unsigned)grid_y, 1),
        dim3(kBlock, 1, 1), 0, stream, static_cast<const T *>(in),
        static_cast<T *>(out), rank, shape[0], shape[1], shape[2], shape[3],
        shape[4], shape[5], shape[6], shape[7], dim0, dim1, total);
    return static_cast<int>(hipGetLastError());
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
    if (dim0 == dim1) {
        // Identity: bytewise copy of total elements.
        int64_t total = 1;
        for (int i = 0; i < rank; ++i)
            total *= in_shape[i];
        size_t elem_size = 0;
        switch (hip_dtype) {
        case HIP_DTYPE_FLOAT32:
        case HIP_DTYPE_INT32:
            elem_size = 4;
            break;
        case HIP_DTYPE_FLOAT16:
        case HIP_DTYPE_BFLOAT16:
            elem_size = 2;
            break;
        case HIP_DTYPE_INT64:
        case HIP_DTYPE_FLOAT64:
            elem_size = 8;
            break;
        default:
            fprintf(stderr,
                    "[custom_kernels] hip_transpose_nd: unsupported dtype %d\n",
                    hip_dtype);
            return -1;
        }
        return static_cast<int>(hipMemcpyAsync(
            output, input, total * elem_size, hipMemcpyDeviceToDevice,
            static_cast<hipStream_t>(stream)));
    }

    hipStream_t hip_stream = static_cast<hipStream_t>(stream);
    CUSTOM_KERNELS_DEBUG_LOG(
        "[custom_kernels] hip_transpose_nd: dtype=%d, rank=%d, dim0=%d, dim1=%d\n",
        hip_dtype, rank, dim0, dim1);

    int rc = -1;
    switch (hip_dtype) {
    case HIP_DTYPE_FLOAT32:
    case HIP_DTYPE_INT32:
        rc = launch_transpose<int32_t>(hip_stream, input, output, rank,
                                        in_shape, dim0, dim1);
        break;
    case HIP_DTYPE_FLOAT16:
    case HIP_DTYPE_BFLOAT16:
        rc = launch_transpose<int16_t>(hip_stream, input, output, rank,
                                        in_shape, dim0, dim1);
        break;
    case HIP_DTYPE_INT64:
    case HIP_DTYPE_FLOAT64:
        rc = launch_transpose<int64_t>(hip_stream, input, output, rank,
                                        in_shape, dim0, dim1);
        break;
    default:
        fprintf(stderr,
                "[custom_kernels] hip_transpose_nd: unsupported dtype %d\n",
                hip_dtype);
        return -1;
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr,
                "[custom_kernels] hip_transpose_nd launch failed: %s\n",
                hipGetErrorString(err));
        return static_cast<int>(err);
    }
    return rc;
}
