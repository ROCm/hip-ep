/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//
// Strided gather kernel for ONNX Slice.  All slice parameters (starts,
// steps) are normalised on the host into per-dim "input offset = start +
// out_idx * step" relations and passed as element-stride arrays.
//
// We intentionally don't try to specialise per-dtype semantics: the kernel
// only cares about element_size_bytes (1, 2, 4, or 8) and uses the matching
// integer type for the copy.  That keeps slice fast for f16/bf16/f32/i64
// without one specialised kernel per dtype.

#include "debug_log.h"
#include "hip_custom_kernels.h"

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>

#define HIP_SLICE_MAX_RANK 8

namespace {

struct SliceDesc {
    int rank;
    int64_t out_shape[HIP_SLICE_MAX_RANK];
    int64_t in_strides[HIP_SLICE_MAX_RANK]; // elements per step in input
    int64_t in_start_off;                   // sum of starts*in_stride
};

template <typename Word>
__global__ void slice_kernel(const Word *__restrict__ input,
                             Word *__restrict__ output, int64_t num_elements,
                             SliceDesc desc) {
    int64_t flat =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (flat >= num_elements)
        return;
    int64_t in_off = desc.in_start_off;
    int64_t rem = flat;
    for (int d = desc.rank - 1; d >= 0; --d) {
        int64_t s = desc.out_shape[d];
        int64_t coord = rem % s;
        rem /= s;
        in_off += coord * desc.in_strides[d];
    }
    output[flat] = input[in_off];
}

} // namespace

extern "C" int hip_slice(void *stream, const void *input, void *output,
                         int64_t num_elements, int element_size_bytes,
                         int rank, const int64_t *out_shape,
                         const int64_t *in_strides_elems,
                         const int64_t *starts_elems) {
    if (num_elements <= 0)
        return 0;
    if (!input || !output || !out_shape || !in_strides_elems || !starts_elems) {
        fprintf(stderr, "[custom_kernels] hip_slice: null pointer\n");
        return -1;
    }
    if (rank <= 0 || rank > HIP_SLICE_MAX_RANK) {
        fprintf(stderr,
                "[custom_kernels] hip_slice: rank %d out of range [1, %d]\n",
                rank, HIP_SLICE_MAX_RANK);
        return -1;
    }

    SliceDesc desc{};
    desc.rank = rank;
    desc.in_start_off = 0;
    for (int i = 0; i < rank; ++i) {
        desc.out_shape[i] = out_shape[i];
        desc.in_strides[i] = in_strides_elems[i];
        desc.in_start_off += starts_elems[i] * in_strides_elems[i];
    }

    hipStream_t hip_stream = static_cast<hipStream_t>(stream);
    constexpr int kBlockSize = 256;
    int grid_size =
        static_cast<int>((num_elements + kBlockSize - 1) / kBlockSize);

    CUSTOM_KERNELS_DEBUG_LOG(
        "[custom_kernels] hip_slice: n=%lld, rank=%d, elem=%d\n",
        (long long)num_elements, rank, element_size_bytes);

    switch (element_size_bytes) {
    case 1:
        hipLaunchKernelGGL(slice_kernel<uint8_t>, dim3(grid_size),
                           dim3(kBlockSize), 0, hip_stream,
                           static_cast<const uint8_t *>(input),
                           static_cast<uint8_t *>(output), num_elements, desc);
        break;
    case 2:
        hipLaunchKernelGGL(slice_kernel<uint16_t>, dim3(grid_size),
                           dim3(kBlockSize), 0, hip_stream,
                           static_cast<const uint16_t *>(input),
                           static_cast<uint16_t *>(output), num_elements, desc);
        break;
    case 4:
        hipLaunchKernelGGL(slice_kernel<uint32_t>, dim3(grid_size),
                           dim3(kBlockSize), 0, hip_stream,
                           static_cast<const uint32_t *>(input),
                           static_cast<uint32_t *>(output), num_elements, desc);
        break;
    case 8:
        hipLaunchKernelGGL(slice_kernel<uint64_t>, dim3(grid_size),
                           dim3(kBlockSize), 0, hip_stream,
                           static_cast<const uint64_t *>(input),
                           static_cast<uint64_t *>(output), num_elements, desc);
        break;
    default:
        fprintf(stderr,
                "[custom_kernels] hip_slice: unsupported element_size=%d\n",
                element_size_bytes);
        return -1;
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "[custom_kernels] hip_slice launch failed: %s\n",
                hipGetErrorString(err));
        return static_cast<int>(err);
    }
    return 0;
}
