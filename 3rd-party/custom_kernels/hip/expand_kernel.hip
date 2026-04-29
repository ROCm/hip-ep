/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//
// ONNX Expand: broadcast `input` to a larger shape using numpy-style
// right-aligned broadcasting.
//
// The host-side wrapper does the broadcast bookkeeping:
//   - Right-aligns the input shape to the output rank by virtually
//     prepending size-1 dims.
//   - Builds an `effective_in_strides` array (length = out_rank) where
//     each entry is the input element-stride for that axis, or 0 when
//     the corresponding (aligned) input dim is 1 (broadcast axis).
//
// With those tables in hand the kernel is just one thread per output
// element:
//   in_offset = sum_d (out_coord[d] * effective_in_strides[d])
//   output[flat] = input[in_offset]
//
// Element-type dispatch is by raw byte width (1/2/4/8) so f16/bf16/f32
// and i32/i64 all share the same code paths.

#include "hip_custom_kernels.h"

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>

#define HIP_EXPAND_MAX_RANK 8

namespace {

struct ExpandDesc {
    int rank;
    int64_t out_shape[HIP_EXPAND_MAX_RANK];
    int64_t in_strides[HIP_EXPAND_MAX_RANK]; // effective strides (0 = broadcast)
};

template <typename Word>
__global__ void expand_kernel(const Word *__restrict__ input,
                              Word *__restrict__ output, int64_t num_elements,
                              ExpandDesc desc) {
    int64_t flat = static_cast<int64_t>(blockIdx.y) * gridDim.x * blockDim.x +
                   static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (flat >= num_elements)
        return;
    int64_t in_off = 0;
    int64_t rem = flat;
    for (int d = desc.rank - 1; d >= 0; --d) {
        int64_t s = desc.out_shape[d];
        int64_t coord = rem % s;
        rem /= s;
        in_off += coord * desc.in_strides[d];
    }
    output[flat] = input[in_off];
}

template <typename Word>
int launch_expand(hipStream_t stream, const void *input, void *output,
                  int64_t num_elements, const ExpandDesc &desc) {
    constexpr int kBlockSize = 256;
    int64_t blocks = (num_elements + kBlockSize - 1) / kBlockSize;
    if (blocks <= 0)
        return 0;
    // Tile across grid.x (capped at 65535 like other kernels) and grid.y.
    constexpr int kMaxGridX = 65535;
    int grid_x = static_cast<int>(blocks > kMaxGridX ? kMaxGridX : blocks);
    int grid_y = static_cast<int>((blocks + kMaxGridX - 1) / kMaxGridX);
    hipLaunchKernelGGL(expand_kernel<Word>, dim3(grid_x, grid_y),
                       dim3(kBlockSize), 0, stream,
                       static_cast<const Word *>(input),
                       static_cast<Word *>(output), num_elements, desc);
    return static_cast<int>(hipGetLastError());
}

} // namespace

extern "C" int hip_expand(void *stream, const void *input, void *output,
                          const int64_t *in_shape,
                          const int64_t *in_strides_elems,
                          const int64_t *out_shape, int64_t rank,
                          int hip_dtype) {
    if (!input || !output || !in_shape || !in_strides_elems || !out_shape) {
        fprintf(stderr, "[custom_kernels] hip_expand: null pointer\n");
        return -1;
    }
    if (rank <= 0 || rank > HIP_EXPAND_MAX_RANK) {
        fprintf(stderr,
                "[custom_kernels] hip_expand: rank %lld out of range [1, %d]\n",
                (long long)rank, HIP_EXPAND_MAX_RANK);
        return -1;
    }

    ExpandDesc desc{};
    desc.rank = static_cast<int>(rank);
    int64_t num_elements = 1;
    for (int64_t i = 0; i < rank; ++i) {
        desc.out_shape[i] = out_shape[i];
        desc.in_strides[i] = in_strides_elems[i];
        num_elements *= out_shape[i];
    }
    if (num_elements <= 0)
        return 0;

    // in_shape is currently informational only; the broadcast is fully
    // encoded by `in_strides_elems` (0 == broadcast).  We still accept it
    // so the kernel ABI matches the wrapper's view of the world and so
    // future debug builds can sanity-check it.
    (void)in_shape;

    hipStream_t hip_stream = static_cast<hipStream_t>(stream);

    int element_size_bytes = -1;
    switch (hip_dtype) {
    case HIP_DTYPE_FLOAT32:
    case HIP_DTYPE_INT32:
        element_size_bytes = 4;
        break;
    case HIP_DTYPE_FLOAT16:
    case HIP_DTYPE_BFLOAT16:
        element_size_bytes = 2;
        break;
    case HIP_DTYPE_INT64:
    case HIP_DTYPE_FLOAT64:
        element_size_bytes = 8;
        break;
    default:
        fprintf(stderr, "[custom_kernels] hip_expand: unsupported dtype=%d\n",
                hip_dtype);
        return -1;
    }

    int rc = -1;
    switch (element_size_bytes) {
    case 2:
        rc = launch_expand<uint16_t>(hip_stream, input, output, num_elements,
                                     desc);
        break;
    case 4:
        rc = launch_expand<uint32_t>(hip_stream, input, output, num_elements,
                                     desc);
        break;
    case 8:
        rc = launch_expand<uint64_t>(hip_stream, input, output, num_elements,
                                     desc);
        break;
    default:
        fprintf(stderr,
                "[custom_kernels] hip_expand: unsupported element_size=%d\n",
                element_size_bytes);
        return -1;
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "[custom_kernels] hip_expand launch failed: %s\n",
                hipGetErrorString(err));
        return static_cast<int>(err);
    }
    return rc;
}
