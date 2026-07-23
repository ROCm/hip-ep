/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip_custom_kernels.h"
#include "debug_log.h"

#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>

// Maximum tensor rank supported by the transpose kernel. Empirically, ONNX
// models very rarely exceed 6 dimensions; cap at 8 for headroom while keeping
// the kernel parameter struct small enough to fit in the launch ABI.
static constexpr int kTransposeMaxRank = 8;

namespace {

// Compile-time-bounded permutation metadata passed by value to the kernel.
struct TransposeMeta {
    int rank;
    int64_t input_strides[kTransposeMaxRank];
    int64_t output_shape[kTransposeMaxRank];
    int64_t perm[kTransposeMaxRank];
};

template <typename T>
__global__ void transpose_kernel(const T* __restrict__ input,
                                 T* __restrict__ output,
                                 int64_t num_elements,
                                 TransposeMeta meta) {
    int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= num_elements) return;

    // Decompose tid into output coordinates (row-major: last dim is innermost).
    int64_t out_coords[kTransposeMaxRank];
    int64_t remaining = tid;
    for (int i = meta.rank - 1; i >= 0; --i) {
        out_coords[i] = remaining % meta.output_shape[i];
        remaining /= meta.output_shape[i];
    }

    // Map output coords back to input linear index using perm:
    //   output[i_0, ..., i_{r-1}] = input[i_{perm^{-1}[0]}, ..., i_{perm^{-1}[r-1]}]
    // Equivalently: in_coord[perm[i]] = out_coord[i].
    int64_t in_idx = 0;
    for (int i = 0; i < meta.rank; ++i) {
        int64_t in_dim = meta.perm[i];
        in_idx += out_coords[i] * meta.input_strides[in_dim];
    }
    output[tid] = input[in_idx];
}

// Tiled, fully-coalesced batched 2-D transpose. Transposing is pure data
// movement, so a single kernel templated on an unsigned integer of the element
// width (uint8/16/32/64_t for 1/2/4/8-byte dtypes such as int8, fp16/bf16,
// fp32/int32, int64/fp64) is dtype-agnostic. Each grid.z slice transposes one
// row-major [rows x cols] matrix into a row-major [cols x rows] matrix. Both the
// load and the store go through a bank-conflict-padded LDS tile so every global
// access is coalesced, unlike transpose_kernel above whose per-output-element
// gather issues a strided (uncoalesced) read whenever the innermost dim changes
// -- the case that dominates the linear-attention [1, S, F] <-> [1, F, S]
// transposes and runs ~15-30x slower than memory bandwidth. Mirrors the proven
// tiled transpose used for WMMA layout conversion in matmul_nbits_kernel.hip.
template <typename T, int TILE>
__global__ void transpose_tiled_kernel(const T* __restrict__ src,
                                       T* __restrict__ dst,
                                       int rows, int cols) {
    __shared__ T tile[TILE][TILE + 1];

    const int64_t slice = static_cast<int64_t>(blockIdx.z) * rows * cols;
    const T* s = src + slice;
    T* d = dst + slice;

    // Load a [TILE(rows) x TILE(cols)] block of src, coalesced along cols.
    int x = blockIdx.x * TILE + threadIdx.x; // src column
    int y = blockIdx.y * TILE + threadIdx.y; // src row
    for (int j = 0; j < TILE; j += blockDim.y) {
        int yy = y + j;
        if (x < cols && yy < rows)
            tile[threadIdx.y + j][threadIdx.x] =
                s[static_cast<int64_t>(yy) * cols + x];
    }
    __syncthreads();

    // Store the transposed block, coalesced along rows (dst's innermost dim).
    int xo = blockIdx.y * TILE + threadIdx.x; // dst column (= src row)
    int yo = blockIdx.x * TILE + threadIdx.y; // dst row    (= src col)
    for (int j = 0; j < TILE; j += blockDim.y) {
        int yy = yo + j;
        if (xo < rows && yy < cols)
            d[static_cast<int64_t>(yy) * rows + xo] =
                tile[threadIdx.x][threadIdx.y + j];
    }
}

}  // namespace

// Fast path for a batched last-two-dim transpose of 1/2/4/8-byte elements.
// Returns 0 on success, 1 if it declines the config (caller must fall back to
// the generic hip_transpose), or a hipError_t on launch failure. gridDim.z is
// capped at 65535 by the HW, so batches larger than that decline; unsupported
// element widths also decline.
extern "C" int hip_transpose_2d_tiled(void* stream, const void* input,
                                      void* output, int64_t batch,
                                      int64_t rows, int64_t cols,
                                      int element_size_bytes) {
    if (!input || !output) {
        fprintf(stderr,
                "[custom_kernels] hip_transpose_2d_tiled: null pointer\n");
        return -1;
    }
    if (batch <= 0 || rows <= 0 || cols <= 0)
        return 0; // nothing to do
    if (batch > 65535)
        return 1; // decline: exceeds gridDim.z limit

    constexpr int TILE = 32;
    dim3 block(TILE, 8);
    dim3 grid(static_cast<unsigned>((cols + TILE - 1) / TILE),
              static_cast<unsigned>((rows + TILE - 1) / TILE),
              static_cast<unsigned>(batch));
    hipStream_t hip_stream = static_cast<hipStream_t>(stream);
    const int r = static_cast<int>(rows);
    const int c = static_cast<int>(cols);
    switch (element_size_bytes) {
        case 1:
            hipLaunchKernelGGL(
                (transpose_tiled_kernel<uint8_t, TILE>), grid, block, 0,
                hip_stream, static_cast<const uint8_t*>(input),
                static_cast<uint8_t*>(output), r, c);
            break;
        case 2:
            hipLaunchKernelGGL(
                (transpose_tiled_kernel<uint16_t, TILE>), grid, block, 0,
                hip_stream, static_cast<const uint16_t*>(input),
                static_cast<uint16_t*>(output), r, c);
            break;
        case 4:
            hipLaunchKernelGGL(
                (transpose_tiled_kernel<uint32_t, TILE>), grid, block, 0,
                hip_stream, static_cast<const uint32_t*>(input),
                static_cast<uint32_t*>(output), r, c);
            break;
        case 8:
            hipLaunchKernelGGL(
                (transpose_tiled_kernel<uint64_t, TILE>), grid, block, 0,
                hip_stream, static_cast<const uint64_t*>(input),
                static_cast<uint64_t*>(output), r, c);
            break;
        default:
            return 1; // decline: unsupported element width -> generic fallback
    }
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr,
                "[custom_kernels] hip_transpose_2d_tiled launch failed: %s\n",
                hipGetErrorString(err));
        return static_cast<int>(err);
    }
    return 0;
}

extern "C" int hip_transpose(void* stream,
                             const void* input,
                             void* output,
                             int64_t rank,
                             const int64_t* input_shape,
                             const int64_t* perm,
                             int64_t num_elements,
                             int element_size_bytes) {
    if (!input || !output || !input_shape || !perm) {
        fprintf(stderr, "[custom_kernels] hip_transpose: null pointer argument\n");
        return -1;
    }
    if (num_elements <= 0) {
        return 0;
    }
    if (rank <= 0 || rank > kTransposeMaxRank) {
        fprintf(stderr,
                "[custom_kernels] hip_transpose: rank=%lld out of supported "
                "range [1, %d]\n",
                (long long)rank, kTransposeMaxRank);
        return -1;
    }

    TransposeMeta meta{};
    meta.rank = static_cast<int>(rank);
    // Row-major strides: input_strides[r-1] = 1; input_strides[i] = stride[i+1] * shape[i+1].
    meta.input_strides[rank - 1] = 1;
    for (int i = static_cast<int>(rank) - 2; i >= 0; --i) {
        meta.input_strides[i] = meta.input_strides[i + 1] * input_shape[i + 1];
    }
    for (int i = 0; i < rank; ++i) {
        int64_t p = perm[i];
        if (p < 0 || p >= rank) {
            fprintf(stderr,
                    "[custom_kernels] hip_transpose: perm[%d]=%lld out of range\n",
                    i, (long long)p);
            return -1;
        }
        meta.perm[i] = p;
        meta.output_shape[i] = input_shape[p];
    }

    hipStream_t hip_stream = static_cast<hipStream_t>(stream);
    constexpr int kBlockSize = 256;
    int64_t grid_size_i64 = (num_elements + kBlockSize - 1) / kBlockSize;
    if (grid_size_i64 <= 0) return 0;
    int grid_size = static_cast<int>(grid_size_i64);

    CUSTOM_KERNELS_DEBUG_LOG(
        "[custom_kernels] hip_transpose: rank=%lld, num_elements=%lld, "
        "elem_size=%d, grid=%d, block=%d\n",
        (long long)rank, (long long)num_elements, element_size_bytes,
        grid_size, kBlockSize);

    switch (element_size_bytes) {
        case 1:
            hipLaunchKernelGGL(transpose_kernel<uint8_t>,
                               dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                               static_cast<const uint8_t*>(input),
                               static_cast<uint8_t*>(output),
                               num_elements, meta);
            break;
        case 2:
            hipLaunchKernelGGL(transpose_kernel<uint16_t>,
                               dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                               static_cast<const uint16_t*>(input),
                               static_cast<uint16_t*>(output),
                               num_elements, meta);
            break;
        case 4:
            hipLaunchKernelGGL(transpose_kernel<uint32_t>,
                               dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                               static_cast<const uint32_t*>(input),
                               static_cast<uint32_t*>(output),
                               num_elements, meta);
            break;
        case 8:
            hipLaunchKernelGGL(transpose_kernel<uint64_t>,
                               dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                               static_cast<const uint64_t*>(input),
                               static_cast<uint64_t*>(output),
                               num_elements, meta);
            break;
        default:
            fprintf(stderr,
                    "[custom_kernels] hip_transpose: unsupported "
                    "element_size_bytes=%d\n",
                    element_size_bytes);
            return -1;
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "[custom_kernels] hip_transpose launch failed: %s\n",
                hipGetErrorString(err));
    }
    return static_cast<int>(err);
}
