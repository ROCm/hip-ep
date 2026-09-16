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

// Division by a runtime constant, without dividing. The generic kernel below
// decomposes a flat index with one division per rank, per element; with native
// 64-bit division that is the dominant cost of the whole transpose. Every
// divisor is an output extent, fixed for the launch, so each becomes a
// multiply-high and a shift. Valid for numerators below 2^31, which the caller
// checks before selecting this path.
struct TDiv {
    uint32_t mul; // 0 => a pure shift (divisor is a power of two)
    uint32_t shr;
};

__device__ __forceinline__ uint32_t tFastDiv(uint32_t n, TDiv d) {
    return d.mul ? (__umulhi(n, d.mul) >> d.shr) : (n >> d.shr);
}

static TDiv makeTDiv(int64_t denom) {
    const uint32_t d = static_cast<uint32_t>(denom < 1 ? 1 : denom);
    TDiv r{0u, 0u};
    uint32_t l = 0;
    while ((1u << l) < d) ++l;
    if ((1u << l) == d) {
        r.shr = l;
        return r;
    }
    const uint32_t p = 31 + l;
    r.mul = static_cast<uint32_t>(((1ull << p) + d - 1) / d);
    r.shr = p - 32;
    return r;
}

struct TransposeMetaFD {
    int rank;
    int64_t input_strides[kTransposeMaxRank];
    uint32_t output_shape[kTransposeMaxRank];
    TDiv r_output_shape[kTransposeMaxRank];
    int32_t perm[kTransposeMaxRank];
};

// Generic gather, with the index decode done in 32-bit reciprocal arithmetic.
template <typename T>
__global__ void transpose_kernel_fd(const T* __restrict__ input,
                                    T* __restrict__ output,
                                    uint32_t num_elements,
                                    TransposeMetaFD meta) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_elements) return;

    int64_t in_idx = 0;
    uint32_t remaining = tid;
    for (int i = meta.rank - 1; i >= 0; --i) {
        const uint32_t q = tFastDiv(remaining, meta.r_output_shape[i]);
        const uint32_t c = remaining - q * meta.output_shape[i];
        remaining = q;
        in_idx += static_cast<int64_t>(c) * meta.input_strides[meta.perm[i]];
    }
    output[tid] = input[in_idx];
}

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

    for (int i = 0; i < rank; ++i) {
        if (perm[i] < 0 || perm[i] >= rank) {
            fprintf(stderr,
                    "[custom_kernels] hip_transpose: perm[%d]=%lld out of range\n",
                    i, (long long)perm[i]);
            return -1;
        }
    }

    hipStream_t hip_stream = static_cast<hipStream_t>(stream);

    // -----------------------------------------------------------------------
    // Canonicalise the permutation before choosing a kernel.
    //
    // The shape a transpose arrives in is an artefact of how the graph was
    // written, not of the data movement it asks for. Two rewrites reduce it to
    // the movement's actual rank without changing a single element's
    // destination:
    //
    //   * An axis of extent 1 contributes nothing to any address, so it drops.
    //   * Two input axes that are adjacent *and* stay adjacent and in order in
    //     the output are never separated, so the elements they span are one
    //     contiguous run: they merge into a single axis.
    //
    // This matters because it is what exposes the fast path. The patch-embed
    // output permute arrives as rank 4, [1,64,64,1152] by [0,3,1,2], and the
    // generic gather runs it at a small fraction of memory bandwidth -- its
    // innermost output axis is not the innermost input axis, so every read is a
    // separate transaction. Canonicalised it is [4096,1152] by [1,0]: a plain
    // 2-D transpose, which the tiled kernel above already does at bandwidth
    // through LDS. Same for an identity permute wrapped in reshapes, which
    // collapses to rank 1 and is just a copy.
    // -----------------------------------------------------------------------
    int64_t cShape[kTransposeMaxRank];
    int cPerm[kTransposeMaxRank];
    int cRank = 0;
    {
        // Drop extent-1 axes, renumbering the surviving ones.
        int remap[kTransposeMaxRank];
        for (int a = 0; a < rank; ++a) {
            remap[a] = (input_shape[a] > 1) ? cRank : -1;
            if (input_shape[a] > 1) cShape[cRank++] = input_shape[a];
        }
        int n = 0;
        for (int i = 0; i < rank; ++i)
            if (remap[perm[i]] >= 0) cPerm[n++] = remap[perm[i]];
        // n == cRank: both count the axes with extent > 1.

        // Merge runs that stay adjacent and ascending in the output.
        for (bool changed = true; changed;) {
            changed = false;
            for (int i = 0; i + 1 < cRank; ++i) {
                const int a = cPerm[i];
                if (cPerm[i + 1] != a + 1) continue;
                cShape[a] *= cShape[a + 1];
                for (int j = a + 1; j + 1 < cRank; ++j) cShape[j] = cShape[j + 1];
                for (int j = i + 1; j + 1 < cRank; ++j) cPerm[j] = cPerm[j + 1];
                --cRank;
                for (int j = 0; j < cRank; ++j)
                    if (cPerm[j] > a) --cPerm[j];
                changed = true;
                break;
            }
        }
    }

    // Rank 0 or 1 can only be the identity: the data is already in place.
    if (cRank <= 1) {
        hipError_t err = hipMemcpyAsync(
            output, input,
            static_cast<size_t>(num_elements) * element_size_bytes,
            hipMemcpyDeviceToDevice, hip_stream);
        if (err != hipSuccess) {
            fprintf(stderr, "[custom_kernels] hip_transpose copy failed: %s\n",
                    hipGetErrorString(err));
            return static_cast<int>(err);
        }
        return 0;
    }

    // A batched swap of the two innermost axes is what the tiled kernel does.
    const bool two_d = (cRank == 2 && cPerm[0] == 1 && cPerm[1] == 0);
    const bool batched_two_d =
        (cRank == 3 && cPerm[0] == 0 && cPerm[1] == 2 && cPerm[2] == 1);
    if (two_d || batched_two_d) {
        const int64_t batch = two_d ? 1 : cShape[0];
        const int64_t rows = two_d ? cShape[0] : cShape[1];
        const int64_t cols = two_d ? cShape[1] : cShape[2];
        const int rc = hip_transpose_2d_tiled(stream, input, output, batch, rows,
                                              cols, element_size_bytes);
        if (rc == 0) return 0;
        if (rc < 0) return rc;
        // rc == 1: declined (batch over the gridDim.z cap, or an element width
        // it does not handle). Fall through to the generic gather.
    }

    TransposeMeta meta{};
    meta.rank = cRank;
    // Row-major strides: input_strides[r-1] = 1; input_strides[i] = stride[i+1] * shape[i+1].
    meta.input_strides[cRank - 1] = 1;
    for (int i = cRank - 2; i >= 0; --i) {
        meta.input_strides[i] = meta.input_strides[i + 1] * cShape[i + 1];
    }
    for (int i = 0; i < cRank; ++i) {
        meta.perm[i] = cPerm[i];
        meta.output_shape[i] = cShape[cPerm[i]];
    }

    // 32-bit reciprocal decode where the flat index allows it, which is every
    // tensor that is not enormous.
    if (num_elements < (1LL << 31)) {
        TransposeMetaFD fd{};
        fd.rank = cRank;
        for (int i = 0; i < cRank; ++i) {
            fd.input_strides[i] = meta.input_strides[i];
            fd.perm[i] = cPerm[i];
            fd.output_shape[i] = static_cast<uint32_t>(meta.output_shape[i]);
            fd.r_output_shape[i] = makeTDiv(meta.output_shape[i]);
        }
        const uint32_t n = static_cast<uint32_t>(num_elements);
        constexpr int kBlock = 256;
        const unsigned g = static_cast<unsigned>((num_elements + kBlock - 1) / kBlock);
        switch (element_size_bytes) {
            case 1:
                hipLaunchKernelGGL(transpose_kernel_fd<uint8_t>, dim3(g),
                                   dim3(kBlock), 0, hip_stream,
                                   static_cast<const uint8_t*>(input),
                                   static_cast<uint8_t*>(output), n, fd);
                break;
            case 2:
                hipLaunchKernelGGL(transpose_kernel_fd<uint16_t>, dim3(g),
                                   dim3(kBlock), 0, hip_stream,
                                   static_cast<const uint16_t*>(input),
                                   static_cast<uint16_t*>(output), n, fd);
                break;
            case 4:
                hipLaunchKernelGGL(transpose_kernel_fd<uint32_t>, dim3(g),
                                   dim3(kBlock), 0, hip_stream,
                                   static_cast<const uint32_t*>(input),
                                   static_cast<uint32_t*>(output), n, fd);
                break;
            case 8:
                hipLaunchKernelGGL(transpose_kernel_fd<uint64_t>, dim3(g),
                                   dim3(kBlock), 0, hip_stream,
                                   static_cast<const uint64_t*>(input),
                                   static_cast<uint64_t*>(output), n, fd);
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
