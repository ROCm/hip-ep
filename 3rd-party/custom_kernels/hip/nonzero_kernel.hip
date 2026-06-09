/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip_custom_kernels.h"
#include "debug_log.h"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <cstdio>

// Maximum tensor rank supported by the NonZero kernel. ONNX models very
// rarely exceed 6 dims; cap at 8 (matches kTransposeMaxRank / kTileMaxRank /
// kGatherNDMaxRank) so the per-launch metadata struct stays small.
static constexpr int kNonZeroMaxRank = 8;

// Single-block compaction block size. MUST be a power of two (the in-block
// Hillis-Steele scan below assumes it).
static constexpr int kNonZeroBlockSize = 256;

namespace {

// Compile-time-bounded shape metadata passed by value to the kernel.
struct NonZeroMeta {
    int rank;
    int64_t capacity;
    int64_t dims[kNonZeroMaxRank];
    int64_t strides[kNonZeroMaxRank]; // row-major strides over `dims`
};

// Non-zero predicate. For floats, NaN is non-zero and -0.0 is zero, which is
// exactly what IEEE `!= 0` gives (NaN != 0 -> true, -0.0 != 0 -> false). This
// matches ONNX NonZero semantics. Overloaded so the kernel template can stay
// dtype-agnostic.
__device__ __forceinline__ bool nonzero_value(float v) { return v != 0.0f; }
__device__ __forceinline__ bool nonzero_value(__half v) {
    return __half2float(v) != 0.0f;
}
__device__ __forceinline__ bool nonzero_value(int64_t v) { return v != 0; }
__device__ __forceinline__ bool nonzero_value(int32_t v) { return v != 0; }
__device__ __forceinline__ bool nonzero_value(uint8_t v) { return v != 0; }

// One cooperative block. Walks the input in tiles of blockDim.x elements; for
// each tile it computes a per-thread {0,1} predicate, runs an in-block
// exclusive prefix sum to assign each non-zero element its output column, then
// unravels the flat index into row-major coordinates and writes one int64 per
// axis. A shared running base carries the compacted count across tiles, so the
// global output order is the increasing flat-index order ONNX requires.
template <typename T>
__global__ void nonzero_kernel(const T* __restrict__ input,
                               int64_t* __restrict__ output,
                               int64_t num_elements, NonZeroMeta meta) {
    __shared__ int s_scan[kNonZeroBlockSize];
    __shared__ int64_t s_base; // non-zero elements emitted by prior tiles

    const int tid = threadIdx.x;
    if (tid == 0)
        s_base = 0;
    __syncthreads();

    for (int64_t tile_start = 0; tile_start < num_elements;
         tile_start += blockDim.x) {
        const int64_t idx = tile_start + tid;

        int p = 0;
        if (idx < num_elements)
            p = nonzero_value(input[idx]) ? 1 : 0;
        s_scan[tid] = p;
        __syncthreads();

        // In-block inclusive scan (Hillis-Steele). Power-of-two blockDim.
        for (int offset = 1; offset < blockDim.x; offset <<= 1) {
            int add = 0;
            if (tid >= offset)
                add = s_scan[tid - offset];
            __syncthreads();
            s_scan[tid] += add;
            __syncthreads();
        }

        const int inclusive = s_scan[tid];
        const int exclusive = inclusive - p; // this thread's local column
        const int tile_total = s_scan[blockDim.x - 1];

        if (p) {
            const int64_t out_col = s_base + exclusive;
            // Row-major unravel: coord[axis] = (idx / stride[axis]) % dim[axis].
            for (int axis = 0; axis < meta.rank; ++axis) {
                const int64_t coord =
                    (idx / meta.strides[axis]) % meta.dims[axis];
                output[axis * meta.capacity + out_col] = coord;
            }
        }
        __syncthreads();
        if (tid == 0)
            s_base += tile_total;
        __syncthreads();
    }
}

template <typename T>
void launch_nonzero(hipStream_t stream, const void* input, void* output,
                    int64_t num_elements, const NonZeroMeta& meta) {
    hipLaunchKernelGGL(nonzero_kernel<T>, dim3(1), dim3(kNonZeroBlockSize), 0,
                       stream, static_cast<const T*>(input),
                       static_cast<int64_t*>(output), num_elements, meta);
}

} // namespace

extern "C" int hip_nonzero(void* stream, const void* input, void* output,
                           const int64_t* input_shape, int64_t rank,
                           int64_t num_elements, int64_t capacity,
                           int element_size_bytes, int is_float) {
    if (!input || !output || !input_shape) {
        fprintf(stderr, "[custom_kernels] hip_nonzero: null pointer argument\n");
        return -1;
    }
    if (rank <= 0 || rank > kNonZeroMaxRank) {
        fprintf(stderr,
                "[custom_kernels] hip_nonzero: rank=%lld out of supported "
                "range [1, %d]\n",
                (long long)rank, kNonZeroMaxRank);
        return -1;
    }

    hipStream_t hip_stream = static_cast<hipStream_t>(stream);

    // Zero-fill the (over-allocated) output so the undefined tail [N, capacity)
    // is deterministic. Done even when num_elements == 0.
    hipError_t memset_err = hipMemsetAsync(
        output, 0, static_cast<size_t>(rank) * static_cast<size_t>(capacity) *
                       sizeof(int64_t),
        hip_stream);
    if (memset_err != hipSuccess) {
        fprintf(stderr, "[custom_kernels] hip_nonzero: hipMemsetAsync failed: %s\n",
                hipGetErrorString(memset_err));
        return static_cast<int>(memset_err);
    }
    if (num_elements <= 0)
        return 0;

    NonZeroMeta meta{};
    meta.rank = static_cast<int>(rank);
    meta.capacity = capacity;
    for (int i = 0; i < static_cast<int>(rank); ++i)
        meta.dims[i] = input_shape[i];
    // Row-major strides: strides[r-1] = 1; strides[i] = strides[i+1]*dims[i+1].
    meta.strides[rank - 1] = 1;
    for (int i = static_cast<int>(rank) - 2; i >= 0; --i)
        meta.strides[i] = meta.strides[i + 1] * meta.dims[i + 1];

    CUSTOM_KERNELS_DEBUG_LOG(
        "[custom_kernels] hip_nonzero: rank=%lld, num_elements=%lld, "
        "capacity=%lld, elem_size=%d, is_float=%d\n",
        (long long)rank, (long long)num_elements, (long long)capacity,
        element_size_bytes, is_float);

    // Launch errors are surfaced via hipGetLastError() below, so the launcher
    // returns void; the switches only select the typed instantiation.
    if (is_float) {
        switch (element_size_bytes) {
        case 4:
            launch_nonzero<float>(hip_stream, input, output, num_elements, meta);
            break;
        case 2:
            launch_nonzero<__half>(hip_stream, input, output, num_elements,
                                   meta);
            break;
        default:
            fprintf(stderr,
                    "[custom_kernels] hip_nonzero: unsupported float "
                    "element_size_bytes=%d\n",
                    element_size_bytes);
            return -1;
        }
    } else {
        switch (element_size_bytes) {
        case 8:
            launch_nonzero<int64_t>(hip_stream, input, output, num_elements,
                                    meta);
            break;
        case 4:
            launch_nonzero<int32_t>(hip_stream, input, output, num_elements,
                                    meta);
            break;
        case 1:
            // Covers i1 / i8 / ui8: the != 0 predicate is sign-agnostic.
            launch_nonzero<uint8_t>(hip_stream, input, output, num_elements,
                                    meta);
            break;
        default:
            fprintf(stderr,
                    "[custom_kernels] hip_nonzero: unsupported int "
                    "element_size_bytes=%d\n",
                    element_size_bytes);
            return -1;
        }
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "[custom_kernels] hip_nonzero launch failed: %s\n",
                hipGetErrorString(err));
    }
    return static_cast<int>(err);
}
