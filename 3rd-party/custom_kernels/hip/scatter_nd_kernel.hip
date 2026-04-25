/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//
// ONNX ScatterND (opset 13).  Applies per-index updates to `output` (which
// the host has already populated by copying `data` over).  One thread per
// (slice * inner_block) pair so the same kernel handles both pure overwrite
// (reduction == 0) and atomic add (reduction == 1).
//
// Index layout (matches the ONNX spec):
//   - indices_shape[:-1]                       gives N (number of slices)
//   - indices_shape[-1]                        gives coord_len (<= data_rank)
//   - data_shape[coord_len:]                   gives the inner block layout
//
// For slice i and inner offset j inside that block:
//   coord       = indices[i, :coord_len]
//   block_off   = sum_d (coord[d] * data_strides[d])   (d = 0..coord_len-1)
//   output_off  = block_off + j
//   updates_off = i * inner_block + j

#include "debug_log.h"
#include "hip_custom_kernels.h"

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>

#define HIP_SCATTER_ND_MAX_RANK 8

namespace {

struct ScatterNdDesc {
    int data_rank;
    int coord_len;          // = indices_shape[-1]
    int64_t data_shape[HIP_SCATTER_ND_MAX_RANK];     // for per-axis coord clamp
    int64_t data_strides[HIP_SCATTER_ND_MAX_RANK];   // row-major, in elements
    int64_t inner_block;    // product of data_shape[coord_len:]
    int64_t data_total;     // for global out-of-range guard
};

template <typename T>
__device__ __forceinline__ void scatter_overwrite(T *out, T v) {
    *out = v;
}

template <typename T>
__device__ __forceinline__ void scatter_atomic_add(T *out, T v);

template <>
__device__ __forceinline__ void scatter_atomic_add<float>(float *out, float v) {
    atomicAdd(out, v);
}
template <>
__device__ __forceinline__ void scatter_atomic_add<int32_t>(int32_t *out,
                                                            int32_t v) {
    atomicAdd(out, v);
}
template <>
__device__ __forceinline__ void scatter_atomic_add<int64_t>(int64_t *out,
                                                            int64_t v) {
    // HIP atomicAdd over (unsigned long long *).  Value reinterpret is the
    // standard idiom for 64-bit integer atomic adds on AMD/NV.
    atomicAdd(reinterpret_cast<unsigned long long *>(out),
              static_cast<unsigned long long>(v));
}
template <>
__device__ __forceinline__ void scatter_atomic_add<__half>(__half *out,
                                                           __half v) {
    // Promote to fp32 for the add so we don't truncate intermediate sums.
    // hipDNN/AMD provides atomicAdd for __half on RDNA3+; emulate with a
    // float CAS so this works on every arch.
    unsigned short *u = reinterpret_cast<unsigned short *>(out);
    unsigned short old_h = __half_as_ushort(*out);
    unsigned short assumed;
    do {
        assumed = old_h;
        float new_v = __half2float(__ushort_as_half(assumed)) + __half2float(v);
        old_h = atomicCAS(u, assumed, __half_as_ushort(__float2half(new_v)));
    } while (assumed != old_h);
}
template <>
__device__ __forceinline__ void scatter_atomic_add<__hip_bfloat16>(
    __hip_bfloat16 *out, __hip_bfloat16 v) {
    // Same CAS-loop trick as __half but reading/writing the raw 16-bit
    // pattern via a uint16_t alias.
    unsigned short *u = reinterpret_cast<unsigned short *>(out);
    unsigned short old_h = *u;
    unsigned short assumed;
    do {
        assumed = old_h;
        __hip_bfloat16 cur;
        *reinterpret_cast<unsigned short *>(&cur) = assumed;
        float new_v = __bfloat162float(cur) + __bfloat162float(v);
        __hip_bfloat16 nb = __float2bfloat16(new_v);
        unsigned short nb_u = *reinterpret_cast<unsigned short *>(&nb);
        old_h = atomicCAS(u, assumed, nb_u);
    } while (assumed != old_h);
}

template <typename T, typename IdxT, int Reduction>
__global__ void scatter_nd_kernel(T *__restrict__ output,
                                  const IdxT *__restrict__ indices,
                                  const T *__restrict__ updates,
                                  int64_t num_slices, ScatterNdDesc desc) {
    int64_t slice = static_cast<int64_t>(blockIdx.y) * gridDim.x +
                    static_cast<int64_t>(blockIdx.x);
    int64_t inner = static_cast<int64_t>(threadIdx.x) +
                    static_cast<int64_t>(blockIdx.z) * blockDim.x;
    if (slice >= num_slices || inner >= desc.inner_block)
        return;

    // Resolve the destination block offset from the slice's coord.  Drop
    // out-of-range coords (ONNX leaves behaviour undefined; we just no-op
    // to avoid corrupting unrelated memory).
    int64_t base = 0;
    bool drop = false;
    for (int d = 0; d < desc.coord_len; ++d) {
        IdxT raw = indices[slice * desc.coord_len + d];
        int64_t coord = static_cast<int64_t>(raw);
        if (coord < 0 || coord >= desc.data_shape[d]) {
            drop = true;
            break;
        }
        base += coord * desc.data_strides[d];
    }
    if (drop)
        return;
    int64_t out_off = base + inner;
    if (out_off < 0 || out_off >= desc.data_total)
        return;

    int64_t upd_off = slice * desc.inner_block + inner;
    T value = updates[upd_off];
    if constexpr (Reduction == 0) {
        scatter_overwrite<T>(&output[out_off], value);
    } else {
        scatter_atomic_add<T>(&output[out_off], value);
    }
}

template <typename T, typename IdxT>
int dispatch_scatter_nd(hipStream_t stream, T *output, const IdxT *indices,
                        const T *updates, int64_t num_slices,
                        const ScatterNdDesc &desc, int reduction) {
    if (num_slices <= 0 || desc.inner_block <= 0)
        return 0;
    constexpr int kBlockSize = 256;
    int64_t inner_blocks =
        (desc.inner_block + kBlockSize - 1) / kBlockSize;
    constexpr int64_t kMaxGridX = 65535;
    int64_t grid_y = 1;
    int64_t grid_x = num_slices;
    if (grid_x > kMaxGridX) {
        grid_y = (grid_x + kMaxGridX - 1) / kMaxGridX;
        grid_x = (grid_x + grid_y - 1) / grid_y;
    }
    dim3 grid(static_cast<unsigned>(grid_x), static_cast<unsigned>(grid_y),
              static_cast<unsigned>(inner_blocks));
    dim3 block(kBlockSize, 1, 1);

    if (reduction == 0)
        hipLaunchKernelGGL((scatter_nd_kernel<T, IdxT, 0>), grid, block, 0,
                           stream, output, indices, updates, num_slices, desc);
    else
        hipLaunchKernelGGL((scatter_nd_kernel<T, IdxT, 1>), grid, block, 0,
                           stream, output, indices, updates, num_slices, desc);
    return static_cast<int>(hipGetLastError());
}

template <typename T>
int dispatch_index_type(hipStream_t stream, void *output, const void *indices,
                        const void *updates, int64_t num_slices,
                        const ScatterNdDesc &desc, int indices_dtype,
                        int reduction) {
    switch (indices_dtype) {
    case HIP_DTYPE_INT32:
        return dispatch_scatter_nd<T, int32_t>(
            stream, static_cast<T *>(output),
            static_cast<const int32_t *>(indices),
            static_cast<const T *>(updates), num_slices, desc, reduction);
    case HIP_DTYPE_INT64:
        return dispatch_scatter_nd<T, int64_t>(
            stream, static_cast<T *>(output),
            static_cast<const int64_t *>(indices),
            static_cast<const T *>(updates), num_slices, desc, reduction);
    default:
        fprintf(stderr,
                "[custom_kernels] hip_scatter_nd: unsupported indices dtype=%d\n",
                indices_dtype);
        return -1;
    }
}

} // namespace

extern "C" int hip_scatter_nd(void *stream, void *output, const void *indices,
                              const void *updates, const int64_t *data_shape,
                              int64_t data_rank, const int64_t *indices_shape,
                              int64_t indices_rank, int data_hip_dtype,
                              int indices_hip_dtype, int reduction) {
    if (!output || !indices || !updates || !data_shape || !indices_shape) {
        fprintf(stderr, "[custom_kernels] hip_scatter_nd: null pointer\n");
        return -1;
    }
    if (data_rank <= 0 || data_rank > HIP_SCATTER_ND_MAX_RANK) {
        fprintf(stderr,
                "[custom_kernels] hip_scatter_nd: data_rank %lld out of range\n",
                (long long)data_rank);
        return -1;
    }
    if (indices_rank < 1 || indices_rank > HIP_SCATTER_ND_MAX_RANK) {
        fprintf(stderr,
                "[custom_kernels] hip_scatter_nd: indices_rank %lld out of "
                "range\n",
                (long long)indices_rank);
        return -1;
    }
    if (reduction != 0 && reduction != 1) {
        fprintf(stderr,
                "[custom_kernels] hip_scatter_nd: unsupported reduction=%d "
                "(only 0/none, 1/add are implemented)\n",
                reduction);
        return -1;
    }

    int64_t coord_len = indices_shape[indices_rank - 1];
    if (coord_len <= 0 || coord_len > data_rank) {
        fprintf(stderr,
                "[custom_kernels] hip_scatter_nd: coord_len %lld out of "
                "[1, data_rank=%lld]\n",
                (long long)coord_len, (long long)data_rank);
        return -1;
    }

    // Compute the row-major data strides (in elements) and copy data_shape
    // into the desc for the per-axis coord clamp inside the kernel.
    ScatterNdDesc desc{};
    desc.data_rank = static_cast<int>(data_rank);
    desc.coord_len = static_cast<int>(coord_len);
    for (int64_t d = 0; d < data_rank; ++d)
        desc.data_shape[d] = data_shape[d];
    int64_t stride = 1;
    int64_t data_total = 1;
    for (int64_t d = data_rank - 1; d >= 0; --d) {
        desc.data_strides[d] = stride;
        stride *= data_shape[d];
        data_total *= data_shape[d];
    }
    desc.data_total = data_total;

    // Inner block size = product of data_shape[coord_len:].  When coord_len
    // == data_rank it's a per-element scatter (inner_block == 1).
    int64_t inner_block = 1;
    for (int64_t d = coord_len; d < data_rank; ++d)
        inner_block *= data_shape[d];
    desc.inner_block = inner_block;

    int64_t num_slices = 1;
    for (int64_t d = 0; d < indices_rank - 1; ++d)
        num_slices *= indices_shape[d];

    hipStream_t hip_stream = static_cast<hipStream_t>(stream);
    CUSTOM_KERNELS_DEBUG_LOG(
        "[custom_kernels] hip_scatter_nd: data_rank=%lld, coord_len=%lld, "
        "num_slices=%lld, inner_block=%lld, reduction=%d\n",
        (long long)data_rank, (long long)coord_len, (long long)num_slices,
        (long long)inner_block, reduction);

    int rc = -1;
    switch (data_hip_dtype) {
    case HIP_DTYPE_FLOAT32:
        rc = dispatch_index_type<float>(hip_stream, output, indices, updates,
                                        num_slices, desc, indices_hip_dtype,
                                        reduction);
        break;
    case HIP_DTYPE_FLOAT16:
        rc = dispatch_index_type<__half>(hip_stream, output, indices, updates,
                                         num_slices, desc, indices_hip_dtype,
                                         reduction);
        break;
    case HIP_DTYPE_BFLOAT16:
        rc = dispatch_index_type<__hip_bfloat16>(
            hip_stream, output, indices, updates, num_slices, desc,
            indices_hip_dtype, reduction);
        break;
    case HIP_DTYPE_INT32:
        rc = dispatch_index_type<int32_t>(hip_stream, output, indices, updates,
                                          num_slices, desc, indices_hip_dtype,
                                          reduction);
        break;
    case HIP_DTYPE_INT64:
        rc = dispatch_index_type<int64_t>(hip_stream, output, indices, updates,
                                          num_slices, desc, indices_hip_dtype,
                                          reduction);
        break;
    default:
        fprintf(stderr,
                "[custom_kernels] hip_scatter_nd: unsupported data dtype=%d\n",
                data_hip_dtype);
        return -1;
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "[custom_kernels] hip_scatter_nd launch failed: %s\n",
                hipGetErrorString(err));
        return static_cast<int>(err);
    }
    return rc;
}
