/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//
// ONNX NonZero (opset 13): given an N-D `input`, write the row-major N-D
// coordinates of every nonzero element column-by-column into `output`
// (shape [rank, k_max]).
//
// We use a single-pass atomic counter:
//   1. Host clears the counter to 0 and zero-fills `output` (so unused
//      slots read back as the zero coord).
//   2. Each thread inspects one input element.  Nonzero -> grab a slot
//      via atomicAdd(counter), compute the N-D coord by integer divmod
//      against `in_shape`, and write `output[d, slot] = coord[d]`.
//   3. Slots beyond k_max are silently dropped (matches the
//      "best-effort with static K bound" contract documented in
//      hip_custom_kernels.h).

#include "hip_custom_kernels.h"

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>

#define HIP_NONZERO_MAX_RANK 8

namespace {

struct NonzeroDesc {
    int rank;
    int64_t in_shape[HIP_NONZERO_MAX_RANK];
};

template <typename T>
__device__ __forceinline__ bool is_nonzero(T v);

template <>
__device__ __forceinline__ bool is_nonzero<float>(float v) {
    return v != 0.0f;
}
template <>
__device__ __forceinline__ bool is_nonzero<__half>(__half v) {
    return __half2float(v) != 0.0f;
}
template <>
__device__ __forceinline__ bool is_nonzero<__hip_bfloat16>(__hip_bfloat16 v) {
    return __bfloat162float(v) != 0.0f;
}
template <>
__device__ __forceinline__ bool is_nonzero<int32_t>(int32_t v) {
    return v != 0;
}
template <>
__device__ __forceinline__ bool is_nonzero<int64_t>(int64_t v) {
    return v != 0;
}
template <>
__device__ __forceinline__ bool is_nonzero<int8_t>(int8_t v) {
    return v != 0;
}
template <>
__device__ __forceinline__ bool is_nonzero<uint8_t>(uint8_t v) {
    return v != 0;
}

template <typename T>
__global__ void nonzero_kernel(const T *__restrict__ input,
                               int64_t *__restrict__ output,
                               unsigned long long *__restrict__ counter,
                               int64_t total_elements, int64_t k_max,
                               NonzeroDesc desc) {
    int64_t flat = static_cast<int64_t>(blockIdx.y) * gridDim.x * blockDim.x +
                   static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (flat >= total_elements)
        return;
    if (!is_nonzero<T>(input[flat]))
        return;

    // Reserve a slot.  Cap silently when the worst-case bound is too small.
    unsigned long long slot_u = atomicAdd(counter, 1ULL);
    int64_t slot = static_cast<int64_t>(slot_u);
    if (slot >= k_max)
        return;

    // Decode flat -> per-axis coords (row-major, innermost varies fastest).
    int64_t rem = flat;
    for (int d = desc.rank - 1; d >= 0; --d) {
        int64_t s = desc.in_shape[d];
        int64_t coord = rem % s;
        rem /= s;
        // output is laid out row-major as [rank, k_max] -- writing axis d's
        // coord at column `slot` is `output[d * k_max + slot]`.
        output[d * k_max + slot] = coord;
    }
}

template <typename T>
int launch_nonzero(hipStream_t stream, const void *input, void *output,
                   void *counter, int64_t total_elements, int64_t k_max,
                   const NonzeroDesc &desc) {
    if (total_elements <= 0)
        return 0;
    constexpr int kBlockSize = 256;
    int64_t blocks = (total_elements + kBlockSize - 1) / kBlockSize;
    constexpr int64_t kMaxGridX = 65535;
    int64_t grid_x = blocks > kMaxGridX ? kMaxGridX : blocks;
    int64_t grid_y = (blocks + kMaxGridX - 1) / kMaxGridX;
    dim3 grid(static_cast<unsigned>(grid_x), static_cast<unsigned>(grid_y), 1);
    dim3 block(kBlockSize, 1, 1);
    hipLaunchKernelGGL(nonzero_kernel<T>, grid, block, 0, stream,
                       static_cast<const T *>(input),
                       static_cast<int64_t *>(output),
                       static_cast<unsigned long long *>(counter),
                       total_elements, k_max, desc);
    return static_cast<int>(hipGetLastError());
}

} // namespace

extern "C" int hip_nonzero(void *stream, const void *input, void *output,
                           void *k_dev_counter, const int64_t *in_shape,
                           int64_t rank, int64_t total_elements, int64_t k_max,
                           int hip_dtype) {
    if (!input || !output || !k_dev_counter || !in_shape) {
        fprintf(stderr, "[custom_kernels] hip_nonzero: null pointer\n");
        return -1;
    }
    if (rank <= 0 || rank > HIP_NONZERO_MAX_RANK) {
        fprintf(stderr,
                "[custom_kernels] hip_nonzero: rank %lld out of range [1, %d]\n",
                (long long)rank, HIP_NONZERO_MAX_RANK);
        return -1;
    }
    if (total_elements < 0 || k_max <= 0) {
        fprintf(stderr,
                "[custom_kernels] hip_nonzero: invalid total=%lld / k_max=%lld\n",
                (long long)total_elements, (long long)k_max);
        return -1;
    }

    NonzeroDesc desc{};
    desc.rank = static_cast<int>(rank);
    int64_t check_total = 1;
    for (int64_t i = 0; i < rank; ++i) {
        desc.in_shape[i] = in_shape[i];
        check_total *= in_shape[i];
    }
    if (check_total != total_elements) {
        fprintf(stderr,
                "[custom_kernels] hip_nonzero: in_shape product %lld != "
                "total_elements %lld\n",
                (long long)check_total, (long long)total_elements);
        return -1;
    }

    hipStream_t hip_stream = static_cast<hipStream_t>(stream);

    // The host has zero-filled output and counter; just dispatch.
    int rc = -1;
    switch (hip_dtype) {
    case HIP_DTYPE_FLOAT32:
        rc = launch_nonzero<float>(hip_stream, input, output, k_dev_counter,
                                   total_elements, k_max, desc);
        break;
    case HIP_DTYPE_FLOAT16:
        rc = launch_nonzero<__half>(hip_stream, input, output, k_dev_counter,
                                    total_elements, k_max, desc);
        break;
    case HIP_DTYPE_BFLOAT16:
        rc = launch_nonzero<__hip_bfloat16>(hip_stream, input, output,
                                            k_dev_counter, total_elements,
                                            k_max, desc);
        break;
    case HIP_DTYPE_INT32:
        rc = launch_nonzero<int32_t>(hip_stream, input, output, k_dev_counter,
                                     total_elements, k_max, desc);
        break;
    case HIP_DTYPE_INT64:
        rc = launch_nonzero<int64_t>(hip_stream, input, output, k_dev_counter,
                                     total_elements, k_max, desc);
        break;
    default:
        // Treat any other 1-byte type (including bool / ui8) as int8.
        // Kokoro's NonZero feeds on a bool produced by Greater and ONNX
        // serialises booleans as ui8.
        rc = launch_nonzero<uint8_t>(hip_stream, input, output, k_dev_counter,
                                     total_elements, k_max, desc);
        break;
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "[custom_kernels] hip_nonzero launch failed: %s\n",
                hipGetErrorString(err));
        return static_cast<int>(err);
    }
    return rc;
}
