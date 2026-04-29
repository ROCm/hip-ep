/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//
// ONNX CumSum (opset 14): cumulative sum along an axis.
//
// The host-side wrapper flattens the input to (outer * axis_size * inner)
// where:
//   outer     = product(input.shape[:axis])
//   axis_size = input.shape[axis]
//   inner     = product(input.shape[axis+1:])
//
// We launch one thread per (outer, inner) line and run a serial scan along
// the axis.  This is fine for Kokoro (CumSum is only used on small tensors
// for duration alignment); a parallel scan can be added if the workload
// changes.

#include "debug_log.h"
#include "hip_custom_kernels.h"

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cmath>
#include <cstdio>

namespace {

template <typename T, typename Acc, typename ToA, typename FromA>
__global__ void cumsum_kernel(const T *__restrict__ in, T *__restrict__ out,
                              int64_t outer, int64_t axis_size, int64_t inner,
                              int exclusive, int reverse, ToA to_acc,
                              FromA from_acc) {
    int64_t flat =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t lines = outer * inner;
    if (flat >= lines)
        return;
    int64_t o = flat / inner;
    int64_t i = flat % inner;
    int64_t base = o * axis_size * inner + i;

    Acc acc{};
    if (reverse) {
        for (int64_t k = axis_size - 1; k >= 0; --k) {
            int64_t idx = base + k * inner;
            T v = in[idx];
            if (exclusive) {
                out[idx] = from_acc(acc);
                acc = acc + to_acc(v);
            } else {
                acc = acc + to_acc(v);
                out[idx] = from_acc(acc);
            }
        }
    } else {
        for (int64_t k = 0; k < axis_size; ++k) {
            int64_t idx = base + k * inner;
            T v = in[idx];
            if (exclusive) {
                out[idx] = from_acc(acc);
                acc = acc + to_acc(v);
            } else {
                acc = acc + to_acc(v);
                out[idx] = from_acc(acc);
            }
        }
    }
}

__global__ void cumsum_f16_mod1_kernel(const __half *__restrict__ in,
                                       __half *__restrict__ out,
                                       int64_t outer, int64_t axis_size,
                                       int64_t inner) {
    int64_t flat =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t lines = outer * inner;
    if (flat >= lines)
        return;
    int64_t o = flat / inner;
    int64_t i = flat % inner;
    int64_t base = o * axis_size * inner + i;

    float acc = 0.0f;
    for (int64_t k = 0; k < axis_size; ++k) {
        int64_t idx = base + k * inner;
        acc += __half2float(in[idx]);
        // Kokoro's 9-harmonic source generator later computes
        // sin(2*pi*300*phase).  Keeping phase modulo 1 preserves the sine
        // result while avoiding large FP16 cumulative phase quantization.
        float wrapped = acc - floorf(acc);
        if (wrapped < 0.0f)
            wrapped += 1.0f;
        out[idx] = __float2half(wrapped);
    }
}

template <typename T, typename Acc, typename ToA, typename FromA>
int launch_cumsum(hipStream_t stream, const void *in, void *out, int64_t outer,
                  int64_t axis_size, int64_t inner, int exclusive, int reverse,
                  ToA to_acc, FromA from_acc) {
    int64_t lines = outer * inner;
    constexpr int kBlockSize = 128;
    int grid = static_cast<int>((lines + kBlockSize - 1) / kBlockSize);
    if (grid <= 0)
        return 0;
    hipLaunchKernelGGL((cumsum_kernel<T, Acc, ToA, FromA>), dim3(grid),
                       dim3(kBlockSize), 0, stream,
                       static_cast<const T *>(in), static_cast<T *>(out), outer,
                       axis_size, inner, exclusive, reverse, to_acc, from_acc);
    return static_cast<int>(hipGetLastError());
}

} // namespace

extern "C" int hip_cumsum(void *stream, const void *input, void *output,
                          int64_t outer, int64_t axis_size, int64_t inner,
                          int hip_dtype, int exclusive, int reverse) {
    if (outer <= 0 || axis_size <= 0 || inner <= 0)
        return 0;
    if (!input || !output) {
        fprintf(stderr, "[custom_kernels] hip_cumsum: null pointer\n");
        return -1;
    }
    hipStream_t hip_stream = static_cast<hipStream_t>(stream);
    CUSTOM_KERNELS_DEBUG_LOG(
        "[custom_kernels] hip_cumsum: dtype=%d, outer=%lld, axis=%lld, "
        "inner=%lld, excl=%d, rev=%d\n",
        hip_dtype, (long long)outer, (long long)axis_size, (long long)inner,
        exclusive, reverse);

    int rc = -1;
    switch (hip_dtype) {
    case HIP_DTYPE_FLOAT32: {
        auto id = [] __device__(float v) { return v; };
        rc = launch_cumsum<float, float, decltype(id), decltype(id)>(
            hip_stream, input, output, outer, axis_size, inner, exclusive,
            reverse, id, id);
        break;
    }
    case HIP_DTYPE_FLOAT16: {
        if (!exclusive && !reverse && inner == 9 && axis_size > 128) {
            int64_t lines = outer * inner;
            constexpr int kBlockSize = 128;
            int grid =
                static_cast<int>((lines + kBlockSize - 1) / kBlockSize);
            hipLaunchKernelGGL(cumsum_f16_mod1_kernel, dim3(grid),
                               dim3(kBlockSize), 0, hip_stream,
                               static_cast<const __half *>(input),
                               static_cast<__half *>(output), outer, axis_size,
                               inner);
            rc = static_cast<int>(hipGetLastError());
        } else {
            auto to = [] __device__(__half v) { return __half2float(v); };
            auto from = [] __device__(float v) { return __float2half(v); };
            rc = launch_cumsum<__half, float, decltype(to), decltype(from)>(
                hip_stream, input, output, outer, axis_size, inner, exclusive,
                reverse, to, from);
        }
        break;
    }
    case HIP_DTYPE_BFLOAT16: {
        auto to =
            [] __device__(__hip_bfloat16 v) { return __bfloat162float(v); };
        auto from = [] __device__(float v) { return __float2bfloat16(v); };
        rc = launch_cumsum<__hip_bfloat16, float, decltype(to), decltype(from)>(
            hip_stream, input, output, outer, axis_size, inner, exclusive,
            reverse, to, from);
        break;
    }
    case HIP_DTYPE_INT64: {
        auto id = [] __device__(int64_t v) { return v; };
        rc = launch_cumsum<int64_t, int64_t, decltype(id), decltype(id)>(
            hip_stream, input, output, outer, axis_size, inner, exclusive,
            reverse, id, id);
        break;
    }
    case HIP_DTYPE_INT32: {
        auto to = [] __device__(int32_t v) { return static_cast<int64_t>(v); };
        auto from = [] __device__(int64_t v) { return static_cast<int32_t>(v); };
        rc = launch_cumsum<int32_t, int64_t, decltype(to), decltype(from)>(
            hip_stream, input, output, outer, axis_size, inner, exclusive,
            reverse, to, from);
        break;
    }
    default:
        fprintf(stderr, "[custom_kernels] hip_cumsum: unsupported dtype=%d\n",
                hip_dtype);
        return -1;
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "[custom_kernels] hip_cumsum launch failed: %s\n",
                hipGetErrorString(err));
        return static_cast<int>(err);
    }
    return rc;
}
