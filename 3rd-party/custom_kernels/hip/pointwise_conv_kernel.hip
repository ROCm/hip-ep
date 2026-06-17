/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip_custom_kernels.h"
#include "debug_log.h"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include <cstdint>
#include <cstdio>

// =============================================================================
// Pointwise (1x1) convolution: fused GEMM + bias
// =============================================================================
//
// A pointwise (1x1, stride 1, no pad, no dilation, group 1) convolution is a
// per-pixel channel mixing:
//
//   Y[n, co, p] = bias[co] + sum_{ci} X[n, ci, p] * W[co, ci]
//
// where p ranges over the HW = H*W spatial positions. The weight tensor
// [Cout, Cin, 1, 1] collapses to [Cout, Cin] and the activation
// [N, Cin, H, W] collapses to [N, Cin, HW] (both are zero-cost metadata
// reshapes upstream). This is mathematically a batched GEMM
// W[Cout,Cin] @ X[Cin,HW] -> Y[Cout,HW] per batch element.
//
// Why a dedicated kernel instead of hipBLASLt + a separate bias-add op:
//   - For the small channel counts typical of decoder head/projection 1x1
//     convs (Cin on the order of tens), the GEMM is extremely "skinny" in the
//     reduction dimension. A general GEMM library's tiling is tuned for large
//     K and pays disproportionate per-call launch / workspace / algorithm
//     overhead here, so the millisecond-scale launch cost dwarfs the
//     microsecond-scale FLOPs.
//   - Folding the bias add into the same kernel removes one full-output kernel
//     launch and one full-output read+write pass over the (largest) tensor.
//
// The op is memory bound (the output write dominates), so the design just
// maximizes coalesced traffic and keeps the reduction in registers:
//   - grid  = (ceil(HW / BLOCK), Cout, N); block = BLOCK threads along HW.
//   - One block owns a single (n, co); all its threads share the same weight
//     row W[co, :], staged once into LDS (Cin elements, tiny).
//   - Thread t computes output position p = blockIdx.x*BLOCK + t. Adjacent
//     threads touch adjacent p, so both the per-channel X reads X[n,ci,p] and
//     the Y[n,co,p] write are fully coalesced.
//   - Accumulation is in float regardless of storage dtype to avoid fp16/bf16
//     precision loss across the channel reduction.
//
// Restriction: this kernel is selected upstream only for small Cin (see the
// HIP->LLVM lowering); larger Cin is routed to the GEMM-library path where its
// tiling wins. There is no correctness dependence on Cin being small — the
// loop handles any Cin — it is purely a performance routing choice.

namespace {

constexpr int kBlock = 256;

template <typename T> __device__ inline float pwc_to_float(T v);
template <> __device__ inline float pwc_to_float<float>(float v) { return v; }
template <> __device__ inline float pwc_to_float<__half>(__half v) {
  return __half2float(v);
}
template <>
__device__ inline float pwc_to_float<hip_bfloat16>(hip_bfloat16 v) {
  return static_cast<float>(v);
}

template <typename T> __device__ inline T pwc_from_float(float v);
template <> __device__ inline float pwc_from_float<float>(float v) { return v; }
template <> __device__ inline __half pwc_from_float<__half>(float v) {
  return __float2half(v);
}
template <>
__device__ inline hip_bfloat16 pwc_from_float<hip_bfloat16>(float v) {
  return hip_bfloat16(v);
}

template <typename T>
__global__ void pointwise_conv_kernel(const T* __restrict__ x,
                                      const T* __restrict__ w,
                                      const T* __restrict__ bias,
                                      T* __restrict__ y, int64_t Cin,
                                      int64_t Cout, int64_t HW) {
  // Stage this block's weight row W[co, :] into LDS once. Cin is small (the
  // routing precondition), so a fixed upper bound keeps the allocation static.
  __shared__ float w_row[1024];

  const int64_t co = blockIdx.y;
  const int64_t n = blockIdx.z;

  for (int64_t c = threadIdx.x; c < Cin; c += blockDim.x)
    w_row[c] = pwc_to_float<T>(w[co * Cin + c]);
  __syncthreads();

  const int64_t p = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (p >= HW)
    return;

  // X[n, :, p] and Y[n, co, p] base offsets. X is [N, Cin, HW] so successive
  // channels are HW apart; Y is [N, Cout, HW].
  const T* x_np = x + n * Cin * HW + p;
  float acc = (bias != nullptr) ? pwc_to_float<T>(bias[co]) : 0.0f;
  for (int64_t c = 0; c < Cin; ++c)
    acc += pwc_to_float<T>(x_np[c * HW]) * w_row[c];

  y[(n * Cout + co) * HW + p] = pwc_from_float<T>(acc);
}

template <typename T>
int launch_for_dtype(hipStream_t stream, const void* x, const void* w,
                     const void* bias, void* y, int64_t N, int64_t Cin,
                     int64_t Cout, int64_t HW) {
  dim3 grid(static_cast<unsigned>((HW + kBlock - 1) / kBlock),
            static_cast<unsigned>(Cout), static_cast<unsigned>(N));
  dim3 block(kBlock);
  hipLaunchKernelGGL((pointwise_conv_kernel<T>), grid, block, 0, stream,
                     static_cast<const T*>(x), static_cast<const T*>(w),
                     static_cast<const T*>(bias), static_cast<T*>(y), Cin, Cout,
                     HW);
  return 0;
}

} // namespace

extern "C" int hip_pointwise_conv(void* stream, const void* x, const void* w,
                                  const void* bias, void* y, int64_t N,
                                  int64_t Cin, int64_t Cout, int64_t HW,
                                  int hip_dtype) {
  if (N <= 0 || Cin <= 0 || Cout <= 0 || HW <= 0)
    return 0;

  // LDS staging of the weight row uses a fixed-size buffer; reject Cin beyond
  // it so we never index out of bounds. The upstream routing keeps Cin well
  // under this bound, so this is a defensive guard, not the normal path.
  if (Cin > 1024) {
    fprintf(stderr,
            "[custom_kernels] hip_pointwise_conv: Cin=%lld exceeds LDS bound\n",
            (long long)Cin);
    return -1;
  }

  hipStream_t hip_stream = static_cast<hipStream_t>(stream);

  CUSTOM_KERNELS_DEBUG_LOG(
      "[custom_kernels] hip_pointwise_conv: N=%lld Cin=%lld Cout=%lld HW=%lld "
      "dtype=%d bias=%s\n",
      (long long)N, (long long)Cin, (long long)Cout, (long long)HW, hip_dtype,
      bias ? "yes" : "null");

  int rc = -1;
  switch (hip_dtype) {
  case HIP_DTYPE_FLOAT32:
    rc = launch_for_dtype<float>(hip_stream, x, w, bias, y, N, Cin, Cout, HW);
    break;
  case HIP_DTYPE_FLOAT16:
    rc = launch_for_dtype<__half>(hip_stream, x, w, bias, y, N, Cin, Cout, HW);
    break;
  case HIP_DTYPE_BFLOAT16:
    rc = launch_for_dtype<hip_bfloat16>(hip_stream, x, w, bias, y, N, Cin, Cout,
                                        HW);
    break;
  default:
    fprintf(stderr,
            "[custom_kernels] hip_pointwise_conv: unsupported dtype=%d\n",
            hip_dtype);
    return -1;
  }

  if (rc != 0)
    return rc;

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr, "[custom_kernels] hip_pointwise_conv launch failed: %s\n",
            hipGetErrorString(err));
  }
  return static_cast<int>(err);
}
