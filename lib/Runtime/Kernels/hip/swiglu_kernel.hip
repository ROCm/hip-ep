/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "debug_log.h"
#include "hip_custom_kernels.h"

#include <cstdint>
#include <cstdio>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

template <typename T> struct SwigluTraits;

template <> struct SwigluTraits<float> {
  using ComputeType = float;
  __device__ static ComputeType load(float value) { return value; }
  __device__ static float store(ComputeType value) { return value; }
  __device__ static ComputeType expFn(ComputeType value) { return expf(value); }
};

template <> struct SwigluTraits<__half> {
  using ComputeType = float;
  __device__ static ComputeType load(__half value) {
    return __half2float(value);
  }
  __device__ static __half store(ComputeType value) {
    return __float2half(value);
  }
  __device__ static ComputeType expFn(ComputeType value) { return expf(value); }
};

template <> struct SwigluTraits<__hip_bfloat16> {
  using ComputeType = float;
  __device__ static ComputeType load(__hip_bfloat16 value) {
    return __bfloat162float(value);
  }
  __device__ static __hip_bfloat16 store(ComputeType value) {
    return __float2bfloat16(value);
  }
  __device__ static ComputeType expFn(ComputeType value) { return expf(value); }
};

template <> struct SwigluTraits<double> {
  using ComputeType = double;
  __device__ static ComputeType load(double value) { return value; }
  __device__ static double store(ComputeType value) { return value; }
  __device__ static ComputeType expFn(ComputeType value) { return exp(value); }
};

template <typename T, bool PROMOTE = false>
__device__ __forceinline__ T computeSwiglu(T gateValue, T upValue) {
  using Traits = SwigluTraits<T>;
  using ComputeType = typename Traits::ComputeType;

  ComputeType gate = Traits::load(gateValue);
  ComputeType up = Traits::load(upValue);
  ComputeType sigmoid =
      ComputeType(1) / (ComputeType(1) + Traits::expFn(-gate));

  // Match the exported Sigmoid -> Mul -> Mul graph. Each ONNX node produces
  // a tensor of T, so round at both intermediate tensor boundaries.
  if constexpr (!PROMOTE)
    sigmoid = Traits::load(Traits::store(sigmoid));
  ComputeType activated = gate * sigmoid;
  if constexpr (!PROMOTE)
    activated = Traits::load(Traits::store(activated));
  return Traits::store(activated * up);
}

// The activations the kernels below apply element by element. Each names the
// thread count its kernels are built for.
//
// Plain SwiGLU; PROMOTE keeps FP32 intermediates (hip_swiglu_promoted).
template <typename T, bool PROMOTE> struct PlainSwiglu {
  static constexpr int kMaxThreads = PROMOTE ? 256 : 1024;
  __device__ T operator()(T gate, T up) const {
    return computeSwiglu<T, PROMOTE>(gate, up);
  }
};

// Clamped SwiGLU, in FP32 with one rounding at the store: the form hip.qmoe
// applies between its projections (GPT-OSS), written as GGML and llama.cpp
// write it, g / (1 + exp(-alpha * g)) rather than g * sigmoid(alpha * g).
template <typename T> struct ClampedSwiglu {
  static constexpr int kMaxThreads = 256;
  float alpha, beta, limit;
  __device__ T operator()(T gateValue, T upValue) const {
    using Traits = SwigluTraits<T>;
    const float g = fminf(static_cast<float>(Traits::load(gateValue)), limit);
    const float u =
        fminf(fmaxf(static_cast<float>(Traits::load(upValue)), -limit), limit);
    return Traits::store(g / (1.0f + expf(alpha * -g)) * (u + beta));
  }
};

template <typename T, typename Op>
__global__ __launch_bounds__(Op::kMaxThreads) void swigluKernel(
    const T *__restrict__ gate, const T *__restrict__ up,
    T *__restrict__ output, int64_t numElements, Op op) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < numElements)
    output[idx] = op(gate[idx], up[idx]);
}

// 16 bytes per thread: 8 halves or 4 floats. Scalar half loads on a decode
// SwiGLU (a few dozen KiB) never fill the memory pipe. The math stays per
// element so the fp16 rounding matches the scalar kernel.
template <typename T, int V> union SwigluVec {
  static_assert(sizeof(T) * V == sizeof(uint4),
                "SwigluVec covers one 16-byte load");
  uint4 raw;
  T elements[V];
};

template <typename T, int V, typename Op>
__global__ __launch_bounds__(Op::kMaxThreads) void swigluVecKernel(
    const T *__restrict__ gate, const T *__restrict__ up,
    T *__restrict__ output, int64_t numVectors, Op op) {
  int64_t vecIdx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (vecIdx >= numVectors)
    return;

  const int64_t base = vecIdx * V;
  SwigluVec<T, V> gateVec, upVec, outVec;
  gateVec.raw = *reinterpret_cast<const uint4 *>(gate + base);
  upVec.raw = *reinterpret_cast<const uint4 *>(up + base);
#pragma unroll
  for (int i = 0; i < V; ++i)
    outVec.elements[i] = op(gateVec.elements[i], upVec.elements[i]);
  *reinterpret_cast<uint4 *>(output + base) = outVec.raw;
}

static bool swigluAligned16(const void *gate, const void *up,
                            const void *output) {
  constexpr uintptr_t kAlign = sizeof(uint4);
  auto aligned = [](const void *ptr) {
    return reinterpret_cast<uintptr_t>(ptr) % kAlign == 0;
  };
  return aligned(gate) && aligned(up) && aligned(output);
}

// Vector path (V > 0) when the buffers are 16-byte aligned and the length
// fills whole vectors. A length that does not, and V = 0 (bf16, f64), is
// computed entirely by the scalar kernel.
template <typename T, int V, typename Op>
static void launchSwiglu(hipStream_t hipStream, const void *gate,
                         const void *up, void *output, int64_t numElements,
                         Op op) {
  constexpr int kBlockSize = 256;
  if constexpr (V > 0) {
    if (numElements % V == 0 && swigluAligned16(gate, up, output)) {
      const int64_t numVectors = numElements / V;
      const int grid =
          static_cast<int>((numVectors + kBlockSize - 1) / kBlockSize);
      hipLaunchKernelGGL(
          (swigluVecKernel<T, V, Op>), dim3(grid), dim3(kBlockSize), 0,
          hipStream, static_cast<const T *>(gate), static_cast<const T *>(up),
          static_cast<T *>(output), numVectors, op);
      return;
    }
  }

  const int grid =
      static_cast<int>((numElements + kBlockSize - 1) / kBlockSize);
  hipLaunchKernelGGL((swigluKernel<T, Op>), dim3(grid), dim3(kBlockSize), 0,
                     hipStream, static_cast<const T *>(gate),
                     static_cast<const T *>(up), static_cast<T *>(output),
                     numElements, op);
}

extern "C" int hip_swiglu(void *stream, const void *gate, const void *up,
                          void *output, int64_t numElements, int hipDtype) {
  if (numElements <= 0)
    return 0;

  hipStream_t hipStream = static_cast<hipStream_t>(stream);

  // Clear a stale launch error so the returned status belongs to this launch.
  (void)hipGetLastError();
  switch (hipDtype) {
  case HIP_DTYPE_FLOAT16:
    launchSwiglu<__half, 8>(hipStream, gate, up, output, numElements,
                            PlainSwiglu<__half, false>{});
    break;
  case HIP_DTYPE_FLOAT32:
    launchSwiglu<float, 4>(hipStream, gate, up, output, numElements,
                           PlainSwiglu<float, false>{});
    break;
  case HIP_DTYPE_BFLOAT16:
    launchSwiglu<__hip_bfloat16, 0>(hipStream, gate, up, output, numElements,
                                    PlainSwiglu<__hip_bfloat16, false>{});
    break;
  case HIP_DTYPE_FLOAT64:
    launchSwiglu<double, 0>(hipStream, gate, up, output, numElements,
                            PlainSwiglu<double, false>{});
    break;
  default:
    fprintf(stderr, "[custom_kernels] hip_swiglu: unsupported dtype=%d\n",
            hipDtype);
    return -1;
  }

  CUSTOM_KERNELS_DEBUG_LOG("[custom_kernels] hip_swiglu: dtype=%d, num=%lld\n",
                           hipDtype, (long long)numElements);
  hipError_t error = hipGetLastError();
  if (error != hipSuccess)
    fprintf(stderr, "[custom_kernels] hip_swiglu launch failed: %s\n",
            hipGetErrorString(error));
  return static_cast<int>(error);
}

extern "C" int hip_swiglu_promoted(void *stream, const void *gate,
                                   const void *up, void *output,
                                   int64_t numElements, int hipDtype) {
  if (hipDtype != HIP_DTYPE_FLOAT16)
    return -1;
  if (numElements <= 0)
    return 0;
  (void)hipGetLastError();
  launchSwiglu<__half, 8>(static_cast<hipStream_t>(stream), gate, up, output,
                          numElements, PlainSwiglu<__half, true>{});
  return static_cast<int>(hipGetLastError());
}

extern "C" int hip_swiglu_clamped(void *stream, const void *gate,
                                  const void *up, void *output,
                                  int64_t numElements, int hipDtype,
                                  float alpha, float beta, float limit) {
  if (numElements <= 0)
    return 0;
  hipStream_t hipStream = static_cast<hipStream_t>(stream);
  (void)hipGetLastError();
  switch (hipDtype) {
  case HIP_DTYPE_FLOAT32:
    launchSwiglu<float, 4>(hipStream, gate, up, output, numElements,
                           ClampedSwiglu<float>{alpha, beta, limit});
    break;
  case HIP_DTYPE_FLOAT16:
    launchSwiglu<__half, 8>(hipStream, gate, up, output, numElements,
                            ClampedSwiglu<__half>{alpha, beta, limit});
    break;
  case HIP_DTYPE_BFLOAT16:
    launchSwiglu<__hip_bfloat16, 0>(
        hipStream, gate, up, output, numElements,
        ClampedSwiglu<__hip_bfloat16>{alpha, beta, limit});
    break;
  default:
    fprintf(stderr,
            "[custom_kernels] hip_swiglu_clamped: unsupported dtype=%d\n",
            hipDtype);
    return -1;
  }
  hipError_t error = hipGetLastError();
  if (error != hipSuccess)
    fprintf(stderr, "[custom_kernels] hip_swiglu_clamped launch failed: %s\n",
            hipGetErrorString(error));
  return static_cast<int>(error);
}
