/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "activation_quant_i8.h"
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

template <typename T>
__device__ __forceinline__ T computeSwiglu(T gateValue, T upValue) {
  using Traits = SwigluTraits<T>;
  using ComputeType = typename Traits::ComputeType;

  ComputeType gate = Traits::load(gateValue);
  ComputeType up = Traits::load(upValue);
  ComputeType sigmoid =
      ComputeType(1) / (ComputeType(1) + Traits::expFn(-gate));

  // Match the exported Sigmoid -> Mul -> Mul graph. Each ONNX node produces
  // a tensor of T, so round at both intermediate tensor boundaries.
  sigmoid = Traits::load(Traits::store(sigmoid));
  ComputeType activated = Traits::load(Traits::store(gate * sigmoid));
  return Traits::store(activated * up);
}

template <typename T>
__global__ void swigluKernel(const T *__restrict__ gate,
                             const T *__restrict__ up, T *__restrict__ output,
                             int64_t numElements) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < numElements)
    output[idx] = computeSwiglu<T>(gate[idx], up[idx]);
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

template <typename T, int V>
__global__ void swigluVecKernel(const T *__restrict__ gate,
                                const T *__restrict__ up,
                                T *__restrict__ output, int64_t numVectors) {
  int64_t vecIdx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (vecIdx >= numVectors)
    return;

  const int64_t base = vecIdx * V;
  SwigluVec<T, V> gateVec, upVec, outVec;
  gateVec.raw = *reinterpret_cast<const uint4 *>(gate + base);
  upVec.raw = *reinterpret_cast<const uint4 *>(up + base);
#pragma unroll
  for (int i = 0; i < V; ++i)
    outVec.elements[i] =
        computeSwiglu<T>(gateVec.elements[i], upVec.elements[i]);
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

// Vector path when the buffers are 16-byte aligned and the length fills whole
// vectors. A length that does not is computed entirely by the scalar kernel.
template <typename T, int V>
static void launchSwiglu(hipStream_t hipStream, const void *gate,
                         const void *up, void *output, int64_t numElements) {
  constexpr int kBlockSize = 256;
  if (numElements % V == 0 && swigluAligned16(gate, up, output)) {
    const int64_t numVectors = numElements / V;
    const int grid =
        static_cast<int>((numVectors + kBlockSize - 1) / kBlockSize);
    hipLaunchKernelGGL((swigluVecKernel<T, V>), dim3(grid), dim3(kBlockSize), 0,
                       hipStream, static_cast<const T *>(gate),
                       static_cast<const T *>(up), static_cast<T *>(output),
                       numVectors);
    return;
  }

  const int grid =
      static_cast<int>((numElements + kBlockSize - 1) / kBlockSize);
  hipLaunchKernelGGL(swigluKernel<T>, dim3(grid), dim3(kBlockSize), 0,
                     hipStream, static_cast<const T *>(gate),
                     static_cast<const T *>(up), static_cast<T *>(output),
                     numElements);
}

extern "C" int hip_swiglu(void *stream, const void *gate, const void *up,
                          void *output, int64_t numElements, int hipDtype) {
  if (numElements <= 0)
    return 0;

  hipStream_t hipStream = static_cast<hipStream_t>(stream);
  constexpr int kBlockSize = 256;
  int gridSize = static_cast<int>((numElements + kBlockSize - 1) / kBlockSize);

  // Clear a stale launch error so the returned status belongs to this launch.
  (void)hipGetLastError();
  switch (hipDtype) {
  case HIP_DTYPE_FLOAT16:
    launchSwiglu<__half, 8>(hipStream, gate, up, output, numElements);
    break;
  case HIP_DTYPE_FLOAT32:
    launchSwiglu<float, 4>(hipStream, gate, up, output, numElements);
    break;
  case HIP_DTYPE_BFLOAT16:
    hipLaunchKernelGGL(swigluKernel<__hip_bfloat16>, dim3(gridSize),
                       dim3(kBlockSize), 0, hipStream,
                       static_cast<const __hip_bfloat16 *>(gate),
                       static_cast<const __hip_bfloat16 *>(up),
                       static_cast<__hip_bfloat16 *>(output), numElements);
    break;
  case HIP_DTYPE_FLOAT64:
    hipLaunchKernelGGL(swigluKernel<double>, dim3(gridSize), dim3(kBlockSize),
                       0, hipStream, static_cast<const double *>(gate),
                       static_cast<const double *>(up),
                       static_cast<double *>(output), numElements);
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

// hip.sigmoid_mul: y = x * sigmoid(gate), rounding sigmoid(gate) to fp16 as
// hip.sigmoid followed by hip.mul do.
__device__ __forceinline__ float sigmoidMulF16(float x, float gate) {
  const float s = __half2float(__float2half(1.0f / (1.0f + expf(-gate))));
  return x * s;
}

__global__ void sigmoidMulKernel(const __half *__restrict__ x,
                                 const __half *__restrict__ gate,
                                 __half *__restrict__ output,
                                 int64_t numElements) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < numElements)
    output[idx] = __float2half(
        sigmoidMulF16(__half2float(x[idx]), __half2float(gate[idx])));
}

// One __half2 pair per thread. QUANT also writes y's int8 quantization: a
// row is a whole number of 32-element groups, so every group is 16
// consecutive, 16-aligned lanes, all of them in range.
template <bool QUANT>
__global__ void sigmoidMulPairKernel(const __half2 *__restrict__ x,
                                     const __half2 *__restrict__ gate,
                                     __half2 *__restrict__ output,
                                     int8_t *__restrict__ qOut,
                                     float *__restrict__ qScale,
                                     int64_t numPairs, int64_t rowPairs) {
  int64_t p = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (p >= numPairs)
    return;
  const float2 xv = __half22float2(x[p]);
  const float2 g = __half22float2(gate[p]);
  const __half2 y = __float22half2_rn(
      make_float2(sigmoidMulF16(xv.x, g.x), sigmoidMulF16(xv.y, g.y)));
  output[p] = y;
  if constexpr (QUANT) {
    const int64_t row = p / rowPairs;
    quant_act_i8_pair(__half22float2(y), p - row * rowPairs,
                      qOut + row * 2 * rowPairs,
                      qScale + row * (2 * rowPairs / 32));
  }
}

extern "C" int hip_sigmoid_mul(void *stream, const void *x, const void *gate,
                               void *output, void *q_out, void *q_scale,
                               int64_t num_elements, int64_t row_size,
                               int64_t quant_block_size) {
  if (num_elements <= 0)
    return 0;
  hipStream_t hipStream = static_cast<hipStream_t>(stream);
  constexpr int kBlockSize = 256;
  auto aligned4 = [](const void *ptr) {
    return reinterpret_cast<uintptr_t>(ptr) % 4 == 0;
  };
  const bool pairs = num_elements % 2 == 0 && row_size % 2 == 0 &&
                     aligned4(x) && aligned4(gate) && aligned4(output);
  const bool quant = q_out && q_scale;
  if (quant && (!pairs || quant_block_size != 32 || row_size % 32 != 0 ||
                num_elements % row_size != 0))
    return hipErrorNotSupported;

  (void)hipGetLastError();
  if (pairs) {
    const int64_t numPairs = num_elements / 2;
    const int grid = static_cast<int>((numPairs + kBlockSize - 1) / kBlockSize);
    auto kernel = quant ? sigmoidMulPairKernel<true> : sigmoidMulPairKernel<false>;
    hipLaunchKernelGGL(kernel, dim3(grid), dim3(kBlockSize), 0, hipStream,
                       static_cast<const __half2 *>(x),
                       static_cast<const __half2 *>(gate),
                       static_cast<__half2 *>(output),
                       static_cast<int8_t *>(q_out),
                       static_cast<float *>(q_scale), numPairs, row_size / 2);
  } else {
    const int grid =
        static_cast<int>((num_elements + kBlockSize - 1) / kBlockSize);
    hipLaunchKernelGGL(sigmoidMulKernel, dim3(grid), dim3(kBlockSize), 0,
                       hipStream, static_cast<const __half *>(x),
                       static_cast<const __half *>(gate),
                       static_cast<__half *>(output), num_elements);
  }

  CUSTOM_KERNELS_DEBUG_LOG(
      "[custom_kernels] hip_sigmoid_mul: num=%lld, row=%lld, pairs=%d, "
      "quant=%d\n",
      (long long)num_elements, (long long)row_size, pairs ? 1 : 0,
      quant ? 1 : 0);
  hipError_t error = hipGetLastError();
  if (error != hipSuccess)
    fprintf(stderr, "[custom_kernels] hip_sigmoid_mul launch failed: %s\n",
            hipGetErrorString(error));
  return static_cast<int>(error);
}
