/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// hip.random_normal_like: one Normal(mean, scale) sample per output element.
//
// The input tensor is not loaded. `shape` is the output shape, including
// dynamic sizes. Rank 0 writes a single element. A zero dimension writes
// nothing and returns success.
//
// ONNX does not specify the generator. Each element is one Box-Muller
// sample from a counter hash of (seed, index), so the same seed and the
// same element count always refill the same values, and a different runtime
// shape does not require a new binary. Values are not bit-identical to ORT.

#include "debug_log.h"
#include "hip_custom_kernels.h"

#include <cstdint>
#include <cstdio>
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <limits>

namespace {

constexpr int kMaxRank = 8;
constexpr int kBlockSize = 256;

__device__ uint32_t mix64(uint64_t x) {
  x ^= x >> 30;
  x *= 0xBF58476D1CE4E5B9ull;
  x ^= x >> 27;
  x *= 0x94D049BB133111EBull;
  x ^= x >> 31;
  return static_cast<uint32_t>(x);
}

// (0, 1), exclusive, so log() stays finite.
__device__ float uniform_f32(uint64_t seed, uint64_t index, uint64_t stream) {
  uint32_t bits = mix64(seed ^ (index * 0x9E3779B97F4A7C15ull) ^
                        (stream * 0xBF58476D1CE4E5B9ull));
  return (static_cast<float>(bits >> 8) + 0.5f) * (1.0f / 16777216.0f);
}

__device__ double uniform_f64(uint64_t seed, uint64_t index, uint64_t stream) {
  uint32_t hi = mix64(seed ^ (index * 0x9E3779B97F4A7C15ull) ^
                      (stream * 0xBF58476D1CE4E5B9ull));
  uint32_t lo = mix64(seed ^ (index * 0x94D049BB133111EBull) ^
                      ((stream + 1) * 0xBF58476D1CE4E5B9ull));
  uint64_t bits = (static_cast<uint64_t>(hi) << 32) | lo;
  // 53 mantissa bits, shifted so the value is in (0, 1).
  return (static_cast<double>(bits >> 11) + 0.5) * (1.0 / 9007199254740992.0);
}

__device__ float box_muller_f32(uint64_t seed, uint64_t index) {
  float u1 = uniform_f32(seed, index, 1);
  float u2 = uniform_f32(seed, index, 2);
  float radius = sqrtf(-2.0f * logf(u1));
  return radius * cosf(6.283185307179586f * u2);
}

__device__ double box_muller_f64(uint64_t seed, uint64_t index) {
  double u1 = uniform_f64(seed, index, 1);
  double u2 = uniform_f64(seed, index, 2);
  double radius = sqrt(-2.0 * log(u1));
  return radius * cos(6.283185307179586 * u2);
}

__device__ void store_sample(float *out, int64_t index, float value) {
  out[index] = value;
}

__device__ void store_sample(double *out, int64_t index, double value) {
  out[index] = value;
}

__device__ void store_sample(__half *out, int64_t index, float value) {
  out[index] = __float2half(value);
}

__device__ void store_sample(hip_bfloat16 *out, int64_t index, float value) {
  out[index] = hip_bfloat16(value);
}

template <typename OutT>
__global__ void random_normal_f32_kernel(OutT *out, int64_t n, float mean,
                                         float scale, uint64_t seed) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= n)
    return;
  float sample =
      mean + scale * box_muller_f32(seed, static_cast<uint64_t>(index));
  store_sample(out, index, sample);
}

__global__ void random_normal_f64_kernel(double *out, int64_t n, double mean,
                                         double scale, uint64_t seed) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= n)
    return;
  out[index] =
      mean + scale * box_muller_f64(seed, static_cast<uint64_t>(index));
}

int64_t output_numel(int64_t rank, const int64_t *shape) {
  if (rank == 0)
    return 1;
  int64_t numel = 1;
  for (int64_t i = 0; i < rank; ++i) {
    int64_t dim = shape[i];
    if (dim < 0)
      return -1;
    if (dim == 0)
      return 0;
    if (numel > std::numeric_limits<int64_t>::max() / dim)
      return -1;
    numel *= dim;
  }
  return numel;
}

template <typename OutT>
int launch_f32(hipStream_t stream, void *output, int64_t numel, float mean,
               float scale, uint64_t seed) {
  uint64_t grid64 =
      (static_cast<uint64_t>(numel) + kBlockSize - 1) / kBlockSize;
  if (grid64 > std::numeric_limits<uint32_t>::max()) {
    fprintf(stderr,
            "[custom_kernels] hip_random_normal_like: element count %lld "
            "exceeds the launch grid\n",
            (long long)numel);
    return -1;
  }
  // Drop a stale error so a successful launch is not reported as failed.
  (void)hipGetLastError();
  hipLaunchKernelGGL(random_normal_f32_kernel<OutT>,
                     dim3(static_cast<uint32_t>(grid64)), dim3(kBlockSize), 0,
                     stream, static_cast<OutT *>(output), numel, mean, scale,
                     seed);
  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr,
            "[custom_kernels] hip_random_normal_like: launch failed: %s\n",
            hipGetErrorString(err));
    return -1;
  }
  return 0;
}

} // namespace

extern "C" int hip_random_normal_like(void *stream, void *output, int64_t rank,
                                      const int64_t *shape, float mean,
                                      float scale, uint64_t seed,
                                      int hip_dtype) {
  if (rank < 0 || rank > kMaxRank) {
    fprintf(
        stderr,
        "[custom_kernels] hip_random_normal_like: rank %lld not in [0, %d]\n",
        (long long)rank, kMaxRank);
    return -1;
  }
  // A zero-extent tensor may have no allocation. Decide that from the shape
  // before requiring a data pointer. Rank 0 is one element and still needs
  // storage.
  if (rank > 0 && !shape) {
    fprintf(stderr, "[custom_kernels] hip_random_normal_like: null argument\n");
    return -1;
  }

  int64_t numel = output_numel(rank, shape);
  if (numel < 0) {
    fprintf(stderr,
            "[custom_kernels] hip_random_normal_like: invalid or overflowing "
            "shape\n");
    return -1;
  }
  if (numel == 0)
    return 0;
  if (!output) {
    fprintf(stderr, "[custom_kernels] hip_random_normal_like: null argument\n");
    return -1;
  }

  CUSTOM_KERNELS_DEBUG_LOG(
      "[custom_kernels] hip_random_normal_like: rank=%lld, numel=%lld, "
      "dtype=%d\n",
      (long long)rank, (long long)numel, hip_dtype);

  hipStream_t hip_stream = static_cast<hipStream_t>(stream);
  switch (hip_dtype) {
  case HIP_DTYPE_FLOAT16:
    return launch_f32<__half>(hip_stream, output, numel, mean, scale, seed);
  case HIP_DTYPE_BFLOAT16:
    return launch_f32<hip_bfloat16>(hip_stream, output, numel, mean, scale,
                                    seed);
  case HIP_DTYPE_FLOAT32:
    return launch_f32<float>(hip_stream, output, numel, mean, scale, seed);
  case HIP_DTYPE_FLOAT64: {
    uint64_t grid64 =
        (static_cast<uint64_t>(numel) + kBlockSize - 1) / kBlockSize;
    if (grid64 > std::numeric_limits<uint32_t>::max()) {
      fprintf(stderr,
              "[custom_kernels] hip_random_normal_like: element count %lld "
              "exceeds the launch grid\n",
              (long long)numel);
      return -1;
    }
    // Drop a stale error so a successful launch is not reported as failed.
    (void)hipGetLastError();
    hipLaunchKernelGGL(
        random_normal_f64_kernel, dim3(static_cast<uint32_t>(grid64)),
        dim3(kBlockSize), 0, hip_stream, static_cast<double *>(output), numel,
        static_cast<double>(mean), static_cast<double>(scale), seed);
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
      fprintf(stderr,
              "[custom_kernels] hip_random_normal_like: launch failed: %s\n",
              hipGetErrorString(err));
      return -1;
    }
    return 0;
  }
  default:
    fprintf(
        stderr,
        "[custom_kernels] hip_random_normal_like: unsupported hip_dtype=%d\n",
        hip_dtype);
    return -1;
  }
}
