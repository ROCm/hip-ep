/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * RMS normalization (ONNX RMSNormalization / SimplifiedLayerNormalization):
 * per-row root-mean-square normalization followed by an elementwise scale.
 *
 *   y = x * rsqrt(mean(x^2) + epsilon) * scale
 *
 * Layout: input is reshaped as `[outer, norm_size]` on the host, where
 *         norm_size is the ONNX reduction width (the product of the input
 *         dims from `axis` on) and outer = input_num_elements / norm_size.
 *         `scale` holds `scale_rows` gain vectors of that width; row r uses
 *         group r % scale_rows. scale_rows == 1 is the ordinary case where
 *         every row shares one gain vector.
 *
 * Algorithm: one block per row, two passes with a shared-memory block
 * reduction.
 *   Pass A: each block sums x^2 over `norm_size`, in FP32.
 *   Pass B: each thread writes y in the input dtype, using FP32 math.
 *
 * FP32 accumulation is not optional: a 4096-wide sum of squares in FP16
 * saturates well before the row ends.
 *
 * The VEC2 template parameter selects a packed __half2 body. RMS norm is
 * memory-bound, so 2-wide loads/stores are the main lever; the launcher picks
 * it only when norm_size is even and every pointer is 4-byte aligned.
 *
 * Structurally mirrors layer_norm_kernel.hip (same block size, same reduction
 * shape) minus the mean subtraction and the bias term.
 */

#include "hip_custom_kernels.h"
#include "debug_log.h"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cstdio>

static constexpr int kRmsNormBlockSize = 256;

namespace {

__device__ inline float rms_load(const float *p) { return *p; }
__device__ inline float rms_load(const __half *p) { return __half2float(*p); }

__device__ inline void rms_store(float *p, float v) { *p = v; }
__device__ inline void rms_store(__half *p, float v) { *p = __float2half(v); }

// Block-wide sum. Every thread supplies a partial; every thread gets the total.
__device__ inline float rms_block_sum(float partial, float *s_reduce) {
  s_reduce[threadIdx.x] = partial;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride)
      s_reduce[threadIdx.x] += s_reduce[threadIdx.x + stride];
    __syncthreads();
  }
  return s_reduce[0];
}

// One block per row. blockDim.x == kRmsNormBlockSize.
// VEC2 requires norm_size % 2 == 0 and 4-byte aligned rows; the launcher
// guarantees both before selecting this instantiation.
template <typename T, bool VEC2>
__global__ void hip_rms_norm_kernel(const T *__restrict__ input,
                                    const T *__restrict__ scale,
                                    T *__restrict__ output, int64_t outer,
                                    int64_t norm_size, int64_t scale_rows,
                                    float epsilon) {
  int64_t row = blockIdx.x;
  if (row >= outer)
    return;

  const T *x_row = input + row * norm_size;
  T *y_row = output + row * norm_size;
  // A scale that spans several rows (grouped norm: scale [G, D] against input
  // [N, G, D] with axis = -1) tiles over the row index, and row-major order
  // makes the group the fastest-varying leading dim -- hence r % scale_rows.
  const T *g_row = scale + (row % scale_rows) * norm_size;

  __shared__ float s_reduce[kRmsNormBlockSize];
  __shared__ float s_inv_rms;

  float local_sumsq = 0.0f;

  if constexpr (VEC2) {
    const int64_t norm_size2 = norm_size / 2;
    const __half2 *x2 = reinterpret_cast<const __half2 *>(x_row);
    for (int64_t i = threadIdx.x; i < norm_size2; i += blockDim.x) {
      float2 v = __half22float2(x2[i]);
      local_sumsq += v.x * v.x + v.y * v.y;
    }
  } else {
    for (int64_t i = threadIdx.x; i < norm_size; i += blockDim.x) {
      float v = rms_load(x_row + i);
      local_sumsq += v * v;
    }
  }

  float sumsq = rms_block_sum(local_sumsq, s_reduce);
  if (threadIdx.x == 0)
    s_inv_rms = rsqrtf(sumsq / static_cast<float>(norm_size) + epsilon);
  __syncthreads();

  const float inv_rms = s_inv_rms;

  if constexpr (VEC2) {
    const int64_t norm_size2 = norm_size / 2;
    const __half2 *x2 = reinterpret_cast<const __half2 *>(x_row);
    const __half2 *g2 = reinterpret_cast<const __half2 *>(g_row);
    __half2 *y2 = reinterpret_cast<__half2 *>(y_row);
    for (int64_t i = threadIdx.x; i < norm_size2; i += blockDim.x) {
      float2 v = __half22float2(x2[i]);
      float2 g = __half22float2(g2[i]);
      float2 r;
      r.x = v.x * inv_rms * g.x;
      r.y = v.y * inv_rms * g.y;
      y2[i] = __float22half2_rn(r);
    }
  } else {
    for (int64_t i = threadIdx.x; i < norm_size; i += blockDim.x) {
      float v = rms_load(x_row + i);
      float g = rms_load(g_row + i);
      rms_store(y_row + i, v * inv_rms * g);
    }
  }
}

// One logical wave per narrow row, eight rows per block. The shuffle tree
// reproduces the original 256-thread VEC2 reduction: unused lanes contributed
// zero, then offsets64/32 (width256) or32 (width128) joined each lane's pairs.
// Keep loaded pairs in registers for the output pass; no block barriers/LDS.
template <int NORM>
__global__ __launch_bounds__(256)
void hip_rms_norm_narrow_kernel(const __half *__restrict__ input,
                                const __half *__restrict__ scale,
                                __half *__restrict__ output, int64_t outer,
                                int64_t scale_rows, float epsilon) {
  static_assert(NORM == 128 || NORM == 256, "narrow VEC2 reduction width");
  constexpr int kWave = 32;
  constexpr int kRowsPerBlock = 8;
  constexpr int kPairsPerLane = NORM / (2 * kWave);
  const int lane = threadIdx.x % kWave;
  const int64_t row = static_cast<int64_t>(blockIdx.x) * kRowsPerBlock +
                      threadIdx.x / kWave;
  if (row >= outer)
    return;
  const __half2 *x = reinterpret_cast<const __half2 *>(input + row * NORM);
  const __half2 *g = reinterpret_cast<const __half2 *>(
      scale + (scale_rows == 1 ? 0 : row % scale_rows) * NORM);
  __half2 *y = reinterpret_cast<__half2 *>(output + row * NORM);
  float2 values[kPairsPerLane];
  float partial[kPairsPerLane];
#pragma unroll
  for (int i = 0; i < kPairsPerLane; ++i) {
    values[i] = __half22float2(x[lane + i * kWave]);
    // Match the existing per-thread half2 contribution before the tree.
    float local_sumsq = 0.0f;
    local_sumsq += values[i].x * values[i].x + values[i].y * values[i].y;
    partial[i] = local_sumsq;
  }
  float sumsq;
  if constexpr (NORM == 128)
    sumsq = partial[0] + partial[1];
  else
    sumsq = (partial[0] + partial[2]) + (partial[1] + partial[3]);
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1)
    sumsq += __shfl_down(sumsq, offset, kWave);
  sumsq = __shfl(sumsq, 0, kWave);
  const float inv_rms = rsqrtf(sumsq / static_cast<float>(NORM) + epsilon);
#pragma unroll
  for (int i = 0; i < kPairsPerLane; ++i) {
    float2 gain = __half22float2(g[lane + i * kWave]);
    float2 result;
    result.x = values[i].x * inv_rms * gain.x;
    result.y = values[i].y * inv_rms * gain.y;
    y[lane + i * kWave] = __float22half2_rn(result);
  }
}

// One block per row; blockDim.x is a power of two <= kRmsNormBlockSize.
template <typename T, bool SILU>
__global__ void hip_gated_rms_norm_kernel(const T *__restrict__ input,
                                          const T *__restrict__ scale,
                                          const T *__restrict__ gate,
                                          T *__restrict__ output, int64_t outer,
                                          int64_t norm_size, float epsilon) {
  int64_t row = blockIdx.x;
  if (row >= outer) {
    return;
  }

  const T *x_row = input + row * norm_size;
  const T *z_row = gate + row * norm_size;
  T *y_row = output + row * norm_size;

  __shared__ float s_reduce[kRmsNormBlockSize];
  __shared__ float s_inv_rms;

  float local_sumsq = 0.0f;
  for (int64_t i = threadIdx.x; i < norm_size; i += blockDim.x) {
    float v = rms_load(x_row + i);
    local_sumsq += v * v;
  }
  float sumsq = rms_block_sum(local_sumsq, s_reduce);
  if (threadIdx.x == 0) {
    s_inv_rms = rsqrtf(sumsq / static_cast<float>(norm_size) + epsilon);
  }
  __syncthreads();
  const float inv_rms = s_inv_rms;

  for (int64_t i = threadIdx.x; i < norm_size; i += blockDim.x) {
    float v = rms_load(x_row + i);
    float g = rms_load(scale + i);
    float z = rms_load(z_row + i);
    float act = 1.0f / (1.0f + expf(-z));
    if constexpr (SILU) {
      act *= z;
    }
    rms_store(y_row + i, v * inv_rms * g * act);
  }
}

// rms_block_sum with a trailing barrier, so the next reduction may reuse
// s_reduce while slower threads are still reading this total.
__device__ inline float rms_block_sum_reuse(float partial, float *s_reduce) {
  float total = rms_block_sum(partial, s_reduce);
  __syncthreads();
  return total;
}

// One block per (b, s, g) row; blockDim.x == kRmsNormBlockSize.
template <typename T>
__global__ void hip_engram_gate_kernel(
    const T *__restrict__ key, const T *__restrict__ query,
    const T *__restrict__ value, const T *__restrict__ key_scale,
    const T *__restrict__ query_scale, const T *__restrict__ conv_scale,
    T *__restrict__ output, T *__restrict__ normed, int64_t rows, int64_t hc,
    int64_t hidden, float epsilon) {
  int64_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  const int64_t g = row % hc;
  const T *k_row = key + row * hidden;
  const T *q_row = query + row * hidden;
  const T *v_row = value + (row / hc) * hidden;
  const T *ks_row = key_scale + g * hidden;
  const T *qs_row = query_scale + g * hidden;
  T *o_row = output + row * hidden;

  __shared__ float s_reduce[kRmsNormBlockSize];
  __shared__ float s_gate;

  float k_sumsq = 0.0f;
  float q_sumsq = 0.0f;
  float dot_num = 0.0f;
  for (int64_t c = threadIdx.x; c < hidden; c += blockDim.x) {
    float k = rms_load(k_row + c);
    float q = rms_load(q_row + c);
    k_sumsq += k * k;
    q_sumsq += q * q;
    dot_num += k * rms_load(ks_row + c) * q * rms_load(qs_row + c);
  }
  k_sumsq = rms_block_sum_reuse(k_sumsq, s_reduce);
  q_sumsq = rms_block_sum_reuse(q_sumsq, s_reduce);
  dot_num = rms_block_sum_reuse(dot_num, s_reduce);

  if (threadIdx.x == 0) {
    const float h = static_cast<float>(hidden);
    const float k_inv = 1.0f / sqrtf(k_sumsq / h + epsilon);
    const float q_inv = 1.0f / sqrtf(q_sumsq / h + epsilon);
    const float dot = dot_num * k_inv * q_inv / sqrtf(h);
    float arg = 0.0f;
    if (dot != 0.0f) {
      const float magnitude = sqrtf(fmaxf(fabsf(dot), 1.0e-6f));
      arg = dot < 0.0f ? -magnitude : magnitude;
    }
    if (arg > 0.0f) {
      s_gate = 1.0f / (1.0f + expf(-arg));
    } else {
      const float e = expf(arg);
      s_gate = e / (1.0f + e);
    }
  }
  __syncthreads();
  const float gate = s_gate;

  // The second output normalizes the already-rounded first output.
  float o_sumsq = 0.0f;
  for (int64_t c = threadIdx.x; c < hidden; c += blockDim.x) {
    rms_store(o_row + c, gate * rms_load(v_row + c));
    float o = rms_load(o_row + c);
    o_sumsq += o * o;
  }
  if (normed == nullptr) {
    return;
  }

  o_sumsq = rms_block_sum(o_sumsq, s_reduce);
  const float o_inv = 1.0f / sqrtf(o_sumsq / static_cast<float>(hidden) +
                                   epsilon);
  const T *cs_row = conv_scale + g * hidden;
  T *n_row = normed + row * hidden;
  for (int64_t c = threadIdx.x; c < hidden; c += blockDim.x) {
    rms_store(n_row + c, rms_load(o_row + c) * o_inv * rms_load(cs_row + c));
  }
}

// Packed __half2 requires an even row and 4-byte aligned bases. Row starts stay
// aligned across rows when norm_size is even, so checking the bases is enough.
inline bool rms_can_vectorize(const void *input, const void *scale,
                              const void *output, int64_t norm_size) {
  if (norm_size % 2 != 0)
    return false;
  auto aligned = [](const void *p) {
    return (reinterpret_cast<uintptr_t>(p) % 4) == 0;
  };
  return aligned(input) && aligned(scale) && aligned(output);
}

} // namespace

// Keep the narrow launch local to the target chip.
static bool rms_narrow_enabled() {
#ifdef _WIN32
  char control[8];
  const auto n = GetEnvironmentVariableA("HIPDNN_RMS_QWEN_NARROW", control,
                                        sizeof(control));
  if (n > 0 && n < sizeof(control) && std::atoi(control) == 0) return false;
#else
  const char *control = std::getenv("HIPDNN_RMS_QWEN_NARROW");
  if (control && std::atoi(control) == 0) return false;
#endif
  thread_local int cached_device = -1;
  thread_local bool supported = false;
  int device = -1;
  if (hipGetDevice(&device) != hipSuccess) return false;
  if (device != cached_device) {
    hipDeviceProp_t prop{};
    supported = hipGetDeviceProperties(&prop, device) == hipSuccess &&
                std::strncmp(prop.gcnArchName, "gfx1151", 7) == 0 &&
                (prop.gcnArchName[7] == '\0' || prop.gcnArchName[7] == ':');
    cached_device = device;
  }
  return supported;
}

extern "C" int hip_rms_norm(void *stream, const void *input, const void *scale,
                            void *output, int64_t outer, int64_t norm_size,
                            int64_t scale_rows, float epsilon, int hip_dtype) {
  if (outer <= 0 || norm_size <= 0)
    return 0;
  if (scale_rows <= 0) {
    fprintf(stderr, "[custom_kernels] hip_rms_norm: scale_rows=%lld\n",
            (long long)scale_rows);
    return -1;
  }

  hipStream_t hip_stream = static_cast<hipStream_t>(stream);
  dim3 grid(static_cast<unsigned int>(outer));
  dim3 block(kRmsNormBlockSize);

  // Clear any error left sticky by a prior kernel whose launcher didn't check
  // its own status, so the hipGetLastError() below attributes a failure to THIS
  // launch only (otherwise an upstream error is misreported as
  // "hip_rms_norm launch failed").
  (void)hipGetLastError();

  const bool vec2 =
      (hip_dtype == HIP_DTYPE_FLOAT16) &&
      rms_can_vectorize(input, scale, output, norm_size);

  CUSTOM_KERNELS_DEBUG_LOG(
      "[custom_kernels] hip_rms_norm: dtype=%d, outer=%lld, norm_size=%lld, "
      "scale_rows=%lld, epsilon=%f, vec2=%s\n",
      hip_dtype, (long long)outer, (long long)norm_size, (long long)scale_rows,
      epsilon, vec2 ? "yes" : "no");

  const bool narrow = vec2 && outer >= 128 &&
                      (norm_size == 128 || norm_size == 256) &&
                      rms_narrow_enabled();
  CUSTOM_KERNELS_DEBUG_LOG(
      "[custom_kernels] rms_narrow: dtype=%d outer=%lld width=%lld "
      "scale_rows=%lld vec2=%d narrow=%d\n",
      hip_dtype, (long long)outer, (long long)norm_size, (long long)scale_rows,
      (int)vec2, (int)narrow);
  if (hip_dtype == HIP_DTYPE_FLOAT16) {
    if (narrow) {
      dim3 narrow_grid(static_cast<unsigned int>((outer + 7) / 8));
      if (norm_size == 128) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_rms_norm_narrow_kernel<128>),
                           narrow_grid, block, 0, hip_stream,
                           static_cast<const __half *>(input),
                           static_cast<const __half *>(scale),
                           static_cast<__half *>(output), outer, scale_rows,
                           epsilon);
      } else {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_rms_norm_narrow_kernel<256>),
                           narrow_grid, block, 0, hip_stream,
                           static_cast<const __half *>(input),
                           static_cast<const __half *>(scale),
                           static_cast<__half *>(output), outer, scale_rows,
                           epsilon);
      }
    } else if (vec2) {
      hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_rms_norm_kernel<__half, true>),
                         grid, block, 0, hip_stream,
                         static_cast<const __half *>(input),
                         static_cast<const __half *>(scale),
                         static_cast<__half *>(output), outer, norm_size,
                         scale_rows, epsilon);
    } else {
      hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_rms_norm_kernel<__half, false>),
                         grid, block, 0, hip_stream,
                         static_cast<const __half *>(input),
                         static_cast<const __half *>(scale),
                         static_cast<__half *>(output), outer, norm_size,
                         scale_rows, epsilon);
    }
  } else if (hip_dtype == HIP_DTYPE_FLOAT32) {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_rms_norm_kernel<float, false>), grid,
                       block, 0, hip_stream, static_cast<const float *>(input),
                       static_cast<const float *>(scale),
                       static_cast<float *>(output), outer, norm_size,
                       scale_rows, epsilon);
  } else {
    fprintf(stderr, "[custom_kernels] hip_rms_norm: unsupported dtype=%d\n",
            hip_dtype);
    return -1;
  }

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr, "[custom_kernels] hip_rms_norm launch failed: %s\n",
            hipGetErrorString(err));
  }
  return static_cast<int>(err);
}

extern "C" int hip_gated_rms_norm(void *stream, const void *input,
                                  const void *scale, const void *gate,
                                  void *output, int64_t outer,
                                  int64_t norm_size, int activation,
                                  float epsilon, int hip_dtype) {
  if (outer <= 0 || norm_size <= 0) {
    return 0;
  }

  // Size the block to the row so narrow groups (e.g. 128 wide) do not leave
  // most of a 256-thread block idle.
  int block_size = 32;
  while (block_size < norm_size && block_size < kRmsNormBlockSize) {
    block_size *= 2;
  }
  hipStream_t hip_stream = static_cast<hipStream_t>(stream);
  dim3 grid(static_cast<unsigned int>(outer));
  dim3 block(block_size);
  const bool silu = activation == 1;

  (void)hipGetLastError();

  CUSTOM_KERNELS_DEBUG_LOG(
      "[custom_kernels] hip_gated_rms_norm: dtype=%d, outer=%lld, "
      "norm_size=%lld, activation=%d, block=%d\n",
      hip_dtype, (long long)outer, (long long)norm_size, activation,
      block_size);

  if (hip_dtype == HIP_DTYPE_FLOAT16) {
    auto *x = static_cast<const __half *>(input);
    auto *s = static_cast<const __half *>(scale);
    auto *z = static_cast<const __half *>(gate);
    auto *y = static_cast<__half *>(output);
    if (silu) {
      hipLaunchKernelGGL(
          HIP_KERNEL_NAME(hip_gated_rms_norm_kernel<__half, true>), grid,
          block, 0, hip_stream, x, s, z, y, outer, norm_size, epsilon);
    } else {
      hipLaunchKernelGGL(
          HIP_KERNEL_NAME(hip_gated_rms_norm_kernel<__half, false>), grid,
          block, 0, hip_stream, x, s, z, y, outer, norm_size, epsilon);
    }
  } else if (hip_dtype == HIP_DTYPE_FLOAT32) {
    auto *x = static_cast<const float *>(input);
    auto *s = static_cast<const float *>(scale);
    auto *z = static_cast<const float *>(gate);
    auto *y = static_cast<float *>(output);
    if (silu) {
      hipLaunchKernelGGL(
          HIP_KERNEL_NAME(hip_gated_rms_norm_kernel<float, true>), grid, block,
          0, hip_stream, x, s, z, y, outer, norm_size, epsilon);
    } else {
      hipLaunchKernelGGL(
          HIP_KERNEL_NAME(hip_gated_rms_norm_kernel<float, false>), grid,
          block, 0, hip_stream, x, s, z, y, outer, norm_size, epsilon);
    }
  } else {
    fprintf(stderr,
            "[custom_kernels] hip_gated_rms_norm: unsupported dtype=%d\n",
            hip_dtype);
    return -1;
  }

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr, "[custom_kernels] hip_gated_rms_norm launch failed: %s\n",
            hipGetErrorString(err));
  }
  return static_cast<int>(err);
}

extern "C" int hip_engram_gate(void *stream, const void *key, const void *query,
                               const void *value, const void *key_scale,
                               const void *query_scale, const void *conv_scale,
                               void *output, void *normed, int64_t rows,
                               int64_t hc, int64_t hidden, float epsilon,
                               int hip_dtype) {
  if (rows <= 0 || hidden <= 0) {
    return 0;
  }
  if (hc <= 0 || (normed != nullptr && conv_scale == nullptr)) {
    fprintf(stderr,
            "[custom_kernels] hip_engram_gate: hc=%lld, normed without "
            "conv_scale=%d\n",
            (long long)hc, normed != nullptr && conv_scale == nullptr);
    return -1;
  }

  hipStream_t hip_stream = static_cast<hipStream_t>(stream);
  dim3 grid(static_cast<unsigned int>(rows));
  dim3 block(kRmsNormBlockSize);

  (void)hipGetLastError();

  CUSTOM_KERNELS_DEBUG_LOG(
      "[custom_kernels] hip_engram_gate: dtype=%d, rows=%lld, hc=%lld, "
      "hidden=%lld, normed=%s\n",
      hip_dtype, (long long)rows, (long long)hc, (long long)hidden,
      normed ? "yes" : "no");

  if (hip_dtype == HIP_DTYPE_FLOAT16) {
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(hip_engram_gate_kernel<__half>), grid, block, 0,
        hip_stream, static_cast<const __half *>(key),
        static_cast<const __half *>(query), static_cast<const __half *>(value),
        static_cast<const __half *>(key_scale),
        static_cast<const __half *>(query_scale),
        static_cast<const __half *>(conv_scale), static_cast<__half *>(output),
        static_cast<__half *>(normed), rows, hc, hidden, epsilon);
  } else if (hip_dtype == HIP_DTYPE_FLOAT32) {
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(hip_engram_gate_kernel<float>), grid, block, 0,
        hip_stream, static_cast<const float *>(key),
        static_cast<const float *>(query), static_cast<const float *>(value),
        static_cast<const float *>(key_scale),
        static_cast<const float *>(query_scale),
        static_cast<const float *>(conv_scale), static_cast<float *>(output),
        static_cast<float *>(normed), rows, hc, hidden, epsilon);
  } else {
    fprintf(stderr, "[custom_kernels] hip_engram_gate: unsupported dtype=%d\n",
            hip_dtype);
    return -1;
  }

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr, "[custom_kernels] hip_engram_gate launch failed: %s\n",
            hipGetErrorString(err));
  }
  return static_cast<int>(err);
}
