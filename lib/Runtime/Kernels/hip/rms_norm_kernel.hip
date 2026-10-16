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

#include "activation_quant_i8.h"
#include "hip_custom_kernels.h"
#include "debug_log.h"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <cstdint>
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

// hip.gated_rms_norm: every `group`-wide slice of the input is one item, R
// threads per item and kRmsNormBlockSize / R items per block. R covers the
// half2 pairs of a group in one stride when the group fits (R >= group / 2),
// so the R-wide tree adds the same partials in the same order as
// hip_rms_norm_kernel's 256-wide one, whose other lanes hold zeros: n is
// bit-identical. The gate then repeats hip.sigmoid, hip.mul and hip.cast in
// fp32.
//
// QUANT (group % 32 == 0) also quantizes y as hip_skip_rms_norm_kernel does:
// a 32-element quantization group is 16 lanes of one item, aligned to 16.
template <int R, bool QUANT>
__global__ void __launch_bounds__(kRmsNormBlockSize)
    hip_gated_rms_norm_kernel(const __half2 *__restrict__ input,
                              const __half2 *__restrict__ gate,
                              const __half2 *__restrict__ weight,
                              __half2 *__restrict__ output,
                              int8_t *__restrict__ q_out,
                              float *__restrict__ q_scale, int64_t items,
                              int64_t groups_per_row, int64_t half_group,
                              int64_t row_size, float epsilon) {
  static_assert(R >= 16 && R % 16 == 0 && kRmsNormBlockSize % R == 0,
                "an item spans whole 16-lane quantization groups");
  __shared__ float s_reduce[kRmsNormBlockSize];
  const int lane = threadIdx.x % R;
  const int64_t item = static_cast<int64_t>(blockIdx.x) *
                           (kRmsNormBlockSize / R) +
                       threadIdx.x / R;
  const bool active = item < items;
  const __half2 *x2 = input + item * half_group;

  float local_sumsq = 0.0f;
  if (active)
    for (int64_t i = lane; i < half_group; i += R) {
      float2 v = __half22float2(x2[i]);
      local_sumsq += v.x * v.x + v.y * v.y;
    }
  s_reduce[threadIdx.x] = local_sumsq;
  __syncthreads();
  for (int stride = R / 2; stride > 0; stride >>= 1) {
    if (lane < stride)
      s_reduce[threadIdx.x] += s_reduce[threadIdx.x + stride];
    __syncthreads();
  }
  if (!active)
    return;
  const float inv_rms =
      rsqrtf(s_reduce[threadIdx.x - lane] /
                 static_cast<float>(2 * half_group) +
             epsilon);

  const __half2 *z2 = gate + item * half_group;
  __half2 *y2 = output + item * half_group;
  const int64_t row = item / groups_per_row;
  const int64_t row_pair = (item % groups_per_row) * half_group;
  for (int64_t i = lane; i < half_group; i += R) {
    float2 v = __half22float2(x2[i]);
    float2 g = __half22float2(weight[i]);
    float2 r;
    r.x = v.x * inv_rms * g.x;
    r.y = v.y * inv_rms * g.y;
    const float2 n = __half22float2(__float22half2_rn(r));
    const float2 z = __half22float2(z2[i]);
    const float sx = 1.0f / (1.0f + expf(-z.x));
    const float sy = 1.0f / (1.0f + expf(-z.y));
    float2 o;
    o.x = n.x * (z.x * sx);
    o.y = n.y * (z.y * sy);
    const __half2 yh = __float22half2_rn(o);
    y2[i] = yh;
    if constexpr (QUANT)
      quant_act_i8_pair(__half22float2(yh), row_pair + i,
                        q_out + row * row_size,
                        q_scale + row * (row_size / 32));
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

  if (hip_dtype == HIP_DTYPE_FLOAT16) {
    if (vec2) {
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

namespace {

template <int R>
void launch_gated_rms_norm(hipStream_t stream, const void *input,
                           const void *gate, const void *weight, void *output,
                           void *q_out, void *q_scale, int64_t items,
                           int64_t groups_per_row, int64_t half_group,
                           int64_t row_size, float epsilon) {
  constexpr int64_t kItemsPerBlock = kRmsNormBlockSize / R;
  dim3 grid(
      static_cast<unsigned int>((items + kItemsPerBlock - 1) / kItemsPerBlock));
  auto args = [&](auto kernel) {
    hipLaunchKernelGGL(kernel, grid, dim3(kRmsNormBlockSize), 0, stream,
                       static_cast<const __half2 *>(input),
                       static_cast<const __half2 *>(gate),
                       static_cast<const __half2 *>(weight),
                       static_cast<__half2 *>(output),
                       static_cast<int8_t *>(q_out),
                       static_cast<float *>(q_scale), items, groups_per_row,
                       half_group, row_size, epsilon);
  };
  if (q_out)
    args(hip_gated_rms_norm_kernel<R, true>);
  else
    args(hip_gated_rms_norm_kernel<R, false>);
}

} // namespace

extern "C" int hip_gated_rms_norm(void *stream, const void *input,
                                  const void *gate, const void *weight,
                                  void *output, void *q_out, void *q_scale,
                                  int64_t rows, int64_t row_size,
                                  int64_t group_size, float epsilon,
                                  int64_t quant_block_size) {
  auto aligned = [](const void *p) {
    return (reinterpret_cast<uintptr_t>(p) % 4) == 0;
  };
  if (group_size <= 0 || group_size % 2 != 0 || row_size % group_size != 0 ||
      !aligned(input) || !aligned(gate) || !aligned(weight) || !aligned(output))
    return hipErrorNotSupported;
  if ((q_out != nullptr) != (q_scale != nullptr) ||
      (q_out && (quant_block_size != 32 || group_size % 32 != 0)))
    return hipErrorNotSupported;
  if (rows <= 0)
    return 0;

  const int64_t half_group = group_size / 2;
  const int64_t groups_per_row = row_size / group_size;
  const int64_t items = rows * groups_per_row;
  hipStream_t hip_stream = static_cast<hipStream_t>(stream);
  (void)hipGetLastError();
  if (half_group <= 32)
    launch_gated_rms_norm<32>(hip_stream, input, gate, weight, output, q_out,
                              q_scale, items, groups_per_row, half_group,
                              row_size, epsilon);
  else if (half_group <= 64)
    launch_gated_rms_norm<64>(hip_stream, input, gate, weight, output, q_out,
                              q_scale, items, groups_per_row, half_group,
                              row_size, epsilon);
  else if (half_group <= 128)
    launch_gated_rms_norm<128>(hip_stream, input, gate, weight, output, q_out,
                               q_scale, items, groups_per_row, half_group,
                               row_size, epsilon);
  else
    launch_gated_rms_norm<256>(hip_stream, input, gate, weight, output, q_out,
                               q_scale, items, groups_per_row, half_group,
                               row_size, epsilon);

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr, "[custom_kernels] hip_gated_rms_norm launch failed: %s\n",
            hipGetErrorString(err));
  }
  return static_cast<int>(err);
}
