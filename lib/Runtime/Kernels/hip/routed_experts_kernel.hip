// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
// Diagnostic prototype. Sparse execution requires finite inactive experts.
#include "routed_experts.h"
#include "hip_arch_compat.h"
#include "matmul_nbits_wmma_core.h"
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

// The direct test can compare tiles without changing the default runtime.
#ifndef NEMOTRON_ROUTED_W4A8_TILE
#define NEMOTRON_ROUTED_W4A8_TILE 2
#endif
static_assert(NEMOTRON_ROUTED_W4A8_TILE == 1 ||
              NEMOTRON_ROUTED_W4A8_TILE == 2 ||
              NEMOTRON_ROUTED_W4A8_TILE == 4 ||
              NEMOTRON_ROUTED_W4A8_TILE == 8);

namespace {
constexpr int kThreads = 128;

// Match the per-group activation quantization of the default MatMulNBits path.
__global__ void quantize_input(const __half *input, int8_t *quant, float *scales,
                              int width, int block_size) {
  const int row = blockIdx.y, group = blockIdx.x, tid = threadIdx.x;
  const int begin = group * block_size;
  const int groups = (width + block_size - 1) / block_size;
  float maximum = 0.0f;
  for (int j = tid; j < block_size && begin + j < width; j += kThreads)
    maximum = fmaxf(maximum, fabsf(__half2float(input[size_t(row) * width + begin + j])));
  for (int offset = warpSize / 2; offset; offset /= 2)
    maximum = fmaxf(maximum, __shfl_down(maximum, offset));
  __shared__ float shared[4];
  if (tid % warpSize == 0) shared[tid / warpSize] = maximum;
  __syncthreads();
  if (tid == 0) {
    float value = 0.0f;
    for (int wave = 0; wave < kThreads / warpSize; ++wave)
      value = fmaxf(value, shared[wave]);
    shared[0] = value;
    scales[row * groups + group] = value > 0 ? value * (1.0f / 127.0f) : 1.0f;
  }
  __syncthreads();
  const float inverse = shared[0] > 0 ? 127.0f / shared[0] : 0.0f;
  for (int j = tid; j < block_size && begin + j < width; j += kThreads) {
    const int column = begin + j;
    int q = __float2int_rn(__half2float(input[size_t(row) * width + column]) * inverse);
    q = max(-127, min(127, q));
    const int position = column % 8;
    const int offset = column / 8 * 8 + (position & 1) * 4 + position / 2;
    quant[size_t(row) * width + offset] = static_cast<int8_t>(q);
  }
}

// Mode 0 uses float dequantization. Mode 1 uses W4A8. Mode 2 rounds weights.
template <bool Up, int Mode>
__global__ void routed_gemv(
    const __half *__restrict__ input, const int64_t *__restrict__ indices,
    const float *__restrict__ gates, RoutedMatrixTable table,
    __half *__restrict__ output, int contraction, int columns, int experts,
    int k, int block_size, const int8_t *quant, const float *quant_scales) {
  constexpr int threads = Mode == 1 ? 32 : kThreads;
  constexpr int kTile = Mode == 1 ? NEMOTRON_ROUTED_W4A8_TILE : 8;
  const int pair = blockIdx.y;
  const int token = pair / k;
  const int expert = indices[pair];
  if (expert < 0 || expert >= experts)
    return;
  const int tid = threadIdx.x;
  const int column = blockIdx.x * kTile;
  const int groups = (contraction + block_size - 1) / block_size;
  const int row_bytes = groups * (block_size / 2);
  const int zp_bytes = (groups + 1) / 2;
  const __half *a = input + static_cast<size_t>(Up ? token : pair) * contraction;
  const uint8_t *w = table.weights[expert];
  const __half *scales = static_cast<const __half *>(table.scales[expert]);
  const uint8_t *zp = table.zero_points[expert];
  float sum[kTile] = {};

  if constexpr (Mode == 1) {
    const int row = Up ? token : pair;
    for (int chunk = tid; chunk < contraction / 32; chunk += threads) {
      const int group = chunk * 32 / block_size;
      const int8_t *aq = quant + size_t(row) * contraction + chunk * 32;
      int low[4], high[4], asum = 0;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        low[j] = *reinterpret_cast<const int *>(aq + j * 8);
        high[j] = *reinterpret_cast<const int *>(aq + j * 8 + 4);
        asum = hipdnn_sudot4(low[j], 0x01010101, asum);
        asum = hipdnn_sudot4(high[j], 0x01010101, asum);
      }
      const float ascale = quant_scales[row * groups + group];
#pragma unroll
      for (int t = 0; t < kTile; ++t) {
        const int n = column + t;
        if (n >= columns) continue;
        int dot = 0;
        const auto *packed = reinterpret_cast<const uint32_t *>(
            w + size_t(n) * row_bytes + chunk * 16);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          dot = hipdnn_sudot4(low[j], packed[j] & 0x0f0f0f0f, dot);
          dot = hipdnn_sudot4(high[j], (packed[j] >> 4) & 0x0f0f0f0f, dot);
        }
        const int zero = zp ? ((zp[n * zp_bytes + group / 2] >> ((group & 1) * 4)) & 15) : 8;
        sum[t] += ascale * __half2float(scales[n * groups + group]) * float(dot - asum * zero);
      }
    }
  } else for (int base = tid * 8; base < contraction; base += kThreads * 8) {
    float values[8];
#pragma unroll
    for (int j = 0; j < 8; ++j)
      values[j] = base + j < contraction ? __half2float(a[base + j]) : 0.0f;
    const int group = base / block_size;
#pragma unroll
    for (int t = 0; t < kTile; ++t) {
      const int n = column + t;
      if (n >= columns)
        continue;
      const uint32_t packed = *reinterpret_cast<const uint32_t *>(
          w + static_cast<size_t>(n) * row_bytes + base / 2);
      const int zero = zp ? ((zp[n * zp_bytes + group / 2] >>
                             ((group & 1) * 4)) & 15) : 8;
      const float scale = __half2float(scales[n * groups + group]);
      float dot = 0.0f;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        float weight = float(int((packed >> (j * 4)) & 15) - zero);
        if constexpr (Mode == 2)
          weight = __half2float(__float2half(weight * scale));
        dot = __fmaf_rn(values[j], weight, dot);
      }
      sum[t] += Mode == 2 ? dot : dot * scale;
    }
  }

  const int lane = tid % warpSize;
  const int wave = tid / warpSize;
  const int waves = (threads + warpSize - 1) / warpSize;
  const int reductionWidth = min(threads, warpSize);
  __shared__ float partial[kTile * 4];
#pragma unroll
  for (int t = 0; t < kTile; ++t) {
    for (int offset = reductionWidth / 2; offset; offset /= 2)
      sum[t] += __shfl_down(sum[t], offset);
    if (lane == 0)
      partial[t * waves + wave] = sum[t];
  }
  __syncthreads();
  if (wave != 0)
    return;
#pragma unroll
  for (int t = 0; t < kTile; ++t) {
    float value = lane < waves ? partial[t * waves + lane] : 0.0f;
    for (int offset = reductionWidth / 2; offset; offset /= 2)
      value += __shfl_down(value, offset);
    const int n = column + t;
    if (lane == 0 && n < columns) {
      // Keep the original MatMulNBits result rounding before activation.
      value = __half2float(__float2half(value));
      if (Up) {
        value = fmaxf(value, 0.0f);
        value *= value;
      } else {
        value *= __half2float(__float2half(gates[pair]));
      }
      output[static_cast<size_t>(pair) * columns + n] = __float2half(value);
    }
  }
}

// Build compact expert tiles on the GPU. Keep token order within each expert.
template <int TileRows>
__global__ void group_routes(const int64_t *indices, int *routing,
                             int pairs, int experts, int k) {
  const int expert = threadIdx.x;
  __shared__ int counts[128];
  int count = 0;
  if (expert < experts)
    for (int pair = 0; pair < pairs; ++pair)
      count += indices[pair] == expert;
  counts[expert] = count;
  __syncthreads();
  if (expert >= experts) return;
  int first = 0, tile_first = 0;
  for (int e = 0; e < expert; ++e) {
    first += counts[e];
    tile_first += (counts[e] + TileRows - 1) / TileRows;
  }
  int *rows = routing + 1;
  int *inverse = rows + pairs;
  int *tiles = inverse + pairs;
  int row = first;
  for (int pair = 0; pair < pairs; ++pair) {
    if (indices[pair] == expert) {
      rows[row] = pair / k;
      inverse[pair] = row++;
    }
  }
  for (int offset = 0; offset < count; offset += TileRows) {
    int *tile = tiles + (tile_first + offset / TileRows) * 3;
    tile[0] = expert;
    tile[1] = first + offset;
    tile[2] = min(TileRows, count - offset);
  }
  if (expert == experts - 1)
    routing[0] = tile_first + (count + TileRows - 1) / TileRows;
}

template <bool Up, int TileRows>
__global__ void grouped_gemm(const _Float16 *input, _Float16 *output,
                             RoutedMatrixTable table, const int *routing,
                             int pairs, int columns, int contraction, int block) {
  if (blockIdx.y >= routing[0]) return;
  const int *tile = routing + 1 + pairs * 2 + blockIdx.y * 3;
  const int expert = tile[0];
  // Reuse dense prefill arithmetic, with original packed weights and zeros.
  GemmFp16U4Impl<TileRows, 64, TileRows / 16, 1, /*USE_ZEROS=*/true,
      /*BOUNDS_CHECK=*/true, /*FUSED_DQ=*/true, 32,
      /*ZEROS_U8=*/true, /*ADD_BIAS=*/false, /*EXPLICIT_TILE=*/true,
      /*CHECK_K_TAIL=*/true, /*RUNTIME_OPTIONAL=*/true,
      /*ZEROS_PACKED=*/true>(
      pairs, columns, contraction, input, contraction, table.weights[expert],
      static_cast<const _Float16 *>(table.scales[expert]), table.zero_points[expert],
      block, (contraction + block - 1) / block, output, columns, 1, 0,
      tile[1], blockIdx.x * 64, tile[2], Up ? routing + 1 : nullptr);
}

__global__ void relu_squared(__half *data, size_t size) {
  const size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < size) {
    const float value = fmaxf(__half2float(data[index]), 0.0f);
    data[index] = __float2half(value * value);
  }
}

__global__ void ordered_sum(const __half *slots, const int64_t *indices,
                            __half *output, int hidden, int k,
                            const int *inverse, const float *gates) {
  const int token = blockIdx.y;
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ int order[128];
  if (threadIdx.x < k) {
    const int slot = threadIdx.x;
    int rank = 0;
    for (int other = 0; other < k; ++other)
      rank += indices[token * k + other] < indices[token * k + slot];
    order[rank] = slot;
  }
  __syncthreads();
  if (n >= hidden)
    return;
  // The source adds experts in increasing ID order and rounds each sum.
  __half sum = __float2half(0.0f);
  for (int rank = 0; rank < k; ++rank) {
    const size_t pair = static_cast<size_t>(token) * k + order[rank];
    __half value = slots[(inverse ? size_t(inverse[pair]) : pair) * hidden + n];
    if (inverse)
      value = __hmul_rn(value, __float2half(gates[pair]));
    // The source has separate multiply and add kernels. Do not fuse them.
    sum = __hadd_rn(sum, value);
  }
  output[static_cast<size_t>(token) * hidden + n] = sum;
}
} // namespace

template <int TileRows>
static hipError_t launch_grouped_tiled(hipStream_t stream, const void *input,
    const int64_t *indices, const RoutedMatrixTable *up,
    const RoutedMatrixTable *down, void *activation, void *slots, int *routing,
    int tokens, int hidden, int intermediate, int experts, int k, int block) {
  const int pairs = tokens * k;
  const int maxTiles = pairs / TileRows + experts;
  hipLaunchKernelGGL((group_routes<TileRows>), dim3(1), dim3(128), 0, stream,
                    indices, routing, pairs, experts, k);
  auto error = hipGetLastError();
  if (error != hipSuccess) return error;
  hipLaunchKernelGGL((grouped_gemm<true, TileRows>),
                    dim3((intermediate + 63) / 64, maxTiles),
                    dim3(128), 0, stream, static_cast<const _Float16 *>(input),
                    static_cast<_Float16 *>(activation), *up, routing,
                    pairs, intermediate, hidden, block);
  error = hipGetLastError();
  if (error != hipSuccess) return error;
  const size_t values = size_t(pairs) * intermediate;
  hipLaunchKernelGGL(relu_squared, dim3((values + 255) / 256), dim3(256),
                    0, stream, static_cast<__half *>(activation), values);
  error = hipGetLastError();
  if (error != hipSuccess) return error;
  hipLaunchKernelGGL((grouped_gemm<false, TileRows>),
                    dim3((hidden + 63) / 64, maxTiles),
                    dim3(128), 0, stream, static_cast<const _Float16 *>(activation),
                    static_cast<_Float16 *>(slots), *down, routing,
                    pairs, hidden, intermediate, block);
  return hipGetLastError();
}

static hipError_t launch_grouped(hipStream_t stream, const void *input,
    const int64_t *indices, const RoutedMatrixTable *up,
    const RoutedMatrixTable *down, void *activation, void *slots, int *routing,
    int tokens, int hidden, int intermediate, int experts, int k, int block) {
  // Reuse each weight tile for more rows in large prefill chunks. Keep the
  // smaller tile for sparse groups. Each output keeps the same K reduction.
  if (tokens * k >= experts * 16)
    return launch_grouped_tiled<32>(stream, input, indices, up, down, activation,
        slots, routing, tokens, hidden, intermediate, experts, k, block);
  return launch_grouped_tiled<16>(stream, input, indices, up, down, activation,
      slots, routing, tokens, hidden, intermediate, experts, k, block);
}

template <int Mode>
static hipError_t launch_stages(hipStream_t s, const void *input,
    const int64_t *indices, const float *gates, const RoutedMatrixTable *up,
    const RoutedMatrixTable *down, void *activation, void *slots,
    int tokens, int hidden, int intermediate, int experts, int k, int block_size,
    void *quant, float *quant_scales) {
  constexpr int kTile = Mode == 1 ? NEMOTRON_ROUTED_W4A8_TILE : 8;
  auto quantize = [&](const void *data, int rows, int width) {
    hipLaunchKernelGGL(quantize_input, dim3((width + block_size - 1) / block_size, rows),
                      dim3(kThreads), 0, s, static_cast<const __half *>(data),
                      static_cast<int8_t *>(quant), quant_scales, width, block_size);
    return hipGetLastError();
  };
  if constexpr (Mode == 1) {
    auto error = quantize(input, tokens, hidden);
    if (error != hipSuccess) return error;
  }
  hipLaunchKernelGGL((routed_gemv<true, Mode>), dim3((intermediate + kTile - 1) / kTile, tokens * k),
                    dim3(Mode == 1 ? 32 : kThreads), 0, s, static_cast<const __half *>(input),
                    indices, gates, *up, static_cast<__half *>(activation), hidden,
                    intermediate, experts, k, block_size,
                    static_cast<const int8_t *>(quant), quant_scales);
  auto error = hipGetLastError();
  if (error != hipSuccess) return error;
  if constexpr (Mode == 1) {
    error = quantize(activation, tokens * k, intermediate);
    if (error != hipSuccess) return error;
  }
  hipLaunchKernelGGL((routed_gemv<false, Mode>), dim3((hidden + kTile - 1) / kTile, tokens * k),
                    dim3(Mode == 1 ? 32 : kThreads), 0, s, static_cast<const __half *>(activation),
                    indices, gates, *down, static_cast<__half *>(slots),
                    intermediate, hidden, experts, k, block_size,
                    static_cast<const int8_t *>(quant), quant_scales);
  return hipGetLastError();
}

extern "C" HIP_KERNEL_API int hip_routed_experts(
    void *stream, const void *input, const int64_t *indices,
    const float *gates, const RoutedMatrixTable *up,
    const RoutedMatrixTable *down, void *activation, void *slots, void *output,
    int tokens, int hidden, int intermediate, int experts, int k, int block_size,
    int mode, void *quant, float *quant_scales) {
  if (tokens == 0)
    return 0;
  if (!input || !indices || !gates || !up || !down || !activation || !slots ||
      !output || tokens < 0 || hidden <= 0 || intermediate <= 0 || experts < 1 ||
      experts > 128 || k < 1 || k > experts || tokens > 65535 / k ||
      (block_size != 32 && block_size != 64 && block_size != 128) || mode < 0 || mode > 3 ||
      (mode == 1 && (!quant || !quant_scales || hidden % 32 || intermediate % 32)) ||
      (mode == 3 && (!quant || hidden % 32 || intermediate % 32)))
    return hipErrorInvalidValue;
  for (int e = 0; e < experts; ++e)
    if (!up->weights[e] || !up->scales[e] || !down->weights[e] || !down->scales[e])
      return hipErrorInvalidValue;
  auto s = static_cast<hipStream_t>(stream);
  hipError_t error;
#define LAUNCH(MODE) launch_stages<MODE>(s, input, indices, gates, up, down, activation, \
    slots, tokens, hidden, intermediate, experts, k, block_size, quant, quant_scales)
  if (mode == 3)
    error = launch_grouped(s, input, indices, up, down, activation, slots,
        static_cast<int *>(quant), tokens, hidden, intermediate, experts, k, block_size);
  else if (mode == 1) error = LAUNCH(1);
  else if (mode == 2) error = LAUNCH(2);
  else error = LAUNCH(0);
#undef LAUNCH
  if (error != hipSuccess)
    return error;
  hipLaunchKernelGGL(ordered_sum, dim3((hidden + 255) / 256, tokens), dim3(256),
                    0, s, static_cast<const __half *>(slots), indices,
                    static_cast<__half *>(output), hidden, k,
                    mode == 3 ? static_cast<int *>(quant) + 1 + tokens * k : nullptr,
                    gates);
  return hipGetLastError();
}
