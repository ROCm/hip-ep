/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
// FusedOp(x, w) = Q(Conv1x1(DQ(x), DQ(w)))    -- W4A16
//
// x = (X - z_x) * s_x
// w = (W - z_w[co]) * s_w[co]
// OUT = saturate(round((conv(x, w) + bias) / s_out) + z_out)
//
// A 1x1 kernel with unit stride, unit dilation, no padding and no grouping has
// no window left to slide: each output position reads exactly one input
// position, straight down the channel axis. The convolution collapses to a
// matrix multiply,
//
//   acc[co, p] = sum over ci of (X[ci, p] - z_x) * (W[co, ci] - z_w[co])
//
// with M = out_channels, K = in_channels, N = spatial_size. The caller has
// already established that geometry (see QDQConvFusion.pdll), so no im2col
// gather and no per-axis extents appear here.
//
// Both scales leave the sum because neither depends on ci:
//
//   OUT = saturate(round(M[co] * acc + bias[co] * inv_output_scale) + z_out)
//   M[co] = act_out_scale_ratio * weight_scales[co]
//
// act_out_scale_ratio (= s_x / s_out) and inv_output_scale (= 1 / s_out) are
// folded by the caller, so the kernel never divides. The per-channel factor
// cannot be folded away: weight_scales is a device array of one f32 per output
// channel.
//
// Zero points
// -----------
// Unlike hip_qmatmul, which carries its zero points as rowA / colB correction
// terms, this kernel subtracts w's zero point during staging and keeps it there,
// so the inner loop is one multiply-add per output channel. z_x cannot be folded
// the same way -- a 16-bit activation minus its zero point needs 17 bits -- so it
// is subtracted once per k, hoisted out of the per-channel loop.
//
// Accumulator width
// -----------------
// |W - z_w| is at most 15 for a 4-bit weight but 255 for an 8-bit one, so the
// staged difference is int16 rather than int8: a signed 8-bit weight minus its
// zero point spans -255..255 and does not fit a byte.
//
// With |X - z_x| <= 65535 that bounds one term at 1.7e7, so int32 overflows
// after about 128 terms. K is 2048 or 3072 on the shapes this was written for
// and nothing bounds it in general, so the tiled path accumulates one K-tile in
// int32 (kTileK = 32 terms, ~5.4e8, exact) and flushes into an int64 total per
// tile. The result is exact for any K while the inner loop stays 32-bit.
//
// The GEMV path accumulates in int64 directly: it is bandwidth-bound, so the
// wider adds are free and the tiling would only add work.

#include "debug_log.h"
#include "hip_custom_kernels.h"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>

namespace {

// The HIP compile flags include `-Xclang -fno-cuda-host-device-constexpr`
// (see lib/Runtime/Kernels/cmake/hip_utils.cmake), which makes
// `std::numeric_limits<T>` members host-only, hence the explicit bounds.
template <typename T> struct QConvBounds;

template <> struct QConvBounds<uint16_t> {
  __device__ static float lowest() { return 0.0f; }
  __device__ static float highest() { return static_cast<float>(UINT16_MAX); }
};

template <typename T> __device__ inline T q_saturate(float v) {
  return static_cast<T>(
      fminf(QConvBounds<T>::highest(), fmaxf(QConvBounds<T>::lowest(), v)));
}

// Read one LOGICAL element of a weight or weight-zero-point buffer.
//
// `bits` is 4 or 8. At 4, two values share a byte, low nibble first over the
// flattened row-major sequence -- the ONNX INT4/UINT4 convention, matching
// qdq_kernel.hip and gather_block_quantized_kernel.hip. `lin` counts logical
// elements, never bytes.
//
// `is_signed` decides how a nibble widens, and the two disagree on every value
// above 7, so it must track the weight's element type rather than being
// assumed.
__device__ inline int qconv_read_quant(const uint8_t *__restrict__ buf,
                                       int64_t lin, int bits, bool is_signed) {
  if (bits == 4) {
    const uint8_t byte = buf[lin >> 1];
    const uint8_t nibble = (lin & 1) ? (byte >> 4) : (byte & 0xF);
    if (is_signed) {
      // Shifting into the high nibble of an int8 and back propagates bit 3.
      return static_cast<int>(static_cast<int8_t>(nibble << 4)) >> 4;
    }
    return static_cast<int>(nibble);
  }
  if (is_signed)
    return static_cast<int>(static_cast<int8_t>(buf[lin]));
  return static_cast<int>(buf[lin]);
}

// Requantize one accumulated output element. The narrowing to float is safe
// because m_co immediately brings the value back into the output's quantized
// range, leaving float's ~6e-8 relative error well under one LSB of a 16-bit
// output.
template <typename TAct>
__device__ inline TAct qconv_epilogue(int64_t acc, float m_co,
                                      const float *__restrict__ bias,
                                      int64_t co, float inv_output_scale,
                                      int32_t output_zp) {
  float scaled = m_co * static_cast<float>(acc);
  if (bias)
    scaled += inv_output_scale * bias[co];
  return q_saturate<TAct>(rintf(scaled) + static_cast<float>(output_zp));
}

//===----------------------------------------------------------------------===//
// Tiled path (spatial_size > kGemvSpatialThreshold, i.e. prefill)
//===----------------------------------------------------------------------===//

// kThreadCo trades register blocking against parallelism, and on this GPU
// parallelism wins by a wide margin. Total threads launched is
// (Cout / kThreadCo) * P, while the device holds 20 CUs * 2048 threads = 40960
// resident. At kThreadCo = 8 a Cout = 1024 layer launches only 16384 threads and
// leaves 60% of the machine idle for the whole kernel; at 2 it launches 65536 and
// oversubscribes, which is what lets latency hiding work at all.
constexpr int kTileP = 64;
constexpr int kTileCo = 8;
constexpr int kTileK = 32;
constexpr int kThreadCo = 2;
constexpr int kCoGroups = kTileCo / kThreadCo;  // 4
constexpr int kTiledBlock = kTileP * kCoGroups; // 256

// Staging repetitions: how many elements of each LDS tile one thread fetches.
constexpr int kXRepeats = (kTileK * kTileP) / kTiledBlock;  // 8
constexpr int kWRepeats = (kTileCo * kTileK) / kTiledBlock; // 1
static_assert(kTileK * kTileP == kXRepeats * kTiledBlock, "x tile must divide");
static_assert(kTileCo * kTileK == kWRepeats * kTiledBlock, "w tile must divide");

// OUT[n, co, p] = requantize(sum_ci (X[n, ci, p] - z_x) * (W[co, ci] - z_w[co]))
//
// Grid: (ceil(P/kTileP), ceil(Cout/kTileCo), batch). Block: (kTileP, kCoGroups).
// Each thread owns one p and kThreadCo output channels, so a block stages one
// [kTileK x kTileP] slab of X and one [kTileCo x kTileK] slab of W per K-tile
// and every staged element is reused: an X element by kTileCo channels, a W
// element by kTileP positions.
//
// threadIdx.x is the position within the tile, which puts consecutive threads on
// consecutive p. That is what makes both the X staging reads and the output
// writes coalesced -- X is [ci][p] and OUT is [co][p], so p is the contiguous
// axis of both.
//
// Out-of-range K is staged as a zero WEIGHT, not a zero activation: the
// activation has z_x subtracted from it, so a zero there would contribute
// -z_x * w instead of nothing.
template <typename TAct>
__global__ __launch_bounds__(kTiledBlock) void qconv_tiled_kernel(
    const TAct *__restrict__ input, const uint8_t *__restrict__ weights,
    const float *__restrict__ weight_scales,
    const uint8_t *__restrict__ weight_zero_points,
    const float *__restrict__ bias, TAct *__restrict__ output,
    int64_t in_channels, int64_t out_channels, int64_t spatial_size,
    int weight_bits, bool weight_signed, float act_out_scale_ratio,
    int32_t input_zp, float inv_output_scale, int32_t output_zp) {
  __shared__ TAct x_lds[kTileK][kTileP];
  // int16, not int8: an 8-bit weight minus its zero point reaches +-255.
  __shared__ int16_t w_lds[kTileCo][kTileK];

  const int64_t n = blockIdx.z;
  const int64_t p_base = static_cast<int64_t>(blockIdx.x) * kTileP;
  const int64_t co_base = static_cast<int64_t>(blockIdx.y) * kTileCo;

  const int tx = threadIdx.x; // position within the tile
  const int ty = threadIdx.y; // output-channel group
  const int tid = ty * kTileP + tx;

  const TAct *x_n = input + n * in_channels * spatial_size;
  TAct *y_n = output + n * out_channels * spatial_size;

  // z_w depends only on the output channel, so each thread's staging slots keep
  // the same zero point for every K-tile. Hoisting it out of the loop turns a
  // per-tile reload into one read.
  int zw_stage[kWRepeats];
#pragma unroll
  for (int rep = 0; rep < kWRepeats; ++rep) {
    const int lin = rep * kTiledBlock + tid;
    const int64_t co = co_base + lin / kTileK;
    zw_stage[rep] =
        co < out_channels
            ? qconv_read_quant(weight_zero_points, co, weight_bits,
                               weight_signed)
            : 0;
  }

  int64_t acc[kThreadCo] = {};

  for (int64_t k0 = 0; k0 < in_channels; k0 += kTileK) {
#pragma unroll
    for (int rep = 0; rep < kXRepeats; ++rep) {
      const int lin = rep * kTiledBlock + tid;
      const int sk = lin / kTileP;
      const int sp = lin % kTileP;
      const int64_t k = k0 + sk;
      const int64_t p = p_base + sp;
      x_lds[sk][sp] = (k < in_channels && p < spatial_size)
                          ? x_n[k * spatial_size + p]
                          : TAct{0};
    }

#pragma unroll
    for (int rep = 0; rep < kWRepeats; ++rep) {
      const int lin = rep * kTiledBlock + tid;
      const int sc = lin / kTileK;
      const int sk = lin % kTileK;
      const int64_t co = co_base + sc;
      const int64_t k = k0 + sk;
      int16_t v = 0;
      if (co < out_channels && k < in_channels) {
        const int wq = qconv_read_quant(weights, co * in_channels + k,
                                        weight_bits, weight_signed);
        v = static_cast<int16_t>(wq - zw_stage[rep]);
      }
      w_lds[sc][sk] = v;
    }

    __syncthreads();

    // Exact in int32 for one tile: kTileK terms of at most ~1.7e7 each.
    int32_t chunk[kThreadCo] = {};
#pragma unroll
    for (int kk = 0; kk < kTileK; ++kk) {
      // Hoisted out of the channel loop: z_x is per-tensor, so this is the same
      // subtraction for all kThreadCo channels.
      const int32_t xv = static_cast<int32_t>(x_lds[kk][tx]) - input_zp;
#pragma unroll
      for (int c = 0; c < kThreadCo; ++c)
        chunk[c] += xv * static_cast<int32_t>(w_lds[ty * kThreadCo + c][kk]);
    }
#pragma unroll
    for (int c = 0; c < kThreadCo; ++c)
      acc[c] += chunk[c];

    __syncthreads();
  }

  const int64_t p = p_base + tx;
  if (p >= spatial_size)
    return;
#pragma unroll
  for (int c = 0; c < kThreadCo; ++c) {
    const int64_t co = co_base + ty * kThreadCo + c;
    if (co >= out_channels)
      break;
    const float m_co = act_out_scale_ratio * weight_scales[co];
    y_n[co * spatial_size + p] = qconv_epilogue<TAct>(
        acc[c], m_co, bias, co, inv_output_scale, output_zp);
  }
}

//===----------------------------------------------------------------------===//
// GEMV path (small spatial_size, i.e. LLM decode)
//===----------------------------------------------------------------------===//

constexpr int kGemvBlock = 256;

// One block per (n, co, p), reducing over K across the block.
//
// The tiled path degenerates at small P: it covers kTileP positions per block,
// so a single-token decode leaves 63 of 64 lanes idle. Here the whole block
// works on one output element instead, and at P == 1 both the weight row and
// the activation column are contiguous in ci, so both reads coalesce.
template <typename TAct>
__global__ __launch_bounds__(kGemvBlock) void qconv_gemv_kernel(
    const TAct *__restrict__ input, const uint8_t *__restrict__ weights,
    const float *__restrict__ weight_scales,
    const uint8_t *__restrict__ weight_zero_points,
    const float *__restrict__ bias, TAct *__restrict__ output,
    int64_t in_channels, int64_t out_channels, int64_t spatial_size,
    int weight_bits, bool weight_signed, float act_out_scale_ratio,
    int32_t input_zp, float inv_output_scale, int32_t output_zp) {
  __shared__ int64_t reduce[kGemvBlock];

  // The grid is sized from the extents, so all three are in range.
  const int64_t co = blockIdx.x;
  const int64_t p = blockIdx.y;
  const int64_t n = blockIdx.z;

  const TAct *x_n = input + n * in_channels * spatial_size;
  const int zw =
      qconv_read_quant(weight_zero_points, co, weight_bits, weight_signed);

  int64_t acc = 0;
  for (int64_t k = threadIdx.x; k < in_channels; k += kGemvBlock) {
    const int32_t xv =
        static_cast<int32_t>(x_n[k * spatial_size + p]) - input_zp;
    const int wq = qconv_read_quant(weights, co * in_channels + k, weight_bits,
                                    weight_signed);
    acc += static_cast<int64_t>(xv) * (wq - zw);
  }

  reduce[threadIdx.x] = acc;
  __syncthreads();
  for (int stride = kGemvBlock / 2; stride > 0; stride >>= 1) {
    if (static_cast<int>(threadIdx.x) < stride)
      reduce[threadIdx.x] += reduce[threadIdx.x + stride];
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    const float m_co = act_out_scale_ratio * weight_scales[co];
    output[n * out_channels * spatial_size + co * spatial_size + p] =
        qconv_epilogue<TAct>(reduce[0], m_co, bias, co, inv_output_scale,
                             output_zp);
  }
}

//===----------------------------------------------------------------------===//
// Dispatch
//===----------------------------------------------------------------------===//

// Below this many output positions the tiled path wastes more lanes than the
// GEMV path wastes bandwidth re-reading weights per position. A heuristic, not
// a correctness boundary: either kernel computes any shape.
constexpr int64_t kGemvSpatialThreshold = 8;

template <typename TAct>
int launch_qconv(hipStream_t stream, const void *input, const void *weights,
                 const void *weight_scales, const void *weight_zero_points,
                 const void *bias, void *output, int64_t batch,
                 int64_t in_channels, int64_t out_channels,
                 int64_t spatial_size, int weight_bits, bool weight_signed,
                 float act_out_scale_ratio, int32_t input_zp,
                 float inv_output_scale, int32_t output_zp) {
  const auto *x = static_cast<const TAct *>(input);
  const auto *w = static_cast<const uint8_t *>(weights);
  const auto *ws = static_cast<const float *>(weight_scales);
  const auto *wzp = static_cast<const uint8_t *>(weight_zero_points);
  const auto *b = static_cast<const float *>(bias);
  auto *y = static_cast<TAct *>(output);

  (void)hipGetLastError();
  if (spatial_size <= kGemvSpatialThreshold) {
    dim3 grid(static_cast<unsigned>(out_channels),
              static_cast<unsigned>(spatial_size),
              static_cast<unsigned>(batch));
    hipLaunchKernelGGL(HIP_KERNEL_NAME(qconv_gemv_kernel<TAct>), grid,
                       dim3(kGemvBlock), 0, stream, x, w, ws, wzp, b, y,
                       in_channels, out_channels, spatial_size, weight_bits,
                       weight_signed, act_out_scale_ratio, input_zp,
                       inv_output_scale, output_zp);
  } else {
    dim3 grid(
        static_cast<unsigned>((spatial_size + kTileP - 1) / kTileP),
        static_cast<unsigned>((out_channels + kTileCo - 1) / kTileCo),
        static_cast<unsigned>(batch));
    hipLaunchKernelGGL(HIP_KERNEL_NAME(qconv_tiled_kernel<TAct>), grid,
                       dim3(kTileP, kCoGroups), 0, stream, x, w, ws, wzp, b, y,
                       in_channels, out_channels, spatial_size, weight_bits,
                       weight_signed, act_out_scale_ratio, input_zp,
                       inv_output_scale, output_zp);
  }
  return 0;
}

} // namespace

extern "C" int hip_qconv(void *stream, const void *input, const void *weights,
                         const void *weight_scales,
                         const void *weight_zero_points, const void *bias,
                         void *output, int64_t batch, int64_t in_channels,
                         int64_t out_channels, int64_t spatial_size,
                         int act_dtype, int weight_dtype, int weight_bits,
                         float act_out_scale_ratio, int64_t input_zp,
                         float inv_output_scale, int64_t output_zp) {
  // weight_zero_points is required: the fusion only matches a three-operand
  // DequantizeLinear, so an absent zero point cannot reach here.
  if (!input || !weights || !weight_scales || !weight_zero_points || !output) {
    fprintf(stderr, "[custom_kernels] hip_qconv: null tensor argument\n");
    return -1;
  }
  if (batch <= 0 || in_channels <= 0 || out_channels <= 0 || spatial_size <= 0)
    return 0;
  if (weight_bits != 4 && weight_bits != 8) {
    fprintf(stderr, "[custom_kernels] hip_qconv: unsupported weight_bits=%d\n",
            weight_bits);
    return -1;
  }

  bool weight_signed;
  switch (weight_dtype) {
  case HIP_DTYPE_INT8:
    weight_signed = true;
    break;
  case HIP_DTYPE_UINT8:
    weight_signed = false;
    break;
  default:
    fprintf(stderr,
            "[custom_kernels] hip_qconv: unsupported weight_dtype=%d "
            "(expected 8-bit storage)\n",
            weight_dtype);
    return -1;
  }

  CUSTOM_KERNELS_DEBUG_LOG(
      "[custom_kernels] hip_qconv: N=%lld K=%lld M=%lld P=%lld act_dtype=%d "
      "weight_dtype=%d w_bits=%d ratio=%g z_x=%lld inv_s_out=%g z_out=%lld "
      "bias=%s\n",
      (long long)batch, (long long)in_channels, (long long)out_channels,
      (long long)spatial_size, act_dtype, weight_dtype, weight_bits,
      (double)act_out_scale_ratio, (long long)input_zp,
      (double)inv_output_scale, (long long)output_zp, bias ? "yes" : "null");

  int rc;
  switch (act_dtype) {
  case HIP_DTYPE_UINT16:
    rc = launch_qconv<uint16_t>(
        static_cast<hipStream_t>(stream), input, weights, weight_scales,
        weight_zero_points, bias, output, batch, in_channels, out_channels,
        spatial_size, weight_bits, weight_signed, act_out_scale_ratio,
        static_cast<int32_t>(input_zp), inv_output_scale,
        static_cast<int32_t>(output_zp));
    break;
  default:
    fprintf(stderr,
            "[custom_kernels] hip_qconv: unsupported act_dtype=%d "
            "(only UINT16 is implemented)\n",
            act_dtype);
    return -1;
  }
  if (rc != 0)
    return rc;

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr, "[custom_kernels] hip_qconv launch failed: %s\n",
            hipGetErrorString(err));
  }
  return static_cast<int>(err);
}
