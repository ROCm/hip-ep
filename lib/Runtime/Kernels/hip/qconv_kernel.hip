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
// already established that geometry, so no im2col
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
// The GEMV path chunks for the same reason: it stages kGemvVec terms, sums
// them in int32, and flushes into an int64 total.

#include "debug_log.h"
#include "hip_arch_compat.h"
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

// Read one LOGICAL element out of a weight chunk that was staged as whole
// 32-bit words.
//
// Loading by word rather than by element preserves the packing verbatim, so
// the ONNX order still applies within the word: little-endian puts byte 0 in
// the low bits, and byte 0 holds the earlier element in its low nibble, which
// leaves logical element j at bit 4j at 4 bits and bit 8j at 8.
//
// Width and signedness are template parameters, not arguments. As arguments
// they are two unpredictable branches per element in the innermost loop of the
// kernel; as parameters the shift, the mask and the widening all fold away.
template <int WBits, bool WSigned>
__device__ inline int qconv_unpack_word(const uint32_t *__restrict__ words,
                                        int j) {
  if (WBits == 4) {
    const uint32_t nibble = (words[j >> 3] >> ((j & 7) * 4)) & 0xF;
    if (WSigned)
      return static_cast<int>(static_cast<int8_t>(nibble << 4)) >> 4;
    return static_cast<int>(nibble);
  }
  const uint32_t byte = (words[j >> 2] >> ((j & 3) * 8)) & 0xFF;
  if (WSigned)
    return static_cast<int>(static_cast<int8_t>(byte));
  return static_cast<int>(byte);
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

// kTileCo sets how much of the activation traffic is shared rather than
// repeated, and it is the term that dominates this kernel.
//
// A block streams the whole of K for the kTileP positions it owns, whatever
// number of channels it produces, so that column is re-read once per co-tile:
// ceil(Cout / kTileCo) * K * P * sizeof(TAct) bytes against the
// ceil(P / kTileP) * Cout * K * bits/8 the weights cost. At a 5120-deep,
// 17920-wide layer with P = 512 and 4-bit weights the two are 11.7 GB and
// 367 MB, so 97% of what the kernel moves is the same activations over again,
// and dividing kTileCo into that is worth far more than anything done to the
// weight side.
//
// The cost is grid granularity: the grid is ceil(P / kTileP) by
// ceil(Cout / kTileCo), so raising kTileCo divides the block count by the same
// factor, and a narrow layer runs out of blocks to fill the machine before the
// saved traffic pays for it. 64 keeps 280 co-tiles on the widest layer here and
// still leaves a 1024-wide one with 16, which at 8 p-tiles is 128 blocks.
constexpr int kTileP = 64;
constexpr int kTileCo = 64;
constexpr int kTileK = 32;
constexpr int kThreadCo = 16;
constexpr int kCoGroups = kTileCo / kThreadCo;  // 4
constexpr int kTiledBlock = kTileP * kCoGroups; // 256

// Staging repetitions: how many elements of each LDS tile one thread fetches.
constexpr int kXRepeats = (kTileK * kTileP) / kTiledBlock;  // 8
constexpr int kWRepeats = (kTileCo * kTileK) / kTiledBlock; // 8
static_assert(kTileK * kTileP == kXRepeats * kTiledBlock, "x tile must divide");
static_assert(kTileCo * kTileK == kWRepeats * kTiledBlock, "w tile must divide");

// Padding on the channel axis of the staged weights, in elements.
//
// The inner loop wants the kThreadCo weights of one k contiguous so they arrive
// as 128-bit reads instead of kThreadCo separate ones, which needs the row
// stride to stay a multiple of 16 bytes, and 4 is the smallest padding that
// keeps that. No padding can also spread the staging writes, which step by a
// whole row: every stride that is a multiple of four dwords lands them on one
// bank. The reads outnumber those writes by more than an order of magnitude, so
// the read alignment is what the padding is chosen for.
constexpr int kWPad = 4;
static_assert((kTileCo + kWPad) * sizeof(int32_t) % 16 == 0,
              "staged weight rows must stay 16-byte aligned");

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
//
// This is the general fallback. It reads the weight buffer one logical element
// at a time, which costs a wave 64 byte loads to collect the 32 bytes of a
// staging round at 4 bits because two lanes share every byte. The dot-product
// kernel below covers the shapes where the buffer can instead be read by the
// word.
template <typename TAct, int WBits, bool WSigned>
__global__ __launch_bounds__(kTiledBlock) void qconv_tiled_kernel(
    const TAct *__restrict__ input, const uint8_t *__restrict__ weights,
    const float *__restrict__ weight_scales,
    const uint8_t *__restrict__ weight_zero_points,
    const float *__restrict__ bias, TAct *__restrict__ output,
    int64_t in_channels, int64_t out_channels, int64_t spatial_size,
    float act_out_scale_ratio, int32_t input_zp, float inv_output_scale,
    int32_t output_zp) {
  __shared__ TAct x_lds[kTileK][kTileP];
  // Indexed [k][co], the transpose of the weight layout in memory, so that the
  // channels one thread multiplies against a single activation are adjacent.
  //
  // Stored one weight per dword even though a weight minus its zero point only
  // needs 16 bits. The inner loop feeds these to a 24-bit multiply-add, which
  // reads the low 24 bits of a register, so a packed pair costs a shift or a
  // bitfield extract per weight before it can be multiplied -- as many extra
  // instructions as there are multiplies. Widening here moves that sign
  // extension to the staging write, where it happens once per staged element
  // instead of once per use, and doubles the shared-memory footprint of the
  // slab, which has headroom because occupancy is register-limited.
  __shared__ int32_t w_lds[kTileK][kTileCo + kWPad];

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
  // per-tile reload into one read. The two staging layouts land a thread on
  // different channels, so each keeps its own copy.
  int zw_stage[kWRepeats];
#pragma unroll
  for (int rep = 0; rep < kWRepeats; ++rep) {
    const int lin = rep * kTiledBlock + tid;
    const int64_t co = co_base + lin / kTileK;
    zw_stage[rep] =
        co < out_channels
            ? qconv_read_quant(weight_zero_points, co, WBits, WSigned)
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
      int32_t v = 0;
      if (co < out_channels && k < in_channels) {
        const int wq =
            qconv_read_quant(weights, co * in_channels + k, WBits, WSigned);
        v = wq - zw_stage[rep];
      }
      w_lds[sk][sc] = v;
    }

    __syncthreads();

    // Exact in int32 for one tile: kTileK terms of at most ~1.7e7 each.
    //
    // The product goes through the 24-bit multiplier. A full 32x32 integer
    // multiply is quarter rate here and does not fuse with the accumulate,
    // while the 24-bit form is a single full-rate multiply-add, and this loop
    // is the whole inner kernel. Both operands are in range by construction: a
    // zero point is representable in the activation type, so x - z_x lands in
    // +-65535, and a weight minus its zero point in +-255. Their product peaks
    // at 1.7e7, well inside what the low 32 bits hold exactly.
    int32_t chunk[kThreadCo] = {};
    for (int kk = 0; kk < kTileK; ++kk) {
      // Hoisted out of the channel loop: z_x is per-tensor, so this is the same
      // subtraction for all kThreadCo channels.
      const int32_t xv = static_cast<int32_t>(x_lds[kk][tx]) - input_zp;
#pragma unroll
      for (int c = 0; c < kThreadCo; ++c)
        chunk[c] += __mul24(xv, w_lds[kk][ty * kThreadCo + c]);
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
// Tiled path, integer dot product
//===----------------------------------------------------------------------===//

// Same tiling as above, but the reduction runs on v_dot4_i32_iu8, which folds
// four multiply-accumulates into one instruction. The loop is issue-limited, so
// what matters is how few instructions a multiply-accumulate costs; the
// multiply-add form pays roughly two, this pays roughly two thirds of one.
//
// Two things have to change to reach the instruction, and both pay for
// themselves elsewhere.
//
// The operands are four packed BYTES, so a 16-bit activation is split into its
// high and low halves and each half gets its own dot and its own accumulator:
//   sum x*w = 256 * sum xhi*w + sum xlo*w
//
// Both operands are read as UNSIGNED, which rules out subtracting a zero point
// before the dot. Expanding the product instead removes the subtraction from
// the loop entirely:
//   sum (x - zx)(w - zw) = sum x*w - zw * sum x - zx * sum w + K * zx * zw
// The two column sums are the only extra work. sum x is over the position a
// thread already owns, so it is two more dots per tile, and sum w is a
// block-wide reduction over the staged slab that costs a handful of
// instructions per tile. The weights then reach shared memory with no unpacking
// at all: at 4 bits a masked word IS a valid operand.
//
// A signed weight is staged as raw ^ (1 << (WBits - 1)), which maps it onto the
// unsigned range by adding the same bias to the weight and to its zero point.
// The bias cancels in w - zw, so nothing downstream has to know.
//
// K is counted as in_channels, not as the padded tile span: a tail tile stages
// zeros for both operands, and the K * zx * zw term would otherwise charge for
// elements that do not exist.

// Reduction lanes folded into one dot, and the resulting dwords per K-tile.
constexpr int kDotQuad = 4;
constexpr int kDotTileKQ = kTileK / kDotQuad;

// Odd stride so that the positions a wave reads land on distinct banks: a
// thread reads down one row of [p][quad] and consecutive threads are
// consecutive p.
constexpr int kXQPad = 1;

// Slots of the staged activation each thread assembles per K-tile.
constexpr int kXQRepeats = (kTileP * kDotTileKQ) / kTiledBlock;
static_assert(kTileP * kDotTileKQ == kXQRepeats * kTiledBlock,
              "x quad slab must divide");
static_assert(kTiledBlock == kCoGroups * kTileCo,
              "column-sum reduction assigns one thread per (group, channel)");
static_assert(kDotTileKQ % kCoGroups == 0,
              "column-sum reduction splits the quad rows across the groups");

__device__ inline int qconv_dot4(uint32_t a, uint32_t b, int acc) {
  return hipdnn_dot4<false, false>(static_cast<int>(a), static_cast<int>(b),
                                   acc);
}

// Which k of the tile lands in byte i of quad q.
//
// At 8 bits a word already holds four consecutive weights, so a quad is a word.
// At 4 bits masking a word to one nibble per byte necessarily takes every other
// nibble, so a word splits into an even quad and an odd quad.
template <int WBits> __device__ constexpr int qconv_dot_k(int q, int i) {
  return WBits == 4 ? (q >> 1) * 8 + (q & 1) + 2 * i : q * kDotQuad + i;
}

template <typename TAct, int WBits, bool WSigned>
__global__ __launch_bounds__(kTiledBlock) void qconv_tiled_dot_kernel(
    const TAct *__restrict__ input, const uint8_t *__restrict__ weights,
    const float *__restrict__ weight_scales,
    const uint8_t *__restrict__ weight_zero_points,
    const float *__restrict__ bias, TAct *__restrict__ output,
    int64_t in_channels, int64_t out_channels, int64_t spatial_size,
    float act_out_scale_ratio, int32_t input_zp, float inv_output_scale,
    int32_t output_zp) {
  static_assert(sizeof(TAct) <= 2, "activation splits into two bytes");
  constexpr int kWElems = 32 / WBits;
  constexpr int kWordsPerRow = kTileK / kWElems;
  constexpr int kWordRepeats = (kTileCo * kWordsPerRow) / kTiledBlock;
  static_assert(kTileCo * kWordsPerRow == kWordRepeats * kTiledBlock,
                "w word slab must divide");
  // Maps a raw signed weight onto the unsigned range the dot product reads.
  constexpr uint32_t kSignFlip =
      WSigned ? (WBits == 4 ? 0x88888888u : 0x80808080u) : 0u;
  constexpr int kZeroBias = WSigned ? (1 << (WBits - 1)) : 0;
  constexpr uint32_t kOnes = 0x01010101u;

  __shared__ uint32_t xh_lds[kTileP][kDotTileKQ + kXQPad];
  __shared__ uint32_t xl_lds[kTileP][kDotTileKQ + kXQPad];
  __shared__ uint32_t w_lds[kDotTileKQ][kTileCo + kWPad];
  __shared__ int32_t cw_lds[kCoGroups][kTileCo];

  const int64_t n = blockIdx.z;
  const int64_t p_base = static_cast<int64_t>(blockIdx.x) * kTileP;
  const int64_t co_base = static_cast<int64_t>(blockIdx.y) * kTileCo;

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int tid = ty * kTileP + tx;

  const TAct *x_n = input + n * in_channels * spatial_size;
  TAct *y_n = output + n * out_channels * spatial_size;

  // The channel this thread folds the weight column sum for, and the quad rows
  // it covers. Fixed for the whole K loop, so the partial stays in a register
  // and the block only exchanges once, after the loop.
  const int cw_co = tid % kTileCo;
  const int cw_row0 = (tid / kTileCo) * (kDotTileKQ / kCoGroups);

  int32_t acc_hi[kThreadCo] = {};
  int32_t acc_lo[kThreadCo] = {};
  int32_t sx_hi = 0;
  int32_t sx_lo = 0;
  int32_t cw_part = 0;

  for (int64_t k0 = 0; k0 < in_channels; k0 += kTileK) {
#pragma unroll
    for (int rep = 0; rep < kXQRepeats; ++rep) {
      const int lin = rep * kTiledBlock + tid;
      const int sq = lin / kTileP;
      const int sp = lin % kTileP;
      const int64_t p = p_base + sp;
      uint32_t hi = 0;
      uint32_t lo = 0;
#pragma unroll
      for (int i = 0; i < kDotQuad; ++i) {
        const int64_t k = k0 + qconv_dot_k<WBits>(sq, i);
        const uint32_t v = (k < in_channels && p < spatial_size)
                               ? static_cast<uint32_t>(x_n[k * spatial_size + p])
                               : 0u;
        hi |= (v >> 8) << (8 * i);
        lo |= (v & 0xFFu) << (8 * i);
      }
      xh_lds[sp][sq] = hi;
      xl_lds[sp][sq] = lo;
    }

#pragma unroll
    for (int rep = 0; rep < kWordRepeats; ++rep) {
      const int widx = rep * kTiledBlock + tid;
      const int sc = widx / kWordsPerRow;
      const int wq = widx % kWordsPerRow;
      const int64_t co = co_base + sc;
      const int64_t k = k0 + wq * kWElems;
      const uint32_t word =
          (co < out_channels && k < in_channels)
              ? reinterpret_cast<const uint32_t *>(
                    weights)[(co * in_channels + k) / kWElems] ^
                    kSignFlip
              : 0u;
      if constexpr (WBits == 4) {
        w_lds[2 * wq][sc] = word & 0x0F0F0F0Fu;
        w_lds[2 * wq + 1][sc] = (word >> 4) & 0x0F0F0F0Fu;
      } else {
        w_lds[wq][sc] = word;
      }
    }

    __syncthreads();

    for (int q = 0; q < kDotTileKQ; ++q) {
      const uint32_t xh = xh_lds[tx][q];
      const uint32_t xl = xl_lds[tx][q];
      sx_hi = qconv_dot4(xh, kOnes, sx_hi);
      sx_lo = qconv_dot4(xl, kOnes, sx_lo);
#pragma unroll
      for (int c = 0; c < kThreadCo; ++c) {
        const uint32_t wv = w_lds[q][ty * kThreadCo + c];
        acc_hi[c] = qconv_dot4(xh, wv, acc_hi[c]);
        acc_lo[c] = qconv_dot4(xl, wv, acc_lo[c]);
      }
    }

#pragma unroll
    for (int r = 0; r < kDotTileKQ / kCoGroups; ++r)
      cw_part = qconv_dot4(w_lds[cw_row0 + r][cw_co], kOnes, cw_part);

    __syncthreads();
  }

  cw_lds[tid / kTileCo][cw_co] = cw_part;
  __syncthreads();

  const int64_t p = p_base + tx;
  if (p >= spatial_size)
    return;

  const int64_t sx = 256LL * sx_hi + sx_lo;
#pragma unroll
  for (int c = 0; c < kThreadCo; ++c) {
    const int lane = ty * kThreadCo + c;
    const int64_t co = co_base + lane;
    if (co >= out_channels)
      break;
    int32_t sw = 0;
#pragma unroll
    for (int g = 0; g < kCoGroups; ++g)
      sw += cw_lds[g][lane];
    const int64_t zw =
        qconv_read_quant(weight_zero_points, co, WBits, WSigned) + kZeroBias;
    const int64_t total = 256LL * acc_hi[c] + acc_lo[c] - zw * sx -
                          static_cast<int64_t>(input_zp) * sw +
                          in_channels * static_cast<int64_t>(input_zp) * zw;
    const float m_co = act_out_scale_ratio * weight_scales[co];
    y_n[co * spatial_size + p] =
        qconv_epilogue<TAct>(total, m_co, bias, co, inv_output_scale, output_zp);
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

// Logical weights one thread covers per step of the packed GEMV body. Sixteen
// is eight bytes at 4 bits and sixteen at 8, so the whole step's weights come
// in as a couple of dwords either way.
//
// The constant trades two things off. Raising it shortens the chain of waits a
// thread walks -- that is the whole point of the rewrite -- but it also
// coarsens the split of K across the block, and the last step of a K that does
// not divide evenly leaves part of the block idle. At 16 a block covers 4096
// logical weights per step, which puts the common decode widths at two or five
// steps instead of the twenty or seventy the scalar loop needed.
constexpr int kGemvVec = 16;
constexpr int kGemvWords = kGemvVec / 8; // 4-bit; 8-bit needs twice this

// Output channels one block covers.
//
// A block reads the whole activation column whatever number of channels it
// produces, so at one channel per block the column is re-read once per output
// and those reads, not the weights, set the rate: per channel and per k the
// column costs two bytes against the half byte a 4-bit weight costs. The
// block's reduction is fixed for the same reason, paid once however much the
// block produces. Covering several channels at once divides both terms by the
// count and leaves the weights, which are the only traffic that is actually
// per channel.
//
// Eight is where the two directions meet. Raising it keeps dividing the shared
// traffic, but it divides the grid by the same count, and a narrow output layer
// runs out of blocks to fill the machine before the saving pays for the
// registers and shared memory the extra accumulators cost.
constexpr int kGemvCoPerBlock = 8;

// Same reduction as qconv_gemv_kernel, but reading weights by the word.
//
// The scalar loop above fetches one weight per iteration and dereferences it
// in the same breath, so every iteration ends in a wait for a single byte and
// a thread walks K/kGemvBlock of those waits end to end. Nothing else is in
// flight to cover them: the loads are one deep, and occupancy is already at
// the maximum, so there are no further waves to interleave. The kernel spends
// most of its time stalled while the memory system runs far below its rate.
//
// So stage first, use second. A step loads its weights and its activations
// into registers and touches neither until both have arrived, which leaves the
// loads outstanding together and collapses the step's waits into one. Reading
// weights as words rather than bytes shortens the chain again, by the packing
// factor, and makes the access a contiguous 128 bytes per wave.
//
// A block also covers kGemvCoPerBlock output channels rather than one, so the
// activation chunk a step stages is read once and spent against every one of
// those channels, and the block's reduction is walked once for all of them.
//
// Requires kGemvVec to divide in_channels, which the caller checks. That is
// what guarantees a thread's chunk lies wholly inside the row or wholly past
// its end -- never straddling -- so no step needs a partial vector and the
// nibble order never shifts.
template <typename TAct, int WBits, bool WSigned>
__global__ __launch_bounds__(kGemvBlock) void qconv_gemv_packed_kernel(
    const TAct *__restrict__ input, const uint8_t *__restrict__ weights,
    const float *__restrict__ weight_scales,
    const uint8_t *__restrict__ weight_zero_points,
    const float *__restrict__ bias, TAct *__restrict__ output,
    int64_t in_channels, int64_t out_channels, int64_t spatial_size,
    float act_out_scale_ratio, int32_t input_zp, float inv_output_scale,
    int32_t output_zp) {
  constexpr int kWords = WBits == 4 ? kGemvWords : kGemvWords * 2;
  constexpr int kCo = kGemvCoPerBlock;

  __shared__ int64_t reduce[kCo][kGemvBlock];

  const int64_t co0 = static_cast<int64_t>(blockIdx.x) * kCo;
  const int64_t p = blockIdx.y;
  const int64_t n = blockIdx.z;

  // Channels this block really owns. A tail block folds the rest onto its
  // first channel, which costs a repeated row and is thrown away below; the
  // point is that no lane addresses past the weight buffer.
  const int live = static_cast<int>(out_channels - co0) < kCo
                       ? static_cast<int>(out_channels - co0)
                       : kCo;

  const TAct *x_n = input + n * in_channels * spatial_size;

  // Uniform across the block, so these stay in scalar registers.
  int zw[kCo];
  const uint8_t *w_row[kCo];
#pragma unroll
  for (int c = 0; c < kCo; ++c) {
    // Byte where the channel's weights start. Divisibility of in_channels by
    // kGemvVec keeps it a multiple of four, so the word reads below stay
    // naturally aligned.
    const int64_t co = co0 + (c < live ? c : 0);
    zw[c] = qconv_read_quant(weight_zero_points, co, WBits, WSigned);
    w_row[c] =
        weights + (WBits == 4 ? (co * in_channels) / 2 : co * in_channels);
  }

  int64_t acc[kCo] = {};
  for (int64_t k0 = static_cast<int64_t>(threadIdx.x) * kGemvVec;
       k0 < in_channels; k0 += static_cast<int64_t>(kGemvBlock) * kGemvVec) {
    TAct xs[kGemvVec];
    if (spatial_size == 1) {
      // Decode reads straight down the channel axis. Splitting the case out
      // is what lets the compiler see the run as contiguous and widen it.
      const TAct *xp = x_n + k0;
#pragma unroll
      for (int i = 0; i < kGemvVec; ++i)
        xs[i] = xp[i];
    } else {
#pragma unroll
      for (int i = 0; i < kGemvVec; ++i)
        xs[i] = x_n[(k0 + i) * spatial_size + p];
    }

    uint32_t wv[kCo][kWords];
#pragma unroll
    for (int c = 0; c < kCo; ++c) {
      const auto *src = reinterpret_cast<const uint32_t *>(
          w_row[c] + (WBits == 4 ? k0 / 2 : k0));
#pragma unroll
      for (int i = 0; i < kWords; ++i)
        wv[c][i] = src[i];
    }

    // Nothing above this line reads a loaded value, so every load in the step
    // is still outstanding when the first one is finally waited on.
#pragma unroll
    for (int c = 0; c < kCo; ++c) {
      int32_t part = 0;
#pragma unroll
      for (int i = 0; i < kGemvVec; ++i) {
        const int32_t xv = static_cast<int32_t>(xs[i]) - input_zp;
        part += xv * (qconv_unpack_word<WBits, WSigned>(wv[c], i) - zw[c]);
      }
      acc[c] += part;
    }
  }

  // The channels share the tree, so the barriers are walked once rather than
  // once per channel.
#pragma unroll
  for (int c = 0; c < kCo; ++c)
    reduce[c][threadIdx.x] = acc[c];
  __syncthreads();
  for (int stride = kGemvBlock / 2; stride > 0; stride >>= 1) {
    if (static_cast<int>(threadIdx.x) < stride) {
#pragma unroll
      for (int c = 0; c < kCo; ++c)
        reduce[c][threadIdx.x] += reduce[c][threadIdx.x + stride];
    }
    __syncthreads();
  }

  if (static_cast<int>(threadIdx.x) < live) {
    const int c = static_cast<int>(threadIdx.x);
    const int64_t co = co0 + c;
    const float m_co = act_out_scale_ratio * weight_scales[co];
    output[n * out_channels * spatial_size + co * spatial_size + p] =
        qconv_epilogue<TAct>(reduce[c][0], m_co, bias, co, inv_output_scale,
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

    // The packed body reads the weight row by the word, so the row has to
    // start on one and a thread's chunk must never straddle the row's end.
    // Both follow from kGemvVec dividing in_channels: it makes the row's byte
    // offset a multiple of four and every chunk either wholly inside the row
    // or wholly past it. The pointer check covers the buffer itself, which the
    // allocator aligns far past this but which nothing here guarantees.
    const bool packed = (in_channels % kGemvVec) == 0 &&
                        (reinterpret_cast<uintptr_t>(w) % sizeof(uint32_t)) == 0;

    // The packed body produces kGemvCoPerBlock channels per block.
    dim3 packed_grid(
        static_cast<unsigned>((out_channels + kGemvCoPerBlock - 1) /
                              kGemvCoPerBlock),
        static_cast<unsigned>(spatial_size), static_cast<unsigned>(batch));

#define QCONV_LAUNCH_GEMV_PACKED(BITS, SIGNED)                                 \
  hipLaunchKernelGGL(                                                          \
      HIP_KERNEL_NAME(qconv_gemv_packed_kernel<TAct, BITS, SIGNED>),           \
      packed_grid, dim3(kGemvBlock), 0, stream, x, w, ws, wzp, b, y,           \
      in_channels, out_channels, spatial_size, act_out_scale_ratio, input_zp,  \
      inv_output_scale, output_zp)

    if (packed && weight_bits == 4 && weight_signed)
      QCONV_LAUNCH_GEMV_PACKED(4, true);
    else if (packed && weight_bits == 4)
      QCONV_LAUNCH_GEMV_PACKED(4, false);
    else if (packed && weight_signed)
      QCONV_LAUNCH_GEMV_PACKED(8, true);
    else if (packed)
      QCONV_LAUNCH_GEMV_PACKED(8, false);
    else
      hipLaunchKernelGGL(HIP_KERNEL_NAME(qconv_gemv_kernel<TAct>), grid,
                         dim3(kGemvBlock), 0, stream, x, w, ws, wzp, b, y,
                         in_channels, out_channels, spatial_size, weight_bits,
                         weight_signed, act_out_scale_ratio, input_zp,
                         inv_output_scale, output_zp);

#undef QCONV_LAUNCH_GEMV_PACKED
  } else {
    dim3 grid(
        static_cast<unsigned>((spatial_size + kTileP - 1) / kTileP),
        static_cast<unsigned>((out_channels + kTileCo - 1) / kTileCo),
        static_cast<unsigned>(batch));

    // The dot-product kernel reads the weight buffer by the word, which needs
    // every channel's row to start on a word boundary and a quad to stay inside
    // one row; both follow from in_channels dividing by the element count of a
    // word. The pointer check covers the buffer itself, which the allocator
    // aligns far past this but which nothing here guarantees.
    //
    // It also keeps its two accumulators in int32 for the whole reduction
    // rather than widening per tile, so K has to stay under what 255 times the
    // largest unsigned weight can sum to without wrapping.
    const int64_t w_elems_per_word = 32 / weight_bits;
    const int64_t dot_max_k =
        (int64_t{1} << 31) / (255 * ((int64_t{1} << weight_bits) - 1));
    const bool dot_capable =
        (in_channels % w_elems_per_word) == 0 && in_channels <= dot_max_k &&
        (reinterpret_cast<uintptr_t>(w) % sizeof(uint32_t)) == 0;

#define QCONV_LAUNCH_TILED(KERNEL, BITS, SIGNED)                               \
  hipLaunchKernelGGL(HIP_KERNEL_NAME(KERNEL<TAct, BITS, SIGNED>), grid,        \
                     dim3(kTileP, kCoGroups), 0, stream, x, w, ws, wzp, b, y,  \
                     in_channels, out_channels, spatial_size,                  \
                     act_out_scale_ratio, input_zp, inv_output_scale,          \
                     output_zp)
#define QCONV_LAUNCH_TILED_BITS(BITS)                                          \
  do {                                                                         \
    if (dot_capable && weight_signed)                                          \
      QCONV_LAUNCH_TILED(qconv_tiled_dot_kernel, BITS, true);                  \
    else if (dot_capable)                                                      \
      QCONV_LAUNCH_TILED(qconv_tiled_dot_kernel, BITS, false);                 \
    else if (weight_signed)                                                    \
      QCONV_LAUNCH_TILED(qconv_tiled_kernel, BITS, true);                      \
    else                                                                       \
      QCONV_LAUNCH_TILED(qconv_tiled_kernel, BITS, false);                     \
  } while (0)

    if (weight_bits == 4)
      QCONV_LAUNCH_TILED_BITS(4);
    else
      QCONV_LAUNCH_TILED_BITS(8);

#undef QCONV_LAUNCH_TILED_BITS
#undef QCONV_LAUNCH_TILED
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
