/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Device-side tiled reduction shared by the quantized GEMM kernels.
 *
 * Only the reduction lives here. Each op keeps its own epilogue, because they
 * disagree on what a result needs -- a bias term, a per-output-channel
 * coefficient -- while agreeing exactly on how the integer sums are formed.
 */

#pragma once

#include "hip_arch_compat.h"

#include <hip/hip_runtime.h>

#include <cstdint>

namespace quant_tile {

// Sums formed here
// ----------------
// The caller's identity is
//
//   acc[m,n] = sum_k (A[m,k] - z_a) * (B[k,n] - z_b)
//
// which this reduction never evaluates directly: subtracting the zero point
// inside the loop would widen both operands before they can be multiplied.
// Expanding it keeps the inner loop on raw operands,
//
//   acc[m,n] = accRaw[m,n] - z_b*rowA[m] - z_a*colB[n] + K*z_a*z_b
//   accRaw[m,n] = sum_k A[m,k]*B[k,n],  rowA[m] = sum_k A[m,k],
//   colB[n] = sum_k B[k,n]
//
// so the three sums below are all the epilogue needs. rowA and colB are only
// accumulated when their zero point can be non-zero, which is why the
// symmetric weights that dominate real QDQ graphs cost nothing extra.
//
// Byte planes
// -----------
// A may be 16-bit while B stays 8-bit. Two things break at that width: an
// int32 accumulator overflows around K = 256 (one u16*u8 product already
// reaches 1.7e7), and the dot4 instruction cannot see a 16-bit operand at all.
// Splitting A into its two bytes solves both. With
//
//   A = hi * 256 + lo,  hi = A >> 8,  lo = A & 0xFF
//
// each half runs on dot4 with its own int32 accumulator -- |lo*B| <= 255*128
// keeps K under 65792, past any real shape -- and the caller combines the two
// halves in int64. The result is exact, at two dot4 instructions per term
// instead of the 4x-plus slowdown a scalar int64 inner loop would bring.

// All-ones byte vector: dot4(x, kOnes4) sums the four bytes of x, which is how
// rowA / colB ride along on the same integer dot pipeline as the products.
constexpr int kOnes4 = 0x01010101;

// Tile geometry: one block owns a [kTileM x kTileN] output tile, one thread a
// [kThreadM x kThreadN] micro-tile.
constexpr int kTileM = 64;
constexpr int kTileN = 64;
constexpr int kTileK = 16;
constexpr int kThreadM = 4;
constexpr int kThreadN = 4;
constexpr int kTiledBlock = (kTileM / kThreadM) * (kTileN / kThreadN); // 256
// Rows of the transposed B tile that one pass over the block can stage.
constexpr int kBStageRows = kTiledBlock / kTileN; // 4

// The LDS tiles hold raw bytes: hipdnn_dot4's template flags decide how each
// operand's bytes are interpreted, so staging never has to know the signedness.
using Byte = uint8_t;

// A 16-bit A is staged as two byte planes (hi, lo); an 8-bit one needs a single
// plane holding the value itself. Indices into the plane dimension:
constexpr int kPlaneHi = 0;
constexpr int kPlaneLo = 1;

template <typename TA> struct APlanes {
  static constexpr bool kWide = sizeof(TA) == 2;
  static constexpr int kCount = kWide ? 2 : 1;
};

// Reads 4 k-consecutive bytes of an LDS tile row as one dot4 operand. The tile
// rows are kTileK bytes wide and the array is 16-byte aligned, so every
// multiple-of-4 offset is a legal aligned int load.
__device__ inline int lds_pack4(const Byte *row, int k) {
  return *reinterpret_cast<const int *>(row + k);
}

template <int kPlanes> struct MicroTile {
  int32_t acc[kPlanes][kThreadM][kThreadN];
  int32_t row_a[kPlanes][kThreadM];
  int32_t col_b[kThreadN];
};

__device__ inline void micro_tile_origin(int tid, int &thread_m,
                                         int &thread_n) {
  thread_m = (tid / (kTileN / kThreadN)) * kThreadM;
  thread_n = (tid % (kTileN / kThreadN)) * kThreadN;
}

// One LOGICAL element of B, widened to the byte the LDS tile holds.
//
// `bits` is 4 or 8, and `lin` counts logical elements, never bytes. At 4 two
// values share a byte, low nibble first over the flattened row-major sequence,
// which is the ONNX INT4/UINT4 convention. A signed nibble is sign-extended to
// a full byte so that dot4's signed reading of that byte agrees with the
// nibble's value.
template <bool kBSigned>
__device__ inline Byte read_b(const uint8_t *__restrict__ B, int64_t lin,
                              int bits) {
  if (bits == 4) {
    const uint8_t byte = B[lin >> 1];
    const uint8_t nibble = (lin & 1) ? (byte >> 4) : (byte & 0xF);
    if constexpr (kBSigned)
      return static_cast<Byte>(static_cast<int8_t>(nibble << 4) >> 4);
    return nibble;
  }
  return B[lin];
}

// Reduces A[m_base.., k_begin:k_end] x B[k_begin:k_end, n_base..] into this
// thread's micro-tile. K is the full logical extent because it is also the
// operand stride.
//
// The range is a parameter rather than always [0, K) so split-K can hand each
// block a slice. Staging zero-fills anything past k_end and zero is the
// identity for all three sums, so a slice needs no separate tail handling; it
// only has to start on a kTileK boundary to keep the staging maps coalesced.
//
// Both LDS tiles are k-major because dot4 needs four k-consecutive bytes in one
// register. For a [K, N] B that means staging it transposed; for a [N, K] one
// (trans_b) the tile is already in that order and staging is a straight copy.
//
// trans_a / trans_b / b_bits are runtime flags rather than template parameters:
// they only pick a load stride or an unpack during staging, which is
// block-uniform and off the dot4 inner loop, and templating them would multiply
// the instantiations the dtype pair already costs.
template <typename TA, bool kASigned, bool kBSigned>
__device__ MicroTile<APlanes<TA>::kCount>
accumulate_k_range(const TA *__restrict__ A, const uint8_t *__restrict__ B,
                   int64_t M, int64_t N, int64_t K, int64_t m_base,
                   int64_t n_base, int64_t k_begin, int64_t k_end, int b_bits,
                   bool need_row_a, bool need_col_b, bool trans_a,
                   bool trans_b) {
  constexpr bool kAWide = APlanes<TA>::kWide;
  constexpr int kAPlanes = APlanes<TA>::kCount;

  __shared__ __align__(16) Byte a_lds[kAPlanes][kTileM][kTileK];
  __shared__ __align__(16) Byte b_lds[kTileN][kTileK];

  const int tid = threadIdx.x;
  int thread_m, thread_n;
  micro_tile_origin(tid, thread_m, thread_n);

  // Per-plane accumulators. |lo * B| <= 255*128 bounds the widest plane, so
  // int32 holds K up to 65792 -- past any shape the runtime admits.
  MicroTile<kAPlanes> out = {};

  // Staging maps, each coalescing the reads of the layout it serves.
  //
  // `row4_*` walks a tile row-major, one 4-byte run per thread, which is what
  // A wants and also what a [N, K] B wants since its k-runs are contiguous.
  // kTileM == kTileN, so the one map covers both tiles.
  //
  // `b_*` walks a [K, N] B, where the k stride is N: one byte per thread per
  // k, so the 64 threads sharing a k cover a contiguous 64-byte run.
  static_assert(kTileM == kTileN, "row4 staging map is shared by A and B");
  const int row4_r = (tid * 4) / kTileK;
  const int row4_c = (tid * 4) % kTileK;
  const int b_n = tid % kTileN;
  const int b_k_base = tid / kTileN;

  for (int64_t k0 = k_begin; k0 < k_end; k0 += kTileK) {
    // Out-of-range elements are staged as 0, the identity for all three sums.
    const int64_t a_row = m_base + row4_r;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int64_t k = k0 + row4_c + i;
      const bool in_range = a_row < M && k < k_end;
      // A [K, M] costs a strided read here, unlike the contiguous [M, K] one.
      const int64_t a_idx = trans_a ? k * M + a_row : a_row * K + k;
      if constexpr (kAWide) {
        // Promoting to int first makes `>> 8` arithmetic for int16 and logical
        // for uint16, so hi inherits A's signedness while lo stays an unsigned
        // byte -- exactly the split A = hi*256 + lo needs.
        const int v = in_range ? static_cast<int>(A[a_idx]) : 0;
        a_lds[kPlaneHi][row4_r][row4_c + i] = static_cast<Byte>(v >> 8);
        a_lds[kPlaneLo][row4_r][row4_c + i] = static_cast<Byte>(v & 0xFF);
      } else {
        a_lds[kPlaneHi][row4_r][row4_c + i] =
            in_range ? static_cast<Byte>(A[a_idx]) : Byte{0};
      }
    }

    if (trans_b) {
      const int64_t b_row = n_base + row4_r;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int64_t k = k0 + row4_c + i;
        b_lds[row4_r][row4_c + i] =
            (b_row < N && k < k_end)
                ? read_b<kBSigned>(B, b_row * K + k, b_bits)
                : Byte{0};
      }
    } else {
      const int64_t b_col = n_base + b_n;
#pragma unroll
      for (int rep = 0; rep < kTileK / kBStageRows; ++rep) {
        const int b_k = b_k_base + rep * kBStageRows;
        const int64_t k = k0 + b_k;
        b_lds[b_n][b_k] = (b_col < N && k < k_end)
                              ? read_b<kBSigned>(B, k * N + b_col, b_bits)
                              : Byte{0};
      }
    }

    __syncthreads();

#pragma unroll
    for (int kk = 0; kk < kTileK; kk += 4) {
      int a_frag[kAPlanes][kThreadM];
      int b_frag[kThreadN];
#pragma unroll
      for (int p = 0; p < kAPlanes; ++p) {
#pragma unroll
        for (int i = 0; i < kThreadM; ++i)
          a_frag[p][i] = lds_pack4(a_lds[p][thread_m + i], kk);
      }
#pragma unroll
      for (int j = 0; j < kThreadN; ++j)
        b_frag[j] = lds_pack4(b_lds[thread_n + j], kk);

#pragma unroll
      for (int i = 0; i < kThreadM; ++i) {
#pragma unroll
        for (int j = 0; j < kThreadN; ++j) {
          out.acc[kPlaneHi][i][j] = hipdnn_dot4<kASigned, kBSigned>(
              a_frag[kPlaneHi][i], b_frag[j], out.acc[kPlaneHi][i][j]);
          if constexpr (kAWide)
            out.acc[kPlaneLo][i][j] = hipdnn_dot4<false, kBSigned>(
                a_frag[kPlaneLo][i], b_frag[j], out.acc[kPlaneLo][i][j]);
        }
      }

      // Block-uniform branches, so the skipped dots cost nothing when the
      // corresponding zero point is 0 -- or, under split-K, when this block
      // was not the one elected to contribute that correction.
      if (need_row_a) {
#pragma unroll
        for (int i = 0; i < kThreadM; ++i) {
          out.row_a[kPlaneHi][i] = hipdnn_dot4<kASigned, false>(
              a_frag[kPlaneHi][i], kOnes4, out.row_a[kPlaneHi][i]);
          if constexpr (kAWide)
            out.row_a[kPlaneLo][i] = hipdnn_dot4<false, false>(
                a_frag[kPlaneLo][i], kOnes4, out.row_a[kPlaneLo][i]);
        }
      }
      if (need_col_b) {
#pragma unroll
        for (int j = 0; j < kThreadN; ++j)
          out.col_b[j] =
              hipdnn_dot4<kBSigned, false>(b_frag[j], kOnes4, out.col_b[j]);
      }
    }

    __syncthreads();
  }

  return out;
}

} // namespace quant_tile
