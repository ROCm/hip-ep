/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
// FusedOp(A,B) = Q(DQ(A) @ DQ(B))
//
// a = (A - z_a) * s_a
// b = (B - z_b) * s_b
// Y = saturate(round((a @ b) / s_y) + z_y)
//
// (a @ b)[m,n] = s_a*s_b * sum_k (A[m,k] - z_a) * (B[k,n] - z_b)
//              = s_a*s_b * (acc[m,n] - z_b*rowA[m] - z_a*colB[n] + K*z_a*z_b)
//   acc[m,n] = sum_k A[m,k]*B[k,n]
//   rowA[m]  = sum_k A[m,k]
//   colB[n]  = sum_k B[k,n]
// --->
// let M = s_a * s_b / s_y   (folded by lowering)
// Y = saturate(round(M * (acc - z_b*rowA - z_a*colB + K*z_a*z_b)) + z_y)
//
// A - z_a does not fit back into 8 bits, so the zero points cannot be folded
// into the operands; they are carried as the rowA / colB correction terms
// instead. Each term is only accumulated when its zero point is non-zero, so
// the symmetric weights that dominate real QDQ graphs cost nothing extra.
//
// 16-bit activations (A16W8)
// --------------------------
// A may be 16-bit while B stays 8-bit. Two things break at that width: the
// int32 accumulator overflows at K around 256 (a single u16*u8 product already
// reaches 1.7e7), and hipdnn_dot4 is a 4x8-bit instruction that cannot see a
// 16-bit operand at all.
//
// Both are solved by splitting A into its two bytes. With
//
//   A = hi * 256 + lo,   hi = A >> 8,   lo = A & 0xFF
//
// (an arithmetic shift for int16 and a logical one for uint16, so hi carries
// A's signedness and lo is always unsigned), the dot product separates into
// two 8-bit dot products:
//
//   acc  = sum_k A[k]*B[k] = 256 * sum_k hi[k]*B[k] + sum_k lo[k]*B[k]
//   rowA = sum_k A[k]      = 256 * sum_k hi[k]      + sum_k lo[k]
//
// Each half runs on dot4 with its own int32 accumulator -- |lo*B| <= 255*128
// keeps K under 65792, far past any real shape -- and the two halves are
// combined in int64 at the epilogue. The result is exact, and the cost is two
// dot4 instructions per term instead of one rather than the 4x-plus slowdown a
// scalar int64 inner loop would bring.
//
// Kernel paths
// ------------
// All three share the identity above and differ only in how the reduction is
// parallelized:
//
//   M == 1                       -> GEMV, one thread per output column
//   M > 1                        -> tiled GEMM, one block per output tile
//   M > 1, too few output tiles  -> split-K over the tiled inner loop
//
// The GEMV path is bandwidth-bound and does not use dot4, so it accumulates in
// int64 directly instead of splitting A.

#include "debug_log.h"
#include "hip_arch_compat.h"
#include "hip_custom_kernels.h"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>

namespace {

//===----------------------------------------------------------------------===//
// Element types, epilogue, and tile layout
//===----------------------------------------------------------------------===//

// All-ones byte vector: dot4(x, kOnes4) sums the four bytes of x, which is how
// rowA / colB ride along on the same integer dot pipeline as the products.
constexpr int kOnes4 = 0x01010101;

template <typename T> struct QMatMulTraits;

template <> struct QMatMulTraits<int8_t> {
  static constexpr bool kSigned = true;
  __device__ static float lowest() { return static_cast<float>(INT8_MIN); }
  __device__ static float highest() { return static_cast<float>(INT8_MAX); }
};

template <> struct QMatMulTraits<uint8_t> {
  static constexpr bool kSigned = false;
  __device__ static float lowest() { return 0.0f; }
  __device__ static float highest() { return static_cast<float>(UINT8_MAX); }
};

template <> struct QMatMulTraits<int16_t> {
  static constexpr bool kSigned = true;
  __device__ static float lowest() { return static_cast<float>(INT16_MIN); }
  __device__ static float highest() { return static_cast<float>(INT16_MAX); }
};

template <> struct QMatMulTraits<uint16_t> {
  static constexpr bool kSigned = false;
  __device__ static float lowest() { return 0.0f; }
  __device__ static float highest() { return static_cast<float>(UINT16_MAX); }
};

// The HIP compile flags include `-Xclang -fno-cuda-host-device-constexpr`
// (see lib/Runtime/Kernels/cmake/hip_utils.cmake), which makes
// `std::numeric_limits<T>` members host-only, hence the explicit bounds above.
template <typename T> __device__ inline T q_saturate(float v) {
  return static_cast<T>(
      fminf(QMatMulTraits<T>::highest(), fmaxf(QMatMulTraits<T>::lowest(), v)));
}

// Requantize one accumulated output element.
//
// Everything is combined in int64: with a 16-bit A the four terms reach ~1e12
// between them, and even at 8 bits their sum outgrows int32 for large K. The
// narrowing to float afterwards is safe because m_scale immediately brings the
// value back into the output's quantized range, so float's ~6e-8 relative
// error stays two orders of magnitude below one LSB of a 16-bit output.
template <typename TY>
__device__ inline TY
qmatmul_epilogue(int64_t acc, int64_t row_a, int64_t col_b, int64_t k_zazb,
                 int32_t a_zp, int32_t b_zp, float m_scale, int32_t y_zp) {
  const int64_t corrected = acc - static_cast<int64_t>(b_zp) * row_a -
                            static_cast<int64_t>(a_zp) * col_b + k_zazb;
  const float scaled = m_scale * static_cast<float>(corrected);
  return q_saturate<TY>(rintf(scaled) + static_cast<float>(y_zp));
}

// Tile geometry shared by the tiled and split-K paths: one block owns a
// [kTileM x kTileN] output tile, one thread a [kThreadM x kThreadN] micro-tile.
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
using QMatMulByte = uint8_t;

// A 16-bit A is staged as two byte planes (hi, lo); an 8-bit one needs a single
// plane holding the value itself. Indices into the plane dimension:
constexpr int kPlaneHi = 0;
constexpr int kPlaneLo = 1;

template <typename TA> struct QMatMulAPlanes {
  static constexpr bool kWide = sizeof(TA) == 2;
  static constexpr int kCount = kWide ? 2 : 1;
};

// Reads 4 k-consecutive bytes of an LDS tile row as one dot4 operand. The tile
// rows are kTileK bytes wide and the array is 16-byte aligned, so every
// multiple-of-4 offset is a legal aligned int load.
__device__ inline int lds_pack4(const QMatMulByte *row, int k) {
  return *reinterpret_cast<const int *>(row + k);
}

template <int kPlanes> struct QMatMulMicroTile {
  int32_t acc[kPlanes][kThreadM][kThreadN];
  int32_t row_a[kPlanes][kThreadM];
  int32_t col_b[kThreadN];
};

__device__ inline void qmatmul_micro_tile_origin(int tid, int &thread_m,
                                                 int &thread_n) {
  thread_m = (tid / (kTileN / kThreadN)) * kThreadM;
  thread_n = (tid % (kTileN / kThreadN)) * kThreadN;
}

//===----------------------------------------------------------------------===//
// Shared k-range accumulation (LDS staging + dot4)
//===----------------------------------------------------------------------===//

// Reduces A[m_base.., k_begin:k_end] x B[k_begin:k_end, n_base..] into this
// thread's micro-tile. A and B are already offset to the block's batch; K is
// still the full logical extent because it is also the operand stride.
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
// trans_a / trans_b are runtime flags rather than template parameters: they
// only pick a load stride during staging, which is block-uniform and off the
// dot4 inner loop, and templating them would quadruple the instantiations the
// dtype pair already costs.
template <typename TA, typename TB>
__device__ QMatMulMicroTile<QMatMulAPlanes<TA>::kCount>
qmatmul_accumulate_k_range(const TA *__restrict__ A, const TB *__restrict__ B,
                           int64_t M, int64_t N, int64_t K, int64_t m_base,
                           int64_t n_base, int64_t k_begin, int64_t k_end,
                           bool need_row_a, bool need_col_b, bool trans_a,
                           bool trans_b) {
  constexpr bool kAWide = QMatMulAPlanes<TA>::kWide;
  constexpr int kAPlanes = QMatMulAPlanes<TA>::kCount;
  constexpr bool kASigned = QMatMulTraits<TA>::kSigned;
  constexpr bool kBSigned = QMatMulTraits<TB>::kSigned;

  __shared__ __align__(16) QMatMulByte a_lds[kAPlanes][kTileM][kTileK];
  __shared__ __align__(16) QMatMulByte b_lds[kTileN][kTileK];

  const int tid = threadIdx.x;
  int thread_m, thread_n;
  qmatmul_micro_tile_origin(tid, thread_m, thread_n);

  // Per-plane accumulators. |lo * B| <= 255*128 bounds the widest plane, so
  // int32 holds K up to 65792 -- past any shape the runtime admits.
  QMatMulMicroTile<kAPlanes> out = {};

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
      // Left as is: TransposeMatMulFold folds Transpose into whichever operand
      // the graph transposes, and attention's Q @ K^T only ever stamps transB.
      const int64_t a_idx = trans_a ? k * M + a_row : a_row * K + k;
      if constexpr (kAWide) {
        // Promoting to int first makes `>> 8` arithmetic for int16 and logical
        // for uint16, so hi inherits A's signedness while lo stays an unsigned
        // byte -- exactly the split A = hi*256 + lo needs.
        const int v = in_range ? static_cast<int>(A[a_idx]) : 0;
        a_lds[kPlaneHi][row4_r][row4_c + i] = static_cast<QMatMulByte>(v >> 8);
        a_lds[kPlaneLo][row4_r][row4_c + i] =
            static_cast<QMatMulByte>(v & 0xFF);
      } else {
        a_lds[kPlaneHi][row4_r][row4_c + i] =
            in_range ? static_cast<QMatMulByte>(A[a_idx]) : QMatMulByte{0};
      }
    }

    if (trans_b) {
      const int64_t b_row = n_base + row4_r;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int64_t k = k0 + row4_c + i;
        b_lds[row4_r][row4_c + i] =
            (b_row < N && k < k_end)
                ? static_cast<QMatMulByte>(B[b_row * K + k])
                : QMatMulByte{0};
      }
    } else {
      const int64_t b_col = n_base + b_n;
#pragma unroll
      for (int rep = 0; rep < kTileK / kBStageRows; ++rep) {
        const int b_k = b_k_base + rep * kBStageRows;
        const int64_t k = k0 + b_k;
        b_lds[b_n][b_k] = (b_col < N && k < k_end)
                              ? static_cast<QMatMulByte>(B[k * N + b_col])
                              : QMatMulByte{0};
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

//===----------------------------------------------------------------------===//
// Tiled GEMM (M > 1)
//===----------------------------------------------------------------------===//

// Y[b, m, n] = requantize(sum_k A[b, m, k] * B[b, k, n])
//
// Grid: (ceil(N/kTileN), ceil(M/kTileM), batch_count). One block owns a whole
// output tile and the whole reduction, so it can requantize in place.
template <typename TA, typename TB, typename TY>
__global__ __launch_bounds__(kTiledBlock) void qmatmul_tiled_kernel(
    const TA *__restrict__ A, const TB *__restrict__ B, TY *__restrict__ Y,
    int64_t M, int64_t N, int64_t K, int64_t b_batch_stride, float m_scale,
    int32_t a_zp, int32_t b_zp, int32_t y_zp, int64_t k_zazb, bool need_row_a,
    bool need_col_b, bool trans_a, bool trans_b) {
  constexpr bool kAWide = QMatMulAPlanes<TA>::kWide;

  const int64_t batch = blockIdx.z;
  A += batch * M * K;
  B += batch * b_batch_stride;
  Y += batch * M * N;

  const int64_t m_base = static_cast<int64_t>(blockIdx.y) * kTileM;
  const int64_t n_base = static_cast<int64_t>(blockIdx.x) * kTileN;

  const auto mt = qmatmul_accumulate_k_range<TA, TB>(
      A, B, M, N, K, m_base, n_base, 0, K, need_row_a, need_col_b, trans_a,
      trans_b);

  int thread_m, thread_n;
  qmatmul_micro_tile_origin(threadIdx.x, thread_m, thread_n);

#pragma unroll
  for (int i = 0; i < kThreadM; ++i) {
    const int64_t m = m_base + thread_m + i;
    if (m >= M)
      continue;
    int64_t row_a64;
    if constexpr (kAWide)
      row_a64 = int64_t{256} * mt.row_a[kPlaneHi][i] + mt.row_a[kPlaneLo][i];
    else
      row_a64 = mt.row_a[kPlaneHi][i];
#pragma unroll
    for (int j = 0; j < kThreadN; ++j) {
      const int64_t n = n_base + thread_n + j;
      if (n >= N)
        continue;
      int64_t acc64;
      if constexpr (kAWide)
        acc64 = int64_t{256} * mt.acc[kPlaneHi][i][j] + mt.acc[kPlaneLo][i][j];
      else
        acc64 = mt.acc[kPlaneHi][i][j];
      Y[m * N + n] = qmatmul_epilogue<TY>(acc64, row_a64, mt.col_b[j], k_zazb,
                                          a_zp, b_zp, m_scale, y_zp);
    }
  }
}

//===----------------------------------------------------------------------===//
// Split-K (low output-parallelism shapes)
//===----------------------------------------------------------------------===//
//
// The tiled grid has no K dimension, so its parallelism is bounded by the
// output alone: ceil(N/kTileN) * ceil(M/kTileM) * batch_count blocks. A
// low-rank shape such as M=128, N=32 therefore launches two blocks, leaves the
// rest of the device idle, and makes each block walk the entire reduction.
//
// Split-K restores the missing dimension: each block reduces a k slice into a
// global int64 accumulator, and a second kernel requantizes once every slice
// has landed. Integer addition is associative and exact, so the atomics make
// the result bit-identical to the single-pass path regardless of the order the
// slices complete in -- there is no reduction-order drift to reason about.

__device__ inline void qmatmul_atomic_add_i64(int64_t *addr, int64_t v) {
  atomicAdd(reinterpret_cast<unsigned long long *>(addr),
            static_cast<unsigned long long>(v));
}

// Split-K partial pass: one block per (n-tile, m-tile, batch x split).
template <typename TA, typename TB>
__global__ __launch_bounds__(kTiledBlock) void qmatmul_splitk_partial_kernel(
    const TA *__restrict__ A, const TB *__restrict__ B,
    int64_t *__restrict__ acc_ws, int64_t *__restrict__ row_a_ws,
    int64_t *__restrict__ col_b_ws, int64_t M, int64_t N, int64_t K,
    int64_t b_batch_stride, int splits, int64_t k_chunk, bool need_row_a,
    bool need_col_b, bool trans_a, bool trans_b) {
  constexpr bool kAWide = QMatMulAPlanes<TA>::kWide;

  const int64_t split = blockIdx.z % splits;
  const int64_t batch = blockIdx.z / splits;

  const int64_t k_begin = split * k_chunk;
  if (k_begin >= K)
    return;
  const int64_t k_end = (k_begin + k_chunk < K) ? (k_begin + k_chunk) : K;

  A += batch * M * K;
  B += batch * b_batch_stride;
  acc_ws += batch * M * N;
  row_a_ws += batch * M;
  col_b_ws += batch * N;

  const int64_t m_base = static_cast<int64_t>(blockIdx.y) * kTileM;
  const int64_t n_base = static_cast<int64_t>(blockIdx.x) * kTileN;

  // rowA does not depend on n and colB does not depend on m, so every block
  // column would otherwise add the same rowA slice again (and every block row
  // the same colB slice). Elect one of each; the flags are block-uniform, so
  // the non-electing blocks skip the corresponding dots entirely.
  const bool add_row_a = need_row_a && blockIdx.x == 0;
  const bool add_col_b = need_col_b && blockIdx.y == 0;

  const auto mt = qmatmul_accumulate_k_range<TA, TB>(
      A, B, M, N, K, m_base, n_base, k_begin, k_end, add_row_a, add_col_b,
      trans_a, trans_b);

  int thread_m, thread_n;
  qmatmul_micro_tile_origin(threadIdx.x, thread_m, thread_n);

#pragma unroll
  for (int i = 0; i < kThreadM; ++i) {
    const int64_t m = m_base + thread_m + i;
    if (m >= M)
      continue;
    // Every thread sharing this m holds the same rowA slice, so within the
    // elected block only the first micro-tile column commits it.
    if (add_row_a && thread_n == 0) {
      int64_t row_a64;
      if constexpr (kAWide)
        row_a64 = int64_t{256} * mt.row_a[kPlaneHi][i] + mt.row_a[kPlaneLo][i];
      else
        row_a64 = mt.row_a[kPlaneHi][i];
      qmatmul_atomic_add_i64(&row_a_ws[m], row_a64);
    }
#pragma unroll
    for (int j = 0; j < kThreadN; ++j) {
      const int64_t n = n_base + thread_n + j;
      if (n >= N)
        continue;
      int64_t acc64;
      if constexpr (kAWide)
        acc64 = int64_t{256} * mt.acc[kPlaneHi][i][j] + mt.acc[kPlaneLo][i][j];
      else
        acc64 = mt.acc[kPlaneHi][i][j];
      qmatmul_atomic_add_i64(&acc_ws[m * N + n], acc64);
    }
  }

  // Mirror of the rowA election: only the first micro-tile row commits colB.
  if (add_col_b && thread_m == 0) {
#pragma unroll
    for (int j = 0; j < kThreadN; ++j) {
      const int64_t n = n_base + thread_n + j;
      if (n < N)
        qmatmul_atomic_add_i64(&col_b_ws[n], mt.col_b[j]);
    }
  }
}

constexpr int kSplitKEpilogueBlock = 256;

// Requantizes the completed split-K sums, one thread per output element.
//
// rowA / colB are read unconditionally. When their zero point is zero the
// partial pass never wrote them, but the workspace was zeroed beforehand and
// the epilogue multiplies them by that same zero point, so the branch would
// buy nothing.
template <typename TY>
__global__ __launch_bounds__(kSplitKEpilogueBlock) void
qmatmul_splitk_epilogue_kernel(const int64_t *__restrict__ acc_ws,
                               const int64_t *__restrict__ row_a_ws,
                               const int64_t *__restrict__ col_b_ws,
                               TY *__restrict__ Y, int64_t M, int64_t N,
                               int64_t total, float m_scale, int32_t a_zp,
                               int32_t b_zp, int32_t y_zp, int64_t k_zazb) {
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * kSplitKEpilogueBlock + threadIdx.x;
  if (idx >= total)
    return;

  // acc_ws and Y share the [batch, M, N] row-major order, so one flat index
  // addresses both; only the two correction vectors need unpacking.
  const int64_t mn = M * N;
  const int64_t batch = idx / mn;
  const int64_t rem = idx - batch * mn;
  const int64_t m = rem / N;
  const int64_t n = rem - m * N;

  Y[idx] = qmatmul_epilogue<TY>(acc_ws[idx], row_a_ws[batch * M + m],
                                col_b_ws[batch * N + n], k_zazb, a_zp, b_zp,
                                m_scale, y_zp);
}

// Slice ceiling. Past it the per-slice staging prologue and the atomic traffic
// grow faster than the added parallelism pays back.
constexpr int kSplitKMaxSplits = 32;
// A slice still has to cover enough k-tiles to amortize that prologue and the
// atomic it ends with.
constexpr int64_t kSplitKMinKPerSplit = 4 * kTileK;
// Blocks per CU the plain grid must already reach for split-K to be pointless.
constexpr int kSplitKTargetBlocksPerCU = 2;

// Compute-unit count of the currently selected device, cached per device
// ordinal for the same reason hipdnn_device_wave_size caches it: one host may
// drive several GPUs of different widths. 0 means "could not probe", which
// disables split-K rather than planning against a guessed width.
int qmatmul_device_cu_count() {
  constexpr int kMaxCachedDevices = 16;
  // Zero-initialized (0 == not probed yet), so no thread-safe-statics guard is
  // emitted; concurrent probes race only to write the same value.
  static int cached[kMaxCachedDevices] = {};

  auto probe = [](int d) {
    hipDeviceProp_t p;
    return (hipGetDeviceProperties(&p, d) == hipSuccess) ? p.multiProcessorCount
                                                         : 0;
  };

  int dev = 0;
  if (hipGetDevice(&dev) != hipSuccess || dev < 0)
    return 0;
  if (dev >= kMaxCachedDevices)
    return probe(dev);

  int c = cached[dev];
  if (c == 0) {
    c = probe(dev);
    cached[dev] = c;
  }
  return c;
}

struct QMatMulSplitKPlan {
  bool enabled;
  int splits;
  int64_t k_chunk;
};

QMatMulSplitKPlan qmatmul_plan_splitk(int64_t M, int64_t N, int64_t K,
                                      int64_t batch_count) {
  QMatMulSplitKPlan plan{false, 1, K};
  // M == 1 takes the GEMV kernel, which already parallelizes over N.
  if (M <= 1)
    return plan;

  const int cus = qmatmul_device_cu_count();
  if (cus <= 0)
    return plan;

  const int64_t base_blocks = ((N + kTileN - 1) / kTileN) *
                              ((M + kTileM - 1) / kTileM) * batch_count;
  const int64_t target = static_cast<int64_t>(cus) * kSplitKTargetBlocksPerCU;
  if (base_blocks >= target)
    return plan;

  int64_t want = (target + base_blocks - 1) / base_blocks;
  if (want > kSplitKMaxSplits)
    want = kSplitKMaxSplits;
  const int64_t k_cap = K / kSplitKMinKPerSplit;
  if (want > k_cap)
    want = k_cap;
  if (want < 2)
    return plan;

  // Round the slice up to a whole k-tile so each one starts on the boundary
  // the staging maps assume, then recompute the count so the last slice is not
  // empty.
  int64_t chunk = (K + want - 1) / want;
  chunk = (chunk + kTileK - 1) / kTileK * kTileK;
  const int64_t splits = (K + chunk - 1) / chunk;
  if (splits < 2)
    return plan;

  plan.enabled = true;
  plan.splits = static_cast<int>(splits);
  plan.k_chunk = chunk;
  return plan;
}

size_t qmatmul_splitk_workspace_bytes(int64_t M, int64_t N,
                                      int64_t batch_count) {
  return static_cast<size_t>(batch_count) * (M * N + M + N) * sizeof(int64_t);
}

//===----------------------------------------------------------------------===//
// GEMV (M == 1, the LLM decode shape)
//===----------------------------------------------------------------------===//

constexpr int kGemvBlock = 256;
constexpr int kGemvTileK = 256;

// Y[b, 0, n] = requantize(sum_k A[b, 0, k] * B[b, k, n])
//
// One thread per output column so that, for a fixed k, the block's global reads
// of B are contiguous. A is identical across threads, so it is staged in LDS
// once per k-tile instead of being re-read per column.
//
// A tiled GEMM degenerates badly at M == 1 (63 of 64 rows in each block tile
// are padding), and this shape dominates LLM decode, so it gets its own path.
//
// There is no trans_a flag: at M == 1 both layouts of A are the same K-element
// vector, so the index is unchanged. trans_b costs the coalescing of the B
// reads, since a [N, K] B puts each thread's column on its own row.
template <typename TA, typename TB, typename TY>
__global__ __launch_bounds__(kGemvBlock) void qmatmul_gemv_kernel(
    const TA *__restrict__ A, const TB *__restrict__ B, TY *__restrict__ Y,
    int64_t N, int64_t K, int64_t b_batch_stride, float m_scale, int32_t a_zp,
    int32_t b_zp, int32_t y_zp, int64_t k_zazb, bool need_row_a,
    bool need_col_b, bool trans_b) {
  __shared__ int32_t a_lds[kGemvTileK];

  const int64_t batch = blockIdx.y;
  A += batch * K;
  B += batch * b_batch_stride;
  Y += batch * N;

  const int64_t n = static_cast<int64_t>(blockIdx.x) * kGemvBlock + threadIdx.x;
  const bool active = n < N;

  int64_t acc = 0;
  int64_t row_a = 0;
  int64_t col_b = 0;

  for (int64_t k0 = 0; k0 < K; k0 += kGemvTileK) {
    const int64_t remaining = K - k0;
    const int tile =
        static_cast<int>(remaining < kGemvTileK ? remaining : kGemvTileK);
    for (int t = threadIdx.x; t < tile; t += kGemvBlock)
      a_lds[t] = static_cast<int32_t>(A[k0 + t]);
    __syncthreads();

    if (active) {
      for (int t = 0; t < tile; ++t) {
        const int64_t k = k0 + t;
        const int32_t b_val =
            static_cast<int32_t>(trans_b ? B[n * K + k] : B[k * N + n]);
        acc += a_lds[t] * b_val;
        if (need_col_b)
          col_b += b_val;
      }
      // rowA does not depend on n; every thread recomputes it from LDS rather
      // than paying a second reduction and broadcast.
      if (need_row_a) {
        for (int t = 0; t < tile; ++t)
          row_a += a_lds[t];
      }
    }
    __syncthreads();
  }

  if (active)
    Y[n] = qmatmul_epilogue<TY>(acc, row_a, col_b, k_zazb, a_zp, b_zp, m_scale,
                                y_zp);
}

//===----------------------------------------------------------------------===//
// Host launch and dtype dispatch
//===----------------------------------------------------------------------===//

// Bundled so the three dispatch levels below stay readable: A has 4 element
// types, B has 2 and Y has 4, and threading a dozen scalars through 32 switch
// arms by hand buries the one thing each arm actually says.
struct QMatMulArgs {
  hipStream_t stream;
  const void *A;
  const void *B;
  void *Y;
  int64_t M;
  int64_t N;
  int64_t K;
  int64_t batch_count;
  int64_t b_batch_stride;
  bool trans_a;
  bool trans_b;
  float m_scale;
  int32_t a_zp;
  int32_t b_zp;
  int32_t y_zp;
  void *workspace;
  size_t workspace_bytes;
};

// Picks between the three paths: GEMV at M == 1, then split-K when the plan
// asks for it and the caller supplied enough scratch, otherwise tiled.
template <typename TA, typename TB, typename TY>
int launch_qmatmul(const QMatMulArgs &g) {
  const int64_t k_zazb = static_cast<int64_t>(g.K) * g.a_zp * g.b_zp;
  const bool need_row_a = g.b_zp != 0;
  const bool need_col_b = g.a_zp != 0;

  const auto *a_ptr = static_cast<const TA *>(g.A);
  const auto *b_ptr = static_cast<const TB *>(g.B);
  auto *y_ptr = static_cast<TY *>(g.Y);

  (void)hipGetLastError();
  if (g.M == 1) {
    dim3 grid(static_cast<unsigned>((g.N + kGemvBlock - 1) / kGemvBlock),
              static_cast<unsigned>(g.batch_count));
    hipLaunchKernelGGL(HIP_KERNEL_NAME(qmatmul_gemv_kernel<TA, TB, TY>), grid,
                       dim3(kGemvBlock), 0, g.stream, a_ptr, b_ptr, y_ptr, g.N,
                       g.K, g.b_batch_stride, g.m_scale, g.a_zp, g.b_zp, g.y_zp,
                       k_zazb, need_row_a, need_col_b, g.trans_b);
    return 0;
  }

  const QMatMulSplitKPlan plan =
      qmatmul_plan_splitk(g.M, g.N, g.K, g.batch_count);
  const size_t ws_needed =
      plan.enabled ? qmatmul_splitk_workspace_bytes(g.M, g.N, g.batch_count)
                   : 0;
  bool use_splitk =
      plan.enabled && g.workspace && g.workspace_bytes >= ws_needed;

  if (use_splitk &&
      hipMemsetAsync(g.workspace, 0, ws_needed, g.stream) != hipSuccess) {
    // Half-zeroed accumulators would corrupt the result, so drop to the
    // single-pass kernel, which needs no scratch. Clear the sticky error too,
    // or the fallback's own clean launch would still be reported as a failure.
    (void)hipGetLastError();
    use_splitk = false;
  }

  if (use_splitk) {
    auto *acc_ws = static_cast<int64_t *>(g.workspace);
    auto *row_a_ws = acc_ws + g.batch_count * g.M * g.N;
    auto *col_b_ws = row_a_ws + g.batch_count * g.M;

    CUSTOM_KERNELS_DEBUG_LOG(
        "[custom_kernels] hip_qmatmul: split-K M=%lld N=%lld K=%lld "
        "splits=%d k_chunk=%lld scratch=%zu B\n",
        (long long)g.M, (long long)g.N, (long long)g.K, plan.splits,
        (long long)plan.k_chunk, ws_needed);

    // grid.z stays small: split-K only engages when the plain grid is under
    // kSplitKTargetBlocksPerCU blocks/CU, which bounds batch_count by that
    // same figure, so batch_count * splits cannot approach the 65535 limit.
    dim3 grid(static_cast<unsigned>((g.N + kTileN - 1) / kTileN),
              static_cast<unsigned>((g.M + kTileM - 1) / kTileM),
              static_cast<unsigned>(g.batch_count * plan.splits));
    hipLaunchKernelGGL(HIP_KERNEL_NAME(qmatmul_splitk_partial_kernel<TA, TB>),
                       grid, dim3(kTiledBlock), 0, g.stream, a_ptr, b_ptr,
                       acc_ws, row_a_ws, col_b_ws, g.M, g.N, g.K,
                       g.b_batch_stride, plan.splits, plan.k_chunk, need_row_a,
                       need_col_b, g.trans_a, g.trans_b);

    const int64_t total = g.batch_count * g.M * g.N;
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(qmatmul_splitk_epilogue_kernel<TY>),
        dim3(static_cast<unsigned>((total + kSplitKEpilogueBlock - 1) /
                                   kSplitKEpilogueBlock)),
        dim3(kSplitKEpilogueBlock), 0, g.stream, acc_ws, row_a_ws, col_b_ws,
        y_ptr, g.M, g.N, total, g.m_scale, g.a_zp, g.b_zp, g.y_zp, k_zazb);
    return 0;
  }

  dim3 grid(static_cast<unsigned>((g.N + kTileN - 1) / kTileN),
            static_cast<unsigned>((g.M + kTileM - 1) / kTileM),
            static_cast<unsigned>(g.batch_count));
  hipLaunchKernelGGL(HIP_KERNEL_NAME(qmatmul_tiled_kernel<TA, TB, TY>), grid,
                     dim3(kTiledBlock), 0, g.stream, a_ptr, b_ptr, y_ptr, g.M,
                     g.N, g.K, g.b_batch_stride, g.m_scale, g.a_zp, g.b_zp,
                     g.y_zp, k_zazb, need_row_a, need_col_b, g.trans_a,
                     g.trans_b);
  return 0;
}

// Y is 8- or 16-bit: a 16-bit activation is usually requantized back to 16-bit
// to keep the precision it was chosen for, but an 8-bit output is legal too.
template <typename TA, typename TB>
int dispatch_y(int y_dtype, const QMatMulArgs &args) {
  switch (y_dtype) {
  case HIP_DTYPE_INT8:
    return launch_qmatmul<TA, TB, int8_t>(args);
  case HIP_DTYPE_UINT8:
    return launch_qmatmul<TA, TB, uint8_t>(args);
  case HIP_DTYPE_INT16:
    return launch_qmatmul<TA, TB, int16_t>(args);
  case HIP_DTYPE_UINT16:
    return launch_qmatmul<TA, TB, uint16_t>(args);
  default:
    return -1;
  }
}

// B stays 8-bit because only A is byte-split, leaving a 16-bit weight no path
// through the dot4 pipeline. Splitting B the same way would work -- four
// plane products instead of two, with the widest still bounded by 255*255 --
// but is not implemented: 16-bit integer weights save neither space nor
// bandwidth over fp16, which has hardware matrix instructions.
template <typename TA>
int dispatch_b(int b_dtype, int y_dtype, const QMatMulArgs &args) {
  switch (b_dtype) {
  case HIP_DTYPE_INT8:
    return dispatch_y<TA, int8_t>(y_dtype, args);
  case HIP_DTYPE_UINT8:
    return dispatch_y<TA, uint8_t>(y_dtype, args);
  default:
    return -1;
  }
}

int dispatch_a(int a_dtype, int b_dtype, int y_dtype, const QMatMulArgs &args) {
  switch (a_dtype) {
  case HIP_DTYPE_INT8:
    return dispatch_b<int8_t>(b_dtype, y_dtype, args);
  case HIP_DTYPE_UINT8:
    return dispatch_b<uint8_t>(b_dtype, y_dtype, args);
  case HIP_DTYPE_INT16:
    return dispatch_b<int16_t>(b_dtype, y_dtype, args);
  case HIP_DTYPE_UINT16:
    return dispatch_b<uint16_t>(b_dtype, y_dtype, args);
  default:
    return -1;
  }
}

} // namespace

//===----------------------------------------------------------------------===//
// C API
//===----------------------------------------------------------------------===//

extern "C" size_t hip_qmatmul_workspace_bytes(int64_t M, int64_t N, int64_t K,
                                              int64_t batch_count) {
  if (M <= 0 || N <= 0 || K <= 0 || batch_count <= 0)
    return 0;
  const QMatMulSplitKPlan plan = qmatmul_plan_splitk(M, N, K, batch_count);
  return plan.enabled ? qmatmul_splitk_workspace_bytes(M, N, batch_count) : 0;
}

extern "C" int hip_qmatmul(void *stream, const void *A, const void *B, void *Y,
                           int64_t M, int64_t N, int64_t K, int64_t batch_count,
                           int64_t b_batch_stride, int trans_a, int trans_b,
                           int a_dtype, int b_dtype, int y_dtype, float M_scale,
                           int64_t a_zp, int64_t b_zp, int64_t y_zp,
                           void *workspace, size_t workspace_bytes) {
  if (!A || !B || !Y) {
    fprintf(stderr, "[custom_kernels] hip_qmatmul: null tensor argument\n");
    return -1;
  }
  if (M <= 0 || N <= 0 || K <= 0 || batch_count <= 0)
    return 0;

  CUSTOM_KERNELS_DEBUG_LOG(
      "[custom_kernels] hip_qmatmul: M=%lld N=%lld K=%lld batch=%lld "
      "b_batch_stride=%lld trans=(%d,%d) dtypes=(%d,%d,%d) M_scale=%g "
      "zp=(%lld,%lld,%lld)\n",
      (long long)M, (long long)N, (long long)K, (long long)batch_count,
      (long long)b_batch_stride, trans_a, trans_b, a_dtype, b_dtype, y_dtype,
      (double)M_scale, (long long)a_zp, (long long)b_zp, (long long)y_zp);

  const QMatMulArgs args{static_cast<hipStream_t>(stream),
                         A,
                         B,
                         Y,
                         M,
                         N,
                         K,
                         batch_count,
                         b_batch_stride,
                         trans_a != 0,
                         trans_b != 0,
                         M_scale,
                         static_cast<int32_t>(a_zp),
                         static_cast<int32_t>(b_zp),
                         static_cast<int32_t>(y_zp),
                         workspace,
                         workspace_bytes};
  const int rc = dispatch_a(a_dtype, b_dtype, y_dtype, args);

  if (rc != 0) {
    fprintf(stderr,
            "[custom_kernels] hip_qmatmul: unsupported dtype triple "
            "(a=%d, b=%d, y=%d)\n",
            a_dtype, b_dtype, y_dtype);
    return rc;
  }

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr, "[custom_kernels] hip_qmatmul launch failed: %s\n",
            hipGetErrorString(err));
  }
  return static_cast<int>(err);
}
