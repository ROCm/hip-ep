/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip_custom_kernels.h"
#include "debug_log.h"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include <cstdint>
#include <cstdio>

// =============================================================================
// Conv — general forward convolution (1D / 2D / 3D spatial)
// =============================================================================
//
// Forward ONNX Conv over an (N, C_in, D_1[, D_2[, D_3]]) input against a
// (C_out, C_in/group, k_1[, k_2[, k_3]]) filter, with arbitrary stride,
// dilation, group and asymmetric padding, and the bias add fused in.
//
// Per group g, the convolution is a matrix product:
//
//   Y[n, g*M + m, o] = bias[g*M + m] + sum_{ic,kv} A[m, (ic,kv)] * B[(ic,kv), o]
//
//   M = C_out / group                 rows      (output channels in the group)
//   K = (C_in / group) * prod(kernel) contraction
//   N = prod(output spatial)          columns   (output positions)
//
// A is the filter, already laid out row-major as [M][K] inside the group, so it
// needs no gather at all. B is the im2col matrix, which is never materialised:
// each element is addressed straight out of the input as
//
//   x_j = o_j * stride_j - pad_begin_j + i_j * dilation_j
//
// and reads as zero when any x_j falls outside the input. That is what padding
// means, and it is why only pad_begin appears in the ABI -- pad_end shows up
// only in the output extent, which the caller passes in.
//
// Three kernels, because the shapes that matter are three different problems:
//
//   conv_wmma   -- fp16, large output-channel count. Runs the same blocked GEMM
//                  as conv_tiled but on the matrix cores. This exists because
//                  the part cannot reach MIOpen's throughput any other way: at
//                  16 WGPs and 2.7 GHz the scalar FMA ceiling is ~11 TFLOP/s,
//                  and MIOpen measures above that on the Whisper 1D shapes, so
//                  it is provably on the matrix cores too.
//
//   conv_tiled  -- the same blocking with scalar FMAs, for fp32 and bf16 and
//                  for fp16 shapes too small for the WMMA tiles. A BM x BN
//                  output tile per block, staged through LDS in BK-deep slices,
//                  with a TM x TN register tile per thread. The register tile
//                  is what keeps the inner loop compute-bound rather than
//                  load-bound: TM+TN LDS reads feed TM*TN multiply-adds.
//
//   conv_direct -- one thread per output element. Used when M is too small to
//                  fill a tile row block, which is the depthwise and narrow
//                  grouped case (M == 1 for depthwise): there the tiled kernel
//                  would compute a whole row block to keep one row of it, and
//                  there is no output-channel reuse for a register tile to
//                  exploit anyway.
//
// Accumulation is always float, whatever the storage dtype, so fp16 and bf16
// convolutions do not lose the reduction to rounding -- K reaches into the
// thousands on the Whisper 1D shapes.
//
// The second thing that decides this kernel's speed, after the arithmetic, is
// the index math around the im2col gather: every staged element has to turn a
// flat contraction index into a channel and a kernel tap, and a flat output
// index into spatial coordinates. Both are divisions by runtime values, and
// with native 64-bit division that is hundreds of cycles against a few cycles
// of useful work. Hence FastDiv below, and hence the staging assignments are
// chosen so that everything that can be hoisted out of the reduction loop is.

namespace {

// ---------------------------------------------------------------------------
// Division by a runtime constant, without dividing
// ---------------------------------------------------------------------------
//
// The GPU has no integer divide, so `a / b` with a runtime b expands inline to
// a long reciprocal-refinement sequence -- and to a far longer one for 64-bit
// operands. Every divisor here (kernel volume, spatial extents) is fixed for
// the whole launch, so each one is turned on the host into a multiply-high and
// a shift.
//
// Valid for numerators below 2^31. Both numerators used here -- a contraction
// index below K and an output position below prod(output spatial) -- are
// bounded by tensor extents that must already fit in memory, so this is not a
// practical restriction. Address arithmetic stays 64-bit; only the division
// narrows.
struct FastDiv {
  uint32_t mul; // 0 => a pure shift (divisor is a power of two)
  uint32_t shr;
};

__device__ __forceinline__ uint32_t fastDiv(uint32_t n, FastDiv d) {
  return d.mul ? (__umulhi(n, d.mul) >> d.shr) : (n >> d.shr);
}

static FastDiv makeFastDiv(int64_t denom) {
  const uint32_t d = static_cast<uint32_t>(denom < 1 ? 1 : denom);
  FastDiv r{0u, 0u};
  uint32_t l = 0;
  while ((1u << l) < d)
    ++l; // ceil(log2(d))
  if ((1u << l) == d) {
    r.shr = l; // exact power of two, including d == 1
    return r;
  }
  const uint32_t p = 31 + l;
  r.mul = static_cast<uint32_t>(((1ull << p) + d - 1) / d);
  r.shr = p - 32;
  return r;
}

// ---------------------------------------------------------------------------
// dtype conversion
// ---------------------------------------------------------------------------

template <typename T> __device__ __forceinline__ float to_float(T v);
template <> __device__ __forceinline__ float to_float<float>(float v) {
  return v;
}
template <> __device__ __forceinline__ float to_float<__half>(__half v) {
  return __half2float(v);
}
template <>
__device__ __forceinline__ float to_float<hip_bfloat16>(hip_bfloat16 v) {
  return static_cast<float>(v);
}

template <typename T> __device__ __forceinline__ T from_float(float v);
template <> __device__ __forceinline__ float from_float<float>(float v) {
  return v;
}
template <> __device__ __forceinline__ __half from_float<__half>(float v) {
  return __float2half(v);
}
template <>
__device__ __forceinline__ hip_bfloat16 from_float<hip_bfloat16>(float v) {
  return hip_bfloat16(v);
}

// ---------------------------------------------------------------------------
// Problem descriptor
//
// Passed by value so the kernels take one argument instead of thirty, and so
// the derived quantities (volumes, per-group extents) are computed once on the
// host rather than by every thread.
// ---------------------------------------------------------------------------

struct ConvParams {
  int64_t N;         // batch
  int64_t Cin, Cout; // total channels
  int64_t group;
  int64_t CinPerGroup, MPerGroup;

  int32_t in0i, in1i, in2i; // input spatial extents, for the bounds test
  int64_t in0, in1, in2;    // input spatial extents (unused axes = 1)
  int64_t out0, out1, out2; // output spatial extents (unused axes = 1)
  int64_t k0, k1, k2;
  int64_t s0, s1, s2;
  int64_t p0, p1, p2; // pad_begin only; pad_end is implied by out_d*
  int64_t d0, d1, d2;

  int64_t kvol;     // k0*k1*k2
  int64_t K;        // CinPerGroup * kvol
  int64_t Nout;     // out0*out1*out2
  int64_t inPlane;  // in0*in1*in2 (elements per input channel)
  int64_t k1k2;     // k1*k2
  int64_t out1out2; // out1*out2

  // Reciprocals of the five divisors the index decode needs.
  FastDiv rKvol, rK1k2, rK2, rOut1out2, rOut2;

  // Tile-grid geometry for the block swizzle below. Depends on the tile shape,
  // so it is filled in per launch rather than with the rest of the descriptor.
  int32_t nBlkM, nBlkN;
  int32_t gmLog2; // log2 of the swizzle group height
  FastDiv rGroupSpan;
};

// The input coordinate that kernel tap (0,0,0) of an output position reads.
// May be negative when pad_begin > 0. `lin` is the same point as a flat offset
// within the channel, which is what makes the split in KSlice below work.
struct PosOrigin {
  int32_t x0, x1, x2;
  int64_t lin; // (x0*in1 + x1)*in2 + x2
};

__device__ __forceinline__ PosOrigin decodeOrigin(const ConvParams &pr,
                                                  int64_t o) {
  const uint32_t ou = static_cast<uint32_t>(o);
  const uint32_t o0 = fastDiv(ou, pr.rOut1out2);
  const uint32_t r = ou - o0 * static_cast<uint32_t>(pr.out1out2);
  const uint32_t o1 = fastDiv(r, pr.rOut2);
  const uint32_t o2 = r - o1 * static_cast<uint32_t>(pr.out2);
  PosOrigin org;
  org.x0 = static_cast<int32_t>(static_cast<int64_t>(o0) * pr.s0 - pr.p0);
  org.x1 = static_cast<int32_t>(static_cast<int64_t>(o1) * pr.s1 - pr.p1);
  org.x2 = static_cast<int32_t>(static_cast<int64_t>(o2) * pr.s2 - pr.p2);
  org.lin = (static_cast<int64_t>(org.x0) * pr.in1 + org.x1) * pr.in2 + org.x2;
  return org;
}

// ---------------------------------------------------------------------------
// The im2col address, split by what it depends on
// ---------------------------------------------------------------------------
//
// An im2col element is B[(ic, tap), o] = x[ic, org(o) + tap*dilation], whose
// flat offset within the batch/group is
//
//   ic*inPlane + ((org.x0 + c0)*in1 + (org.x1 + c1))*in2 + (org.x2 + c2)
//
// writing c_j for tap_j*dilation_j. That expression is linear in the
// coordinates, so it splits exactly into a part that depends only on the
// contraction index and a part that depends only on the output position:
//
//   off(ic, tap)  = ic*inPlane + (c0*in1 + c1)*in2 + c2
//   org.lin(o)    = (org.x0*in1 + org.x1)*in2 + org.x2
//
// Both halves are loop-invariant in the staging loops -- the position half over
// the reduction, the contraction half across the block's columns -- and the
// second is already hoisted. Hoisting the first is what KSlice is for, and it
// matters because *every thread staging the same k recomputes the same value*:
// with BN=128 columns per slice, the three divisions and three 64-bit
// multiplies below were being done 128 times over for one result. Computing
// each slice once per block instead leaves the per-element gather with three
// adds and a bounds test, and no multiply or division at all.
//
// The tap coordinates have to survive into the gather because the bounds test
// is per position: whether a tap is inside the input depends on the column.
struct KSlice {
  int64_t off;
  int32_t c0, c1, c2;
};

// The contraction half, for one contraction index. A `kidx` past the end of the
// contraction is a zero-padded column, which is expressed as a tap that the
// bounds test is guaranteed to reject.
__device__ __forceinline__ KSlice makeKSlice(const ConvParams &pr,
                                             int64_t kidx) {
  KSlice s;
  if (kidx >= pr.K) {
    s.off = 0;
    s.c0 = pr.in0i; // always out of bounds
    s.c1 = 0;
    s.c2 = 0;
    return s;
  }
  const uint32_t ku = static_cast<uint32_t>(kidx);
  const uint32_t ic = fastDiv(ku, pr.rKvol);
  const uint32_t kr = ku - ic * static_cast<uint32_t>(pr.kvol);
  const uint32_t i0 = fastDiv(kr, pr.rK1k2);
  const uint32_t kr2 = kr - i0 * static_cast<uint32_t>(pr.k1k2);
  const uint32_t i1 = fastDiv(kr2, pr.rK2);
  const uint32_t i2 = kr2 - i1 * static_cast<uint32_t>(pr.k2);

  s.c0 = static_cast<int32_t>(i0 * static_cast<uint32_t>(pr.d0));
  s.c1 = static_cast<int32_t>(i1 * static_cast<uint32_t>(pr.d1));
  s.c2 = static_cast<int32_t>(i2 * static_cast<uint32_t>(pr.d2));
  s.off = static_cast<int64_t>(ic) * pr.inPlane +
          (static_cast<int64_t>(s.c0) * pr.in1 + s.c1) * pr.in2 + s.c2;
  return s;
}

// One im2col element, given both halves of its address. Out-of-bounds taps read
// as zero, which is what padding means. `xChan` already carries the batch and
// group part of the address.
template <typename T>
__device__ __forceinline__ float gatherSplit(const T *__restrict__ xChan,
                                             const ConvParams &pr,
                                             const PosOrigin &org,
                                             const KSlice &ks) {
  const int32_t x0 = org.x0 + ks.c0;
  if (x0 < 0 || x0 >= pr.in0i) return 0.0f;
  const int32_t x1 = org.x1 + ks.c1;
  if (x1 < 0 || x1 >= pr.in1i) return 0.0f;
  const int32_t x2 = org.x2 + ks.c2;
  if (x2 < 0 || x2 >= pr.in2i) return 0.0f;
  return to_float<T>(xChan[org.lin + ks.off]);
}

// One im2col element B[(ic, kv), o], decoding the contraction index on the
// spot. Only the direct kernel wants this: there every thread holds a different
// output position, so there is nothing to share a decode with.
template <typename T>
__device__ __forceinline__ float gatherB(const T *__restrict__ xChan,
                                         const ConvParams &pr,
                                         const PosOrigin &org, int64_t kidx) {
  return gatherSplit<T>(xChan, pr, org, makeKSlice(pr, kidx));
}

// The batch/group base of the input, and of the filter, for one block.
__device__ __forceinline__ void blockBases(const ConvParams &pr, int64_t z,
                                           int64_t &nbatch, int64_t &g) {
  nbatch = z / pr.group;
  g = z - nbatch * pr.group;
}

// ---------------------------------------------------------------------------
// Block swizzle — which output tile a block owns
// ---------------------------------------------------------------------------
//
// This is the difference between the tiled kernels being DRAM-bound and being
// cache-bound, and on the Whisper 1D shapes it is worth more than every other
// tuning knob in this file combined.
//
// A block reads a BM-row slice of the filter and a BN-column slice of the
// im2col matrix. Two blocks in the same tile row share the whole filter slice;
// two in the same tile column share the whole im2col slice. Per BK-deep step
// each slice is only a few KB, so if the blocks that are resident together
// cover a compact square of the tile grid, one DRAM read of a slice serves
// every block in its row or column and the rest hit L2. If they cover a long
// thin strip, one of the two operands is re-fetched from DRAM by every block.
//
// The natural dispatch order is exactly that thin strip: with the tile grid
// mapped to (blockIdx.x = column, blockIdx.y = row), blocks launch x-major, so
// the resident set is a handful of full tile rows -- good sharing of the
// filter, almost none of the im2col matrix. On whisper_l1 that costs ~350 MB of
// traffic against ~21 MB of unique data.
//
// So the tile grid is flattened into blockIdx.x and walked in column-major
// groups of kGroupM rows instead: kGroupM consecutive blocks share an im2col
// slice, and successive groups walk across the columns sharing filter slices.
// kGroupM is a power of two so the group-local split is a mask and a shift, and
// only the group index costs a (reciprocal) division.
//
// The last group is left ragged rather than clamped -- clamping would need a
// division by a runtime group height. The blocks above nBlkM exit immediately,
// which costs at most gm-1 empty blocks per tile column; capping gm at the
// number of tile rows keeps that bounded for the short-and-wide shapes, where a
// fixed height would launch a mostly-empty grid.

constexpr int kGroupM = 8;

// Tile-grid extent to launch in x. Rounds the tile rows up to a whole number of
// groups, so every launched pid decodes to a group-local row.
static inline unsigned swizzledGridX(const ConvParams &pr) {
  const int64_t gm = 1 << pr.gmLog2;
  return static_cast<unsigned>(((pr.nBlkM + gm - 1) / gm) * gm * pr.nBlkN);
}

// False when this block is one of the ragged tail's empty ones.
__device__ __forceinline__ bool swizzledTile(const ConvParams &pr, int &blkM,
                                             int &blkN) {
  const uint32_t pid = blockIdx.x;
  const uint32_t span = static_cast<uint32_t>(pr.nBlkN) << pr.gmLog2;
  const uint32_t groupId = fastDiv(pid, pr.rGroupSpan);
  const uint32_t inGroup = pid - groupId * span;
  const uint32_t gmMask = (1u << pr.gmLog2) - 1u;
  blkM = static_cast<int>((groupId << pr.gmLog2) + (inGroup & gmMask));
  blkN = static_cast<int>(inGroup >> pr.gmLog2);
  return blkM < pr.nBlkM;
}

// Fill in the tile-grid geometry the swizzle needs for one tile shape.
static inline ConvParams withTileGrid(ConvParams pr, int64_t bm, int64_t bn) {
  pr.nBlkM = static_cast<int32_t>((pr.MPerGroup + bm - 1) / bm);
  pr.nBlkN = static_cast<int32_t>((pr.Nout + bn - 1) / bn);
  // Largest power of two that is both <= kGroupM and <= the tile rows.
  int lg = 0;
  while ((1 << (lg + 1)) <= kGroupM && (1 << (lg + 1)) <= pr.nBlkM) ++lg;
  pr.gmLog2 = lg;
  pr.rGroupSpan =
      makeFastDiv(static_cast<int64_t>(pr.nBlkN) << lg);
  return pr;
}

// ---------------------------------------------------------------------------
// Scalar tiled kernel
// ---------------------------------------------------------------------------
//
// The two staging loops are deliberately assigned differently, because the two
// operands are laid out differently and each wants the axis it is contiguous
// along to run across the wavefront:
//
//   * The filter is [M][K] row-major, so a thread takes a contiguous run of K
//     for one output channel. Assigning threads down M instead would make
//     neighbouring lanes read addresses K apart -- one memory transaction per
//     lane, for a tile that is read every slice.
//   * The im2col matrix is contiguous along the output position (for unit
//     stride), so a thread takes one position and walks K. That also lets the
//     position decode -- the expensive half of the address math -- be hoisted
//     out of the reduction loop entirely.
//
// Both then write LDS transposed into k-major order, which is what the inner
// loop wants: a fixed k, contiguous in the tile index, so TM and TN reads
// vectorise.

template <typename T, int BM, int BN, int BK, int TM, int TN>
__global__ __launch_bounds__((BM / TM) * (BN / TN)) void
conv_tiled_kernel(const T *__restrict__ x, const T *__restrict__ w,
                  const T *__restrict__ bias, T *__restrict__ y,
                  ConvParams pr) {
  constexpr int kThreads = (BM / TM) * (BN / TN);
  static_assert(kThreads % BN == 0, "B column tiling");
  static_assert((BK * BM) % kThreads == 0 && (BK * BN) % kThreads == 0,
                "stage tiling");
  // Per-thread filter run length, and the resulting threads-per-row. The
  // identity kThreads * kVecA == BK * BM makes the block cover exactly BM
  // rows, so no row is loaded twice and none is missed.
  constexpr int kVecA = (BK * BM) / kThreads;
  constexpr int kRowThreadsA = BK / kVecA;
  static_assert(kVecA <= BK && BK % kVecA == 0, "filter run divides BK");

  __shared__ float As[BK][BM]; // filter slice, k-major for the inner loop
  __shared__ float Bs[BK][BN]; // im2col slice
  // The contraction half of the im2col address, one entry per k in the slice,
  // decoded once for the whole block. Double-buffered so the next slice's
  // decode overlaps this slice's arithmetic and costs no extra barrier.
  __shared__ KSlice ks[2][BK];

  const int tid = threadIdx.x;
  int blkM, blkN;
  if (!swizzledTile(pr, blkM, blkN)) return;
  const int64_t n0 = static_cast<int64_t>(blkN) * BN; // first position
  const int64_t m0 = static_cast<int64_t>(blkM) * BM; // first channel
  int64_t nbatch, g;
  blockBases(pr, blockIdx.z, nbatch, g);

  // Filter staging: one contiguous run of K per thread, resolved once.
  const int rowA = tid / kRowThreadsA;
  const int kOffA = (tid % kRowThreadsA) * kVecA;
  const int64_t mRowA = m0 + rowA;
  const bool rowValidA = mRowA < pr.MPerGroup;
  const int64_t wRowBase = (g * pr.MPerGroup + mRowA) * pr.K;

  // im2col staging: column -> output position -> input origin, also once.
  constexpr int kLoadsB = (BK * BN) / kThreads;
  constexpr int kStepB = kThreads / BN;
  const int colB = tid % BN;
  const int kkB = tid / BN;
  const int64_t oPos = n0 + colB;
  const bool colValid = oPos < pr.Nout;
  const PosOrigin org = colValid ? decodeOrigin(pr, oPos) : PosOrigin{0, 0, 0};
  const T *xChan = x + (nbatch * pr.Cin + g * pr.CinPerGroup) * pr.inPlane;

  const int ty = tid / (BN / TN); // register-tile row block
  const int tx = tid % (BN / TN); // register-tile column block
  const int aOff = ty * TM;
  const int bOff = tx * TN;

  float acc[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j) acc[i][j] = 0.0f;

  int cur = 0;
  if (tid < BK) ks[0][tid] = makeKSlice(pr, tid);
  __syncthreads();

  for (int64_t kBase = 0; kBase < pr.K; kBase += BK) {
    const int64_t wBase = wRowBase + kBase + kOffA;
    const int64_t kRem = pr.K - kBase - kOffA; // taps left in this thread's run
#pragma unroll
    for (int i = 0; i < kVecA; ++i)
      As[kOffA + i][rowA] =
          (rowValidA && i < kRem) ? to_float<T>(w[wBase + i]) : 0.0f;

#pragma unroll
    for (int l = 0; l < kLoadsB; ++l) {
      const int kk = kkB + l * kStepB;
      Bs[kk][colB] =
          colValid ? gatherSplit<T>(xChan, pr, org, ks[cur][kk]) : 0.0f;
    }
    __syncthreads();

#pragma unroll
    for (int kk = 0; kk < BK; ++kk) {
      float a[TM], b[TN];
#pragma unroll
      for (int i = 0; i < TM; ++i) a[i] = As[kk][aOff + i];
#pragma unroll
      for (int j = 0; j < TN; ++j) b[j] = Bs[kk][bOff + j];
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] += a[i] * b[j];
    }
    if (tid < BK) ks[cur ^ 1][tid] = makeKSlice(pr, kBase + BK + tid);
    __syncthreads();
    cur ^= 1;
  }

  // Store, folding in the bias.
  const int64_t yBatchBase = nbatch * pr.Cout * pr.Nout;
  for (int i = 0; i < TM; ++i) {
    const int64_t m = m0 + aOff + i;
    if (m >= pr.MPerGroup) break;
    const int64_t oc = g * pr.MPerGroup + m;
    const float bv = bias ? to_float<T>(bias[oc]) : 0.0f;
    T *row = y + yBatchBase + oc * pr.Nout;
    for (int j = 0; j < TN; ++j) {
      const int64_t o = n0 + bOff + j;
      if (o >= pr.Nout) break;
      row[o] = from_float<T>(acc[i][j] + bv);
    }
  }
}

// ---------------------------------------------------------------------------
// WMMA tiled kernel (fp16)
// ---------------------------------------------------------------------------

typedef _Float16 half16 __attribute__((ext_vector_type(16)));
typedef _Float16 half8 __attribute__((ext_vector_type(8)));
typedef float float8 __attribute__((ext_vector_type(8)));

// RDNA3-family wave32 builtin. Where it is absent, trap: this file still has
// to compile for other architectures, and a silent wrong answer is worse than
// an abort that names the reason.
#if !defined(__has_builtin) ||                                                 \
    !__has_builtin(__builtin_amdgcn_wmma_f32_16x16x16_f16_w32)
static __device__ __forceinline__ float8 conv_wmma_unavailable(half16, half16,
                                                              float8 c) {
  __builtin_trap();
  return c;
}
#define __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c)                    \
  conv_wmma_unavailable((a), (b), (c))
#endif

constexpr int kWmma = 16; // 16x16x16 tile, so BK is fixed at 16

// The blocked GEMM again, with the register tile replaced by WMMA fragments.
//
// Fragment layout (wave32), matching the in-tree WMMA GEMM:
//   a_frag[e] at lane m  = A_tile[m][k = e]      lane -> M row, element -> K
//   b_frag[e] at lane n  = B_tile[k = e][n]      element -> K row, lane -> N
//   c_frag[e]            -> row (e*2 + pair), col lane,  pair = (tid % 32) / 16
// Lanes 0-15 and 16-31 must hold identical a/b data, which indexing by
// `tid % 16` gives for free.
//
// The LDS tiles are stored K-contiguous (`[tile][BK + pad]`) rather than
// K-major as in the scalar kernel, because a fragment is 16 consecutive K for
// one row or column -- that way loading one is two 16-byte reads instead of
// sixteen 2-byte ones. The padding is what keeps that from serialising: it
// makes the row stride an odd multiple of 16 bytes, so the 32 lanes of a
// fragment load land on 32 distinct banks.
template <int BM, int BN, int WM, int WN>
__global__ __launch_bounds__((BM / (WM * kWmma)) * (BN / (WN * kWmma)) * 32)
void conv_wmma_kernel(const __half *__restrict__ x,
                      const __half *__restrict__ w,
                      const __half *__restrict__ bias, __half *__restrict__ y,
                      ConvParams pr) {
  constexpr int kWavesM = BM / (WM * kWmma);
  constexpr int kWavesN = BN / (WN * kWmma);
  constexpr int kThreads = kWavesM * kWavesN * 32;
  constexpr int kLdsK = kWmma + 8; // 24 halves = 48 B: 16-byte aligned, odd
  static_assert((BM * kWmma) % kThreads == 0, "A stage tiling");
  static_assert((BN * kWmma) % kThreads == 0, "B stage tiling");
  static_assert(kThreads % BN == 0, "B column tiling");

  __shared__ _Float16 As[BM][kLdsK];
  __shared__ _Float16 Bs[BN][kLdsK];
  __shared__ KSlice ks[2][kWmma]; // see conv_tiled_kernel

  const int tid = threadIdx.x;
  const int wave = tid / 32;
  const int lane16 = tid % kWmma;
  const int pair = (tid % 32) / kWmma;
  const int waveM = wave / kWavesN;
  const int waveN = wave % kWavesN;

  int blkM, blkN;
  if (!swizzledTile(pr, blkM, blkN)) return;
  const int64_t n0 = static_cast<int64_t>(blkN) * BN;
  const int64_t m0 = static_cast<int64_t>(blkM) * BM;
  int64_t nbatch, g;
  blockBases(pr, blockIdx.z, nbatch, g);

  // Filter staging: contiguous run of K per row, as in the scalar kernel.
  constexpr int kVecA = (BM * kWmma) / kThreads;
  constexpr int kRowThreadsA = kWmma / kVecA;
  const int rowA = tid / kRowThreadsA;
  const int kOffA = (tid % kRowThreadsA) * kVecA;
  const bool rowValidA = m0 + rowA < pr.MPerGroup;
  const int64_t wRowBase = (g * pr.MPerGroup + m0 + rowA) * pr.K;

  // im2col staging: one output position per thread, coalesced across lanes.
  constexpr int kLoadsB = (BN * kWmma) / kThreads;
  constexpr int kStepB = kThreads / BN;
  const int colB = tid % BN;
  const int kkB = tid / BN;
  const int64_t oPos = n0 + colB;
  const bool colValid = oPos < pr.Nout;
  const PosOrigin org = colValid ? decodeOrigin(pr, oPos) : PosOrigin{0, 0, 0};
  const __half *xChan =
      x + (nbatch * pr.Cin + g * pr.CinPerGroup) * pr.inPlane;

  float8 acc[WM][WN];
#pragma unroll
  for (int i = 0; i < WM; ++i)
#pragma unroll
    for (int j = 0; j < WN; ++j) acc[i][j] = (float8)0.0f;

  int cur = 0;
  if (tid < kWmma) ks[0][tid] = makeKSlice(pr, tid);
  __syncthreads();

  for (int64_t kBase = 0; kBase < pr.K; kBase += kWmma) {
    const int64_t wBase = wRowBase + kBase + kOffA;
    const int64_t kRem = pr.K - kBase - kOffA;
#pragma unroll
    for (int i = 0; i < kVecA; ++i)
      As[rowA][kOffA + i] =
          (rowValidA && i < kRem) ? static_cast<_Float16>(w[wBase + i])
                                  : static_cast<_Float16>(0.0f);

#pragma unroll
    for (int l = 0; l < kLoadsB; ++l) {
      const int kk = kkB + l * kStepB;
      Bs[colB][kk] =
          colValid ? static_cast<_Float16>(
                         gatherSplit<__half>(xChan, pr, org, ks[cur][kk]))
                   : static_cast<_Float16>(0.0f);
    }
    __syncthreads();

    half16 aFrag[WM], bFrag[WN];
#pragma unroll
    for (int i = 0; i < WM; ++i) {
      const _Float16 *p = &As[waveM * WM * kWmma + i * kWmma + lane16][0];
      *reinterpret_cast<half8 *>(&aFrag[i]) =
          *reinterpret_cast<const half8 *>(p);
      *(reinterpret_cast<half8 *>(&aFrag[i]) + 1) =
          *reinterpret_cast<const half8 *>(p + 8);
    }
#pragma unroll
    for (int j = 0; j < WN; ++j) {
      const _Float16 *p = &Bs[waveN * WN * kWmma + j * kWmma + lane16][0];
      *reinterpret_cast<half8 *>(&bFrag[j]) =
          *reinterpret_cast<const half8 *>(p);
      *(reinterpret_cast<half8 *>(&bFrag[j]) + 1) =
          *reinterpret_cast<const half8 *>(p + 8);
    }
#pragma unroll
    for (int i = 0; i < WM; ++i)
#pragma unroll
      for (int j = 0; j < WN; ++j)
        acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
            aFrag[i], bFrag[j], acc[i][j]);
    if (tid < kWmma) ks[cur ^ 1][tid] = makeKSlice(pr, kBase + kWmma + tid);
    __syncthreads();
    cur ^= 1;
  }

  const int64_t yBatchBase = nbatch * pr.Cout * pr.Nout;
#pragma unroll
  for (int i = 0; i < WM; ++i) {
#pragma unroll
    for (int j = 0; j < WN; ++j) {
      const int64_t o = n0 + waveN * WN * kWmma + j * kWmma + lane16;
      if (o >= pr.Nout) continue;
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int64_t m =
            m0 + waveM * WM * kWmma + i * kWmma + e * 2 + pair;
        if (m >= pr.MPerGroup) continue;
        const int64_t oc = g * pr.MPerGroup + m;
        const float bv = bias ? to_float<__half>(bias[oc]) : 0.0f;
        y[yBatchBase + oc * pr.Nout + o] = from_float<__half>(acc[i][j][e] + bv);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Direct kernel — one thread per output element
// ---------------------------------------------------------------------------

template <typename T>
__global__ void conv_direct_kernel(const T *__restrict__ x,
                                   const T *__restrict__ w,
                                   const T *__restrict__ bias,
                                   T *__restrict__ y, ConvParams pr,
                                   int64_t total) {
  int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid >= total) return;

  // tid indexes Y as [N][Cout][Nout], row-major.
  const int64_t o = tid % pr.Nout;
  const int64_t rest = tid / pr.Nout;
  const int64_t oc = rest % pr.Cout;
  const int64_t nbatch = rest / pr.Cout;
  const int64_t g = oc / pr.MPerGroup;

  const PosOrigin org = decodeOrigin(pr, o);
  const T *wRow = w + oc * pr.K;
  const T *xChan = x + (nbatch * pr.Cin + g * pr.CinPerGroup) * pr.inPlane;

  float sum = bias ? to_float<T>(bias[oc]) : 0.0f;
  for (int64_t kidx = 0; kidx < pr.K; ++kidx)
    sum += to_float<T>(wRow[kidx]) * gatherB<T>(xChan, pr, org, kidx);
  y[tid] = from_float<T>(sum);
}

// ---------------------------------------------------------------------------
// Launch
// ---------------------------------------------------------------------------

// Compute units the driver reports. On RDNA this counts WGPs rather than CUs (a
// 40-CU 8060S reports 20), so it is only ever usable as a relative "how wide is
// this part" figure -- never to derive an absolute CU count.
static int deviceUnitCount() {
  static const int units = [] {
    int dev = 0;
    hipDeviceProp_t prop{};
    if (hipGetDevice(&dev) != hipSuccess ||
        hipGetDeviceProperties(&prop, dev) != hipSuccess ||
        prop.multiProcessorCount <= 0)
      return 16; // assume a small part rather than over-tiling a big one
    return prop.multiProcessorCount;
  }();
  return units;
}

constexpr int kBK = 16;

template <typename T, int BM, int BN, int TM, int TN>
static void launchTile(hipStream_t stream, const void *x, const void *w,
                       const void *bias, void *y, const ConvParams &base) {
  const ConvParams pr = withTileGrid(base, BM, BN);
  dim3 grid(swizzledGridX(pr), 1u,
            static_cast<unsigned>(pr.N * pr.group));
  hipLaunchKernelGGL((conv_tiled_kernel<T, BM, BN, kBK, TM, TN>), grid,
                     dim3((BM / TM) * (BN / TN)), 0, stream,
                     static_cast<const T *>(x), static_cast<const T *>(w),
                     static_cast<const T *>(bias), static_cast<T *>(y), pr);
}

template <int BM, int BN, int WM, int WN>
static void launchWmma(hipStream_t stream, const void *x, const void *w,
                       const void *bias, void *y, const ConvParams &base) {
  constexpr int threads = (BM / (WM * kWmma)) * (BN / (WN * kWmma)) * 32;
  const ConvParams pr = withTileGrid(base, BM, BN);
  dim3 grid(swizzledGridX(pr), 1u,
            static_cast<unsigned>(pr.N * pr.group));
  hipLaunchKernelGGL((conv_wmma_kernel<BM, BN, WM, WN>), grid, dim3(threads), 0,
                     stream, static_cast<const __half *>(x),
                     static_cast<const __half *>(w),
                     static_cast<const __half *>(bias),
                     static_cast<__half *>(y), pr);
}

// Whether a tile shape is worth using: its row block has to be fully live (a
// block computes its whole tile whether or not the rows exist) and the grid it
// produces has to cover the device.
static bool tileFits(const ConvParams &pr, int64_t bm, int64_t bn,
                     int64_t units) {
  if (pr.MPerGroup < bm) return false;
  const int64_t blocks = ((pr.MPerGroup + bm - 1) / bm) *
                         ((pr.Nout + bn - 1) / bn) * pr.N * pr.group;
  return blocks >= units;
}

// fp16 gets first refusal on the matrix cores; everything else, and every fp16
// shape too small for a WMMA tile, falls through to the scalar ladder.
template <typename T>
static bool launchWmmaIfProfitable(hipStream_t, const void *, const void *,
                                   const void *, void *, const ConvParams &,
                                   int64_t) {
  return false;
}

template <>
bool launchWmmaIfProfitable<__half>(hipStream_t stream, const void *x,
                                    const void *w, const void *bias, void *y,
                                    const ConvParams &pr, int64_t units) {
  // The 4x4 wave tile issues 16 WMMAs off 8 fragment loads where 4x2 issues 8
  // off 6, so it asks much less of LDS per unit of matrix-core work -- but it
  // doubles the accumulator register count, which costs occupancy, and it
  // halves the number of blocks.
  //
  // Which of those wins is decided by the row count, and measurably so. Below
  // a few hundred output channels there are few tile rows, so the swizzle group
  // is shallow and every block re-reads much of the im2col matrix: doubling BN
  // cuts that in half and the LDS relief is free (whisper80 384/512 gain
  // 15-20%). With a thousand-odd channels there is no shortage of blocks, the
  // reduction is long enough that hiding the filter loads is what matters, and
  // the lost occupancy dominates (whisper_l1 loses 33%).
  if (pr.MPerGroup <= 512 && tileFits(pr, 128, 256, units)) {
    launchWmma<128, 256, 4, 4>(stream, x, w, bias, y, pr);
    return true;
  }
  // A taller row tile, for the shapes with many output channels. What this buys
  // is not arithmetic but traffic: the im2col matrix is re-read once per tile
  // row, and it is never materialised, so each re-read is a fresh gather out of
  // the input. Halving the number of tile rows halves the dominant cost of the
  // kernel -- on whisper_l1 the gather is 51% of the runtime, moving 236 MB for
  // 21 MB of unique data at close to what the memory system can deliver, so
  // there is nothing to win by making it faster, only by doing less of it.
  if (tileFits(pr, 256, 128, units)) {
    launchWmma<256, 128, 4, 2>(stream, x, w, bias, y, pr);
    return true;
  }
  if (tileFits(pr, 128, 128, units)) {
    launchWmma<128, 128, 4, 2>(stream, x, w, bias, y, pr);
    return true;
  }
  // Narrow row block for the grouped and low-channel-count shapes -- the audio
  // encoder's second conv has 32 output channels and would throw away three
  // quarters of a 128-row tile.
  if (tileFits(pr, 32, 256, units)) {
    launchWmma<32, 256, 2, 2>(stream, x, w, bias, y, pr);
    return true;
  }
  return false;
}

template <typename T>
static int launch(hipStream_t stream, const void *x, const void *w,
                  const void *bias, void *y, const ConvParams &pr) {
  const int64_t totalOut = pr.N * pr.Cout * pr.Nout;
  if (totalOut <= 0 || pr.K <= 0) return 0;

  const int64_t units = deviceUnitCount();

  if (!launchWmmaIfProfitable<T>(stream, x, w, bias, y, pr, units)) {
    // Widest scalar tile that fits, then narrower ones. A wider tile does more
    // FMAs per LDS read -- 8x8 does 64 per 16, where 4x4 does 16 per 8 -- so
    // the widest one that fits the problem is the fastest.
    if (tileFits(pr, 128, 128, units)) {
      launchTile<T, 128, 128, 8, 8>(stream, x, w, bias, y, pr);
    } else if (tileFits(pr, 64, 64, units)) {
      launchTile<T, 64, 64, 4, 4>(stream, x, w, bias, y, pr);
    } else if (pr.MPerGroup >= 24 && tileFits(pr, 32, 128, units)) {
      launchTile<T, 32, 128, 2, 8>(stream, x, w, bias, y, pr);
    } else {
      const int block = 256;
      const int64_t grid = (totalOut + block - 1) / block;
      hipLaunchKernelGGL(conv_direct_kernel<T>,
                         dim3(static_cast<unsigned>(grid)), dim3(block), 0,
                         stream, static_cast<const T *>(x),
                         static_cast<const T *>(w),
                         static_cast<const T *>(bias), static_cast<T *>(y), pr,
                         totalOut);
    }
  }

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr, "[custom_kernels] hip_conv launch failed: %s\n",
            hipGetErrorString(err));
  }
  return static_cast<int>(err);
}

} // namespace

extern "C" int hip_conv(void *stream, const void *input, const void *weights,
                        const void *bias, void *output, int hip_dtype,
                        int spatial_rank, int64_t N, int64_t Cin, int64_t Cout,
                        int64_t in_d0, int64_t in_d1, int64_t in_d2,
                        int64_t out_d0, int64_t out_d1, int64_t out_d2,
                        int64_t k0, int64_t k1, int64_t k2, int64_t s0,
                        int64_t s1, int64_t s2, int64_t p0, int64_t p1,
                        int64_t p2, int64_t dil0, int64_t dil1, int64_t dil2,
                        int64_t group) {
  if (spatial_rank < 1 || spatial_rank > 3) {
    fprintf(stderr, "[custom_kernels] hip_conv: unsupported spatial_rank=%d\n",
            spatial_rank);
    return -1;
  }
  if (!input || !weights || !output) {
    fprintf(stderr, "[custom_kernels] hip_conv: null input/weights/output\n");
    return -1;
  }
  if (group < 1 || Cin % group != 0 || Cout % group != 0) {
    fprintf(stderr,
            "[custom_kernels] hip_conv: group=%lld does not divide Cin=%lld / "
            "Cout=%lld\n",
            (long long)group, (long long)Cin, (long long)Cout);
    return -1;
  }

  ConvParams pr{};
  pr.N = N;
  pr.Cin = Cin;
  pr.Cout = Cout;
  pr.group = group;
  pr.CinPerGroup = Cin / group;
  pr.MPerGroup = Cout / group;
  pr.in0 = in_d0;  pr.in1 = in_d1;  pr.in2 = in_d2;
  pr.in0i = static_cast<int32_t>(in_d0);
  pr.in1i = static_cast<int32_t>(in_d1);
  pr.in2i = static_cast<int32_t>(in_d2);
  pr.out0 = out_d0; pr.out1 = out_d1; pr.out2 = out_d2;
  pr.k0 = k0; pr.k1 = k1; pr.k2 = k2;
  pr.s0 = s0; pr.s1 = s1; pr.s2 = s2;
  pr.p0 = p0; pr.p1 = p1; pr.p2 = p2;
  pr.d0 = dil0; pr.d1 = dil1; pr.d2 = dil2;
  pr.kvol = k0 * k1 * k2;
  pr.K = pr.CinPerGroup * pr.kvol;
  pr.Nout = out_d0 * out_d1 * out_d2;
  pr.inPlane = in_d0 * in_d1 * in_d2;
  pr.k1k2 = k1 * k2;
  pr.out1out2 = out_d1 * out_d2;
  pr.rKvol = makeFastDiv(pr.kvol);
  pr.rK1k2 = makeFastDiv(pr.k1k2);
  pr.rK2 = makeFastDiv(k2);
  pr.rOut1out2 = makeFastDiv(pr.out1out2);
  pr.rOut2 = makeFastDiv(out_d2);

  CUSTOM_KERNELS_DEBUG_LOG(
      "[custom_kernels] hip_conv: dtype=%d spatial_rank=%d N=%lld Cin=%lld "
      "Cout=%lld group=%lld in=[%lld,%lld,%lld] out=[%lld,%lld,%lld] "
      "k=[%lld,%lld,%lld] s=[%lld,%lld,%lld] pbegin=[%lld,%lld,%lld] "
      "dil=[%lld,%lld,%lld] bias=%s\n",
      hip_dtype, spatial_rank, (long long)N, (long long)Cin, (long long)Cout,
      (long long)group, (long long)in_d0, (long long)in_d1, (long long)in_d2,
      (long long)out_d0, (long long)out_d1, (long long)out_d2, (long long)k0,
      (long long)k1, (long long)k2, (long long)s0, (long long)s1,
      (long long)s2, (long long)p0, (long long)p1, (long long)p2,
      (long long)dil0, (long long)dil1, (long long)dil2,
      bias ? "yes" : "null");

  hipStream_t hip_stream = static_cast<hipStream_t>(stream);
  switch (hip_dtype) {
    case HIP_DTYPE_FLOAT32:
      return launch<float>(hip_stream, input, weights, bias, output, pr);
    case HIP_DTYPE_FLOAT16:
      return launch<__half>(hip_stream, input, weights, bias, output, pr);
    case HIP_DTYPE_BFLOAT16:
      return launch<hip_bfloat16>(hip_stream, input, weights, bias, output, pr);
    default:
      fprintf(stderr, "[custom_kernels] hip_conv: unsupported dtype=%d\n",
              hip_dtype);
      return -1;
  }
}
