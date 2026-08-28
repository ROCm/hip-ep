/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip_custom_kernels.h"
#include "debug_log.h"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <cstdio>

// ===========================================================================
// Winograd F(2x2, 3x3) for fp16 3x3 stride-1 convolution
// ===========================================================================
//
// Why F(2,3) and not F(4,3)
// -------------------------
// F(2,3)'s transform constants are {0, +/-1, +/-1/2} -- every one exactly
// representable in fp16, so the transforms themselves introduce no rounding
// beyond the additions. F(4,3) needs 1/6 and 1/24, which are not, and the
// larger transform amplifies input error by a much bigger factor. bevformer
// already sits close enough to fp16's range to saturate the CPU reference, so
// the extra headroom is not available to spend. F(2,3) gives 16 multiplies per
// 4 outputs against 36 for direct, a 2.25x arithmetic reduction, and is what
// MIOpen's `f2x3` assembly kernel implements.
//
// What this stage does and does not buy
// -------------------------------------
// This is the non-fused form: the input transform materialises V in scratch,
// a batched GEMM reads it back, and the output transform reads the result
// back. It is the correctness and measurement stage, and it is worth being
// explicit that on the shapes it targets it is *not* expected to be much
// faster, because of where the cost actually is.
//
// For esrgan's dominant convolution (Cin=192, Cout=64, 250x250, 3x3 s1) the
// direct path moves about 278 MB -- 216 MB of it re-deriving the im2col matrix
// -- and measures 1.08 ms, which is the bandwidth bound for that traffic.
// Non-fused Winograd moves:
//
//   V write   16 * Cin  * ntiles * 2B   =  96 MB
//   V read    (once per GEMM row block) =  96 MB
//   M write   16 * Cout * ntiles * 2B   =  32 MB
//   M read                              =  32 MB
//   x read, y write                      =  32 MB
//                                          ------
//                                          288 MB
//
// which is not an improvement. The 2.25x is in the multiplies, and these
// shapes are not limited by multiplies. Materialising V is what costs it: the
// transform's output is 16 values per (channel, tile) where the input was 4x4
// overlapping, so V is larger than the input it came from.
//
// The win requires computing V inside the GEMM's B stage and never writing it
// down -- then x is read once per GEMM row block instead of V being written
// and read, and with the tile halo shared through LDS the traffic falls to
// roughly 80-130 MB depending on the column block, which is the range MIOpen's
// measured 3.4x advantage sits in. That is a separate change and is sequenced
// after this one deliberately: it needs these transforms to be known correct
// first, and a fused kernel that is also wrong is not debuggable.
//
// Layout
// ------
// The four stages and their scratch, for one block of `tiles` spatial tiles:
//
//   U[xi][cout][cin]   filter transform, 16 * Cout * Cin, hoisted and cached
//                      by the caller across calls (it depends only on w)
//   V[xi][cin][t]      input transform,  16 * Cin * tiles
//   M[xi][cout][t]     16 independent GEMMs U[xi] @ V[xi]
//   y                  output transform of M, bias folded in
//
// `xi` is the position within the 4x4 transform domain, so each of the 16 is
// an independent (Cout x Cin) @ (Cin x tiles) GEMM. They are batched on the
// grid's z axis rather than launched separately: at 16 launches per tile block
// per convolution, launch overhead would be a large fraction of a model like
// esrgan with 351 of them.
//
// Scratch is blocked over tiles rather than channels. Blocking over channels
// would force M to be accumulated across blocks and so to exist at full size;
// blocking over tiles makes every block independent end to end, which is what
// keeps the working set bounded on esrgan's 1000x1000 layers where a full V
// would be 512 MB.

namespace {

constexpr int kTileIn = 4;   // input tile extent
constexpr int kTileOut = 2;  // output tile extent
constexpr int kXi = kTileIn * kTileIn;

// One row of the input transform B^T, and of its transpose applied to columns.
// B^T = [[1,0,-1,0],[0,1,1,0],[0,-1,1,0],[0,1,0,-1]]
__device__ __forceinline__ void btdb4(float a0, float a1, float a2, float a3,
                                      float &b0, float &b1, float &b2,
                                      float &b3) {
  b0 = a0 - a2;
  b1 = a1 + a2;
  b2 = a2 - a1;
  b3 = a1 - a3;
}

// One row of the output transform A^T = [[1,1,1,0],[0,1,-1,-1]].
__device__ __forceinline__ void atma4(float a0, float a1, float a2, float a3,
                                      float &b0, float &b1) {
  b0 = a0 + a1 + a2;
  b1 = a1 - a2 - a3;
}

// One column of the filter transform G = [[1,0,0],[.5,.5,.5],[.5,-.5,.5],
// [0,0,1]]. The halves are exact in fp16, which is the reason for F(2,3).
__device__ __forceinline__ void gg3(float g0, float g1, float g2, float &u0,
                                    float &u1, float &u2, float &u3) {
  u0 = g0;
  u1 = 0.5f * (g0 + g1 + g2);
  u2 = 0.5f * (g0 - g1 + g2);
  u3 = g2;
}

// ---------------------------------------------------------------------------
// Filter transform: w[Cout][Cin][3][3] -> U[16][Cout][Cin]
// ---------------------------------------------------------------------------
//
// One thread per (cout, cin). Hoisted out of the convolution entirely by the
// caller, so its cost is paid once per weight tensor rather than per call and
// its access pattern does not matter much.
__global__ void winograd_filter_kernel(const __half *__restrict__ w,
                                       __half *__restrict__ u, int64_t Cout,
                                       int64_t Cin) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = Cout * Cin;
  if (idx >= total) return;

  const int64_t cin = idx % Cin;
  const int64_t cout = idx / Cin;

  const __half *g = w + idx * 9;
  float gv[9];
  for (int i = 0; i < 9; ++i) gv[i] = __half2float(g[i]);

  // G g: 3 columns of the 3x3 filter become 4 rows.
  float t[4][3];
  for (int c = 0; c < 3; ++c)
    gg3(gv[0 * 3 + c], gv[1 * 3 + c], gv[2 * 3 + c], t[0][c], t[1][c], t[2][c],
        t[3][c]);

  // (G g) G^T: each of the 4 rows becomes 4 wide.
  for (int r = 0; r < 4; ++r) {
    float o0, o1, o2, o3;
    gg3(t[r][0], t[r][1], t[r][2], o0, o1, o2, o3);
    const float vals[4] = {o0, o1, o2, o3};
    for (int c = 0; c < 4; ++c) {
      const int xi = r * 4 + c;
      u[(static_cast<int64_t>(xi) * Cout + cout) * Cin + cin] =
          __float2half(vals[c]);
    }
  }
}

// ---------------------------------------------------------------------------
// Input transform: x -> V[16][Cin][tiles]
// ---------------------------------------------------------------------------
//
// One thread per (channel, tile). `tile0` is the first tile of this block in
// the flattened (batch, tileRow, tileCol) order, so a block of tiles can span
// batch images without the caller having to split on them.
//
// Writes are contiguous in the tile index, which is the fast axis of V and the
// axis the GEMM's B stage walks. Reads are a 4x4 window stepping by 2, so
// neighbouring threads' windows overlap by half -- the overlap is what the
// fused form will exploit and this form simply pays for.
__global__ void winograd_input_kernel(const __half *__restrict__ x,
                                      __half *__restrict__ v, int64_t Cin,
                                      int64_t inH, int64_t inW, int64_t padTop,
                                      int64_t padLeft, int64_t ntW,
                                      int64_t ntPerImage, int64_t tile0,
                                      int64_t tiles) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = Cin * tiles;
  if (idx >= total) return;

  const int64_t t = idx % tiles;      // tile within this block
  const int64_t c = idx / tiles;      // input channel
  const int64_t gt = tile0 + t;       // global tile index

  const int64_t n = gt / ntPerImage;
  const int64_t rem = gt - n * ntPerImage;
  const int64_t th = rem / ntW;
  const int64_t tw = rem - th * ntW;

  const int64_t ih0 = th * kTileOut - padTop;
  const int64_t iw0 = tw * kTileOut - padLeft;

  const __half *xc = x + (n * Cin + c) * inH * inW;

  float d[4][4];
  for (int i = 0; i < 4; ++i) {
    const int64_t ih = ih0 + i;
    const bool okH = ih >= 0 && ih < inH;
    for (int j = 0; j < 4; ++j) {
      const int64_t iw = iw0 + j;
      d[i][j] = (okH && iw >= 0 && iw < inW)
                    ? __half2float(xc[ih * inW + iw])
                    : 0.0f;
    }
  }

  // B^T d, then (B^T d) B.
  float r[4][4];
  for (int j = 0; j < 4; ++j)
    btdb4(d[0][j], d[1][j], d[2][j], d[3][j], r[0][j], r[1][j], r[2][j],
          r[3][j]);

  for (int i = 0; i < 4; ++i) {
    float o0, o1, o2, o3;
    btdb4(r[i][0], r[i][1], r[i][2], r[i][3], o0, o1, o2, o3);
    const float vals[4] = {o0, o1, o2, o3};
    for (int j = 0; j < 4; ++j) {
      const int xi = i * 4 + j;
      v[(static_cast<int64_t>(xi) * Cin + c) * tiles + t] = __float2half(vals[j]);
    }
  }
}

// ---------------------------------------------------------------------------
// Batched GEMM: M[xi] = U[xi] @ V[xi], 16 batches on the grid's z axis
// ---------------------------------------------------------------------------
//
// Register-tiled with both operands staged through LDS, accumulating in fp32.
// Deliberately not a matrix-core kernel: the arithmetic here is 2.25x less
// than the direct path's to begin with and this stage is limited by the V it
// has to read, so matrix cores would speed up the part that is not the
// problem. If the fused form changes that balance it can be revisited.
template <int BM, int BN, int TM, int TN, int BK>
__global__ void winograd_gemm_kernel(const __half *__restrict__ u,
                                     const __half *__restrict__ v,
                                     __half *__restrict__ m, int32_t Cout,
                                     int32_t Cin, int32_t tiles) {
  static_assert(BM % TM == 0 && BN % TN == 0, "tile must divide the block");
  constexpr int kThreads = (BM / TM) * (BN / TN);
  static_assert(BK * BM % kThreads == 0 && BK * BN % kThreads == 0,
                "staging must divide evenly across the block");

  __shared__ float As[BK][BM];
  __shared__ float Bs[BK][BN];

  const int tid = static_cast<int>(threadIdx.x);
  const int tx = tid % (BN / TN);
  const int ty = tid / (BN / TN);

  const int32_t n0 = static_cast<int32_t>(blockIdx.x) * BN;
  const int32_t m0 = static_cast<int32_t>(blockIdx.y) * BM;
  const int32_t xi = static_cast<int32_t>(blockIdx.z);

  const __half *uXi = u + static_cast<int64_t>(xi) * Cout * Cin;
  const __half *vXi = v + static_cast<int64_t>(xi) * Cin * tiles;
  __half *mXi = m + static_cast<int64_t>(xi) * Cout * tiles;

  const int stageRow = tid % BM;   // this thread's row of A
  const int stageCol = tid % BN;   // ... and column of B
  const int stageKA = tid / BM;
  const int stageKB = tid / BN;
  constexpr int kStepA = kThreads / BM;
  constexpr int kStepB = kThreads / BN;

  float acc[TM][TN];
  for (int i = 0; i < TM; ++i)
    for (int j = 0; j < TN; ++j) acc[i][j] = 0.0f;

  for (int32_t k0 = 0; k0 < Cin; k0 += BK) {
    for (int i = 0; i < BK * BM / kThreads; ++i) {
      const int kk = stageKA + i * kStepA;
      const int32_t k = k0 + kk;
      const int32_t row = m0 + stageRow;
      As[kk][stageRow] =
          (k < Cin && row < Cout)
              ? __half2float(uXi[static_cast<int64_t>(row) * Cin + k])
              : 0.0f;
    }
    for (int i = 0; i < BK * BN / kThreads; ++i) {
      const int kk = stageKB + i * kStepB;
      const int32_t k = k0 + kk;
      const int32_t col = n0 + stageCol;
      Bs[kk][stageCol] =
          (k < Cin && col < tiles)
              ? __half2float(vXi[static_cast<int64_t>(k) * tiles + col])
              : 0.0f;
    }
    __syncthreads();

    for (int kk = 0; kk < BK; ++kk) {
      float a[TM], b[TN];
      for (int i = 0; i < TM; ++i) a[i] = As[kk][ty * TM + i];
      for (int j = 0; j < TN; ++j) b[j] = Bs[kk][tx * TN + j];
      for (int i = 0; i < TM; ++i)
        for (int j = 0; j < TN; ++j) acc[i][j] += a[i] * b[j];
    }
    __syncthreads();
  }

  for (int i = 0; i < TM; ++i) {
    const int32_t row = m0 + ty * TM + i;
    if (row >= Cout) break;
    for (int j = 0; j < TN; ++j) {
      const int32_t col = n0 + tx * TN + j;
      if (col < tiles)
        mXi[static_cast<int64_t>(row) * tiles + col] = __float2half(acc[i][j]);
    }
  }
}

// ---------------------------------------------------------------------------
// Output transform: M[16][Cout][tiles] -> y, bias folded in
// ---------------------------------------------------------------------------
//
// One thread per (output channel, tile). Each writes the tile's kTileOut x
// kTileOut output cells, dropping the ones a partial edge tile puts outside
// the output.
__global__ void winograd_output_kernel(const __half *__restrict__ m,
                                       const __half *__restrict__ bias,
                                       __half *__restrict__ y, int64_t Cout,
                                       int64_t outH, int64_t outW, int64_t ntW,
                                       int64_t ntPerImage, int64_t tile0,
                                       int64_t tiles) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = Cout * tiles;
  if (idx >= total) return;

  const int64_t t = idx % tiles;
  const int64_t co = idx / tiles;
  const int64_t gt = tile0 + t;

  const int64_t n = gt / ntPerImage;
  const int64_t rem = gt - n * ntPerImage;
  const int64_t th = rem / ntW;
  const int64_t tw = rem - th * ntW;

  float mv[4][4];
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j) {
      const int xi = i * 4 + j;
      mv[i][j] = __half2float(
          m[(static_cast<int64_t>(xi) * Cout + co) * tiles + t]);
    }

  // A^T M, then (A^T M) A.
  float r[2][4];
  for (int j = 0; j < 4; ++j)
    atma4(mv[0][j], mv[1][j], mv[2][j], mv[3][j], r[0][j], r[1][j]);

  const float bv = bias ? __half2float(bias[co]) : 0.0f;
  __half *yc = y + (n * Cout + co) * outH * outW;

  for (int i = 0; i < 2; ++i) {
    float o0, o1;
    atma4(r[i][0], r[i][1], r[i][2], r[i][3], o0, o1);
    const int64_t oh = th * kTileOut + i;
    if (oh >= outH) continue;
    const float vals[2] = {o0, o1};
    for (int j = 0; j < 2; ++j) {
      const int64_t ow = tw * kTileOut + j;
      if (ow < outW) yc[oh * outW + ow] = __float2half(vals[j] + bv);
    }
  }
}

// ===========================================================================
// Fused Winograd: one kernel, V never reaches memory
// ===========================================================================
//
// What the non-fused form got wrong, measured
// ------------------------------------------
// The four-stage form above is 2.2x *slower* than the direct path on esrgan
// (conv 472 ms against 215 ms). Splitting the stages means V and M are written
// to memory and read back, and on these shapes that scratch traffic is larger
// than the im2col traffic it was meant to replace.
//
// Where the win actually is
// -------------------------
// It is not the 2.25x in multiplies. Measuring the direct path on esrgan's
// 64->32 250x250 layer: 0.268 ms at 8.6 TFLOP/s, which is 29% of this part's
// ~29.7 TFLOP/s fp16 WMMA peak, and it gets that at 12-16 waves/SIMD -- full
// occupancy, 96-104 VGPRs, nothing to reclaim. So it is not arithmetic-bound
// and not occupancy-bound. Its B stage reads K * Nout = (64*9) * 62500 halves
// = 72 MB per call, and 72 MB in 0.268 ms is 269 GB/s, which is this part's
// LPDDR5X bandwidth. It is bandwidth-bound, at the limit, and the cause is
// im2col's 9x amplification: every input element is read once per tap.
//
// Winograd's advantage over that is a traffic advantage. A 2x2 output tile
// needs a 4x4 input window, 16 reads per 4 outputs against 36, and
// neighbouring windows overlap by half. Staging a *patch* of tiles in LDS and
// sharing the halo brings it down further: a 4x4 patch of tiles spans 10x10
// inputs, 100 reads for 64 outputs, against im2col's 576.
//
//   direct        72 MB   measured 0.268 ms
//   fused         ~17 MB  predicted ~0.10 ms
//
// So the halo sharing is not an optimisation here, it is the entire point --
// which is why this reads x through LDS rather than letting each thread gather
// its own windows.
//
// Structure
// ---------
// One block owns one output patch: BM output channels x (TPH x TPW) output
// tiles, and carries all 16 transform components, so x is read once.
//
//   for each chunk of BK input channels:
//     stage the raw input patch          x -> Xs        (halo shared here)
//     transform it                       Xs -> Vs      (V stays in LDS)
//     stage the transformed filter       U  -> Us
//     accumulate the 16 GEMMs into registers
//   output transform of the 16, bias folded, straight from registers -> y
//
// Every thread holds all 16 components of the outputs it owns, so the output
// transform reads them from registers and there is no third LDS buffer and no
// second phase. That costs 16*TM*TN accumulators, which is why TM and TN are
// 2 and not the 8x2 a plain GEMM would want.
//
// Why the tile block is wide
// --------------------------
// The block needs all 16 filter planes, so it stages 16*BK*BM filter elements
// against only BK*BN input elements, and does 16*BM*BN*BK multiplies. Its
// arithmetic intensity is therefore
//
//   16*BM*BN*BK / (16*BK*BM + BK*BN)  =  16*BM*BN / (16*BM + BN)
//
// which is dominated by the filter term until BN is large. A first cut at
// BM=32, BN=16 gave 15.5 multiplies per staged element -- *worse* than the
// direct kernel's 32x256 tile at 28 -- and measured accordingly: no faster
// than the non-fused form. BM=32, BN=64 gives 57, and 64 tiles is also what
// makes the halo cheap, an 8x8 patch of tiles spanning 18x18 inputs, 324 reads
// for 256 outputs against im2col's 2304.
//
// The wave's LDS access falls out of that geometry: 32 lanes of a wave share
// one `ty` and span `tx` 0..31, so the filter read is a broadcast of one
// address and the input read is contiguous. Neither conflicts.
//
// Why the staging is K-contiguous and the inner loop is a dot product
// ------------------------------------------------------------------
// This kernel is instruction-bound, not bandwidth-bound: on esrgan's 64->32
// 250x250 layer it moves ~31 MB, which is 92 GB/s against the ~270 GB/s the
// direct path saturates. So what matters is instructions per multiply.
//
// Staged K-major, with one scalar fp16 read per operand and a convert before
// the FMA, a group of 4 multiplies costs 2 LDS reads + 4 converts + 4 FMAs --
// 10 instructions for 4 multiplies, and it measured as such (0.339 ms against
// the direct path's 0.268 ms).
//
// Staged K-contiguous instead, one 16-byte LDS read fetches a whole BK slice
// of one row, and v_dot2_f32_f16 consumes it two multiplies at a time with the
// fp32 accumulation built in and no converts at all. The same work becomes
// 4 LDS reads + 256 dot2 for 512 multiplies, 0.625 instructions per multiply
// against 2.5.
//
// It also makes the filter staging a single 16-byte load per thread instead of
// eight scalar ones, because a thread's 8 channels are now contiguous in both
// the source and the destination.

constexpr int kFBM = 32;   // output channels per block
constexpr int kFTPH = 8;   // output tiles per block, rows
constexpr int kFTPW = 8;   // ... and columns
constexpr int kFBN = kFTPH * kFTPW;
constexpr int kFTM = 2;    // output channels per thread
constexpr int kFTN = 2;    // output tiles per thread
constexpr int kFBK = 8;    // input channels per chunk
constexpr int kFThreads = (kFBM / kFTM) * (kFBN / kFTN);
constexpr int kFPH = kFTPH * kTileOut + 2; // staged input patch extent
constexpr int kFPW = kFTPW * kTileOut + 2;

static_assert(kFThreads == 512, "block size");
static_assert(kFBK * kFBN == kFThreads,
              "the input transform is one window per thread");
static_assert(kXi * kFBM == kFThreads,
              "the filter staging is one BK slice per thread");
static_assert(kFBN % kFTN == 0 && kFBM % kFTM == 0, "thread tile divides");
static_assert(kFBK % 2 == 0, "dot2 consumes K in pairs");

// Where this kernel beats the direct path, measured
// --------------------------------------------------
// It does not beat it everywhere, and the boundary is sharp enough to gate on.
// Over 49 fp16 3x3 stride-1 shapes taken from esrgan, resnet50, detr,
// bevformer, mb1-ssd and yolov5lu, the ratio against `hip_conv` runs from
// 1.85x down to 0.36x, and what separates them is the number of blocks the
// whole convolution launches:
//
//   blocks <= ~3 per CU     1.0x to 1.85x   (resnet50's 3x3 stack: 1.43x)
//   blocks >  ~3 per CU     0.36x to 0.94x  (esrgan's 250x250 layers)
//
// The reason is that the two are limited by different things. Above the
// threshold both reach steady state, and there this kernel loses: it is bound
// by LDS bandwidth, reading one operand element per multiply (~2 bytes/MAC),
// which caps it at 1.86 T MAC/s against the direct path's 4.52 -- the 2.25x
// fewer multiplies do not make up a 2.4x lower rate. Below the threshold the
// convolution is too small to fill the machine, and there the direct path's
// wide tiles leave it parallelism-starved (Cin=Cout=256 at 14x14 launches two
// blocks and takes as long as the same shape at 50x50), while this kernel's
// narrower blocks give the scheduler more to work with.
//
// Raising the ceiling would mean a bigger register tile to amortise the
// operand reads, and that is what the 16 live components rule out: acc is
// 16*TM*TN, so TM=TN=2 is already 64 VGPRs and 4x4 would be 256. Selecting on
// the block count instead lands within 1.5% of picking the better path per
// shape (22.75 ms against a 22.40 ms oracle, where always-direct is 24.85).
//
// The threshold is in blocks per CU rather than absolute blocks so it tracks
// the part; the plateau is wide (2.5 to 3.1 per CU all measure the same), so
// it is not knife-edge.
constexpr int kResidentBlocksPerUnit = 3;

// A whole BK slice of one row, which is what one LDS instruction fetches.
typedef _Float16 halfK __attribute__((ext_vector_type(kFBK)));
typedef _Float16 half2v __attribute__((ext_vector_type(2)));

// v_dot2_f32_f16: two fp16 multiplies accumulated into fp32, one instruction,
// no converts. Falling back to the arithmetic it stands for keeps this file
// compiling for architectures without it -- correct, just slower.
#if !defined(__has_builtin) || !__has_builtin(__builtin_amdgcn_fdot2)
__device__ __forceinline__ float wgDot2(half2v a, half2v b, float c) {
  return c + static_cast<float>(a[0]) * static_cast<float>(b[0]) +
         static_cast<float>(a[1]) * static_cast<float>(b[1]);
}
#else
__device__ __forceinline__ float wgDot2(half2v a, half2v b, float c) {
  return __builtin_amdgcn_fdot2(a, b, c, false);
}
#endif

__global__ __launch_bounds__(kFThreads) void winograd_fused_kernel(
    const __half *__restrict__ x, const __half *__restrict__ u,
    const __half *__restrict__ bias, __half *__restrict__ y, int32_t Cin,
    int32_t Cout, int32_t inH, int32_t inW, int32_t outH, int32_t outW,
    int32_t padTop, int32_t padLeft, int32_t ntH, int32_t ntW,
    int32_t patchesW, int32_t patchesPerImage) {
  // 16-byte aligned base, and kFPW is even, so a window row's packed pairs are
  // always 4-byte aligned wherever the even column origin puts them.
  __shared__ __align__(16) _Float16 Xs[kFBK][kFPH][kFPW];
  static_assert(kFPW % 2 == 0, "packed window rows need an even patch width");
  __shared__ __align__(16) _Float16 Us[kXi][kFBM][kFBK];
  __shared__ __align__(16) _Float16 Vs[kXi][kFBN][kFBK];

  const int tid = static_cast<int>(threadIdx.x);

  // This block's patch of output tiles, and the output channels it covers.
  const int patch = static_cast<int>(blockIdx.x);
  const int n = patch / patchesPerImage;
  const int pin = patch - n * patchesPerImage;
  const int tRow0 = (pin / patchesW) * kFTPH;
  const int tCol0 = (pin % patchesW) * kFTPW;
  const int m0 = static_cast<int>(blockIdx.y) * kFBM;

  // Top-left input element of the staged patch, in the padded frame.
  const int pih0 = tRow0 * kTileOut - padTop;
  const int piw0 = tCol0 * kTileOut - padLeft;

  const __half *xImg = x + static_cast<int64_t>(n) * Cin * inH * inW;

  float acc[kXi][kFTM][kFTN];
#pragma unroll
  for (int e = 0; e < kXi; ++e)
#pragma unroll
    for (int i = 0; i < kFTM; ++i)
#pragma unroll
      for (int j = 0; j < kFTN; ++j) acc[e][i][j] = 0.0f;

  // GEMM operand positions. The N side indexes tiles, the M side channels.
  // ty is constant across a wave and tx spans it, which is what makes the
  // filter read a broadcast and the input read contiguous.
  const int ty = tid / (kFBN / kFTN);
  const int tx = tid % (kFBN / kFTN);

  // Input-transform assignment: one (channel, tile) window per thread.
  const int vK = tid / kFBN;
  const int vT = tid - vK * kFBN;
  const int vTh = vT / kFTPW;
  const int vTw = vT - vTh * kFTPW;

  // Filter-staging assignment: one (component, channel-slice) row per thread.
  const int uXi = tid / kFBM;
  const int uM = tid - uXi * kFBM;

  // Patch-staging assignment: one (row, column) of the patch per thread, held
  // across every channel and every chunk. All of it -- the decomposition, the
  // bounds test, and the in-plane offset -- is invariant in the channel loop,
  // so it is computed once here. Left in the loop it was two integer divisions
  // by an extent that is not a power of two, per element, per chunk.
  const int sPy = tid / kFPW;
  const int sPx = tid - sPy * kFPW;
  const bool sActive = tid < kFPH * kFPW;
  const int sIh = pih0 + sPy;
  const int sIw = piw0 + sPx;
  const bool sInside =
      sActive && sIh >= 0 && sIh < inH && sIw >= 0 && sIw < inW;
  const int64_t sOff = static_cast<int64_t>(sIh) * inW + sIw;
  const int64_t sPlane = static_cast<int64_t>(inH) * inW;

  for (int32_t k0 = 0; k0 < Cin; k0 += kFBK) {
    // --- stage the raw input patch; the halo is shared from here on --------
    if (sActive) {
      const __half *xp = xImg + static_cast<int64_t>(k0) * sPlane + sOff;
#pragma unroll
      for (int pk = 0; pk < kFBK; ++pk, xp += sPlane)
        Xs[pk][sPy][sPx] = (sInside && k0 + pk < Cin)
                               ? static_cast<_Float16>(*xp)
                               : static_cast<_Float16>(0.0f);
    }

    // --- stage the transformed filter -------------------------------------
    //
    // One BK slice per thread, so this is a single 16-byte load and a single
    // 16-byte store. The whole-slice fast path needs the slice to be inside
    // the tensor; the tail re-reads elementwise rather than over-reading.
    {
      const int mm = uM;
      const int32_t row = m0 + mm;
      halfK slice;
      if (row < Cout && k0 + kFBK <= Cin) {
        slice = *reinterpret_cast<const halfK *>(
            u + (static_cast<int64_t>(uXi) * Cout + row) * Cin + k0);
      } else {
#pragma unroll
        for (int kk = 0; kk < kFBK; ++kk) {
          const int32_t col = k0 + kk;
          slice[kk] = (row < Cout && col < Cin)
                          ? static_cast<_Float16>(
                                u[(static_cast<int64_t>(uXi) * Cout + row) *
                                      Cin +
                                  col])
                          : static_cast<_Float16>(0.0f);
        }
      }
      *reinterpret_cast<halfK *>(&Us[uXi][mm][0]) = slice;
    }
    __syncthreads();

    // --- transform the patch into all 16 components, in LDS ---------------
    {
      // Two packed pairs per window row rather than four scalar reads: the
      // row is contiguous in the patch and the column origin is even, so the
      // pairs are always 4-byte aligned.
      float d[4][4];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const _Float16 *row =
            &Xs[vK][vTh * kTileOut + i][vTw * kTileOut];
        const half2v lo = *reinterpret_cast<const half2v *>(row);
        const half2v hi = *reinterpret_cast<const half2v *>(row + 2);
        d[i][0] = static_cast<float>(lo[0]);
        d[i][1] = static_cast<float>(lo[1]);
        d[i][2] = static_cast<float>(hi[0]);
        d[i][3] = static_cast<float>(hi[1]);
      }

      float r[4][4];
#pragma unroll
      for (int j = 0; j < 4; ++j)
        btdb4(d[0][j], d[1][j], d[2][j], d[3][j], r[0][j], r[1][j], r[2][j],
              r[3][j]);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        float o0, o1, o2, o3;
        btdb4(r[i][0], r[i][1], r[i][2], r[i][3], o0, o1, o2, o3);
        Vs[i * 4 + 0][vT][vK] = static_cast<_Float16>(o0);
        Vs[i * 4 + 1][vT][vK] = static_cast<_Float16>(o1);
        Vs[i * 4 + 2][vT][vK] = static_cast<_Float16>(o2);
        Vs[i * 4 + 3][vT][vK] = static_cast<_Float16>(o3);
      }
    }
    __syncthreads();

    // --- accumulate all 16 components -------------------------------------
    //
    // One 16-byte read per operand row covers the whole BK slice, then dot2
    // walks it in pairs straight into the fp32 accumulators.
#pragma unroll
    for (int xi = 0; xi < kXi; ++xi) {
      half2v av[kFTM][kFBK / 2], bv[kFTN][kFBK / 2];
#pragma unroll
      for (int i = 0; i < kFTM; ++i)
        *reinterpret_cast<halfK *>(&av[i][0]) =
            *reinterpret_cast<const halfK *>(&Us[xi][ty * kFTM + i][0]);
#pragma unroll
      for (int j = 0; j < kFTN; ++j)
        *reinterpret_cast<halfK *>(&bv[j][0]) =
            *reinterpret_cast<const halfK *>(&Vs[xi][tx * kFTN + j][0]);

#pragma unroll
      for (int p = 0; p < kFBK / 2; ++p)
#pragma unroll
        for (int i = 0; i < kFTM; ++i)
#pragma unroll
          for (int j = 0; j < kFTN; ++j)
            acc[xi][i][j] = wgDot2(av[i][p], bv[j][p], acc[xi][i][j]);
    }
    __syncthreads();
  }

  // --- output transform, bias folded in, straight from registers -----------
#pragma unroll
  for (int i = 0; i < kFTM; ++i) {
    const int32_t co = m0 + ty * kFTM + i;
    if (co >= Cout) continue;
    const float bv = bias ? __half2float(bias[co]) : 0.0f;
    __half *yc = y + (static_cast<int64_t>(n) * Cout + co) *
                         static_cast<int64_t>(outH) * outW;
#pragma unroll
    for (int j = 0; j < kFTN; ++j) {
      const int t = tx * kFTN + j;
      const int th = tRow0 + t / kFTPW;
      const int tw = tCol0 + t % kFTPW;
      if (th >= ntH || tw >= ntW) continue;

      float r[2][4];
#pragma unroll
      for (int c = 0; c < 4; ++c)
        atma4(acc[0 * 4 + c][i][j], acc[1 * 4 + c][i][j], acc[2 * 4 + c][i][j],
              acc[3 * 4 + c][i][j], r[0][c], r[1][c]);

#pragma unroll
      for (int a = 0; a < 2; ++a) {
        float o0, o1;
        atma4(r[a][0], r[a][1], r[a][2], r[a][3], o0, o1);
        const int oh = th * kTileOut + a;
        if (oh >= outH) continue;
        const int ow0 = tw * kTileOut;
        yc[static_cast<int64_t>(oh) * outW + ow0] = __float2half(o0 + bv);
        if (ow0 + 1 < outW)
          yc[static_cast<int64_t>(oh) * outW + ow0 + 1] = __float2half(o1 + bv);
      }
    }
  }
}

// Scratch is blocked over tiles against a fixed byte budget rather than sized
// to the shape, because the shapes this runs on differ by two orders of
// magnitude in tile count -- esrgan alone spans 250x250 and 1000x1000, whose
// full V would be 96 MB and 512 MB.
constexpr int64_t kScratchBudgetBytes = 32ll << 20;
constexpr int64_t kMinTilesPerBlock = 256;

int64_t tilesPerBlock(int64_t Cin, int64_t Cout, int64_t ntiles) {
  const int64_t perTile =
      kXi * (Cin + Cout) * static_cast<int64_t>(sizeof(__half));
  int64_t tb = kScratchBudgetBytes / (perTile > 0 ? perTile : 1);
  if (tb < kMinTilesPerBlock) tb = kMinTilesPerBlock;
  tb = (tb / 64) * 64;
  if (tb < 64) tb = 64;
  if (tb > ntiles) tb = ntiles;
  return tb;
}

} // namespace

extern "C" int hip_conv_winograd_eligible(int hip_dtype, int64_t spatial_rank,
                                          int64_t k0, int64_t k1, int64_t s0,
                                          int64_t s1, int64_t d0, int64_t d1,
                                          int64_t group, int64_t Cin,
                                          int64_t Cout) {
  // fp16 only: the transform constants are chosen to be exact in fp16 and the
  // accumulation is fp32 regardless, so there is nothing here for fp32 that
  // the direct path does not already do better.
  if (hip_dtype != HIP_DTYPE_FLOAT16) return 0;
  if (spatial_rank != 2) return 0;
  if (k0 != 3 || k1 != 3) return 0;
  if (s0 != 1 || s1 != 1) return 0;
  if (d0 != 1 || d1 != 1) return 0;
  // Grouped convolution would need the transforms and the GEMM to work inside
  // a group; every 3x3 stride-1 shape in the guard set is group 1, so this is
  // left out rather than written untested.
  if (group != 1) return 0;
  if (Cin < 1 || Cout < 1) return 0;
  return 1;
}

extern "C" int64_t hip_conv_winograd_filter_elems(int64_t Cout, int64_t Cin) {
  return kXi * Cout * Cin;
}

namespace {

int winogradUnitCount() {
  static const int units = [] {
    int dev = 0;
    hipDeviceProp_t prop{};
    if (hipGetDevice(&dev) != hipSuccess ||
        hipGetDeviceProperties(&prop, dev) != hipSuccess ||
        prop.multiProcessorCount <= 0)
      return 16;
    return prop.multiProcessorCount;
  }();
  return units;
}

} // namespace

extern "C" int hip_conv_winograd_profitable(int64_t N, int64_t Cin,
                                            int64_t Cout, int64_t out0,
                                            int64_t out1) {
  (void)Cin;
  const int64_t ntH = (out0 + kTileOut - 1) / kTileOut;
  const int64_t ntW = (out1 + kTileOut - 1) / kTileOut;
  if (ntH <= 0 || ntW <= 0 || N <= 0) return 0;

  const int64_t patches = ((ntH + kFTPH - 1) / kFTPH) *
                          ((ntW + kFTPW - 1) / kFTPW) * N;
  const int64_t blocks = patches * ((Cout + kFBM - 1) / kFBM);

  return blocks <= static_cast<int64_t>(kResidentBlocksPerUnit) *
                       winogradUnitCount()
             ? 1
             : 0;
}

extern "C" int64_t hip_conv_winograd_scratch_bytes(int64_t Cin, int64_t Cout,
                                                   int64_t out0, int64_t out1) {
  const int64_t ntH = (out0 + kTileOut - 1) / kTileOut;
  const int64_t ntW = (out1 + kTileOut - 1) / kTileOut;
  const int64_t ntiles = ntH * ntW;
  if (ntiles <= 0) return 0;
  const int64_t tb = tilesPerBlock(Cin, Cout, ntiles);
  const int64_t vBytes = kXi * Cin * tb * static_cast<int64_t>(sizeof(__half));
  const int64_t mBytes = kXi * Cout * tb * static_cast<int64_t>(sizeof(__half));
  // 256-byte gap so M's base stays over-aligned for vector stores.
  return ((vBytes + 255) & ~255ll) + mBytes;
}

extern "C" int hip_conv_winograd_filter(void *stream, const void *weights,
                                        void *u, int64_t Cout, int64_t Cin) {
  if (!weights || !u) return -1;
  const int64_t total = Cout * Cin;
  if (total <= 0) return 0;
  constexpr int kBlock = 256;
  const int64_t grid = (total + kBlock - 1) / kBlock;
  hipLaunchKernelGGL(winograd_filter_kernel,
                     dim3(static_cast<unsigned>(grid)), dim3(kBlock), 0,
                     static_cast<hipStream_t>(stream),
                     static_cast<const __half *>(weights),
                     static_cast<__half *>(u), Cout, Cin);
  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr,
            "[custom_kernels] hip_conv_winograd_filter: launch failed (%s)\n",
            hipGetErrorString(err));
    return static_cast<int>(err);
  }
  return 0;
}

extern "C" int hip_conv_winograd_fused(void *stream, const void *input,
                                       const void *u_transformed,
                                       const void *bias, void *output,
                                       int64_t N, int64_t Cin, int64_t Cout,
                                       int64_t inH, int64_t inW, int64_t outH,
                                       int64_t outW, int64_t padTop,
                                       int64_t padLeft) {
  if (!input || !u_transformed || !output) return -1;

  const int64_t ntH = (outH + kTileOut - 1) / kTileOut;
  const int64_t ntW = (outW + kTileOut - 1) / kTileOut;
  if (ntH <= 0 || ntW <= 0 || N <= 0) return 0;

  const int64_t patchesH = (ntH + kFTPH - 1) / kFTPH;
  const int64_t patchesW = (ntW + kFTPW - 1) / kFTPW;
  const int64_t patchesPerImage = patchesH * patchesW;
  const int64_t patches = patchesPerImage * N;
  const int64_t mBlocks = (Cout + kFBM - 1) / kFBM;

  // int32 indexing inside the kernel: the grid is bounded by the launch limit
  // and the tensor offsets are formed in int64, but the per-block scalars are
  // not, so refuse rather than wrap.
  if (patches > 0x7fffffffll || Cin > 0x7fffffffll || Cout > 0x7fffffffll ||
      inH > 0x7fffffffll || inW > 0x7fffffffll)
    return -1;

  const dim3 grid(static_cast<unsigned>(patches),
                  static_cast<unsigned>(mBlocks), 1);
  hipLaunchKernelGGL(winograd_fused_kernel, grid, dim3(kFThreads), 0,
                     static_cast<hipStream_t>(stream),
                     static_cast<const __half *>(input),
                     static_cast<const __half *>(u_transformed),
                     static_cast<const __half *>(bias),
                     static_cast<__half *>(output), static_cast<int32_t>(Cin),
                     static_cast<int32_t>(Cout), static_cast<int32_t>(inH),
                     static_cast<int32_t>(inW), static_cast<int32_t>(outH),
                     static_cast<int32_t>(outW), static_cast<int32_t>(padTop),
                     static_cast<int32_t>(padLeft), static_cast<int32_t>(ntH),
                     static_cast<int32_t>(ntW),
                     static_cast<int32_t>(patchesW),
                     static_cast<int32_t>(patchesPerImage));

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr,
            "[custom_kernels] hip_conv_winograd_fused: launch failed (%s)\n",
            hipGetErrorString(err));
    return static_cast<int>(err);
  }
  return 0;
}

extern "C" int hip_conv_winograd(void *stream, const void *input,
                                 const void *u_transformed, const void *bias,
                                 void *output, void *scratch, int64_t N,
                                 int64_t Cin, int64_t Cout, int64_t inH,
                                 int64_t inW, int64_t outH, int64_t outW,
                                 int64_t padTop, int64_t padLeft) {
  if (!input || !u_transformed || !output || !scratch) return -1;

  const int64_t ntH = (outH + kTileOut - 1) / kTileOut;
  const int64_t ntW = (outW + kTileOut - 1) / kTileOut;
  const int64_t ntPerImage = ntH * ntW;
  const int64_t ntiles = ntPerImage * N;
  if (ntiles <= 0) return 0;

  const int64_t tb = tilesPerBlock(Cin, Cout, ntPerImage);
  const int64_t vBytes = kXi * Cin * tb * static_cast<int64_t>(sizeof(__half));
  __half *V = static_cast<__half *>(scratch);
  __half *M = reinterpret_cast<__half *>(static_cast<char *>(scratch) +
                                         ((vBytes + 255) & ~255ll));

  hipStream_t s = static_cast<hipStream_t>(stream);
  const __half *x = static_cast<const __half *>(input);
  const __half *U = static_cast<const __half *>(u_transformed);
  const __half *b = static_cast<const __half *>(bias);
  __half *y = static_cast<__half *>(output);

  constexpr int BM = 64, BN = 64, TM = 4, TN = 4, BK = 16;
  constexpr int kGemmThreads = (BM / TM) * (BN / TN);
  constexpr int kBlock = 256;

  for (int64_t tile0 = 0; tile0 < ntiles; tile0 += tb) {
    const int64_t n = ntiles - tile0 < tb ? ntiles - tile0 : tb;

    const int64_t inThreads = Cin * n;
    hipLaunchKernelGGL(
        winograd_input_kernel,
        dim3(static_cast<unsigned>((inThreads + kBlock - 1) / kBlock)),
        dim3(kBlock), 0, s, x, V, Cin, inH, inW, padTop, padLeft, ntW,
        ntPerImage, tile0, n);

    const dim3 gemmGrid(static_cast<unsigned>((n + BN - 1) / BN),
                        static_cast<unsigned>((Cout + BM - 1) / BM), kXi);
    hipLaunchKernelGGL((winograd_gemm_kernel<BM, BN, TM, TN, BK>), gemmGrid,
                       dim3(kGemmThreads), 0, s, U, V, M,
                       static_cast<int32_t>(Cout), static_cast<int32_t>(Cin),
                       static_cast<int32_t>(n));

    const int64_t outThreads = Cout * n;
    hipLaunchKernelGGL(
        winograd_output_kernel,
        dim3(static_cast<unsigned>((outThreads + kBlock - 1) / kBlock)),
        dim3(kBlock), 0, s, M, b, y, Cout, outH, outW, ntW, ntPerImage, tile0,
        n);
  }

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr, "[custom_kernels] hip_conv_winograd: launch failed (%s)\n",
            hipGetErrorString(err));
    return static_cast<int>(err);
  }
  return 0;
}
