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
#include <cstdlib>

// ---------------------------------------------------------------------------
// ConvTranspose -- forward transposed convolution (2D spatial)
// ---------------------------------------------------------------------------
//
// ONNX defines ConvTranspose by scattering: each input element is multiplied by
// the whole filter and accumulated into a strided, dilated footprint of the
// output,
//
//   y[n][m][ih*sH + kh*dilH - padTop][iw*sW + kw*dilW - padLeft]
//       += x[n][c][ih][iw] * w[c][m_local][kh][kw]
//
// Written that way it needs atomics, because neighbouring input elements land
// on overlapping footprints whenever the stride is smaller than the kernel.
// Inverting it removes them: solving the same equation for the input position
// gives, for a fixed output cell, exactly the taps that can reach it --
//
//   ih = (oh + padTop - kh*dilH) / sH,  valid when that division is exact and
//                                       the quotient is inside the input
//
// so each thread owns one output element, reads only, and the result does not
// depend on scheduling order.
//
// Two things fall out of the ONNX definition rather than being handled:
//
//   * Trailing pads are absent from the ABI, exactly as in `hip_conv`. Padding
//     on a transposed convolution crops the output, and cropping at the far end
//     only removes output positions -- which the caller has already accounted
//     for in `out_h` / `out_w` -- so it changes nothing this kernel reads.
//   * `output_padding` likewise only extends `out_h` / `out_w`. The cells it
//     adds are ones no input reaches, and the divisibility and range tests
//     leave them at just the bias, which is what ONNX asks for.
//
// The weight layout is ONNX ConvTranspose's `[C, M/group, kH, kW]` -- input
// channels first, unlike forward Conv's `[M, C/group, ...]` -- and is indexed
// directly, so nothing has to be transposed first.
//
// Accumulation is in float for every storage dtype, matching `hip_conv`.
//
// Two kernels implement that inversion.
//
// `conv_transpose_direct_kernel` is one thread per output element. It shipped
// on the premise that no model the EP supports contains a ConvTranspose, so
// the only shape that mattered was the one where correctness is obvious. That
// premise was wrong: sam2.1/decoder has two. The formulation has no reuse at
// all -- every thread walks the whole channel reduction itself, so the
// CinPerGroup input column and CinPerGroup filter column behind each output
// element are read once per output element rather than once per block. On
// sam2.1's 256->64 2x2 stride-2 layer that is a gigabyte of traffic to serve
// two megabytes of unique data. It is kept for the shapes the tiled kernel
// declines (see `useTiled`), where the missing reuse costs nothing because
// there is none to be had.
//
// `conv_transpose_tiled_kernel` is the one that runs. Fixing the reuse means
// recognising what the inverted form actually is: with the contraction taken
// over (input channel, filter tap) jointly and the output position as the
// other free index, it is a GEMM,
//
//   C[m][p] = sum over (c,kh,kw) of  A[m][(c,kh,kw)] * B[(c,kh,kw)][p]
//
// where A is the filter read in its native ConvTranspose layout and B is the
// input gathered through the divisibility test above -- the transpose of the
// im2col matrix that forward convolution builds. So it is the same kernel
// shape as `hip_conv`: register-tiled, both operands staged through LDS, the
// gather folded into the B stage. A block loads each input and filter element
// once for BM x BN outputs instead of once per output.
//
// B is materialised with an explicit zero where the divisibility test fails,
// which is what lets the taps that cannot reach a given output cell take part
// in the reduction harmlessly. That also means `kernel == stride` needs no
// special case: exactly one tap per output cell passes the test, the other
// kH*kW-1 stage as zero, and the wasted lanes cost LDS and arithmetic rather
// than memory traffic -- which is the resource that was short. Keeping one
// path rather than two is worth more here than the lanes it gives up.
//
// Measured on sam2.1/decoder's two layers, per inference:
//
//   direct                    2.80 ms
//   tiled, flat contraction   1.30 ms
//   tiled, tap-major          0.80 ms
//
// The gap between the last two is all integer division; see the note on the
// loop nesting below.

namespace {

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

struct ConvTransposeParams {
  int64_t N, Cin, Cout;
  // The direct kernel recovers the group from the output channel it owns; the
  // tiled kernel is launched with one block per (batch, group) and so needs it
  // as a grid dimension.
  int64_t group;
  int64_t CinPerGroup, MPerGroup;
  int64_t inH, inW;
  int64_t outH, outW;
  int64_t kH, kW;
  int64_t sH, sW;
  int64_t padTop, padLeft;
  int64_t dilH, dilW;
};

template <typename T>
__global__ void conv_transpose_direct_kernel(const T *__restrict__ x,
                                             const T *__restrict__ w,
                                             const T *__restrict__ bias,
                                             T *__restrict__ y,
                                             ConvTransposeParams pr,
                                             int64_t total) {
  int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid >= total)
    return;

  // tid indexes Y as [N][Cout][outH][outW], row-major.
  const int64_t ow = tid % pr.outW;
  int64_t rest = tid / pr.outW;
  const int64_t oh = rest % pr.outH;
  rest /= pr.outH;
  const int64_t m = rest % pr.Cout;
  const int64_t n = rest / pr.Cout;

  const int64_t g = m / pr.MPerGroup;
  const int64_t mLocal = m - g * pr.MPerGroup;

  const int64_t inPlane = pr.inH * pr.inW;
  const int64_t wChanStride = pr.MPerGroup * pr.kH * pr.kW;
  // Both bases are already advanced to this group's first input channel, so the
  // reduction below indexes channels relative to the group.
  const T *xBase = x + (n * pr.Cin + g * pr.CinPerGroup) * inPlane;
  const T *wBase =
      w + (g * pr.CinPerGroup) * wChanStride + mLocal * pr.kH * pr.kW;

  float acc = bias ? to_float<T>(bias[m]) : 0.0f;

  for (int64_t kh = 0; kh < pr.kH; ++kh) {
    // Exact division is the whole test: a tap only reaches this row if the
    // stride steps onto it. Truncation toward zero makes a negative numerator
    // land on a negative quotient, which the range check then drops.
    const int64_t th = oh + pr.padTop - kh * pr.dilH;
    if (th % pr.sH != 0)
      continue;
    const int64_t ih = th / pr.sH;
    if (ih < 0 || ih >= pr.inH)
      continue;

    for (int64_t kw = 0; kw < pr.kW; ++kw) {
      const int64_t tw = ow + pr.padLeft - kw * pr.dilW;
      if (tw % pr.sW != 0)
        continue;
      const int64_t iw = tw / pr.sW;
      if (iw < 0 || iw >= pr.inW)
        continue;

      const T *xp = xBase + ih * pr.inW + iw;
      const T *wp = wBase + kh * pr.kW + kw;
      for (int64_t c = 0; c < pr.CinPerGroup; ++c)
        acc += to_float<T>(xp[c * inPlane]) * to_float<T>(wp[c * wChanStride]);
    }
  }

  y[tid] = from_float<T>(acc);
}

// ---------------------------------------------------------------------------
// Tiled path
// ---------------------------------------------------------------------------
//
// One block owns a BM x BN tile of (output channel) x (output position) inside
// one (batch, group), and walks the joint (channel, tap) contraction BK at a
// time. Each thread keeps a TM x TN register tile, so the LDS traffic per unit
// of arithmetic falls with the register tile rather than with the block.
//
// The spatial index is flattened to a single `p` over outH*outW and split back
// out into (oh, ow) once per thread. Flattening rather than tiling h and w
// separately keeps the store contiguous -- consecutive threads write
// consecutive output columns.
//
// The contraction is nested tap-major, one loop over the kH*kW taps and an
// inner one over channels, rather than run as a single flat index. That is not
// cosmetic. Written flat, recovering (channel, tap) from the index and then
// (kh, kw) from the tap costs several divisions by runtime values *per staged
// element*, and the divisibility test that decides whether a tap reaches an
// output cell costs two more. Integer division is a multi-instruction sequence
// here, and at 1024 staged elements per stage that arithmetic, not the memory
// it was hiding, sets the kernel's speed. Nested, the tap is a loop variable:
// (kh, kw) and the whole reachability test -- which depends only on the tap
// and the thread's own output column -- are computed once per tap and reused
// across every channel. The inner loops then contain no division at all.
//
// All the spatial arithmetic is 32-bit; `useTiled` refuses shapes whose
// extents do not fit.
template <typename T, int BM, int BN, int TM, int TN, int BK>
__global__ void conv_transpose_tiled_kernel(const T *__restrict__ x,
                                            const T *__restrict__ w,
                                            const T *__restrict__ bias,
                                            T *__restrict__ y,
                                            ConvTransposeParams pr,
                                            int32_t Nsp) {
  static_assert(BM % TM == 0 && BN % TN == 0, "tile must divide the block");
  constexpr int kThreads = (BM / TM) * (BN / TN);
  static_assert(BK * BM % kThreads == 0 && BK * BN % kThreads == 0,
                "staging must divide evenly across the block");
  // Each thread stages one fixed column of B and one fixed row of A, which is
  // what lets the reachability test be hoisted out of the channel loop.
  static_assert(kThreads % BN == 0 && kThreads % BM == 0,
                "staging must not move a thread between columns");

  __shared__ float As[BK][BM];
  __shared__ float Bs[BK][BN];

  const int tid = static_cast<int>(threadIdx.x);
  const int tx = tid % (BN / TN);
  const int ty = tid / (BN / TN);

  const int32_t p0 = static_cast<int32_t>(blockIdx.x) * BN;
  const int32_t m0 = static_cast<int32_t>(blockIdx.y) * BM;
  const int64_t nz = blockIdx.z;
  const int64_t n = nz / pr.group;
  const int64_t g = nz - n * pr.group;

  const int32_t MPerGroup = static_cast<int32_t>(pr.MPerGroup);
  const int32_t CinPerGroup = static_cast<int32_t>(pr.CinPerGroup);
  const int32_t kH = static_cast<int32_t>(pr.kH);
  const int32_t kW = static_cast<int32_t>(pr.kW);
  const int32_t kHkW = kH * kW;
  const int32_t outW = static_cast<int32_t>(pr.outW);
  const int32_t inH = static_cast<int32_t>(pr.inH);
  const int32_t inW = static_cast<int32_t>(pr.inW);
  const int32_t sH = static_cast<int32_t>(pr.sH);
  const int32_t sW = static_cast<int32_t>(pr.sW);
  const int32_t dilH = static_cast<int32_t>(pr.dilH);
  const int32_t dilW = static_cast<int32_t>(pr.dilW);
  const int32_t padTop = static_cast<int32_t>(pr.padTop);
  const int32_t padLeft = static_cast<int32_t>(pr.padLeft);
  const int32_t inPlane = inH * inW;

  // Both bases are advanced to this group's first channel, so the contraction
  // below indexes channels relative to the group -- the same convention the
  // direct kernel uses.
  const T *xBase = x + (n * pr.Cin + g * pr.CinPerGroup) * inPlane;
  const T *wBase = w + (g * pr.CinPerGroup) * (pr.MPerGroup * kHkW);

  // The staging positions this thread owns, both fixed for the whole kernel.
  const int stageCol = tid % BN;   // its column of B
  const int stageRow = tid % BM;   // its row of A
  const int stageKB = tid / BN;    // first contraction offset it stages into B
  const int stageKA = tid / BM;    // ... and into A
  constexpr int kStepB = kThreads / BN;
  constexpr int kStepA = kThreads / BM;

  // The output cell this thread's B column belongs to. One division, once.
  const int32_t pCol = p0 + stageCol;
  const int32_t ohCol = pCol < Nsp ? pCol / outW : 0;
  const int32_t owCol = pCol - ohCol * outW;

  float acc[TM][TN];
  for (int i = 0; i < TM; ++i)
    for (int j = 0; j < TN; ++j)
      acc[i][j] = 0.0f;

  for (int32_t kh = 0; kh < kH; ++kh) {
    // Reachability for this tap row, for this thread's column only.
    const int32_t th = ohCol + padTop - kh * dilH;
    const int32_t ihq = th / sH;
    const bool okH = (th % sH == 0) && ihq >= 0 && ihq < inH && pCol < Nsp;

    for (int32_t kw = 0; kw < kW; ++kw) {
      const int32_t tw = owCol + padLeft - kw * dilW;
      const int32_t iwq = tw / sW;
      const bool ok = okH && (tw % sW == 0) && iwq >= 0 && iwq < inW;
      // Where this thread's column reads from, if it reads at all. Constant
      // across the entire channel loop below, which is the point of nesting.
      const int64_t xCol =
          static_cast<int64_t>(ihq) * inW + iwq;
      const int32_t t = kh * kW + kw;

      for (int32_t c0 = 0; c0 < CinPerGroup; c0 += BK) {
        // A stage: the filter in its native [C][M/group][kH][kW] layout.
        for (int i = 0; i < BK * BM / kThreads; ++i) {
          const int kk = stageKA + i * kStepA;
          const int32_t c = c0 + kk;
          const int32_t m = m0 + stageRow;
          float v = 0.0f;
          if (c < CinPerGroup && m < MPerGroup)
            v = to_float<T>(wBase[static_cast<int64_t>(c) * MPerGroup * kHkW +
                                  static_cast<int64_t>(m) * kHkW + t]);
          As[kk][stageRow] = v;
        }

        // B stage: the gather. A tap reaches this output cell only if the
        // stride steps exactly onto it; the zero written otherwise is what
        // lets the reduction run over every tap without a special case.
        for (int i = 0; i < BK * BN / kThreads; ++i) {
          const int kk = stageKB + i * kStepB;
          const int32_t c = c0 + kk;
          float v = 0.0f;
          if (ok && c < CinPerGroup)
            v = to_float<T>(
                xBase[static_cast<int64_t>(c) * inPlane + xCol]);
          Bs[kk][stageCol] = v;
        }

        __syncthreads();

        for (int kk = 0; kk < BK; ++kk) {
          float a[TM], bb[TN];
          for (int i = 0; i < TM; ++i)
            a[i] = As[kk][ty * TM + i];
          for (int j = 0; j < TN; ++j)
            bb[j] = Bs[kk][tx * TN + j];
          for (int i = 0; i < TM; ++i)
            for (int j = 0; j < TN; ++j)
              acc[i][j] += a[i] * bb[j];
        }

        __syncthreads();
      }
    }
  }

  const int64_t outPlane = pr.outH * pr.outW;
  for (int i = 0; i < TM; ++i) {
    const int32_t m = m0 + ty * TM + i;
    if (m >= MPerGroup)
      break;
    const int64_t mGlobal = g * pr.MPerGroup + m;
    const float b0 = bias ? to_float<T>(bias[mGlobal]) : 0.0f;
    T *row = y + (n * pr.Cout + mGlobal) * outPlane;
    for (int j = 0; j < TN; ++j) {
      const int32_t p = p0 + tx * TN + j;
      if (p < Nsp)
        row[p] = from_float<T>(acc[i][j] + b0);
    }
  }
}

// The tiled kernel needs enough of both tile dimensions to be worth its
// staging, and needs the spatial arithmetic to fit in 32 bits.
//
// The output-channel bound is the real one. A block covers BM output channels,
// so a group with fewer than a handful wastes most of every A stage and most
// of every register tile -- and a ConvTranspose that narrow (sam2.1's decoder
// ends at 1 output channel per group) has no reuse for a tile to find anyway,
// which is exactly the case the direct kernel is good at.
inline bool useTiled(const ConvTransposeParams &pr, int64_t Nsp, int64_t K) {
  const int64_t kInt32Max = 2147483647;
  if (Nsp > kInt32Max || K > kInt32Max)
    return false;
  if (pr.inH * pr.inW > kInt32Max || pr.outH * pr.outW > kInt32Max)
    return false;
  if (pr.MPerGroup * pr.kH * pr.kW > kInt32Max)
    return false;
  // Below this there is not enough of the reduction to amortise two LDS
  // stages, and not enough output channels to fill a row block.
  return pr.MPerGroup >= 8 && K >= 8 && Nsp >= 64;
}

// HIPDNN_EP_CONVT_DIRECT=1 forces the untiled kernel. The two paths compute
// the same sum in a different order, so the fallback doubles as the reference
// the tiled kernel can be checked against -- and as the only way to measure
// what the tiling is worth on a given shape, which is how the figures at the
// top of this file were established rather than assumed.
inline bool forceDirect() {
  static const bool v = [] {
    const char *s = std::getenv("HIPDNN_EP_CONVT_DIRECT");
    return s && *s && *s != '0';
  }();
  return v;
}

template <typename T>
int launch(hipStream_t stream, const void *input, const void *weights,
           const void *bias, void *output, const ConvTransposeParams &pr,
           int64_t total) {
  const int64_t Nsp = pr.outH * pr.outW;
  const int64_t K = pr.CinPerGroup * pr.kH * pr.kW;

  if (!forceDirect() && useTiled(pr, Nsp, K)) {
    constexpr int BM = 64, BN = 64, TM = 4, TN = 4, BK = 16;
    constexpr int kThreads = (BM / TM) * (BN / TN);
    const dim3 grid(static_cast<unsigned>((Nsp + BN - 1) / BN),
                    static_cast<unsigned>((pr.MPerGroup + BM - 1) / BM),
                    static_cast<unsigned>(pr.N * pr.group));
    hipLaunchKernelGGL(
        (conv_transpose_tiled_kernel<T, BM, BN, TM, TN, BK>), grid,
        dim3(kThreads), 0, stream, static_cast<const T *>(input),
        static_cast<const T *>(weights), static_cast<const T *>(bias),
        static_cast<T *>(output), pr, static_cast<int32_t>(Nsp));
  } else {
    constexpr int kBlock = 256;
    const int64_t grid = (total + kBlock - 1) / kBlock;
    hipLaunchKernelGGL(conv_transpose_direct_kernel<T>, dim3(grid),
                       dim3(kBlock), 0, stream, static_cast<const T *>(input),
                       static_cast<const T *>(weights),
                       static_cast<const T *>(bias), static_cast<T *>(output),
                       pr, total);
  }

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr, "[custom_kernels] hip_conv_transpose: launch failed (%s)\n",
            hipGetErrorString(err));
    return static_cast<int>(err);
  }
  return 0;
}

} // namespace

extern "C" int hip_conv_transpose(void *stream, const void *input,
                                  const void *weights, const void *bias,
                                  void *output, int hip_dtype, int64_t N,
                                  int64_t Cin, int64_t Cout, int64_t in_h,
                                  int64_t in_w, int64_t out_h, int64_t out_w,
                                  int64_t k_h, int64_t k_w, int64_t s_h,
                                  int64_t s_w, int64_t pad_top,
                                  int64_t pad_left, int64_t dil_h,
                                  int64_t dil_w, int64_t group) {
  if (!input || !weights || !output) {
    fprintf(stderr,
            "[custom_kernels] hip_conv_transpose: null input/weights/output\n");
    return -1;
  }
  if (group < 1 || Cin % group != 0 || Cout % group != 0) {
    fprintf(stderr,
            "[custom_kernels] hip_conv_transpose: group=%lld does not divide "
            "Cin=%lld / Cout=%lld\n",
            (long long)group, (long long)Cin, (long long)Cout);
    return -1;
  }
  if (s_h < 1 || s_w < 1 || dil_h < 1 || dil_w < 1) {
    fprintf(stderr,
            "[custom_kernels] hip_conv_transpose: stride=[%lld,%lld] "
            "dilation=[%lld,%lld] must all be at least 1\n",
            (long long)s_h, (long long)s_w, (long long)dil_h,
            (long long)dil_w);
    return -1;
  }

  const int64_t total = N * Cout * out_h * out_w;
  if (total <= 0)
    return 0;

  ConvTransposeParams pr{};
  pr.N = N;
  pr.Cin = Cin;
  pr.Cout = Cout;
  pr.group = group;
  pr.CinPerGroup = Cin / group;
  pr.MPerGroup = Cout / group;
  pr.inH = in_h;
  pr.inW = in_w;
  pr.outH = out_h;
  pr.outW = out_w;
  pr.kH = k_h;
  pr.kW = k_w;
  pr.sH = s_h;
  pr.sW = s_w;
  pr.padTop = pad_top;
  pr.padLeft = pad_left;
  pr.dilH = dil_h;
  pr.dilW = dil_w;

  CUSTOM_KERNELS_DEBUG_LOG(
      "[custom_kernels] hip_conv_transpose: dtype=%d N=%lld Cin=%lld "
      "Cout=%lld group=%lld in=[%lld,%lld] out=[%lld,%lld] k=[%lld,%lld] "
      "s=[%lld,%lld] pbegin=[%lld,%lld] dil=[%lld,%lld] bias=%s\n",
      hip_dtype, (long long)N, (long long)Cin, (long long)Cout,
      (long long)group, (long long)in_h, (long long)in_w, (long long)out_h,
      (long long)out_w, (long long)k_h, (long long)k_w, (long long)s_h,
      (long long)s_w, (long long)pad_top, (long long)pad_left,
      (long long)dil_h, (long long)dil_w, bias ? "yes" : "null");

  hipStream_t hip_stream = static_cast<hipStream_t>(stream);
  switch (hip_dtype) {
  case HIP_DTYPE_FLOAT32:
    return launch<float>(hip_stream, input, weights, bias, output, pr, total);
  case HIP_DTYPE_FLOAT16:
    return launch<__half>(hip_stream, input, weights, bias, output, pr, total);
  case HIP_DTYPE_BFLOAT16:
    return launch<hip_bfloat16>(hip_stream, input, weights, bias, output, pr,
                                total);
  default:
    fprintf(stderr,
            "[custom_kernels] hip_conv_transpose: unsupported dtype=%d\n",
            hip_dtype);
    return -1;
  }
}
