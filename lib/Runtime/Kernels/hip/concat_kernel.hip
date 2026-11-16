/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// Concat of N contiguous inputs in one launch (`hip.concat`).
//
// Every input and the output are viewed as 2-D [outer, width]: `outer` is the
// product of the dims before the concat axis and an input's width is its
// axis extent times the product of the dims after it. The output row is the
// inputs' rows laid side by side. Copying input by input costs one strided
// copy per input; for thin rows (an interleave concat has width 1) each of
// those is a launch over the whole output height. Here each thread owns output
// elements, finds the input that covers its column and reads from it, so the
// writes stay coalesced and a concat is one launch.
//
// Inputs beyond CC_MAX_INPUTS are handled in further launches, each writing
// its own column window of the output.

#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>

#include "hip_custom_kernels.h"

namespace {
constexpr int CC_MAX_INPUTS = 16;
constexpr int CC_BLOCK = 256;
constexpr int CC_ELEMS_PER_THREAD = 4;

template <typename IndexT> struct ConcatParams {
  const void *srcs[CC_MAX_INPUTS];
  // Column where input i starts within this launch's window; begins[n] is the
  // window width.
  IndexT begins[CC_MAX_INPUTS + 1];
  int numInputs;
  IndexT outer;
  IndexT dstWidth;  // full output row width
  IndexT colOffset; // first output column of this launch's window
};

template <typename T, typename IndexT>
__global__ void __launch_bounds__(CC_BLOCK)
    concat_kernel(T *__restrict__ dst, ConcatParams<IndexT> p) {
  const IndexT window = p.begins[p.numInputs];
  const IndexT total = p.outer * window;
  const IndexT tile = (IndexT)CC_BLOCK * CC_ELEMS_PER_THREAD;
  const IndexT gridTile = (IndexT)gridDim.x * tile;
  for (IndexT base = (IndexT)blockIdx.x * tile; base < total;
       base += gridTile) {
#pragma unroll
    for (int k = 0; k < CC_ELEMS_PER_THREAD; ++k) {
      const IndexT t = base + (IndexT)k * CC_BLOCK + (IndexT)threadIdx.x;
      if (t >= total)
        break;
      const IndexT row = t / window;
      const IndexT col = t - row * window;
      int i = 0;
      while (i + 1 < p.numInputs && col >= p.begins[i + 1])
        ++i;
      const IndexT width = p.begins[i + 1] - p.begins[i];
      const T *src = static_cast<const T *>(p.srcs[i]);
      dst[row * p.dstWidth + p.colOffset + col] =
          src[row * width + (col - p.begins[i])];
    }
  }
}

template <typename IndexT>
int launchConcatWindow(hipStream_t s, void *dst, int64_t vecBytes,
                       int numInputs, const void *const *srcs,
                       const int64_t *widths, int64_t outer, int64_t dstWidth,
                       int64_t colOffset) {
  ConcatParams<IndexT> p{};
  p.numInputs = numInputs;
  p.outer = (IndexT)outer;
  p.dstWidth = (IndexT)dstWidth;
  p.colOffset = (IndexT)colOffset;
  int64_t begin = 0;
  for (int i = 0; i < numInputs; ++i) {
    p.srcs[i] = srcs[i];
    p.begins[i] = (IndexT)begin;
    begin += widths[i];
  }
  p.begins[numInputs] = (IndexT)begin;

  const int64_t total = outer * begin;
  const int64_t tile = int64_t(CC_BLOCK) * CC_ELEMS_PER_THREAD;
  int64_t grid64 = (total + tile - 1) / tile;
  if (grid64 > (int64_t(1) << 20))
    grid64 = int64_t(1) << 20; // grid-stride loop covers the remainder
  dim3 grid((unsigned)grid64), blk(CC_BLOCK);
  switch (vecBytes) {
  case 16:
    hipLaunchKernelGGL((concat_kernel<uint4, IndexT>), grid, blk, 0, s,
                       (uint4 *)dst, p);
    break;
  case 8:
    hipLaunchKernelGGL((concat_kernel<uint64_t, IndexT>), grid, blk, 0, s,
                       (uint64_t *)dst, p);
    break;
  case 4:
    hipLaunchKernelGGL((concat_kernel<uint32_t, IndexT>), grid, blk, 0, s,
                       (uint32_t *)dst, p);
    break;
  case 2:
    hipLaunchKernelGGL((concat_kernel<uint16_t, IndexT>), grid, blk, 0, s,
                       (uint16_t *)dst, p);
    break;
  case 1:
    hipLaunchKernelGGL((concat_kernel<uint8_t, IndexT>), grid, blk, 0, s,
                       (uint8_t *)dst, p);
    break;
  default:
    return -2;
  }
  return 0;
}
} // namespace

extern "C" int hip_concat(void *stream, void *dst, int num_inputs,
                          const void *const *srcs, const int64_t *src_widths,
                          int64_t outer, int64_t elem_size) {
  if (elem_size != 1 && elem_size != 2 && elem_size != 4 && elem_size != 8)
    return -2;
  if (num_inputs <= 0 || outer < 0)
    return -1;

  // Zero-width inputs contribute nothing and may carry a null pointer.
  const void *liveSrcs[CC_MAX_INPUTS];
  int64_t liveWidths[CC_MAX_INPUTS];
  int64_t dstWidth = 0;
  for (int i = 0; i < num_inputs; ++i) {
    if (src_widths[i] < 0)
      return -1;
    dstWidth += src_widths[i];
  }
  if (outer == 0 || dstWidth == 0)
    return 0;

  // Widest element every row width and pointer is aligned to.
  int64_t vecBytes = elem_size;
  for (int64_t vec : {int64_t(16), int64_t(8), int64_t(4), int64_t(2)}) {
    if (vec <= elem_size)
      break;
    bool ok = reinterpret_cast<uintptr_t>(dst) % vec == 0;
    for (int i = 0; ok && i < num_inputs; ++i)
      ok = (src_widths[i] * elem_size) % vec == 0 &&
           (src_widths[i] == 0 ||
            reinterpret_cast<uintptr_t>(srcs[i]) % vec == 0);
    if (ok) {
      vecBytes = vec;
      break;
    }
  }
  const int64_t ratio = vecBytes / elem_size;
  const int64_t vecDstWidth = dstWidth / ratio;
  const int64_t tile = int64_t(CC_BLOCK) * CC_ELEMS_PER_THREAD;
  const bool narrow = outer * vecDstWidth + tile <= INT32_MAX;

  hipStream_t s = static_cast<hipStream_t>(stream);
  // Clear a stale error from an unrelated earlier launch so the check below
  // reports only these launches.
  (void)hipGetLastError();
  int64_t colOffset = 0;
  int i = 0;
  while (i < num_inputs) {
    int n = 0;
    int64_t windowWidth = 0;
    for (; i < num_inputs && n < CC_MAX_INPUTS; ++i) {
      if (src_widths[i] == 0)
        continue;
      liveSrcs[n] = srcs[i];
      liveWidths[n] = src_widths[i] / ratio;
      windowWidth += liveWidths[n];
      ++n;
    }
    if (n == 0)
      break;
    int rc = narrow ? launchConcatWindow<int32_t>(s, dst, vecBytes, n, liveSrcs,
                                                  liveWidths, outer,
                                                  vecDstWidth, colOffset)
                    : launchConcatWindow<int64_t>(s, dst, vecBytes, n, liveSrcs,
                                                  liveWidths, outer,
                                                  vecDstWidth, colOffset);
    if (rc != 0)
      return rc;
    colOffset += windowWidth;
  }

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr, "[custom_kernels] hip_concat launch failed: %s\n",
            hipGetErrorString(err));
  }
  return (int)err;
}
