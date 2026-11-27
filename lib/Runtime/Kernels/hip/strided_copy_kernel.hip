/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// Parallel strided device-to-device copy for MLIR `memref.copy` (memrefCopy),
// thin-row pitched copies, and `hip.reverse`.
//
// The runtime's memrefCopy previously walked the outer index space on the host
// and issued one hipMemcpyAsync PER contiguous row. For the Qwen3.5 vision
// encoder that is ~2 million 72-byte launches per prefill (Reshape/Concat/Tile
// chains over ~19k rows x ~108 copies) and dominated GPU prefill time. This
// kernel does the whole strided copy in ONE launch, with the innermost dim
// mapped to consecutive threads (coalesced on the destination).
//
// The host side normalises the geometry before launching:
//   * size-1 dims are dropped and dims that are contiguous on both sides are
//     merged, so the per-element index decomposition walks as few dims as
//     possible (a [100,1,192,49] copy becomes [19200,49]);
//   * the index math runs in 32 bits when every linear index and offset fits;
//   * a contiguous inner dim whose byte extent, strides and base pointers are
//     suitably aligned is copied as 16/8-byte vector elements;
//   * each thread copies several elements.
// Source strides may be negative (reversed axes); `src` then points at the
// element with every index 0.

#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "hip_custom_kernels.h"

namespace {
// One more than the memref rank: hip_strided_copy appends its contiguous row
// as an explicit innermost dim.
constexpr int SC_MAX_RANK = HIPDNN_MAX_MEMREF_RANK + 1;
constexpr int SC_BLOCK = 256;
constexpr int SC_ELEMS_PER_THREAD = 4;

template <typename IndexT> struct StridedCopyParams {
  // Dims [0, rank-1) are decomposed per element; dim rank-1 is the inner dim
  // mapped to consecutive threads.
  IndexT sizes[SC_MAX_RANK];
  IndexT srcStrides[SC_MAX_RANK];
  IndexT dstStrides[SC_MAX_RANK];
  int rank;
  IndexT total;
};

template <typename T, typename IndexT>
__global__ void __launch_bounds__(SC_BLOCK)
    strided_copy_kernel(T *__restrict__ dst, const T *__restrict__ src,
                        StridedCopyParams<IndexT> p) {
  const int inner = p.rank - 1;
  const IndexT innerSize = p.sizes[inner];
  const IndexT innerSrc = p.srcStrides[inner];
  const IndexT innerDst = p.dstStrides[inner];
  const IndexT tile = (IndexT)SC_BLOCK * SC_ELEMS_PER_THREAD;
  const IndexT gridTile = (IndexT)gridDim.x * tile;
  for (IndexT base = (IndexT)blockIdx.x * tile; base < p.total;
       base += gridTile) {
#pragma unroll
    for (int k = 0; k < SC_ELEMS_PER_THREAD; ++k) {
      const IndexT t = base + (IndexT)k * SC_BLOCK + (IndexT)threadIdx.x;
      if (t >= p.total)
        break;
      IndexT r = t / innerSize;
      const IndexT col = t - r * innerSize;
      IndexT srcOff = col * innerSrc;
      IndexT dstOff = col * innerDst;
      for (int i = inner - 1; i >= 0; --i) {
        const IndexT size = p.sizes[i];
        const IndexT q = r / size;
        const IndexT idx = r - q * size;
        r = q;
        srcOff += idx * p.srcStrides[i];
        dstOff += idx * p.dstStrides[i];
      }
      dst[dstOff] = src[srcOff];
    }
  }
}

struct NormalisedCopy {
  int rank = 0;
  int64_t sizes[SC_MAX_RANK];
  int64_t srcStrides[SC_MAX_RANK];
  int64_t dstStrides[SC_MAX_RANK];
  int64_t total = 1;
};

// Drops size-1 dims and merges adjacent dims that are contiguous on both
// sides. Returns false for an empty copy.
bool normaliseCopy(int rank, const int64_t *sizes, const int64_t *srcStrides,
                   const int64_t *dstStrides, NormalisedCopy &out) {
  out.rank = 0;
  out.total = 1;
  for (int i = 0; i < rank; ++i) {
    if (sizes[i] == 0)
      return false;
    if (sizes[i] == 1)
      continue;
    if (out.rank > 0) {
      const int last = out.rank - 1;
      if (out.srcStrides[last] == srcStrides[i] * sizes[i] &&
          out.dstStrides[last] == dstStrides[i] * sizes[i]) {
        out.sizes[last] *= sizes[i];
        out.srcStrides[last] = srcStrides[i];
        out.dstStrides[last] = dstStrides[i];
        out.total *= sizes[i];
        continue;
      }
    }
    out.sizes[out.rank] = sizes[i];
    out.srcStrides[out.rank] = srcStrides[i];
    out.dstStrides[out.rank] = dstStrides[i];
    ++out.rank;
    out.total *= sizes[i];
  }
  if (out.rank == 0) {
    out.sizes[0] = 1;
    out.srcStrides[0] = 1;
    out.dstStrides[0] = 1;
    out.rank = 1;
  }
  return true;
}

// Widest element (bytes) the copy can be reinterpreted in: the inner dim must
// be contiguous on both sides, and its byte extent, every outer stride and both
// base pointers must be multiples of the vector width.
int64_t pickVectorBytes(const NormalisedCopy &c, int64_t elemSize,
                        const void *dst, const void *src) {
  const int inner = c.rank - 1;
  if (c.srcStrides[inner] != 1 || c.dstStrides[inner] != 1)
    return elemSize;
  for (int64_t vec : {int64_t(16), int64_t(8), int64_t(4)}) {
    if (vec <= elemSize)
      break;
    bool ok = (c.sizes[inner] * elemSize) % vec == 0 &&
              reinterpret_cast<uintptr_t>(dst) % vec == 0 &&
              reinterpret_cast<uintptr_t>(src) % vec == 0;
    for (int i = 0; ok && i < inner; ++i)
      ok = (c.srcStrides[i] * elemSize) % vec == 0 &&
           (c.dstStrides[i] * elemSize) % vec == 0;
    if (ok)
      return vec;
  }
  return elemSize;
}

template <typename IndexT>
StridedCopyParams<IndexT> makeParams(const NormalisedCopy &c) {
  StridedCopyParams<IndexT> p{};
  p.rank = c.rank;
  p.total = (IndexT)c.total;
  for (int i = 0; i < c.rank; ++i) {
    p.sizes[i] = (IndexT)c.sizes[i];
    p.srcStrides[i] = (IndexT)c.srcStrides[i];
    p.dstStrides[i] = (IndexT)c.dstStrides[i];
  }
  return p;
}

constexpr int64_t SC_TILE = int64_t(SC_BLOCK) * SC_ELEMS_PER_THREAD;

int64_t gridBlocks(int64_t total) {
  int64_t grid = (total + SC_TILE - 1) / SC_TILE;
  return grid > (int64_t(1) << 20) ? int64_t(1) << 20 : grid;
}

// A block leaves the grid-stride loop only after stepping its base one grid
// past the last tile, so 32-bit indexing needs that step to fit as well.
bool fitsNarrowLoop(int64_t total) {
  const int64_t lastBase = (total - 1) / SC_TILE * SC_TILE;
  return lastBase + gridBlocks(total) * SC_TILE <= INT32_MAX;
}

template <typename IndexT>
int launchStridedCopy(hipStream_t s, void *dst, const void *src,
                      int64_t vecBytes, const NormalisedCopy &c) {
  StridedCopyParams<IndexT> p = makeParams<IndexT>(c);
  dim3 grid((unsigned)gridBlocks(c.total)), blk(SC_BLOCK);
  switch (vecBytes) {
  case 16:
    hipLaunchKernelGGL((strided_copy_kernel<uint4, IndexT>), grid, blk, 0, s,
                       (uint4 *)dst, (const uint4 *)src, p);
    break;
  case 8:
    hipLaunchKernelGGL((strided_copy_kernel<uint64_t, IndexT>), grid, blk, 0,
                       s, (uint64_t *)dst, (const uint64_t *)src, p);
    break;
  case 4:
    hipLaunchKernelGGL((strided_copy_kernel<uint32_t, IndexT>), grid, blk, 0,
                       s, (uint32_t *)dst, (const uint32_t *)src, p);
    break;
  case 2:
    hipLaunchKernelGGL((strided_copy_kernel<uint16_t, IndexT>), grid, blk, 0,
                       s, (uint16_t *)dst, (const uint16_t *)src, p);
    break;
  case 1:
    hipLaunchKernelGGL((strided_copy_kernel<uint8_t, IndexT>), grid, blk, 0, s,
                       (uint8_t *)dst, (const uint8_t *)src, p);
    break;
  default:
    return -2;
  }
  return 0;
}
} // namespace

extern "C" int hip_strided_copy_nd(void *stream, void *dst, const void *src,
                                   int64_t elem_size, int rank,
                                   const int64_t *sizes,
                                   const int64_t *src_strides,
                                   const int64_t *dst_strides) {
  if (elem_size != 1 && elem_size != 2 && elem_size != 4 && elem_size != 8)
    return -2; // odd element size -> caller falls back
  if (rank < 0 || rank > SC_MAX_RANK)
    return -1;

  NormalisedCopy c;
  if (!normaliseCopy(rank, sizes, src_strides, dst_strides, c))
    return 0;

  const int64_t vecBytes = pickVectorBytes(c, elem_size, dst, src);
  if (vecBytes != elem_size) {
    const int64_t ratio = vecBytes / elem_size;
    const int inner = c.rank - 1;
    c.sizes[inner] /= ratio;
    c.total /= ratio;
    for (int i = 0; i < inner; ++i) {
      c.srcStrides[i] /= ratio;
      c.dstStrides[i] /= ratio;
    }
  }

  // 32-bit indexing when the loop index and every offset reached fit.
  int64_t srcReach = 0, dstReach = 0;
  for (int i = 0; i < c.rank; ++i) {
    srcReach += (c.sizes[i] - 1) * std::llabs(c.srcStrides[i]);
    dstReach += (c.sizes[i] - 1) * std::llabs(c.dstStrides[i]);
  }
  const bool narrow = fitsNarrowLoop(c.total) && srcReach <= INT32_MAX &&
                      dstReach <= INT32_MAX;

  hipStream_t s = static_cast<hipStream_t>(stream);
  // Clear any stale error left by a PRIOR unrelated launch so the post-launch
  // check below reflects only this kernel. Without this a sticky error would
  // make hipGetLastError() return non-zero here, the caller would treat the
  // kernel as failed and re-run its fallback path -- double-copying the
  // buffer. (A real launch-config failure IS synchronous and leaves nothing
  // enqueued, so falling back in that case is still correct.)
  (void)hipGetLastError();
  int rc = narrow ? launchStridedCopy<int32_t>(s, dst, src, vecBytes, c)
                  : launchStridedCopy<int64_t>(s, dst, src, vecBytes, c);
  if (rc != 0)
    return rc;

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr, "[custom_kernels] hip_strided_copy launch failed: %s\n",
            hipGetErrorString(err));
  }
  return (int)err;
}

// One-launch strided D2D copy. Pointers are element-aligned bases (already
// offset). outer_sizes/strides describe the OUTER dims [0..outer_rank-1];
// row_elems is the contiguous inner suffix (stride 1 on both sides).
// Returns 0 on success; -2 for an unsupported element size so the caller can
// fall back to the host per-row path.
extern "C" int hip_strided_copy(void *stream, void *dst, const void *src,
                                int64_t elem_size, int outer_rank,
                                const int64_t *outer_sizes,
                                const int64_t *src_outer_strides,
                                const int64_t *dst_outer_strides,
                                int64_t row_elems, int64_t outer_total) {
  if (outer_total <= 0 || row_elems <= 0)
    return 0;
  if (outer_rank < 0 || outer_rank > HIPDNN_MAX_MEMREF_RANK)
    return -1;

  int64_t sizes[SC_MAX_RANK], srcStrides[SC_MAX_RANK], dstStrides[SC_MAX_RANK];
  for (int i = 0; i < outer_rank; ++i) {
    sizes[i] = outer_sizes[i];
    srcStrides[i] = src_outer_strides[i];
    dstStrides[i] = dst_outer_strides[i];
  }
  sizes[outer_rank] = row_elems;
  srcStrides[outer_rank] = 1;
  dstStrides[outer_rank] = 1;
  return hip_strided_copy_nd(stream, dst, src, elem_size, outer_rank + 1,
                             sizes, srcStrides, dstStrides);
}
