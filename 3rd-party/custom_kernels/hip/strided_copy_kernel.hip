/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// Parallel strided device-to-device copy for MLIR `memref.copy` (memrefCopy)
// when neither side is fully contiguous and the copy spans >1 outer dim.
//
// The runtime's memrefCopy previously walked the outer index space on the host
// and issued one hipMemcpyAsync PER contiguous row. For the Qwen3.5 vision
// encoder that is ~2 million 72-byte launches per prefill (Reshape/Concat/Tile
// chains over ~19k rows x ~108 copies) and dominated GPU prefill time. This
// kernel does the whole strided copy in ONE launch: one element per thread,
// with the contiguous inner suffix mapped to consecutive threads (coalesced),
// and the outer multi-index decomposed per thread to apply src/dst strides.

#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>

#include "hip_custom_kernels.h"

namespace {
constexpr int SC_MAX_RANK = HIPDNN_MAX_MEMREF_RANK;

struct StridedCopyParams {
  int64_t outerSizes[SC_MAX_RANK];      // sizes of the outer (non-contiguous) dims
  int64_t srcOuterStride[SC_MAX_RANK];  // src strides (elements) of outer dims
  int64_t dstOuterStride[SC_MAX_RANK];  // dst strides (elements) of outer dims
  int outerRank;
  int64_t rowElems;    // contiguous inner elements per outer row (stride 1)
  int64_t outerTotal;  // product(outerSizes)
};

template <typename T>
__global__ void strided_copy_kernel(T *__restrict__ dst,
                                    const T *__restrict__ src,
                                    StridedCopyParams p) {
  const int64_t total = p.outerTotal * p.rowElems;
  const int64_t gstride = (int64_t)gridDim.x * blockDim.x;
  for (int64_t t = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; t < total;
       t += gstride) {
    const int64_t row = t / p.rowElems;
    const int64_t col = t - row * p.rowElems;  // contiguous within the row
    int64_t srcOff = 0, dstOff = 0, r = row;
    for (int i = p.outerRank - 1; i >= 0; --i) {
      const int64_t idx = r % p.outerSizes[i];
      r /= p.outerSizes[i];
      srcOff += idx * p.srcOuterStride[i];
      dstOff += idx * p.dstOuterStride[i];
    }
    dst[dstOff + col] = src[srcOff + col];
  }
}
}  // namespace

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
  if (outer_rank < 0 || outer_rank > SC_MAX_RANK)
    return -1;

  StridedCopyParams p{};
  p.outerRank = outer_rank;
  p.rowElems = row_elems;
  p.outerTotal = outer_total;
  for (int i = 0; i < outer_rank; ++i) {
    p.outerSizes[i] = outer_sizes[i];
    p.srcOuterStride[i] = src_outer_strides[i];
    p.dstOuterStride[i] = dst_outer_strides[i];
  }

  hipStream_t s = static_cast<hipStream_t>(stream);
  const int64_t total = outer_total * row_elems;
  const int block = 256;
  int64_t grid64 = (total + block - 1) / block;
  if (grid64 > 65535)
    grid64 = 65535;  // grid-stride loop covers the remainder
  dim3 grid((unsigned)grid64), blk(block);

  // Clear any stale error left by a PRIOR unrelated launch so the post-launch
  // check below reflects only this kernel. Without this a sticky error would
  // make hipGetLastError() return non-zero here, the caller would treat the
  // kernel as failed and re-run the host per-row loop -- double-copying the
  // buffer. (A real launch-config failure IS synchronous and leaves nothing
  // enqueued, so falling back to the host loop in that case is still correct.)
  (void)hipGetLastError();

  switch (elem_size) {
  case 8:
    hipLaunchKernelGGL(strided_copy_kernel<uint64_t>, grid, blk, 0, s,
                       (uint64_t *)dst, (const uint64_t *)src, p);
    break;
  case 4:
    hipLaunchKernelGGL(strided_copy_kernel<uint32_t>, grid, blk, 0, s,
                       (uint32_t *)dst, (const uint32_t *)src, p);
    break;
  case 2:
    hipLaunchKernelGGL(strided_copy_kernel<uint16_t>, grid, blk, 0, s,
                       (uint16_t *)dst, (const uint16_t *)src, p);
    break;
  case 1:
    hipLaunchKernelGGL(strided_copy_kernel<uint8_t>, grid, blk, 0, s,
                       (uint8_t *)dst, (const uint8_t *)src, p);
    break;
  default:
    return -2;  // odd element size -> host fallback
  }

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr, "[custom_kernels] hip_strided_copy launch failed: %s\n",
            hipGetErrorString(err));
  }
  return (int)err;
}
