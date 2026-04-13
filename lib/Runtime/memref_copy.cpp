/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- memref_copy.cpp - MLIR memrefCopy runtime for GPU memory -----------===//
//
// Provides the `memrefCopy` function required by MLIR's FinalizeMemRefToLLVM
// when lowering `memref.copy` with strided (non-identity layout) memrefs.
//
// MLIR's lowering generates:
//   declare void @memrefCopy(i64 %elemSize, ptr %srcDesc, ptr %dstDesc)
//
// where srcDesc/dstDesc are MLIR UnrankedMemRefDescriptor pointers:
//   struct { i64 rank; ptr descriptor }
// and descriptor points to the ranked descriptor:
//   struct { ptr allocated; ptr aligned; i64 offset;
//            i64 sizes[rank]; i64 strides[rank] }
//
// This function is called from host-side code within the compiled model DLL.
// The `aligned` pointers inside the descriptors are GPU device pointers
// (allocated via hipMalloc). We use synchronous HIP APIs (hipMemcpy2D) to
// perform the copy, which implicitly synchronizes with all streams.
//
// Design note: memrefCopy has a fixed ABI from MLIR with no access to the
// HIP stream used by the rest of the pipeline. Using synchronous copies is
// correct but introduces a pipeline stall. For production, the better
// approach is to avoid memref.copy by fusing slice+cat into upstream ops
// or adding a stream-aware hip.concat op. This implementation prioritizes
// correctness for bring-up.
//
//===----------------------------------------------------------------------===//

#include <cstdint>
#include <cstring>

// Layout of MLIR's unranked memref descriptor (as passed by LLVM lowering).
struct UnrankedMemRef {
  int64_t rank;
  void *descriptor; // points to the ranked descriptor below
};

// Layout of MLIR's ranked memref descriptor (variable-length tail).
//   { ptr allocated, ptr aligned, i64 offset, i64 sizes[rank], i64
//   strides[rank] }
// We only access individual fields via pointer arithmetic.
struct RankedDescriptor {
  void *allocated;
  void *aligned;
  int64_t offset;
  // int64_t sizes[rank];    -- at (int64_t*)(this + 1)
  // int64_t strides[rank];  -- at (int64_t*)(this + 1) + rank
};

#ifdef __HIP_PLATFORM_AMD__
// =========================================================================
// Real GPU runtime: use HIP APIs for device-to-device memory copies
// =========================================================================
#include <cstdio>
#include <hip/hip_runtime.h>

/// Recursively copy elements between strided GPU buffers.
/// When the innermost contiguous dimension is reached, uses hipMemcpy2D
/// for efficient bulk transfer. Falls back to per-row hipMemcpy for the
/// general case.
static void gpuCopyRecursive(const char *src, char *dst, int64_t elemSize,
                             int64_t rank, const int64_t *sizes,
                             const int64_t *srcStrides,
                             const int64_t *dstStrides) {
  if (rank == 0) {
    hipMemcpy(dst, src, elemSize, hipMemcpyDeviceToDevice);
    return;
  }

  if (rank == 1) {
    if (srcStrides[0] == 1 && dstStrides[0] == 1) {
      // Both contiguous along this dimension — single bulk copy
      hipMemcpy(dst, src, sizes[0] * elemSize, hipMemcpyDeviceToDevice);
    } else {
      // Strided — element-by-element
      for (int64_t i = 0; i < sizes[0]; ++i) {
        hipMemcpy(dst + i * dstStrides[0] * elemSize,
                  src + i * srcStrides[0] * elemSize, elemSize,
                  hipMemcpyDeviceToDevice);
      }
    }
    return;
  }

  // Optimization: if the last dimension has stride 1 in both src and dst,
  // use hipMemcpy2D which handles different pitches efficiently.
  if (rank == 2 && srcStrides[1] == 1 && dstStrides[1] == 1) {
    size_t widthBytes = static_cast<size_t>(sizes[1] * elemSize);
    size_t srcPitch = static_cast<size_t>(srcStrides[0] * elemSize);
    size_t dstPitch = static_cast<size_t>(dstStrides[0] * elemSize);
    size_t height = static_cast<size_t>(sizes[0]);
    hipMemcpy2D(dst, dstPitch, src, srcPitch, widthBytes, height,
                hipMemcpyDeviceToDevice);
    return;
  }

  // General case: iterate over the outermost dimension and recurse.
  for (int64_t i = 0; i < sizes[0]; ++i) {
    gpuCopyRecursive(src + i * srcStrides[0] * elemSize,
                     dst + i * dstStrides[0] * elemSize, elemSize, rank - 1,
                     sizes + 1, srcStrides + 1, dstStrides + 1);
  }
}

extern "C" void memrefCopy(int64_t elemSize, UnrankedMemRef *srcDesc,
                           UnrankedMemRef *dstDesc) {
  int64_t rank = srcDesc->rank;
  auto *src = static_cast<RankedDescriptor *>(srcDesc->descriptor);
  auto *dst = static_cast<RankedDescriptor *>(dstDesc->descriptor);

  auto *srcFields = reinterpret_cast<int64_t *>(src + 1);
  const int64_t *srcSizes = srcFields;
  const int64_t *srcStrides = srcFields + rank;

  auto *dstFields = reinterpret_cast<int64_t *>(dst + 1);
  const int64_t *dstStrides = dstFields + rank;

  auto *srcPtr =
      static_cast<const char *>(src->aligned) + src->offset * elemSize;
  auto *dstPtr = static_cast<char *>(dst->aligned) + dst->offset * elemSize;

  if (rank == 0) {
    hipMemcpy(dstPtr, srcPtr, elemSize, hipMemcpyDeviceToDevice);
    return;
  }

  gpuCopyRecursive(srcPtr, dstPtr, elemSize, rank, srcSizes, srcStrides,
                   dstStrides);
}

#else
// =========================================================================
// Mock runtime: simple CPU memcpy (data is in host memory)
// =========================================================================

static void cpuCopyRecursive(const char *src, char *dst, int64_t elemSize,
                             int64_t rank, const int64_t *sizes,
                             const int64_t *srcStrides,
                             const int64_t *dstStrides) {
  if (rank == 0) {
    std::memcpy(dst, src, elemSize);
    return;
  }
  if (rank == 1 && srcStrides[0] == 1 && dstStrides[0] == 1) {
    std::memcpy(dst, src, sizes[0] * elemSize);
    return;
  }
  if (rank == 1) {
    for (int64_t i = 0; i < sizes[0]; ++i) {
      std::memcpy(dst + i * dstStrides[0] * elemSize,
                  src + i * srcStrides[0] * elemSize, elemSize);
    }
    return;
  }
  for (int64_t i = 0; i < sizes[0]; ++i) {
    cpuCopyRecursive(src + i * srcStrides[0] * elemSize,
                     dst + i * dstStrides[0] * elemSize, elemSize, rank - 1,
                     sizes + 1, srcStrides + 1, dstStrides + 1);
  }
}

extern "C" void memrefCopy(int64_t elemSize, UnrankedMemRef *srcDesc,
                           UnrankedMemRef *dstDesc) {
  int64_t rank = srcDesc->rank;
  auto *src = static_cast<RankedDescriptor *>(srcDesc->descriptor);
  auto *dst = static_cast<RankedDescriptor *>(dstDesc->descriptor);

  auto *srcFields = reinterpret_cast<int64_t *>(src + 1);
  const int64_t *srcSizes = srcFields;
  const int64_t *srcStrides = srcFields + rank;

  auto *dstFields = reinterpret_cast<int64_t *>(dst + 1);
  const int64_t *dstStrides = dstFields + rank;

  auto *srcPtr =
      static_cast<const char *>(src->aligned) + src->offset * elemSize;
  auto *dstPtr = static_cast<char *>(dst->aligned) + dst->offset * elemSize;

  if (rank == 0) {
    std::memcpy(dstPtr, srcPtr, elemSize);
    return;
  }

  cpuCopyRecursive(srcPtr, dstPtr, elemSize, rank, srcSizes, srcStrides,
                   dstStrides);
}

#endif // __HIP_PLATFORM_AMD__
