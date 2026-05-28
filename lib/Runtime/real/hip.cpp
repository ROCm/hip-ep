/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "../hipdnn_ep_runtime.h"
#include "runtime_types.h"

#include <cstdio>

// Note: These simple wrapper functions don't need cleanup, so we use a local
// HIP_CHECK that returns directly instead of the goto-based HIP_CHECK_GOTO
#define HIP_CHECK(cmd)                                                         \
  do {                                                                         \
    hipError_t error = (cmd);                                                  \
    if (error != hipSuccess) {                                                 \
      fprintf(stderr, "HIP error at %s:%d: %s\n", __FILE__, __LINE__,          \
              hipGetErrorString(error));                                       \
      return -1;                                                               \
    }                                                                          \
  } while (0)

// HIP memory wrappers

int wrap_hipMalloc(void **ptr, int64_t size) {
  HIP_CHECK(hipMalloc(ptr, size));
  return 0;
}

int wrap_hipFree(void *ptr) {
  HIP_CHECK(hipFree(ptr));
  return 0;
}

int wrap_hipMemcpyH2D(void *dst, const void *src, int64_t size, void *stream) {
  HIP_CHECK(hipMemcpyAsync(dst, src, size, hipMemcpyHostToDevice,
                           static_cast<hipStream_t>(stream)));
  return 0;
}

int wrap_hipMemcpyD2H(void *dst, const void *src, int64_t size, void *stream) {
  HIP_CHECK(hipMemcpyAsync(dst, src, size, hipMemcpyDeviceToHost,
                           static_cast<hipStream_t>(stream)));
  return 0;
}

int wrap_hipStreamSynchronize(void *stream) {
  HIP_CHECK(hipStreamSynchronize(static_cast<hipStream_t>(stream)));
  return 0;
}

// Single-argument allocator emitted by the standard MLIR memref->LLVM
// lowering for any `memref.alloc` that survived the hip-pool-allocs pass
// (e.g. an alloc whose dynamic size traces back to a value computed by a
// non-hoistable runtime op — too late for inclusion in the pool prelude).
// Signature must match MLIR's convention: `i64 size -> ptr`, with `ptr`
// returned by value (no out-parameter).
extern "C" void *hip_device_malloc(int64_t size) {
  fprintf(stderr, "[hip_device_malloc] size=%lld\n", (long long)size);
  fflush(stderr);
  void *ptr = nullptr;
  hipError_t err = hipMalloc(&ptr, size);
  if (err != hipSuccess) {
    fprintf(stderr, "hip_device_malloc(%lld) failed: %s\n",
            static_cast<long long>(size), hipGetErrorString(err));
    return nullptr;
  }
  return ptr;
}

extern "C" void hip_device_free(void *ptr) {
  if (!ptr)
    return;
  hipError_t err = hipFree(ptr);
  if (err != hipSuccess) {
    fprintf(stderr, "hip_device_free(%p) failed: %s\n", ptr,
            hipGetErrorString(err));
  }
}

// MLIR's standard memref dialect lowering for `memref.copy` calls a runtime
// helper named `memrefCopy(elemSize, srcDescPtr, dstDescPtr)`. The MemRef
// descriptors are ABI-stable structs: { allocPtr, alignedPtr, offset,
// sizes[rank], strides[rank] }. For copying a rank-N strided memref, the
// helper must walk the index space using src/dst strides — a flat
// hipMemcpyAsync on `nelems * elemSize` bytes is only correct when both
// the source and destination are row-major contiguous.
//
// The bufferized `tensor.insert_slice` chains that Concat / Slice / Tile
// produce regularly emit `memref.copy` where the destination is a
// strided `memref.subview` of an output buffer (e.g. Concat axis=-1 on
// rank-2 produces a subview with strides `[stride_outer, 1]` and the
// outer stride > the source's outer stride). A flat memcpy treats both
// sides as packed and writes the wrong bytes — the second insert_slice's
// 1024-byte source clobbers most of the first one's data and the
// resulting layout is shifted by one element relative to what each
// strided slice is supposed to cover. Symptom on Qwen 3.5 vision: a
// Reshape→Unsqueeze→Concat(axis=-1)→Expand→Tile chain that should
// interleave two index streams ends up returning only the second
// stream's values, shifted by one.
//
// The strategy below has three tiers:
//   1. Fast path (both sides fully contiguous): one hipMemcpyAsync.
//   2. Single outer non-contiguous dim (the canonical Concat-on-last-axis
//      pattern): one hipMemcpy2DAsync using GPU DMA, no kernel launch.
//   3. General case (outer rank >= 2): walk the outer index space and
//      issue one hipMemcpyAsync per row. Per-row copies are correct for
//      any rank and stride pattern; if a future hotspot lands here we
//      can replace this with a HIP kernel.
struct UnrankedMemRefHeader {
  int64_t rank;
  void *descPtr;
};
struct StridedMemRefDescriptor {
  void *allocPtr;
  void *alignedPtr;
  int64_t offset;
  // sizesAndStrides[0..rank-1] are the sizes, sizesAndStrides[rank..2*rank-1]
  // are the strides (in elements, not bytes). This layout matches MLIR's
  // standard ranked memref descriptor lowering.
  int64_t sizesAndStrides[];
};
extern "C" void memrefCopy(int64_t elemSize, UnrankedMemRefHeader *src,
                           UnrankedMemRefHeader *dst) {
  if (!src || !dst || !src->descPtr || !dst->descPtr)
    return;
  auto *sd = static_cast<StridedMemRefDescriptor *>(src->descPtr);
  auto *dd = static_cast<StridedMemRefDescriptor *>(dst->descPtr);
  int64_t rank = src->rank;

  const int64_t *srcSizes = sd->sizesAndStrides;
  const int64_t *srcStrides = sd->sizesAndStrides + rank;
  const int64_t *dstSizes = dd->sizesAndStrides;
  const int64_t *dstStrides = dd->sizesAndStrides + rank;

  // Element-aligned base pointers (alignedPtr + offset * elemSize).
  char *srcBase = static_cast<char *>(sd->alignedPtr) + sd->offset * elemSize;
  char *dstBase = static_cast<char *>(dd->alignedPtr) + dd->offset * elemSize;

  // Rank-0: a single element. No strides to honour.
  if (rank == 0) {
    hipError_t err = hipMemcpyAsync(dstBase, srcBase, elemSize,
                                    hipMemcpyDeviceToDevice, 0);
    if (err != hipSuccess) {
      fprintf(stderr, "memrefCopy(rank-0) failed: %s\n",
              hipGetErrorString(err));
    }
    return;
  }

  // Bail on any empty axis (memref of size 0) — nothing to copy.
  for (int64_t i = 0; i < rank; ++i) {
    if (srcSizes[i] == 0)
      return;
  }

  // Find the longest contiguous suffix [rowStart..rank-1] where the strides
  // of BOTH src and dst match the canonical row-major layout (i.e. the
  // suffix can be flat-copied as one chunk per "row"). Iterate from the
  // innermost dim outward, accumulating the expected stride as the running
  // product of inner sizes. As soon as either side's stride disagrees with
  // its expected value the suffix ends.
  int64_t rowStart = rank;
  int64_t expectedSrc = 1;
  int64_t expectedDst = 1;
  for (int64_t i = rank - 1; i >= 0; --i) {
    if (srcStrides[i] != expectedSrc || dstStrides[i] != expectedDst)
      break;
    expectedSrc *= srcSizes[i];
    expectedDst *= dstSizes[i];
    rowStart = i;
  }

  // Bytes contained in one contiguous suffix "row".
  int64_t rowElems = 1;
  for (int64_t i = rowStart; i < rank; ++i)
    rowElems *= srcSizes[i];
  int64_t rowBytes = rowElems * elemSize;

  // Tier 1: fully contiguous on both sides — one flat memcpy.
  if (rowStart == 0) {
    hipError_t err = hipMemcpyAsync(dstBase, srcBase, rowBytes,
                                    hipMemcpyDeviceToDevice, 0);
    if (err != hipSuccess) {
      fprintf(stderr, "memrefCopy(%lld bytes contig) failed: %s\n",
              static_cast<long long>(rowBytes), hipGetErrorString(err));
    }
    return;
  }

  // Tier 2: exactly one outer dim — hipMemcpy2DAsync handles strided
  // destinations natively. This is the canonical Concat-axis-last on
  // rank-2 (and on higher ranks when only the concat axis breaks
  // contiguity) — the GPU DMA engine performs the strided copy with no
  // kernel launch overhead.
  if (rowStart == 1) {
    int64_t height = srcSizes[0];
    int64_t srcPitch = srcStrides[0] * elemSize;
    int64_t dstPitch = dstStrides[0] * elemSize;
    hipError_t err =
        hipMemcpy2DAsync(dstBase, dstPitch, srcBase, srcPitch, rowBytes,
                         height, hipMemcpyDeviceToDevice, 0);
    if (err != hipSuccess) {
      fprintf(stderr, "memrefCopy(rank-2 strided) failed: %s\n",
              hipGetErrorString(err));
    }
    return;
  }

  // Tier 3: outer rank >= 2 — walk the outer index space and issue one
  // hipMemcpyAsync per contiguous row. This is correct for any rank and
  // stride pattern. Max rank is bounded by MLIR's descriptor layout; 12
  // is comfortably above anything we currently generate.
  constexpr int64_t kMaxRank = 12;
  if (rank > kMaxRank) {
    fprintf(stderr,
            "memrefCopy: rank %lld exceeds supported maximum %lld\n",
            static_cast<long long>(rank),
            static_cast<long long>(kMaxRank));
    return;
  }

  int64_t outerRank = rowStart;
  int64_t idx[kMaxRank] = {};
  int64_t outerTotal = 1;
  for (int64_t i = 0; i < outerRank; ++i)
    outerTotal *= srcSizes[i];

  for (int64_t r = 0; r < outerTotal; ++r) {
    int64_t srcByteOff = 0;
    int64_t dstByteOff = 0;
    for (int64_t i = 0; i < outerRank; ++i) {
      srcByteOff += idx[i] * srcStrides[i] * elemSize;
      dstByteOff += idx[i] * dstStrides[i] * elemSize;
    }
    hipError_t err = hipMemcpyAsync(dstBase + dstByteOff, srcBase + srcByteOff,
                                    rowBytes, hipMemcpyDeviceToDevice, 0);
    if (err != hipSuccess) {
      fprintf(stderr, "memrefCopy(strided row) failed: %s\n",
              hipGetErrorString(err));
      return;
    }
    // Increment the outer multi-dim index (least-significant axis first).
    for (int64_t i = outerRank - 1; i >= 0; --i) {
      if (++idx[i] < srcSizes[i])
        break;
      idx[i] = 0;
    }
  }
}
