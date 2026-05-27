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
// helper must walk the index space using src/dst strides. Our generated code
// only produces `memref.copy` between identical-shape, contiguous memrefs
// (the bufferized `tensor.insert_slice` / `memref.subview` chains around the
// runtime shape buffers — those always end up as a row-contiguous prefix of
// the destination), so a flat hipMemcpy on the byte size is correct.
//
// If a future case needs strided/non-contiguous semantics, generalise this
// to honour the descriptor's strides[] (libmlir_c_runner_utils.dylib has a
// reference implementation).
struct UnrankedMemRefHeader {
  int64_t rank;
  void *descPtr;
};
struct StridedMemRefDescriptor {
  void *allocPtr;
  void *alignedPtr;
  int64_t offset;
  int64_t sizesAndStrides[];
};
extern "C" void memrefCopy(int64_t elemSize, UnrankedMemRefHeader *src,
                           UnrankedMemRefHeader *dst) {
  if (!src || !dst || !src->descPtr || !dst->descPtr)
    return;
  auto *sd = static_cast<StridedMemRefDescriptor *>(src->descPtr);
  auto *dd = static_cast<StridedMemRefDescriptor *>(dst->descPtr);
  int64_t rank = src->rank;
  int64_t nelems = 1;
  for (int64_t i = 0; i < rank; ++i)
    nelems *= sd->sizesAndStrides[i];
  int64_t bytes = nelems * elemSize;
  // Element-aligned source/dest pointers (alignedPtr + offset * elemSize).
  void *srcPtr = static_cast<char *>(sd->alignedPtr) + sd->offset * elemSize;
  void *dstPtr = static_cast<char *>(dd->alignedPtr) + dd->offset * elemSize;
  hipError_t err =
      hipMemcpyAsync(dstPtr, srcPtr, bytes, hipMemcpyDeviceToDevice, 0);
  if (err != hipSuccess) {
    fprintf(stderr, "memrefCopy(%lld bytes) failed: %s\n",
            static_cast<long long>(bytes), hipGetErrorString(err));
  }
}
