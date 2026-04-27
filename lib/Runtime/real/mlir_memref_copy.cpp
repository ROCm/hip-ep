/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "mlir/ExecutionEngine/CRunnerUtils.h"

#include <hip/hip_runtime.h>

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

static bool isRowMajorContiguous(const DynamicMemRefType<char> &m) {
  int64_t expectedStride = 1;
  for (int64_t d = m.rank - 1; d >= 0; --d) {
    if (m.strides[d] != expectedStride)
      return false;
    expectedStride *= m.sizes[d];
  }
  return true;
}

static int64_t getNumElements(const DynamicMemRefType<char> &m) {
  int64_t elems = 1;
  for (int64_t d = 0; d < m.rank; ++d)
    elems *= m.sizes[d];
  return elems;
}

static bool memrefCopyTraceEnabled() {
  const char *v = std::getenv("HIPDNN_MEMREFCOPY_TRACE");
  return v && v[0] != '\0' && v[0] != '0';
}

static bool copyBytes(void *dst, const void *src, size_t bytes) {
  if (bytes == 0)
    return true;

  // hipMemcpyDefault lets HIP infer host/device direction and handles
  // both device and host-backed pointers used by runtime memrefs.
  hipError_t err = hipMemcpy(dst, src, bytes, hipMemcpyDefault);
  if (err == hipSuccess)
    return true;

  // Fallback for host-only pointers or environments where hipMemcpyDefault
  // cannot classify one side.
  (void)hipGetLastError();
  std::memcpy(dst, src, bytes);
  return false;
}

static bool copyBytesAsync(void *dst, const void *src, size_t bytes) {
  if (bytes == 0)
    return true;
  return hipMemcpyAsync(dst, src, bytes, hipMemcpyDefault, nullptr) ==
         hipSuccess;
}

static bool copyBytes2D(void *dst, size_t dstPitchBytes, const void *src,
                        size_t srcPitchBytes, size_t widthBytes,
                        size_t height) {
  if (widthBytes == 0 || height == 0)
    return true;

  hipError_t err = hipMemcpy2D(dst, dstPitchBytes, src, srcPitchBytes,
                               widthBytes, height, hipMemcpyDefault);
  if (err == hipSuccess)
    return true;

  (void)hipGetLastError();
  auto *dstBytes = static_cast<char *>(dst);
  auto *srcBytes = static_cast<const char *>(src);
  for (size_t row = 0; row < height; ++row)
    std::memcpy(dstBytes + row * dstPitchBytes, srcBytes + row * srcPitchBytes,
                widthBytes);
  return false;
}

static bool copyBytes2DAsync(void *dst, size_t dstPitchBytes, const void *src,
                             size_t srcPitchBytes, size_t widthBytes,
                             size_t height) {
  if (widthBytes == 0 || height == 0)
    return true;
  return hipMemcpy2DAsync(dst, dstPitchBytes, src, srcPitchBytes, widthBytes,
                          height, hipMemcpyDefault, nullptr) == hipSuccess;
}

} // namespace

extern "C" void memrefCopy(int64_t elemSize, UnrankedMemRefType<char> *srcArg,
                           UnrankedMemRefType<char> *dstArg) {
  DynamicMemRefType<char> src(*srcArg);
  DynamicMemRefType<char> dst(*dstArg);
  uint64_t copyCalls = 0;
  uint64_t hipMemcpyCalls = 0;
  uint64_t memcpyFallbackCalls = 0;
  size_t copiedBytes = 0;
  uint64_t pendingAsyncCopies = 0;
  uint64_t submittedAsyncCopies = 0;

  auto flushAsyncCopies = [&]() {
    if (pendingAsyncCopies == 0)
      return;
    (void)hipStreamSynchronize(nullptr);
    pendingAsyncCopies = 0;
  };

  auto countedCopy = [&](char *dstPtr, const char *srcPtr, size_t bytes) {
    ++copyCalls;
    copiedBytes += bytes;
    if (copyBytesAsync(dstPtr, srcPtr, bytes)) {
      ++hipMemcpyCalls;
      ++pendingAsyncCopies;
      ++submittedAsyncCopies;
      return;
    }
    (void)hipGetLastError();
    if (copyBytes(dstPtr, srcPtr, bytes))
      ++hipMemcpyCalls;
    else
      ++memcpyFallbackCalls;
  };
  auto countedCopy2D = [&](char *dstPtr, size_t dstPitchBytes,
                           const char *srcPtr, size_t srcPitchBytes,
                           size_t widthBytes, size_t height) {
    ++copyCalls;
    copiedBytes += widthBytes * height;
    if (copyBytes2DAsync(dstPtr, dstPitchBytes, srcPtr, srcPitchBytes,
                         widthBytes, height)) {
      ++hipMemcpyCalls;
      ++pendingAsyncCopies;
      ++submittedAsyncCopies;
      return;
    }
    (void)hipGetLastError();
    if (copyBytes2D(dstPtr, dstPitchBytes, srcPtr, srcPitchBytes, widthBytes,
                    height))
      ++hipMemcpyCalls;
    else
      ++memcpyFallbackCalls;
  };

  assert(src.rank == dst.rank && "rank mismatch in memrefCopy");
  if (src.rank != dst.rank)
    return;

  for (int64_t d = 0; d < src.rank; ++d) {
    assert(src.sizes[d] == dst.sizes[d] && "shape mismatch in memrefCopy");
    if (src.sizes[d] != dst.sizes[d])
      return;
  }

  if (src.rank == 0) {
    countedCopy(dst.data + dst.offset * elemSize,
                src.data + src.offset * elemSize,
                static_cast<size_t>(elemSize));
    flushAsyncCopies();
    if (memrefCopyTraceEnabled())
      std::fprintf(stderr,
                   "[memrefCopy] calls=%llu hip=%llu memcpy=%llu bytes=%zu "
                   "mode=rank0 async=%llu\n",
                   static_cast<unsigned long long>(copyCalls),
                   static_cast<unsigned long long>(hipMemcpyCalls),
                   static_cast<unsigned long long>(memcpyFallbackCalls),
                   copiedBytes,
                   static_cast<unsigned long long>(submittedAsyncCopies));
    return;
  }

  int64_t totalElems = getNumElements(src);
  if (totalElems == 0)
    return;

  if (isRowMajorContiguous(src) && isRowMajorContiguous(dst)) {
    countedCopy(dst.data + dst.offset * elemSize,
                src.data + src.offset * elemSize,
                static_cast<size_t>(totalElems * elemSize));
    flushAsyncCopies();
    if (memrefCopyTraceEnabled())
      std::fprintf(stderr,
                   "[memrefCopy] calls=%llu hip=%llu memcpy=%llu bytes=%zu "
                   "mode=fully-contiguous async=%llu\n",
                   static_cast<unsigned long long>(copyCalls),
                   static_cast<unsigned long long>(hipMemcpyCalls),
                   static_cast<unsigned long long>(memcpyFallbackCalls),
                   copiedBytes,
                   static_cast<unsigned long long>(submittedAsyncCopies));
    return;
  }

  // 2D strided fast-path:
  // If innermost dim is contiguous for both src/dst, treat dim(rank-2) as rows
  // and issue one hipMemcpy2D per outer index point. This drastically reduces
  // API-call overhead for common Split subviews.
  if (src.rank >= 2 && src.strides[src.rank - 1] == 1 &&
      dst.strides[dst.rank - 1] == 1 && src.strides[src.rank - 2] > 0 &&
      dst.strides[dst.rank - 2] > 0) {
    int64_t rowDim = src.rank - 2;
    int64_t outerRank = rowDim;
    size_t widthBytes = static_cast<size_t>(src.sizes[src.rank - 1] * elemSize);
    size_t height = static_cast<size_t>(src.sizes[rowDim]);
    size_t srcPitchBytes = static_cast<size_t>(src.strides[rowDim] * elemSize);
    size_t dstPitchBytes = static_cast<size_t>(dst.strides[rowDim] * elemSize);

    std::vector<int64_t> outerIndices(static_cast<size_t>(outerRank), 0);
    while (true) {
      int64_t srcLinear = src.offset;
      int64_t dstLinear = dst.offset;
      for (int64_t d = 0; d < outerRank; ++d) {
        srcLinear += outerIndices[static_cast<size_t>(d)] * src.strides[d];
        dstLinear += outerIndices[static_cast<size_t>(d)] * dst.strides[d];
      }

      countedCopy2D(dst.data + dstLinear * elemSize, dstPitchBytes,
                    src.data + srcLinear * elemSize, srcPitchBytes, widthBytes,
                    height);

      if (outerRank == 0)
        break;
      int64_t dim = outerRank - 1;
      for (; dim >= 0; --dim) {
        int64_t &idx = outerIndices[static_cast<size_t>(dim)];
        ++idx;
        if (idx < src.sizes[dim])
          break;
        idx = 0;
      }
      if (dim < 0)
        break;
    }
    flushAsyncCopies();

    if (memrefCopyTraceEnabled())
      std::fprintf(stderr,
                   "[memrefCopy] calls=%llu hip=%llu memcpy=%llu bytes=%zu "
                   "mode=strided-2d width=%zu height=%zu async=%llu\n",
                   static_cast<unsigned long long>(copyCalls),
                   static_cast<unsigned long long>(hipMemcpyCalls),
                   static_cast<unsigned long long>(memcpyFallbackCalls),
                   copiedBytes, widthBytes, height,
                   static_cast<unsigned long long>(submittedAsyncCopies));
    return;
  }

  // Find the largest common contiguous suffix and copy that suffix as one block
  // per outer-index point instead of element-by-element.
  int64_t suffixStart = src.rank;
  int64_t suffixElems = 1;
  int64_t srcExpectedStride = 1;
  int64_t dstExpectedStride = 1;
  for (int64_t d = src.rank - 1; d >= 0; --d) {
    if (src.strides[d] != srcExpectedStride ||
        dst.strides[d] != dstExpectedStride)
      break;
    suffixStart = d;
    suffixElems *= src.sizes[d];
    srcExpectedStride *= src.sizes[d];
    dstExpectedStride *= dst.sizes[d];
  }

  int64_t outerRank = suffixStart;
  std::vector<int64_t> indices(static_cast<size_t>(outerRank), 0);
  while (true) {
    int64_t srcLinear = src.offset;
    int64_t dstLinear = dst.offset;
    for (int64_t d = 0; d < outerRank; ++d) {
      srcLinear += indices[static_cast<size_t>(d)] * src.strides[d];
      dstLinear += indices[static_cast<size_t>(d)] * dst.strides[d];
    }

    countedCopy(dst.data + dstLinear * elemSize,
                src.data + srcLinear * elemSize,
                static_cast<size_t>(suffixElems * elemSize));

    if (outerRank == 0)
      break;

    int64_t dim = outerRank - 1;
    for (; dim >= 0; --dim) {
      int64_t &idx = indices[static_cast<size_t>(dim)];
      ++idx;
      if (idx < src.sizes[dim])
        break;
      idx = 0;
    }
    if (dim < 0)
      break;
  }
  flushAsyncCopies();

  if (memrefCopyTraceEnabled())
    std::fprintf(stderr,
                 "[memrefCopy] calls=%llu hip=%llu memcpy=%llu bytes=%zu "
                 "mode=strided block_elems=%lld async=%llu\n",
                 static_cast<unsigned long long>(copyCalls),
                 static_cast<unsigned long long>(hipMemcpyCalls),
                 static_cast<unsigned long long>(memcpyFallbackCalls),
                 copiedBytes, static_cast<long long>(suffixElems),
                 static_cast<unsigned long long>(submittedAsyncCopies));
}
