/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "hipdnn_ep_runtime.h"
#include "runtime_types.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <limits>

#ifdef _WIN32
#include <windows.h>
#define MOCK_PRINT(...)                                                        \
  do {                                                                         \
    char buf[512];                                                             \
    snprintf(buf, sizeof(buf), __VA_ARGS__);                                   \
    OutputDebugStringA(buf);                                                   \
    fprintf(stderr, "%s", buf);                                                \
    fflush(stderr);                                                            \
  } while (0)
#else
#define MOCK_PRINT(...)                                                        \
  do {                                                                         \
    printf(__VA_ARGS__);                                                       \
    fflush(stdout);                                                            \
  } while (0)
#endif

namespace {

constexpr int64_t kMaximumMemRefRank = 12;

struct RankedMemRefDescriptor {
  void *allocatedPtr;
  void *alignedPtr;
  int64_t offset;
  int64_t sizesAndStrides[];
};

bool checkedMultiply(int64_t lhs, int64_t rhs, int64_t &result) {
  if (lhs < 0 || rhs < 0 ||
      (lhs != 0 && rhs > std::numeric_limits<int64_t>::max() / lhs)) {
    return false;
  }
  result = lhs * rhs;
  return true;
}

bool checkedAdd(int64_t lhs, int64_t rhs, int64_t &result) {
  if (lhs < 0 || rhs < 0 || lhs > std::numeric_limits<int64_t>::max() - rhs) {
    return false;
  }
  result = lhs + rhs;
  return true;
}

bool validateReachableSpan(int64_t offset, const int64_t *sizes,
                           const int64_t *strides, int64_t rank,
                           int64_t &maximumOffset) {
  if (offset < 0) {
    return false;
  }
  maximumOffset = offset;
  for (int64_t dimension = 0; dimension < rank; ++dimension) {
    if (sizes[dimension] <= 1) {
      continue;
    }
    if (strides[dimension] <= 0) {
      return false;
    }
    int64_t contribution = 0;
    if (!checkedMultiply(sizes[dimension] - 1, strides[dimension],
                         contribution) ||
        !checkedAdd(maximumOffset, contribution, maximumOffset)) {
      return false;
    }
  }
  return true;
}

bool hasNonOverlappingElements(const int64_t *sizes, const int64_t *strides,
                               int64_t rank) {
  std::array<int64_t, kMaximumMemRefRank> activeDimensions{};
  int64_t activeCount = 0;
  for (int64_t dimension = 0; dimension < rank; ++dimension) {
    if (sizes[dimension] > 1) {
      activeDimensions[static_cast<size_t>(activeCount++)] = dimension;
    }
  }
  std::sort(
      activeDimensions.begin(), activeDimensions.begin() + activeCount,
      [&](int64_t lhs, int64_t rhs) { return strides[lhs] < strides[rhs]; });

  int64_t lowerSpan = 1;
  for (int64_t index = 0; index < activeCount; ++index) {
    const int64_t dimension = activeDimensions[static_cast<size_t>(index)];
    if (strides[dimension] < lowerSpan) {
      return false;
    }
    int64_t contribution = 0;
    if (!checkedMultiply(sizes[dimension] - 1, strides[dimension],
                         contribution) ||
        !checkedAdd(lowerSpan, contribution, lowerSpan)) {
      return false;
    }
  }
  return true;
}

} // namespace

int wrap_hipMemrefCopy(RuntimeState *state, int64_t elem_size,
                       const HipdnnEpUnrankedMemRef *src,
                       const HipdnnEpUnrankedMemRef *dst) {
  if (!state || !src || !dst || !src->descriptor || !dst->descriptor ||
      elem_size <= 0 || src->rank != dst->rank || src->rank < 0 ||
      src->rank > kMaximumMemRefRank) {
    fprintf(stderr, "wrap_hipMemrefCopy: invalid argument or rank\n");
    return -1;
  }

  auto *source = static_cast<const RankedMemRefDescriptor *>(src->descriptor);
  auto *destination =
      static_cast<const RankedMemRefDescriptor *>(dst->descriptor);
  const int64_t rank = src->rank;
  const int64_t *srcSizes = source->sizesAndStrides;
  const int64_t *srcStrides = source->sizesAndStrides + rank;
  const int64_t *dstSizes = destination->sizesAndStrides;
  const int64_t *dstStrides = destination->sizesAndStrides + rank;

  bool empty = false;
  for (int64_t dimension = 0; dimension < rank; ++dimension) {
    if (srcSizes[dimension] < 0 || dstSizes[dimension] < 0 ||
        srcSizes[dimension] != dstSizes[dimension]) {
      fprintf(stderr, "wrap_hipMemrefCopy: invalid shape\n");
      return -1;
    }
    empty |= srcSizes[dimension] == 0;
  }
  if (empty) {
    return 0;
  }

  int64_t totalElements = 1;
  for (int64_t dimension = 0; dimension < rank; ++dimension) {
    if (!checkedMultiply(totalElements, srcSizes[dimension], totalElements)) {
      fprintf(stderr, "wrap_hipMemrefCopy: invalid shape\n");
      return -1;
    }
  }

  int64_t srcMaximumOffset = 0;
  int64_t dstMaximumOffset = 0;
  int64_t totalBytes = 0;
  int64_t srcOffsetBytes = 0;
  int64_t dstOffsetBytes = 0;
  int64_t srcSpanElements = 0;
  int64_t dstSpanElements = 0;
  if (!validateReachableSpan(source->offset, srcSizes, srcStrides, rank,
                             srcMaximumOffset) ||
      !validateReachableSpan(destination->offset, dstSizes, dstStrides, rank,
                             dstMaximumOffset) ||
      !checkedMultiply(totalElements, elem_size, totalBytes) ||
      !checkedMultiply(source->offset, elem_size, srcOffsetBytes) ||
      !checkedMultiply(destination->offset, elem_size, dstOffsetBytes) ||
      !checkedAdd(srcMaximumOffset, 1, srcSpanElements) ||
      !checkedAdd(dstMaximumOffset, 1, dstSpanElements) ||
      srcSpanElements > std::numeric_limits<int64_t>::max() / elem_size ||
      dstSpanElements > std::numeric_limits<int64_t>::max() / elem_size) {
    fprintf(stderr,
            "wrap_hipMemrefCopy: invalid stride, offset, or byte range\n");
    return -1;
  }
  if (!hasNonOverlappingElements(dstSizes, dstStrides, rank)) {
    fprintf(
        stderr,
        "wrap_hipMemrefCopy: destination layout has overlapping elements\n");
    return -1;
  }
  if (!source->alignedPtr || !destination->alignedPtr) {
    fprintf(stderr, "wrap_hipMemrefCopy: null data pointer\n");
    return -1;
  }

  const char *srcBase = static_cast<const char *>(source->alignedPtr) +
                        static_cast<size_t>(srcOffsetBytes);
  char *dstBase = static_cast<char *>(destination->alignedPtr) +
                  static_cast<size_t>(dstOffsetBytes);
  int64_t indices[kMaximumMemRefRank] = {};
  for (int64_t element = 0; element < totalElements; ++element) {
    int64_t srcElementOffset = 0;
    int64_t dstElementOffset = 0;
    for (int64_t dimension = 0; dimension < rank; ++dimension) {
      srcElementOffset += indices[dimension] * srcStrides[dimension];
      dstElementOffset += indices[dimension] * dstStrides[dimension];
    }
    memcpy(dstBase + static_cast<size_t>(dstElementOffset * elem_size),
           srcBase + static_cast<size_t>(srcElementOffset * elem_size),
           static_cast<size_t>(elem_size));
    for (int64_t dimension = rank - 1; dimension >= 0; --dimension) {
      if (++indices[dimension] < srcSizes[dimension]) {
        break;
      }
      indices[dimension] = 0;
    }
  }
  MOCK_PRINT("[MOCK] wrap_hipMemrefCopy(rank=%lld, elems=%lld, bytes=%lld)\n",
             static_cast<long long>(rank),
             static_cast<long long>(totalElements),
             static_cast<long long>(totalBytes));
  return 0;
}

// Mock wrap_hipMemcpyAsync (host memcpy)
// Follows opaque RuntimeState pattern - extracts stream internally
int wrap_hipMemcpyAsync(RuntimeState *state, void *dst_ptr, const void *src_ptr,
                        size_t size_bytes) {
  if (!state) {
    fprintf(stderr, "wrap_hipMemcpyAsync: null state\n");
    return -1;
  }
  if (size_bytes == 0)
    return 0;
  if (!dst_ptr || !src_ptr) {
    fprintf(stderr, "wrap_hipMemcpyAsync: null pointer\n");
    return -1;
  }

  MOCK_PRINT("[MOCK] wrap_hipMemcpyAsync(dst=%p, src=%p, size=%zu)\n", dst_ptr,
             src_ptr, size_bytes);

  // Mock: Just use memcpy since we don't have real GPU memory
  memcpy(dst_ptr, src_ptr, size_bytes);

  return 0;
}

int wrap_hipMemcpy2DAsync(RuntimeState *state, void *dst_ptr, size_t dst_pitch,
                          const void *src_ptr, size_t src_pitch, size_t width,
                          size_t height) {
  if (!state) {
    fprintf(stderr, "wrap_hipMemcpy2DAsync: null state\n");
    return -1;
  }
  if (!dst_ptr || !src_ptr) {
    fprintf(stderr, "wrap_hipMemcpy2DAsync: null pointer\n");
    return -1;
  }
  if (width == 0 || height == 0) {
    return 0;
  }

  MOCK_PRINT("[MOCK] wrap_hipMemcpy2DAsync(dst=%p spitch=%zu dpitch=%zu w=%zu "
             "h=%zu)\n",
             dst_ptr, src_pitch, dst_pitch, width, height);

  const char *s = static_cast<const char *>(src_ptr);
  char *d = static_cast<char *>(dst_ptr);
  for (size_t row = 0; row < height; ++row) {
    memcpy(d + row * dst_pitch, s + row * src_pitch, width);
  }
  return 0;
}

// Mock strided copy: same element semantics as the real kernel, on the host.
// `height` rows, each copying `row_elems` contiguous elements; outer strides
// are in elements.
int wrap_strided_copy(RuntimeState *state, void *dst_ptr, const void *src_ptr,
                      int64_t elem_bytes, int64_t height,
                      int64_t src_pitch_elems, int64_t dst_pitch_elems,
                      int64_t row_elems) {
  if (!state) {
    fprintf(stderr, "wrap_strided_copy: null state\n");
    return -1;
  }
  if (!dst_ptr || !src_ptr) {
    fprintf(stderr, "wrap_strided_copy: null pointer\n");
    return -1;
  }
  if (height == 0 || row_elems == 0)
    return 0;

  MOCK_PRINT("[MOCK] wrap_strided_copy(dst=%p elem=%lld h=%lld spitch_e=%lld "
             "dpitch_e=%lld row_e=%lld)\n",
             dst_ptr, (long long)elem_bytes, (long long)height,
             (long long)src_pitch_elems, (long long)dst_pitch_elems,
             (long long)row_elems);

  const char *s = static_cast<const char *>(src_ptr);
  char *d = static_cast<char *>(dst_ptr);
  size_t row_bytes = static_cast<size_t>(row_elems) * elem_bytes;
  for (int64_t row = 0; row < height; ++row) {
    memcpy(d + row * dst_pitch_elems * elem_bytes,
           s + row * src_pitch_elems * elem_bytes, row_bytes);
  }
  return 0;
}
