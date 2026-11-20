/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "hipdnn_ep_runtime.h"
#include "runtime_types.h"

#include <cstdio>
#include <cstring>

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

// Mock N-D strided copy: walks every index on the host. Source strides may be
// negative.
int wrap_strided_copy_nd(RuntimeState *state, void *dst_ptr,
                         const void *src_ptr, int64_t elem_bytes, int64_t rank,
                         const int64_t *sizes, const int64_t *src_strides,
                         const int64_t *dst_strides) {
  if (!state || rank < 0 ||
      (rank > 0 && (!sizes || !src_strides || !dst_strides))) {
    fprintf(stderr, "wrap_strided_copy_nd: invalid argument\n");
    return -1;
  }
  int64_t total = 1;
  for (int64_t i = 0; i < rank; ++i)
    total *= sizes[i];
  MOCK_PRINT("[MOCK] wrap_strided_copy_nd(dst=%p elem=%lld rank=%lld "
             "total=%lld)\n",
             dst_ptr, (long long)elem_bytes, (long long)rank, (long long)total);
  if (total == 0)
    return 0;
  if (!dst_ptr || !src_ptr) {
    fprintf(stderr, "wrap_strided_copy_nd: null pointer\n");
    return -1;
  }
  const char *s = static_cast<const char *>(src_ptr);
  char *d = static_cast<char *>(dst_ptr);
  for (int64_t t = 0; t < total; ++t) {
    int64_t r = t, srcOff = 0, dstOff = 0;
    for (int64_t i = rank - 1; i >= 0; --i) {
      const int64_t idx = r % sizes[i];
      r /= sizes[i];
      srcOff += idx * src_strides[i];
      dstOff += idx * dst_strides[i];
    }
    memcpy(d + dstOff * elem_bytes, s + srcOff * elem_bytes, elem_bytes);
  }
  return 0;
}

// Mock concat: one pitched host copy per input.
int wrap_concat(RuntimeState *state, void *dst_ptr, int64_t num_inputs,
                void *const *src_ptrs, const int64_t *src_widths, int64_t outer,
                int64_t elem_bytes) {
  if (!state || num_inputs <= 0 || !src_ptrs || !src_widths) {
    fprintf(stderr, "wrap_concat: invalid argument\n");
    return -1;
  }
  int64_t dst_width = 0;
  for (int64_t i = 0; i < num_inputs; ++i)
    dst_width += src_widths[i];
  MOCK_PRINT("[MOCK] wrap_concat(dst=%p n=%lld outer=%lld width=%lld "
             "elem=%lld)\n",
             dst_ptr, (long long)num_inputs, (long long)outer,
             (long long)dst_width, (long long)elem_bytes);
  if (outer == 0 || dst_width == 0)
    return 0;
  if (!dst_ptr) {
    fprintf(stderr, "wrap_concat: null output\n");
    return -1;
  }
  char *d = static_cast<char *>(dst_ptr);
  int64_t col = 0;
  for (int64_t i = 0; i < num_inputs; ++i) {
    const int64_t width = src_widths[i];
    if (width == 0)
      continue;
    const char *s = static_cast<const char *>(src_ptrs[i]);
    for (int64_t row = 0; row < outer; ++row)
      memcpy(d + (row * dst_width + col) * elem_bytes,
             s + row * width * elem_bytes, width * elem_bytes);
    col += width;
  }
  return 0;
}
