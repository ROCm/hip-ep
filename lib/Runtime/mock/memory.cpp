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
  if (!dst_ptr || !src_ptr) {
    fprintf(stderr, "wrap_hipMemcpyAsync: null pointer\n");
    return -1;
  }
  if (size_bytes == 0) {
    return 0; // No-op for zero-sized copy
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
