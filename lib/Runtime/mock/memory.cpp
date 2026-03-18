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

// Mock HIP memory copy wrapper (just uses memcpy)
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
