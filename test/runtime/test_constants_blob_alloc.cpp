/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

/* GPU-free tests for the constants-blob allocation policy
 * (lib/Runtime/constants_blob_alloc.h): managed memory is a fallback for
 * device exhaustion only. The HIP entry points are faked here, so the test
 * injects hipMalloc / hipMallocManaged results without a device.
 */
#include "constants_blob_alloc.h"

#include <cstdio>

static int failures = 0;

#define CHECK(cond, msg)                                                       \
  do {                                                                         \
    if (!(cond)) {                                                             \
      std::fprintf(stderr, "FAIL: %s (%s:%d)\n", (msg), __FILE__, __LINE__);   \
      ++failures;                                                              \
    }                                                                          \
  } while (0)

namespace {
constexpr int kOtherError = 101; // e.g. an invalid-device style failure

hipError_t g_malloc_result = hipSuccess;
hipError_t g_managed_result = hipSuccess;
int g_malloc_calls = 0;
int g_managed_calls = 0;
int g_last_error_clears = 0;
char g_backing;

void reset(hipError_t malloc_result, hipError_t managed_result) {
  g_malloc_result = malloc_result;
  g_managed_result = managed_result;
  g_malloc_calls = g_managed_calls = g_last_error_clears = 0;
}
} // namespace

extern "C" hipError_t hipMalloc(void **ptr, size_t) {
  ++g_malloc_calls;
  *ptr = nullptr;
  if (g_malloc_result == hipSuccess)
    *ptr = &g_backing;
  return g_malloc_result;
}

extern "C" hipError_t hipMallocManaged(void **ptr, size_t, unsigned int) {
  ++g_managed_calls;
  *ptr = nullptr;
  if (g_managed_result == hipSuccess)
    *ptr = &g_backing;
  return g_managed_result;
}

extern "C" const char *hipGetErrorString(hipError_t) { return "fake error"; }

extern "C" hipError_t hipGetLastError() {
  ++g_last_error_clears;
  return hipSuccess;
}

static void testDeviceAllocationSucceeds() {
  reset(hipSuccess, hipSuccess);
  void *blob = nullptr;
  CHECK(hipdnn_ep::alloc_constants_blob(&blob, 4096) == 0, "hipMalloc ok");
  CHECK(blob == &g_backing, "blob comes from hipMalloc");
  CHECK(g_managed_calls == 0, "managed memory untouched on success");
}

static void testOutOfMemoryFallsBackToManaged() {
  reset(hipErrorOutOfMemory, hipSuccess);
  void *blob = nullptr;
  CHECK(hipdnn_ep::alloc_constants_blob(&blob, 4096) == 0,
        "OOM falls back to managed memory");
  CHECK(g_managed_calls == 1, "managed allocation attempted once");
  CHECK(blob == &g_backing, "blob comes from hipMallocManaged");
  CHECK(g_last_error_clears >= 1, "sticky OOM error cleared");
}

static void testOtherErrorDoesNotFallBack() {
  reset(kOtherError, hipSuccess);
  void *blob = nullptr;
  CHECK(hipdnn_ep::alloc_constants_blob(&blob, 4096) != 0,
        "a non-OOM failure is reported");
  CHECK(g_managed_calls == 0, "managed allocation not attempted");
  CHECK(blob == nullptr, "no blob on failure");
}

static void testManagedFailureIsReported() {
  reset(hipErrorOutOfMemory, hipErrorOutOfMemory);
  void *blob = nullptr;
  CHECK(hipdnn_ep::alloc_constants_blob(&blob, 4096) != 0,
        "OOM then managed failure is reported");
  CHECK(g_managed_calls == 1, "managed allocation attempted once");
  CHECK(blob == nullptr, "no blob when both allocations fail");
}

int main() {
  testDeviceAllocationSucceeds();
  testOutOfMemoryFallsBackToManaged();
  testOtherErrorDoesNotFallBack();
  testManagedFailureIsReported();
  if (failures) {
    std::fprintf(stderr, "%d check(s) failed\n", failures);
    return 1;
  }
  std::printf("constants blob allocation tests passed\n");
  return 0;
}
