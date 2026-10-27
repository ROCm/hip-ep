/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===----------------------------------------------------------------------===//
// GPU-free unit test for the plugin-visible RuntimeState accessors
// (hip/Runtime/PluginAPI.h, lib/Runtime/runtime_plugin_api.cpp).
//
// Compiles runtime_plugin_api.cpp natively against the MOCK runtime types and
// supplies its own HIP calls below, backed by host memory, so each accessor's
// contract is checked directly: null-state guards, the session stream, the
// grow-only workspace and DP4A scratch (exact first allocation, then at least
// 1.5x, synchronizing before the old buffer is freed), and the device error
// flag. test_plugin_api_c.c compiles the same header as C.
//===----------------------------------------------------------------------===//

#include "hip/Runtime/PluginAPI.h"
#include "runtime_state_internal.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

extern "C" int plugin_api_c_entry_points(void);

namespace {

int g_failures = 0;

#define CHECK(cond)                                                            \
  do {                                                                         \
    if (!(cond)) {                                                             \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);     \
      ++g_failures;                                                            \
    }                                                                          \
  } while (0)

// What the accessors asked of HIP.
struct HipCalls {
  int mallocs = 0;
  size_t last_malloc = 0;
  int frees = 0;
  int syncs = 0;
  int memsets = 0;
  bool fail_malloc = false;
} g_hip;

RuntimeState makeState() {
  RuntimeState st{};
  st.stream = reinterpret_cast<hipStream_t>(static_cast<uintptr_t>(0x5eed));
  return st;
}

void testNullState() {
  CHECK(hipdnn_ep_state_get_stream(nullptr) == nullptr);
  CHECK(hipdnn_ep_state_get_workspace(nullptr) == nullptr);
  CHECK(hipdnn_ep_state_get_workspace_size(nullptr) == 0);
  CHECK(hipdnn_ep_state_ensure_workspace(nullptr, 16) == -1);
  CHECK(hipdnn_ep_state_get_matmul_dp4a_scratch(nullptr) == nullptr);
  CHECK(hipdnn_ep_state_get_matmul_dp4a_scratch_size(nullptr) == 0);
  CHECK(hipdnn_ep_state_ensure_matmul_dp4a_scratch(nullptr, 16) == -1);
  CHECK(hipdnn_ep_state_get_error_flag_device_ptr(nullptr) == nullptr);
  CHECK(hipdnn_ep_state_reset_error_flag(nullptr) == -1);
  CHECK(hipdnn_ep_state_read_and_clear_error_flag(nullptr) == -1);
}

void testStream() {
  RuntimeState st = makeState();
  CHECK(hipdnn_ep_state_get_stream(&st) == st.stream);
}

// The workspace and the DP4A scratch share one growth policy.
template <typename Get, typename Size, typename Ensure>
void testGrowth(RuntimeState &st, Get get, Size size, Ensure ensure) {
  g_hip = HipCalls{};
  CHECK(ensure(&st, 0) == 0); // nothing requested: nothing allocated
  CHECK(g_hip.mallocs == 0 && get(&st) == nullptr && size(&st) == 0);

  CHECK(ensure(&st, 100) == 0); // first allocation is exact
  CHECK(g_hip.mallocs == 1 && g_hip.last_malloc == 100);
  CHECK(get(&st) != nullptr && size(&st) == 100);
  void *first = get(&st);

  CHECK(ensure(&st, 60) == 0); // never shrinks
  CHECK(g_hip.mallocs == 1 && get(&st) == first && size(&st) == 100);

  CHECK(ensure(&st, 120) == 0); // grows to at least 1.5x
  CHECK(g_hip.syncs == 1 && g_hip.frees == 1);
  CHECK(g_hip.mallocs == 2 && g_hip.last_malloc == 150 && size(&st) == 150);

  CHECK(ensure(&st, 400) == 0); // or to the request when that is larger
  CHECK(g_hip.last_malloc == 400 && size(&st) == 400);
}

void testWorkspace() {
  RuntimeState st = makeState();
  testGrowth(st, hipdnn_ep_state_get_workspace,
             hipdnn_ep_state_get_workspace_size,
             hipdnn_ep_state_ensure_workspace);
  std::free(st.workspace);
}

void testDp4aScratch() {
  RuntimeState st = makeState();
  testGrowth(st, hipdnn_ep_state_get_matmul_dp4a_scratch,
             hipdnn_ep_state_get_matmul_dp4a_scratch_size,
             hipdnn_ep_state_ensure_matmul_dp4a_scratch);
  // A failed allocation reports -1 and leaves no stale buffer behind.
  g_hip.fail_malloc = true;
  CHECK(hipdnn_ep_state_ensure_matmul_dp4a_scratch(&st, 1000) == -1);
  CHECK(hipdnn_ep_state_get_matmul_dp4a_scratch(&st) == nullptr);
  CHECK(hipdnn_ep_state_get_matmul_dp4a_scratch_size(&st) == 0);
  g_hip.fail_malloc = false;
}

void testErrorFlag() {
  int flag = 0;
  RuntimeState st = makeState();
  st.device_error_flag = &flag;
  CHECK(hipdnn_ep_state_get_error_flag_device_ptr(&st) == &flag);

  flag = 5;
  CHECK(hipdnn_ep_state_reset_error_flag(&st) == 0);
  CHECK(flag == 0);

  // A clear flag reads as 0.
  CHECK(hipdnn_ep_state_read_and_clear_error_flag(&st) == 0);
  CHECK(flag == 0);

  // A raised flag is returned and left raised until the next reset: the
  // behaviour the accessors had before they moved to the plugin API.
  flag = 7;
  CHECK(hipdnn_ep_state_read_and_clear_error_flag(&st) == 7);
  CHECK(flag == 7);
  CHECK(hipdnn_ep_state_read_and_clear_error_flag(&st) == 7);
  CHECK(hipdnn_ep_state_reset_error_flag(&st) == 0);
  CHECK(hipdnn_ep_state_read_and_clear_error_flag(&st) == 0);

  // Without a flag or a stream there is nothing to read.
  RuntimeState no_flag = makeState();
  CHECK(hipdnn_ep_state_reset_error_flag(&no_flag) == -1);
  CHECK(hipdnn_ep_state_read_and_clear_error_flag(&no_flag) == -1);
  RuntimeState no_stream = makeState();
  no_stream.stream = nullptr;
  no_stream.device_error_flag = &flag;
  CHECK(hipdnn_ep_state_reset_error_flag(&no_stream) == -1);
  CHECK(hipdnn_ep_state_read_and_clear_error_flag(&no_stream) == -1);
}

} // namespace

// HIP, backed by host memory, for the accessors under test.
extern "C" {
hipError_t hipMalloc(void **ptr, size_t size) {
  ++g_hip.mallocs;
  g_hip.last_malloc = size;
  if (g_hip.fail_malloc) {
    *ptr = nullptr;
    return 2;
  }
  *ptr = std::malloc(size);
  return *ptr ? hipSuccess : 2;
}
hipError_t hipFree(void *ptr) {
  ++g_hip.frees;
  std::free(ptr);
  return hipSuccess;
}
hipError_t hipStreamSynchronize(hipStream_t) {
  ++g_hip.syncs;
  return hipSuccess;
}
hipError_t hipMemsetAsync(void *dst, int value, size_t size, hipStream_t) {
  ++g_hip.memsets;
  std::memset(dst, value, size);
  return hipSuccess;
}
hipError_t hipMemcpyAsync(void *dst, const void *src, size_t size, int,
                          hipStream_t) {
  std::memcpy(dst, src, size);
  return hipSuccess;
}
} // extern "C"

int main() {
  testNullState();
  testStream();
  testWorkspace();
  testDp4aScratch();
  testErrorFlag();
  CHECK(plugin_api_c_entry_points() == 10);
  if (g_failures) {
    std::fprintf(stderr, "%d check(s) failed\n", g_failures);
    return 1;
  }
  std::printf("plugin API accessors: all checks passed\n");
  return 0;
}
