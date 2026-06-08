/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===----------------------------------------------------------------------===//
// GPU-free unit test for the Phase 3 output allocator runtime contract.
//
// Compiles lib/Runtime/output_allocator.cpp natively against the MOCK runtime
// types (no HIP), constructs a RuntimeState on the stack, and exercises the two
// entry points directly:
//   * forwarding of (self, out_idx, shape, rank, elem_size) + the returned ptr
//   * null-guard when no allocator is installed
//   * the struct_size min-copy forward/backward-compat logic
//
// The full end-to-end path (generated hip.alloc_output + EP-installed
// allocator) only exists at Phase 4/5; this isolates the runtime contract.
//===----------------------------------------------------------------------===//

#include "hipdnn_ep_runtime.h"
#include "runtime_state_internal.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>

namespace {

int g_failures = 0;

#define CHECK(cond)                                                            \
  do {                                                                         \
    if (!(cond)) {                                                             \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);     \
      ++g_failures;                                                            \
    }                                                                          \
  } while (0)

// Records the arguments of the most recent allocate() call so the test can
// assert the runtime forwarded them verbatim.
struct StubCapture {
  void *self = nullptr;
  int64_t out_idx = -1;
  const int64_t *shape = nullptr;
  int64_t rank = -1;
  int64_t elem_size = -1;
  int calls = 0;
  void *ret = nullptr;
};

StubCapture g_cap;

void *stub_allocate(void *self, int64_t out_idx, const int64_t *shape,
                    int64_t rank, int64_t elem_size) {
  g_cap.self = self;
  g_cap.out_idx = out_idx;
  g_cap.shape = shape;
  g_cap.rank = rank;
  g_cap.elem_size = elem_size;
  ++g_cap.calls;
  return g_cap.ret;
}

// Fresh, zero-initialized state with no allocator installed (mirrors what
// initialize_state_handles leaves before the EP calls the setter).
RuntimeState makeState() {
  RuntimeState st{};
  st.output_allocator.struct_size = sizeof(hipdnn_output_allocator_t);
  st.output_allocator.self = nullptr;
  st.output_allocator.allocate = nullptr;
  return st;
}

// 1. Forwarding: the runtime passes every argument through unchanged and
//    returns exactly what the callback returned.
void test_forwarding() {
  g_cap = StubCapture{};
  int sentinel_ctx = 0;
  int sentinel_buf = 0;
  g_cap.ret = &sentinel_buf;

  RuntimeState st = makeState();
  hipdnn_output_allocator_t alloc;
  alloc.struct_size = sizeof(alloc);
  alloc.self = &sentinel_ctx;
  alloc.allocate = stub_allocate;
  hipdnn_ep_set_output_allocator(&st, &alloc);

  const int64_t shape[2] = {2, 3};
  void *p = hipdnn_ep_alloc_output(&st, /*out_idx=*/7, shape, /*rank=*/2,
                                   /*elem_size=*/4);
  CHECK(p == &sentinel_buf);
  CHECK(g_cap.calls == 1);
  CHECK(g_cap.self == &sentinel_ctx);
  CHECK(g_cap.out_idx == 7);
  CHECK(g_cap.shape == shape); // pointer forwarded as-is, not copied
  CHECK(g_cap.rank == 2);
  CHECK(g_cap.elem_size == 4);
}

// 2. Null-guard: no allocator installed -> returns null, no crash, no call.
void test_null_guard() {
  g_cap = StubCapture{};
  RuntimeState st = makeState();
  const int64_t shape[1] = {5};
  void *p = hipdnn_ep_alloc_output(&st, 0, shape, 1, 2);
  CHECK(p == nullptr);
  CHECK(g_cap.calls == 0);
}

// 3. A nullptr setter argument resets the slot to "none installed".
void test_nullptr_setter_arg() {
  g_cap = StubCapture{};
  int ctx = 0;
  RuntimeState st = makeState();
  hipdnn_output_allocator_t alloc;
  alloc.struct_size = sizeof(alloc);
  alloc.self = &ctx;
  alloc.allocate = stub_allocate;
  hipdnn_ep_set_output_allocator(&st, &alloc);
  hipdnn_ep_set_output_allocator(&st, nullptr); // clear it
  const int64_t shape[1] = {1};
  void *p = hipdnn_ep_alloc_output(&st, 0, shape, 1, 4);
  CHECK(p == nullptr);
  CHECK(g_cap.calls == 0);
  CHECK(st.output_allocator.allocate == nullptr);
  CHECK(st.output_allocator.self == nullptr);
}

// 4. struct_size min-copy, NEWER caller: caller advertises a larger struct (as
//    if it had extra trailing fields). The setter must clamp the copy to our
//    sizeof, still install the callback we DO understand, and normalize
//    struct_size to our layout.
void test_struct_size_newer_caller() {
  g_cap = StubCapture{};
  int ctx = 0;
  int buf = 0;
  g_cap.ret = &buf;
  RuntimeState st = makeState();
  hipdnn_output_allocator_t alloc;
  alloc.struct_size = sizeof(alloc) + 64; // pretend newer / larger
  alloc.self = &ctx;
  alloc.allocate = stub_allocate;
  hipdnn_ep_set_output_allocator(&st, &alloc);
  CHECK(st.output_allocator.allocate == stub_allocate);
  CHECK(st.output_allocator.self == &ctx);
  CHECK(st.output_allocator.struct_size == sizeof(hipdnn_output_allocator_t));
  const int64_t shape[1] = {3};
  void *p = hipdnn_ep_alloc_output(&st, 1, shape, 1, 2);
  CHECK(p == &buf);
  CHECK(g_cap.calls == 1);
}

// 5. struct_size min-copy, OLDER caller: caller advertises a struct that ends
//    before `allocate` (only struct_size + self present). The setter must not
//    read past the advertised prefix, so `allocate` stays null -> alloc_output
//    null-guards.
void test_struct_size_older_caller() {
  g_cap = StubCapture{};
  int ctx = 0;
  RuntimeState st = makeState();
  hipdnn_output_allocator_t alloc;
  // Advertise only up to (not including) the allocate field.
  alloc.struct_size = offsetof(hipdnn_output_allocator_t, allocate);
  alloc.self = &ctx;
  alloc.allocate = stub_allocate; // present in memory but OUTSIDE the prefix
  hipdnn_ep_set_output_allocator(&st, &alloc);
  CHECK(st.output_allocator.self == &ctx);        // self IS within the prefix
  CHECK(st.output_allocator.allocate == nullptr); // allocate is NOT copied
  const int64_t shape[1] = {1};
  void *p = hipdnn_ep_alloc_output(&st, 0, shape, 1, 4);
  CHECK(p == nullptr);
  CHECK(g_cap.calls == 0);
}

// 6. Defensive: a null RuntimeState* into the setter is a silent no-op. The EP
//    resolves the symbol by name and must never crash the model.dll on a bad
//    handle. Exercises the `!state` guard; passes if it does not dereference.
void test_set_null_state() {
  hipdnn_output_allocator_t alloc;
  alloc.struct_size = sizeof(alloc);
  alloc.self = nullptr;
  alloc.allocate = stub_allocate;
  hipdnn_ep_set_output_allocator(nullptr, &alloc); // must not touch state
}

// 7. Defensive: a null RuntimeState* into alloc_output null-guards (returns
//    null, no callback), exactly like an uninstalled allocator. Exercises the
//    `!state` half of the guard (test 2 covers the `!allocate` half).
void test_alloc_null_state() {
  g_cap = StubCapture{};
  const int64_t shape[1] = {1};
  void *p = hipdnn_ep_alloc_output(nullptr, 0, shape, 1, 4);
  CHECK(p == nullptr);
  CHECK(g_cap.calls == 0);
}

} // namespace

int main() {
  test_forwarding();
  test_null_guard();
  test_nullptr_setter_arg();
  test_struct_size_newer_caller();
  test_struct_size_older_caller();
  test_set_null_state();
  test_alloc_null_state();
  if (g_failures == 0) {
    std::printf("output_allocator unit test: ALL PASS\n");
    return 0;
  }
  std::fprintf(stderr, "output_allocator unit test: %d FAILURE(S)\n",
               g_failures);
  return 1;
}
