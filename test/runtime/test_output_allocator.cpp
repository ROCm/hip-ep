/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===----------------------------------------------------------------------===//
// GPU-free unit test for the output allocator runtime contract.
//
// Compiles lib/Runtime/output_allocator.cpp natively against the MOCK runtime
// types (no HIP), constructs a RuntimeState on the stack, and exercises the two
// entry points directly:
//   * forwarding of (self, out_idx, shape, rank, elem_size) + the returned ptr
//   * null-guard when no allocator is installed / cleared / null state
//
// The full end-to-end path (generated hip.alloc_output + EP-installed
// allocator) is not wired up yet; this isolates the runtime contract.
//===----------------------------------------------------------------------===//

#include "hipdnn_ep_runtime.h"
#include "runtime_state_internal.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>

int runtime_errors = 0;

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

// 4. Defensive: a null RuntimeState* into the setter is a silent no-op. The EP
//    resolves the symbol by name and must never crash the model.dll on a bad
//    handle. Exercises the `!state` guard; passes if it does not dereference.
void test_set_null_state() {
  hipdnn_output_allocator_t alloc;
  alloc.self = nullptr;
  alloc.allocate = stub_allocate;
  hipdnn_ep_set_output_allocator(nullptr, &alloc); // must not touch state
}

// 5. Defensive: a null RuntimeState* into alloc_output null-guards (returns
//    null, no callback), exactly like an uninstalled allocator. Exercises the
//    `!state` half of the guard (test 2 covers the `!allocate` half).
void test_alloc_null_state() {
  g_cap = StubCapture{};
  runtime_errors = 0;
  const int64_t shape[1] = {1};
  void *p = hipdnn_ep_alloc_output(nullptr, 0, shape, 1, 4);
  CHECK(p == nullptr);
  CHECK(g_cap.calls == 0);
  CHECK(runtime_errors == 0);
}

void test_invalid_logical_shapes_record_error() {
  g_cap = StubCapture{};
  int ctx = 0;
  RuntimeState st = makeState();
  hipdnn_output_allocator_t alloc{&ctx, stub_allocate};
  hipdnn_ep_set_output_allocator(&st, &alloc);

  auto expectInvalid = [&](const int64_t *shape, int64_t rank,
                           int64_t elemSize) {
    int previousErrors = runtime_errors;
    int previousCalls = g_cap.calls;
    CHECK(hipdnn_ep_alloc_output(&st, 0, shape, rank, elemSize) == nullptr);
    CHECK(runtime_errors == previousErrors + 1);
    CHECK(g_cap.calls == previousCalls);
  };

  runtime_errors = 0;
  const int64_t negativeShape[1] = {-1};
  expectInvalid(negativeShape, 1, sizeof(float));
  expectInvalid(nullptr, 1, sizeof(float));
  expectInvalid(negativeShape, -1, sizeof(float));
  const int64_t overflowShape[2] = {std::numeric_limits<int64_t>::max(), 2};
  expectInvalid(overflowShape, 2, sizeof(int64_t));
  const int64_t validShape[1] = {1};
  expectInvalid(validShape, 1, 0);
}

void test_zero_byte_output_remains_valid() {
  g_cap = StubCapture{};
  runtime_errors = 0;
  int ctx = 0;
  RuntimeState st = makeState();
  hipdnn_output_allocator_t alloc{&ctx, stub_allocate};
  hipdnn_ep_set_output_allocator(&st, &alloc);
  const int64_t zeroShape[2] = {3, 0};
  CHECK(hipdnn_ep_alloc_output(&st, 4, zeroShape, 2, sizeof(float)) == nullptr);
  CHECK(g_cap.calls == 1);
  CHECK(runtime_errors == 0);
}

void test_safe_output_copy() {
  RuntimeState st = makeState();
  float source[6] = {0, 1, 2, 3, 4, 5};
  float target[4] = {};
  int64_t sizes[2] = {2, 2};
  int64_t strides[2] = {3, 1};
  CHECK(hipdnn_ep_copy_output(&st, target, source, 2, sizes, strides,
                              sizeof(float)) == 0);
  CHECK(target[0] == 0 && target[1] == 1);
  CHECK(target[2] == 3 && target[3] == 4);

  int64_t zeroShape[1] = {0};
  CHECK(hipdnn_ep_copy_output(&st, nullptr, nullptr, 1, zeroShape, zeroShape,
                              sizeof(float)) == 0);
}

} // namespace

extern "C" int hipdnn_ep_state_set_error_flag(RuntimeState *) {
  ++runtime_errors;
  return 0;
}

extern "C" int wrap_hipMemcpyAsync(RuntimeState *, void *dst, const void *src,
                                   size_t size_bytes) {
  if (size_bytes != 0)
    std::memcpy(dst, src, size_bytes);
  return 0;
}

int main() {
  test_forwarding();
  test_null_guard();
  test_nullptr_setter_arg();
  test_set_null_state();
  test_alloc_null_state();
  test_invalid_logical_shapes_record_error();
  test_zero_byte_output_remains_valid();
  test_safe_output_copy();
  if (g_failures == 0) {
    std::printf("output_allocator unit test: ALL PASS\n");
    return 0;
  }
  std::fprintf(stderr, "output_allocator unit test: %d FAILURE(S)\n",
               g_failures);
  return 1;
}
