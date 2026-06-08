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

// EP-side pure helper (GPU-free, header-only) shared by the allocator callback
// and the classic marshal path. Tested here so the OGA share-buffer override
// rules have direct coverage without a GPU / ORT session.
#include "output_shape_override.h"

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
  const int64_t shape[1] = {1};
  void *p = hipdnn_ep_alloc_output(nullptr, 0, shape, 1, 4);
  CHECK(p == nullptr);
  CHECK(g_cap.calls == 0);
}

// ---------------------------------------------------------------------------
// EP-side pure helper: apply_present_share_buffer_override. GPU-free, so it is
// exercised in this same binary. These mirror the OGA past_present_share_buffer
// rules that BOTH the allocator callback and the classic marshal path rely on;
// the "multiple shapes" case stands in for a single dynamic-shape EP session
// seeing several shapes back to back without recompilation.
// ---------------------------------------------------------------------------
using mlir_compilation::apply_present_share_buffer_override;

// 8. Dynamic present seq dim bumped up to the larger past (max_length) buffer.
void test_override_basic_dynamic() {
  const int64_t compiled[4] = {1, 8, -1, 64};
  const int64_t past[4] = {1, 8, 128, 64};
  int64_t out[4] = {1, 8, 7, 64};
  bool changed = apply_present_share_buffer_override(compiled, past, out, 4, 4);
  CHECK(changed);
  CHECK(out[2] == 128); // dynamic dim raised to past
  CHECK(out[0] == 1 && out[1] == 8 && out[3] == 64);
}

// 9. Static dims are NEVER overridden, even if past differs.
void test_override_static_dim_untouched() {
  const int64_t compiled[4] = {1, 8, 128, 64}; // dim 2 static in compiled
  const int64_t past[4] = {1, 8, 256, 64};
  int64_t out[4] = {1, 8, 7, 64};
  bool changed = apply_present_share_buffer_override(compiled, past, out, 4, 4);
  CHECK(!changed);
  CHECK(out[2] == 7); // untouched: static dim guard
}

// 10. Separate-buffer mode (past not strictly larger) -> no override.
void test_override_past_not_larger() {
  const int64_t compiled[4] = {1, 8, -1, 64};
  const int64_t past[4] = {1, 8, 128, 64};
  int64_t out[4] = {1, 8, 200, 64};
  bool changed = apply_present_share_buffer_override(compiled, past, out, 4, 4);
  CHECK(!changed);
  CHECK(out[2] == 200);
}

// 11. Rank mismatch -> defensive no-op.
void test_override_rank_mismatch() {
  const int64_t compiled[4] = {1, 8, -1, 64};
  const int64_t past[3] = {1, 8, 128};
  int64_t out[4] = {1, 8, 7, 64};
  bool changed = apply_present_share_buffer_override(compiled, past, out, 4, 3);
  CHECK(!changed);
  CHECK(out[2] == 7);
}

// 12. Two dynamic dims both overridden; static dims preserved.
void test_override_multi_dynamic_dims() {
  const int64_t compiled[4] = {-1, 8, -1, 64};
  const int64_t past[4] = {5, 8, 128, 64};
  int64_t out[4] = {3, 8, 7, 64};
  bool changed = apply_present_share_buffer_override(compiled, past, out, 4, 4);
  CHECK(changed);
  CHECK(out[0] == 5 && out[2] == 128); // both dynamic dims raised
  CHECK(out[1] == 8 && out[3] == 64);  // static dims preserved
}

// 13. One dynamic-shape session, several shapes back to back. The helper is
//     pure / stateless, so each shape resolves independently regardless of any
//     prior call (decode seq=1, prefill seq=128, long-context seq=64, and a
//     separate-buffer case where past is smaller).
void test_override_multiple_shapes_one_session() {
  const int64_t compiled[4] = {1, 8, -1, 64};
  struct Case {
    int64_t out_seq, past_seq, expect;
  };
  const Case cases[] = {
      {1, 128, 128},    // decode into a 128 buffer
      {128, 256, 256},  // prefill into a 256 buffer
      {64, 2048, 2048}, // long-context into a 2048 buffer
      {300, 256, 300},  // past smaller -> no override (separate buffer)
  };
  for (const Case &c : cases) {
    const int64_t past[4] = {1, 8, c.past_seq, 64};
    int64_t out[4] = {1, 8, c.out_seq, 64};
    apply_present_share_buffer_override(compiled, past, out, 4, 4);
    CHECK(out[2] == c.expect);
    CHECK(out[0] == 1 && out[1] == 8 && out[3] == 64);
  }
}

} // namespace

int main() {
  test_forwarding();
  test_null_guard();
  test_nullptr_setter_arg();
  test_set_null_state();
  test_alloc_null_state();
  test_override_basic_dynamic();
  test_override_static_dim_untouched();
  test_override_past_not_larger();
  test_override_rank_mismatch();
  test_override_multi_dynamic_dims();
  test_override_multiple_shapes_one_session();
  if (g_failures == 0) {
    std::printf("output_allocator unit test: ALL PASS\n");
    return 0;
  }
  std::fprintf(stderr, "output_allocator unit test: %d FAILURE(S)\n",
               g_failures);
  return 1;
}
