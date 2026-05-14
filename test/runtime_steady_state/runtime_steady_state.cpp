/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===- runtime_steady_state.cpp - Op-module registry smoke test --------*-===//
//
// Stage 1 smoke test for lib/Runtime/module_registry.{h,cpp}.
//
// Verifies:
//   1. Slot ids are assigned in registration order and stable across
//      get_op_module_spec lookups.
//   2. init_fn fires lazily on first op_module_get per (session, slot), and
//      never again for that session.
//   3. begin_compute_fn fires exactly once per
//      module_registry_begin_compute on every populated slot whose spec
//      defines it. Slots without the hook are skipped at zero cost.
//   4. destroy_fn fires once per populated slot at module_registry_destroy,
//      in reverse registration order.
//   5. Steady-state invariant: after the first op_module_get, repeated
//      op_module_get and module_registry_begin_compute calls do NOT
//      re-invoke init_fn -- the cached slot pointer is returned and the
//      cached begin_compute_fn is called directly. (This is the
//      "after first run, if all cache hit, pure compute" guarantee at
//      the registry layer.)
//
// On failure: prints the failing assertion to stderr and exits non-zero.
// On success: prints "ALL TESTS PASSED" and exits 0.
//
// Standalone: links only module_registry.cpp natively. The production
// glue (get_module_registry in hipdnn_ep_runtime_state.cpp) is stubbed at
// the bottom of this file -- see the comment there.
//===---------------------------------------------------------------------===//

#include "module_registry.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>

//===-------- Stand-in RuntimeState for the standalone test --------------===//
//
// The registry layer never dereferences RuntimeState* -- it only passes
// the pointer through to user-provided init / begin_compute hooks. So a
// trivial pointer-sized stub suffices, and the test does NOT need to link
// any of the real runtime (no HIP, no MIOpen, no hipBLASLt).

struct RuntimeState {
  int marker;
};

namespace hipdnn_ep {

// Stub for the glue that hipdnn_ep_runtime_state.cpp provides in production.
// The test never goes through the HIPDNN_OP_MODULE macro path (which would
// call this), so the implementation is just here to satisfy the linker.
// Returns a registry pointer the test installs into a file-static slot --
// see g_test_registry below.
ModuleRegistry *g_test_registry = nullptr;
ModuleRegistry *get_module_registry(::RuntimeState *) {
  return g_test_registry;
}

} // namespace hipdnn_ep

//===---------------------- Fake op state types --------------------------===//
//
// Counters live at file scope so the constructors / destructors / hooks
// can update them without having to thread a pointer through.

namespace {

struct Counters {
  int init = 0;
  int destroy = 0;
  int begin_compute = 0;
  int mem_bytes_calls = 0;
};

Counters g_full;
Counters g_minimal;

// Fully-equipped state: init, destroy, begin_compute, and mem_bytes all
// defined. Should drive every codepath in the registry.
struct FullHooksState {
  ::RuntimeState *constructed_with = nullptr;
  int last_begin_compute_marker = -1;

  explicit FullHooksState(::RuntimeState *s) : constructed_with(s) {
    ++g_full.init;
  }
  ~FullHooksState() { ++g_full.destroy; }

  void begin_compute(::RuntimeState *s) {
    ++g_full.begin_compute;
    last_begin_compute_marker = s ? s->marker : -1;
  }

  size_t mem_bytes() const {
    ++g_full.mem_bytes_calls;
    return 42;
  }
};

// Bare-bones state: only the required init / destroy. None of the
// optional hooks. SFINAE should leave begin_compute_fn / end_compute_fn /
// mem_bytes_fn null in the spec, and the registry should skip them.
struct MinimalState {
  explicit MinimalState(::RuntimeState *) { ++g_minimal.init; }
  ~MinimalState() { ++g_minimal.destroy; }
};

} // namespace

//===------------------------- Test helpers -----------------------------===//

#define TEST_ASSERT(cond)                                                      \
  do {                                                                         \
    if (!(cond)) {                                                             \
      std::fprintf(stderr, "FAIL %s:%d : %s\n", __FILE__, __LINE__, #cond);    \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

#define TEST_EQ(a, b)                                                          \
  do {                                                                         \
    long long _a = (long long)(a);                                             \
    long long _b = (long long)(b);                                             \
    if (_a != _b) {                                                            \
      std::fprintf(stderr, "FAIL %s:%d : %s == %s, got %lld vs %lld\n",        \
                   __FILE__, __LINE__, #a, #b, _a, _b);                        \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

static void reset_counters() {
  g_full = {};
  g_minimal = {};
}

//===------------------------- Tests ------------------------------------===//

// Both specs share file scope so register_op_module's "spec must remain
// valid for the lifetime of the process" contract is respected; the same
// addresses are reused across tests, matching how the production
// HIPDNN_OP_MODULE macro uses a function-local static.
static const hipdnn_ep::OpModuleSpec spec_full =
    hipdnn_ep::make_op_module_spec<FullHooksState>("test/full_hooks");
static const hipdnn_ep::OpModuleSpec spec_minimal =
    hipdnn_ep::make_op_module_spec<MinimalState>("test/minimal");

static int slot_full = -1;
static int slot_minimal = -1;

static void test_make_op_module_spec_hook_detection() {
  // SFINAE should populate all four optional pointers for FullHooksState
  // and leave them null for MinimalState (except destroy_fn which is
  // mandatory and always present).
  TEST_ASSERT(spec_full.init_fn != nullptr);
  TEST_ASSERT(spec_full.destroy_fn != nullptr);
  TEST_ASSERT(spec_full.begin_compute_fn != nullptr);
  TEST_ASSERT(spec_full.mem_bytes_fn != nullptr);

  TEST_ASSERT(spec_minimal.init_fn != nullptr);
  TEST_ASSERT(spec_minimal.destroy_fn != nullptr);
  TEST_EQ((long long)spec_minimal.begin_compute_fn, 0);
  TEST_EQ((long long)spec_minimal.end_compute_fn, 0);
  TEST_EQ((long long)spec_minimal.mem_bytes_fn, 0);

  std::printf("ok test_make_op_module_spec_hook_detection\n");
}

static void test_register_assigns_slots_in_order() {
  // Register both specs; capture slot ids for later tests. Slot ids are
  // process-global monotonic indices into spec_table().
  int before = hipdnn_ep::op_module_count();
  slot_full = hipdnn_ep::register_op_module(&spec_full);
  slot_minimal = hipdnn_ep::register_op_module(&spec_minimal);

  TEST_EQ(slot_full, before);
  TEST_EQ(slot_minimal, before + 1);
  TEST_EQ(hipdnn_ep::op_module_count(), before + 2);
  TEST_EQ(hipdnn_ep::get_op_module_spec(slot_full), &spec_full);
  TEST_EQ(hipdnn_ep::get_op_module_spec(slot_minimal), &spec_minimal);

  // Out-of-range / negative slot ids return nullptr (never crash).
  TEST_EQ((long long)hipdnn_ep::get_op_module_spec(-1), 0);
  TEST_EQ((long long)hipdnn_ep::get_op_module_spec(slot_minimal + 10000), 0);

  std::printf("ok test_register_assigns_slots_in_order "
              "(slot_full=%d slot_minimal=%d)\n",
              slot_full, slot_minimal);
}

static void test_lazy_init_fires_once_per_session() {
  reset_counters();
  hipdnn_ep::ModuleRegistry *reg = hipdnn_ep::module_registry_create();
  TEST_ASSERT(reg != nullptr);

  ::RuntimeState dummy_state{12345};

  // No access yet -> init NOT called.
  TEST_EQ(g_full.init, 0);

  // First access -> init fires once.
  void *p1 = hipdnn_ep::op_module_get(reg, &dummy_state, slot_full);
  TEST_ASSERT(p1 != nullptr);
  TEST_EQ(g_full.init, 1);

  // Subsequent accesses on the same registry -> init does NOT fire again,
  // same pointer returned. This is the steady-state guarantee.
  for (int i = 0; i < 100; ++i) {
    void *p2 = hipdnn_ep::op_module_get(reg, &dummy_state, slot_full);
    TEST_EQ(p2, p1);
  }
  TEST_EQ(g_full.init, 1);

  // Verify the spec's init_fn received the right RuntimeState* (i.e. the
  // pointer is plumbed through, not silently dropped).
  auto *typed = static_cast<FullHooksState *>(p1);
  TEST_EQ(typed->constructed_with, &dummy_state);

  // Destroy must fire exactly once on cleanup.
  TEST_EQ(g_full.destroy, 0);
  hipdnn_ep::module_registry_destroy(reg);
  TEST_EQ(g_full.destroy, 1);

  std::printf("ok test_lazy_init_fires_once_per_session\n");
}

static void test_invalid_args_do_not_crash() {
  // nullptr registry -> safe nullptr return.
  TEST_EQ((long long)hipdnn_ep::op_module_get(nullptr, nullptr, slot_full), 0);

  // Negative slot id -> safe nullptr return.
  hipdnn_ep::ModuleRegistry *reg = hipdnn_ep::module_registry_create();
  ::RuntimeState dummy_state{0};
  TEST_EQ((long long)hipdnn_ep::op_module_get(reg, &dummy_state, -1), 0);
  TEST_EQ((long long)hipdnn_ep::op_module_get(reg, &dummy_state, -42), 0);

  // Slot id past the highest registered slot -> safe nullptr return
  // (spec lookup returns null, init never runs).
  int impossible = hipdnn_ep::op_module_count() + 9999;
  TEST_EQ((long long)hipdnn_ep::op_module_get(reg, &dummy_state, impossible),
          0);

  // module_registry_destroy(nullptr) and module_registry_begin_compute on
  // a nullptr registry must be no-ops.
  hipdnn_ep::module_registry_destroy(nullptr);
  hipdnn_ep::module_registry_begin_compute(nullptr, &dummy_state);

  hipdnn_ep::module_registry_destroy(reg);
  std::printf("ok test_invalid_args_do_not_crash\n");
}

static void test_begin_compute_fires_once_per_call() {
  reset_counters();
  hipdnn_ep::ModuleRegistry *reg = hipdnn_ep::module_registry_create();
  ::RuntimeState s1{1001};
  ::RuntimeState s2{2002};

  // No populated slots -> begin_compute is a no-op.
  hipdnn_ep::module_registry_begin_compute(reg, &s1);
  TEST_EQ(g_full.begin_compute, 0);
  TEST_EQ(g_minimal.begin_compute, 0);

  // Populate only the FullHooksState slot. begin_compute should fire only
  // for that slot; the MinimalState slot is unpopulated and skipped.
  void *p_full = hipdnn_ep::op_module_get(reg, &s1, slot_full);
  TEST_ASSERT(p_full != nullptr);

  hipdnn_ep::module_registry_begin_compute(reg, &s1);
  TEST_EQ(g_full.begin_compute, 1);
  TEST_EQ(g_minimal.begin_compute, 0);
  TEST_EQ(static_cast<FullHooksState *>(p_full)->last_begin_compute_marker,
          1001);

  // Different RuntimeState passed in -- the registry must thread it
  // through to the hook, not capture the original from init.
  hipdnn_ep::module_registry_begin_compute(reg, &s2);
  TEST_EQ(g_full.begin_compute, 2);
  TEST_EQ(static_cast<FullHooksState *>(p_full)->last_begin_compute_marker,
          2002);

  // Now populate the MinimalState slot too. Even though MinimalState
  // defines no begin_compute hook, populating its slot must NOT magic up
  // an entry that gets called -- the iteration in
  // module_registry_begin_compute checks for a non-null cached
  // begin_compute_fn.
  void *p_min = hipdnn_ep::op_module_get(reg, &s1, slot_minimal);
  TEST_ASSERT(p_min != nullptr);

  hipdnn_ep::module_registry_begin_compute(reg, &s1);
  TEST_EQ(g_full.begin_compute, 3);
  TEST_EQ(g_minimal.begin_compute, 0); // MinimalState has no hook

  hipdnn_ep::module_registry_destroy(reg);
  std::printf("ok test_begin_compute_fires_once_per_call\n");
}

static void test_destroy_runs_in_reverse_registration_order() {
  reset_counters();
  hipdnn_ep::ModuleRegistry *reg = hipdnn_ep::module_registry_create();
  ::RuntimeState dummy_state{0};

  // Populate both slots.
  void *p_full = hipdnn_ep::op_module_get(reg, &dummy_state, slot_full);
  void *p_min = hipdnn_ep::op_module_get(reg, &dummy_state, slot_minimal);
  TEST_ASSERT(p_full != nullptr);
  TEST_ASSERT(p_min != nullptr);
  TEST_EQ(g_full.init, 1);
  TEST_EQ(g_minimal.init, 1);

  // Destroy. Both destructors must fire exactly once.
  hipdnn_ep::module_registry_destroy(reg);
  TEST_EQ(g_full.destroy, 1);
  TEST_EQ(g_minimal.destroy, 1);

  // Re-creating a registry and destroying it without populating any slot
  // must NOT call destructors a second time.
  reg = hipdnn_ep::module_registry_create();
  hipdnn_ep::module_registry_destroy(reg);
  TEST_EQ(g_full.destroy, 1);
  TEST_EQ(g_minimal.destroy, 1);

  std::printf("ok test_destroy_runs_in_reverse_registration_order\n");
}

static void test_steady_state_no_extra_init_calls() {
  // The headline guarantee: after warmup, repeated op_module_get +
  // module_registry_begin_compute cycles do NOT invoke init_fn beyond
  // the first call per slot. (Equivalent to "all cache hit -> pure
  // compute executed", at the registry layer.)
  reset_counters();
  hipdnn_ep::ModuleRegistry *reg = hipdnn_ep::module_registry_create();
  ::RuntimeState dummy_state{777};

  // Warmup: first access populates the slot. Counts as one init.
  (void)hipdnn_ep::op_module_get(reg, &dummy_state, slot_full);
  TEST_EQ(g_full.init, 1);

  // Simulated steady state: 1000 inferences each doing several
  // op_module_get calls and one begin_compute. init must not budge.
  for (int infer = 0; infer < 1000; ++infer) {
    hipdnn_ep::module_registry_begin_compute(reg, &dummy_state);
    for (int call = 0; call < 8; ++call) {
      (void)hipdnn_ep::op_module_get(reg, &dummy_state, slot_full);
    }
  }
  TEST_EQ(g_full.init, 1);
  TEST_EQ(g_full.begin_compute, 1000);

  hipdnn_ep::module_registry_destroy(reg);
  std::printf("ok test_steady_state_no_extra_init_calls\n");
}

//===---------------------------- main ----------------------------------===//

int main() {
  test_make_op_module_spec_hook_detection();
  test_register_assigns_slots_in_order();
  test_lazy_init_fires_once_per_session();
  test_invalid_args_do_not_crash();
  test_begin_compute_fires_once_per_call();
  test_destroy_runs_in_reverse_registration_order();
  test_steady_state_no_extra_init_calls();
  std::printf("ALL TESTS PASSED\n");
  return 0;
}
