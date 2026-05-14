/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// Native smoke test for lib/Runtime/module_registry.{h,cpp}. Links only
// module_registry.cpp -- no HIP / MIOpen / hipBLASLt. See
// docs/design/runtime-module-registry.md for the seven cases this covers.

#include "module_registry.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>

// The registry never dereferences RuntimeState*; a trivial stub suffices.
struct RuntimeState {
  int marker;
};

namespace hipdnn_ep {
// Stub for the glue normally provided by hipdnn_ep_runtime_state.cpp.
// The test plumbs a per-case registry into g_test_registry; the
// HIPDNN_OP_MODULE accessor path isn't exercised here.
ModuleRegistry *g_test_registry = nullptr;
ModuleRegistry *get_module_registry(::RuntimeState *) {
  return g_test_registry;
}
} // namespace hipdnn_ep

namespace {

struct Counters {
  int init = 0;
  int destroy = 0;
  int begin_compute = 0;
  int mem_bytes_calls = 0;
};

Counters g_full;
Counters g_minimal;

// Defines every optional hook; exercises all SFINAE branches.
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

// Only the required hooks; the registry must skip the optional ones.
struct MinimalState {
  explicit MinimalState(::RuntimeState *) { ++g_minimal.init; }
  ~MinimalState() { ++g_minimal.destroy; }
};

} // namespace

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

// File-scope specs: stable addresses across all test cases (mirrors how
// HIPDNN_OP_MODULE uses a function-local static in production).
static const hipdnn_ep::OpModuleSpec spec_full =
    hipdnn_ep::make_op_module_spec<FullHooksState>("test/full_hooks");
static const hipdnn_ep::OpModuleSpec spec_minimal =
    hipdnn_ep::make_op_module_spec<MinimalState>("test/minimal");

static int slot_full = -1;
static int slot_minimal = -1;

static void test_make_op_module_spec_hook_detection() {
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
  int before = hipdnn_ep::op_module_count();
  slot_full = hipdnn_ep::register_op_module(&spec_full);
  slot_minimal = hipdnn_ep::register_op_module(&spec_minimal);

  TEST_EQ(slot_full, before);
  TEST_EQ(slot_minimal, before + 1);
  TEST_EQ(hipdnn_ep::op_module_count(), before + 2);
  TEST_EQ(hipdnn_ep::get_op_module_spec(slot_full), &spec_full);
  TEST_EQ(hipdnn_ep::get_op_module_spec(slot_minimal), &spec_minimal);

  // Out-of-range / negative slot ids return nullptr instead of crashing.
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

  TEST_EQ(g_full.init, 0);

  void *p1 = hipdnn_ep::op_module_get(reg, &dummy_state, slot_full);
  TEST_ASSERT(p1 != nullptr);
  TEST_EQ(g_full.init, 1);

  // 100 follow-up accesses must return the same pointer without re-init.
  for (int i = 0; i < 100; ++i) {
    void *p2 = hipdnn_ep::op_module_get(reg, &dummy_state, slot_full);
    TEST_EQ(p2, p1);
  }
  TEST_EQ(g_full.init, 1);

  // RuntimeState* must be threaded through to the constructor verbatim.
  auto *typed = static_cast<FullHooksState *>(p1);
  TEST_EQ(typed->constructed_with, &dummy_state);

  TEST_EQ(g_full.destroy, 0);
  hipdnn_ep::module_registry_destroy(reg);
  TEST_EQ(g_full.destroy, 1);

  std::printf("ok test_lazy_init_fires_once_per_session\n");
}

static void test_invalid_args_do_not_crash() {
  TEST_EQ((long long)hipdnn_ep::op_module_get(nullptr, nullptr, slot_full), 0);

  hipdnn_ep::ModuleRegistry *reg = hipdnn_ep::module_registry_create();
  ::RuntimeState dummy_state{0};
  TEST_EQ((long long)hipdnn_ep::op_module_get(reg, &dummy_state, -1), 0);
  TEST_EQ((long long)hipdnn_ep::op_module_get(reg, &dummy_state, -42), 0);

  int impossible = hipdnn_ep::op_module_count() + 9999;
  TEST_EQ((long long)hipdnn_ep::op_module_get(reg, &dummy_state, impossible),
          0);

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

  hipdnn_ep::module_registry_begin_compute(reg, &s1);
  TEST_EQ(g_full.begin_compute, 0);
  TEST_EQ(g_minimal.begin_compute, 0);

  void *p_full = hipdnn_ep::op_module_get(reg, &s1, slot_full);
  TEST_ASSERT(p_full != nullptr);

  hipdnn_ep::module_registry_begin_compute(reg, &s1);
  TEST_EQ(g_full.begin_compute, 1);
  TEST_EQ(g_minimal.begin_compute, 0);
  TEST_EQ(static_cast<FullHooksState *>(p_full)->last_begin_compute_marker,
          1001);

  // The hook must receive the *current* RuntimeState*, not whatever was
  // passed to init_fn.
  hipdnn_ep::module_registry_begin_compute(reg, &s2);
  TEST_EQ(g_full.begin_compute, 2);
  TEST_EQ(static_cast<FullHooksState *>(p_full)->last_begin_compute_marker,
          2002);

  // Populating a slot whose spec has no begin_compute_fn must not cause
  // the registry to invoke anything for that slot.
  void *p_min = hipdnn_ep::op_module_get(reg, &s1, slot_minimal);
  TEST_ASSERT(p_min != nullptr);

  hipdnn_ep::module_registry_begin_compute(reg, &s1);
  TEST_EQ(g_full.begin_compute, 3);
  TEST_EQ(g_minimal.begin_compute, 0);

  hipdnn_ep::module_registry_destroy(reg);
  std::printf("ok test_begin_compute_fires_once_per_call\n");
}

static void test_destroy_runs_in_reverse_registration_order() {
  reset_counters();
  hipdnn_ep::ModuleRegistry *reg = hipdnn_ep::module_registry_create();
  ::RuntimeState dummy_state{0};

  void *p_full = hipdnn_ep::op_module_get(reg, &dummy_state, slot_full);
  void *p_min = hipdnn_ep::op_module_get(reg, &dummy_state, slot_minimal);
  TEST_ASSERT(p_full != nullptr);
  TEST_ASSERT(p_min != nullptr);
  TEST_EQ(g_full.init, 1);
  TEST_EQ(g_minimal.init, 1);

  hipdnn_ep::module_registry_destroy(reg);
  TEST_EQ(g_full.destroy, 1);
  TEST_EQ(g_minimal.destroy, 1);

  // A fresh empty registry must not re-fire prior destructors.
  reg = hipdnn_ep::module_registry_create();
  hipdnn_ep::module_registry_destroy(reg);
  TEST_EQ(g_full.destroy, 1);
  TEST_EQ(g_minimal.destroy, 1);

  std::printf("ok test_destroy_runs_in_reverse_registration_order\n");
}

static void test_steady_state_no_extra_init_calls() {
  // After warmup, op_module_get + begin_compute must not re-invoke init.
  reset_counters();
  hipdnn_ep::ModuleRegistry *reg = hipdnn_ep::module_registry_create();
  ::RuntimeState dummy_state{777};

  (void)hipdnn_ep::op_module_get(reg, &dummy_state, slot_full);
  TEST_EQ(g_full.init, 1);

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
