/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===- module_registry.h - Op-module registry (internal) ---------------*-===//
//
// Each operator that owns persistent per-session state ("op module")
// registers an OpModuleSpec once and gets back a stable integer slot id.
// The HIPDNN_OP_MODULE macro generates the accessor.
//
// See docs/design/runtime-module-registry.md for the contract and the
// steady-state-cost guarantees.
//===---------------------------------------------------------------------===//

#ifndef HIPDNN_EP_MODULE_REGISTRY_H
#define HIPDNN_EP_MODULE_REGISTRY_H

#include <cstddef>
#include <type_traits>

struct RuntimeState;

namespace hipdnn_ep {

struct ModuleRegistry;

using OpInitFn = void *(*)(RuntimeState *);
using OpDestroyFn = void (*)(void *);
using OpBeginComputeFn = void (*)(void *, RuntimeState *);

struct OpModuleSpec {
  const char *name = nullptr;
  OpInitFn init_fn = nullptr;
  OpDestroyFn destroy_fn = nullptr;
  OpBeginComputeFn begin_compute_fn = nullptr;
};

// Register a spec. Returns a stable, non-negative slot id. Aborts on
// duplicate name or missing required fields. `spec` must remain valid for
// the process lifetime (typically a function-local static).
int register_op_module(const OpModuleSpec *spec);

const OpModuleSpec *get_op_module_spec(int slot_id);
int op_module_count();

ModuleRegistry *module_registry_create();

// Destroy every populated slot via spec->destroy_fn in reverse
// registration order, then free the registry. nullptr-safe.
void module_registry_destroy(ModuleRegistry *reg);

// Lock-free fan-out called from hipdnn_ep_runtime_begin_compute. Iterates
// populated slots and invokes each cached begin_compute_fn.
void module_registry_begin_compute(ModuleRegistry *reg, RuntimeState *state);

// Hot path: bounds check + load + null branch. Cold path: resize, spec
// lookup, init_fn call.
void *op_module_get(ModuleRegistry *reg, RuntimeState *state, int slot_id);

// Defined in hipdnn_ep_runtime_state.cpp where RuntimeState's layout is
// in scope; declared here so HIPDNN_OP_MODULE can call it without forcing
// every user to pull in runtime_state_internal.h.
ModuleRegistry *get_module_registry(RuntimeState *state);

// SFINAE detector for the optional begin_compute hook. Absent => false_type
// => make_op_module_spec leaves begin_compute_fn null.
template <typename T, typename = void>
struct has_begin_compute_ : std::false_type {};

template <typename T>
struct has_begin_compute_<
    T, std::void_t<decltype(std::declval<T &>().begin_compute(
           std::declval<RuntimeState *>()))>> : std::true_type {};

// Build a spec for state type T. T must be constructible from
// RuntimeState*. begin_compute is wired up only when the SFINAE detector
// is true.
template <typename T>
inline OpModuleSpec make_op_module_spec(const char *name) {
  OpModuleSpec spec{};
  spec.name = name;
  spec.init_fn =
      +[](RuntimeState *s) -> void * { return static_cast<void *>(new T(s)); };
  spec.destroy_fn = +[](void *p) { delete static_cast<T *>(p); };
  if constexpr (has_begin_compute_<T>::value) {
    spec.begin_compute_fn = +[](void *p, RuntimeState *s) {
      static_cast<T *>(p)->begin_compute(s);
    };
  }
  return spec;
}

} // namespace hipdnn_ep

// Strongly-typed accessor generators. Three variants, used in two
// patterns depending on whether the module's state type is touched from
// one TU or many.
//
// Pattern A -- single-TU module (state struct + accessor both file-local):
//   In the op's .cpp, inside an anonymous namespace:
//
//     struct MyState { explicit MyState(RuntimeState *) {}; ... };
//     HIPDNN_OP_MODULE(my_module, "my_op", MyState);
//
//   This is the common case (QmoeState, GqaSeqlensCache, ZpUnpackState,
//   ...). Generates a `static` (file-local) accessor.
//
// Pattern B -- cross-TU module (state struct in a header; callers in
// multiple .cpp TUs):
//   In a header:
//
//     struct MyState { ... };
//     HIPDNN_OP_MODULE_DECLARE(my_module, MyState);
//
//   In exactly one .cpp:
//
//     HIPDNN_OP_MODULE_DEFINE(my_module, "my_op", MyState);
//
//   This is used by WorkspaceState, reached from six wrap_* TUs. The
//   declaration gives an external-linkage accessor; the definition is
//   what HIPDNN_OP_MODULE generates minus the `static` keyword.
//
// In both patterns the slot id is computed lazily on the first call
// (function-local statics): one process-global spec_table mutex on
// first call ever, then bounds check + load + null branch.
//
// Macro arguments:
//   ACCESSOR : function name, e.g. `causal_conv_module`.
//   STATE_T  : state struct; must be constructible from RuntimeState *.
//   NAME     : unique string id used by the registry's duplicate-detect.

#define HIPDNN_OP_MODULE(ACCESSOR, NAME, STATE_T)                              \
  static STATE_T *ACCESSOR(::RuntimeState *state) {                            \
    static const ::hipdnn_ep::OpModuleSpec hipdnn_ep_module_spec_ =            \
        ::hipdnn_ep::make_op_module_spec<STATE_T>(NAME);                       \
    static const int hipdnn_ep_module_slot_ =                                  \
        ::hipdnn_ep::register_op_module(&hipdnn_ep_module_spec_);              \
    return static_cast<STATE_T *>(                                             \
        ::hipdnn_ep::op_module_get(::hipdnn_ep::get_module_registry(state),    \
                                   state, hipdnn_ep_module_slot_));            \
  }

#define HIPDNN_OP_MODULE_DECLARE(ACCESSOR, STATE_T)                            \
  STATE_T *ACCESSOR(::RuntimeState *state)

#define HIPDNN_OP_MODULE_DEFINE(ACCESSOR, NAME, STATE_T)                       \
  STATE_T *ACCESSOR(::RuntimeState *state) {                                   \
    static const ::hipdnn_ep::OpModuleSpec hipdnn_ep_module_spec_ =            \
        ::hipdnn_ep::make_op_module_spec<STATE_T>(NAME);                       \
    static const int hipdnn_ep_module_slot_ =                                  \
        ::hipdnn_ep::register_op_module(&hipdnn_ep_module_spec_);              \
    return static_cast<STATE_T *>(                                             \
        ::hipdnn_ep::op_module_get(::hipdnn_ep::get_module_registry(state),    \
                                   state, hipdnn_ep_module_slot_));            \
  }

#endif // HIPDNN_EP_MODULE_REGISTRY_H
