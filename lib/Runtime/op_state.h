/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- op_state.h - Per-op state slot substrate -------------------------===//
//
// Substrate for the op-state-slots design (see
// docs/design/op-state-slots-design.md). Each stateful operator instance is
// assigned a compiler-allocated slot in RuntimeState::op_states; the slot is
// constructed during inference_init by code each op emits for itself
// (OpStateOpInterface::generateOpStateInit), and torn down generically in
// cleanup via the per-object `deletor`.
//
// This header is intentionally layout-agnostic: it forward-declares
// RuntimeState and reaches the array through the opaque C accessor
// `hipdnn_ep_op_state_get`, so it can be included by op translation units
// without pulling in runtime_state_internal.h.
//
//===----------------------------------------------------------------------===//

#ifndef HIPDNN_EP_OP_STATE_H
#define HIPDNN_EP_OP_STATE_H

#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <utility>

struct RuntimeState;

// Base of every per-op state struct. `deletor` is wired at construction to a
// function that destroys the concrete type, so cleanup can free every slot
// without knowing its type. Slots reference nothing in other slots, so
// construction / teardown order never matters.
struct OpState {
  void (*deletor)(OpState *) = nullptr;
};

// Opaque slot accessors (defined in op_state.cpp where RuntimeState is
// complete). `_get` returns nullptr for an out-of-range slot or an
// unconstructed one; `_set` bounds- and null-checks the store and returns true
// on success. Generated IR never touches the slot array directly -- every state
// stores itself through `_set` (see OpStateT::create), keeping the RuntimeState
// layout opaque to both generated IR and op translation units.
extern "C" void *hipdnn_ep_op_state_get(RuntimeState *state, int slot);
extern "C" bool hipdnn_ep_op_state_set(RuntimeState *state, int32_t slot,
                                       OpState *value);

// CRTP base for per-op state structs: derive `struct FooState :
// OpStateT<FooState>`. The constructor wires `deletor` to destroy the concrete
// type, so a derived state can never forget to set it. `get_slot` reaches the
// instance from an operator's runtime entry, and `create` builds the state and
// stores it into its slot.
template <class T> struct OpStateT : OpState {
  OpStateT() { this->deletor = [](OpState *p) { delete static_cast<T *>(p); }; }

  // Reach this op's state inside its runtime entry.
  static T *get_slot(RuntimeState *state, int slot) {
    return static_cast<T *>(hipdnn_ep_op_state_get(state, slot));
  }

  // Construct T and store it into op_states[slot]; returns 1 on success, 0 if
  // the store is refused (alloc failed / out-of-range slot), freeing the object
  // via its deletor so nothing leaks. Uses a throwing `new` (as the rest of the
  // runtime does): the nothrow placement form pulls in `operator new(size_t,
  // nothrow_t)` / `std::nothrow`, which are not resolvable when the runtime
  // bitcode is JIT-linked into the EP process, so every model's global_ctors
  // would fail to materialize.
  template <class... Args>
  static int8_t create(RuntimeState *state, int32_t slot, Args &&...args) {
    T *st = new T(std::forward<Args>(args)...);
    if (!hipdnn_ep_op_state_set(state, slot, st)) {
      st->deletor(st);
      return 0;
    }
    return 1;
  }
};

// Opt-in sharing for like operators (e.g. an autotuned algorithm table that is
// identical for a device + library version across every session in a process).
// The value lives while some session holds the returned shared_ptr and is
// freed when the last reference is dropped. A single global store per <Key,Val>
// is reached through the Meyers-singleton accessor `storage()`, sidestepping
// static-initialization-order issues; the mutex guards concurrent construction
// from sessions initializing on independent threads.
//
// NOTE: intentionally distinct from `morphizen::utils::WeakStore` -- this one
// adds the mutex (that store is unlocked) and drops the initialize-injection /
// cleanup machinery. Expired entries are not pruned: keys here are device ids,
// so the residual is O(#devices). Do NOT reuse this with an unbounded key space.
template <class Key, class Val> class WeakStore {
public:
  template <class Factory>
  static std::shared_ptr<Val> get_or_create(const Key &key, Factory factory) {
    Storage &s = storage();
    std::lock_guard<std::mutex> guard(s.mutex);
    auto it = s.entries.find(key);
    if (it != s.entries.end()) {
      if (std::shared_ptr<Val> live = it->second.lock())
        return live;
    }
    std::shared_ptr<Val> created = factory();
    s.entries[key] = created;
    return created;
  }

private:
  struct Storage {
    std::mutex mutex;
    std::map<Key, std::weak_ptr<Val>> entries;
  };
  static Storage &storage() {
    static Storage s;
    return s;
  }
};

#endif // HIPDNN_EP_OP_STATE_H
