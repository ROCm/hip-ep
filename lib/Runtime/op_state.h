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

// Slot accessors. `hipdnn_ep_op_state_get` stays `extern "C"`: it is the opaque
// reader (defined in op_state.cpp where RuntimeState is complete) and returns
// nullptr for an out-of-range or unconstructed slot. `hipdnn_ep_op_state_set`
// is plain C++ -- it is only ever called from within the runtime bitcode (by
// the per-op construct_<op> functions), never by generated IR, so it needs no
// C ABI and no failure path: the compiler always hands it a valid slot into an
// already-allocated array. Together they keep the RuntimeState layout opaque to
// the op translation units that include this header.
extern "C" void *hipdnn_ep_op_state_get(RuntimeState *state, int slot);
void hipdnn_ep_op_state_set(RuntimeState *state, int32_t slot, OpState *value);

// CRTP base for per-op state structs: derive `struct FooState :
// OpStateT<FooState>`. The constructor wires `deletor` to destroy the concrete
// type, so a derived state can never forget to set it. `get_op_state` reaches
// this op's instance from its runtime entry; `create` builds an owning instance
// that the construct_<op> function then stores into its slot.
template <class T> struct OpStateT : OpState {
  OpStateT() {
    this->deletor = [](OpState *p) { delete static_cast<T *>(p); };
  }

  // Reach this op's state inside its runtime entry.
  static T *get_op_state(RuntimeState *state, int slot) {
    return static_cast<T *>(hipdnn_ep_op_state_get(state, slot));
  }

  // Build an owning T. The caller (construct_<op>) hands the released pointer
  // to hipdnn_ep_op_state_set. Uses a throwing `new` (as the rest of the
  // runtime does): the nothrow form pulls in `operator new(size_t, nothrow_t)`
  // / `std::nothrow`, which are not resolvable when the runtime bitcode is
  // JIT-linked into the EP process, so every model's global_ctors would fail to
  // materialize.
  template <class... Args> static std::unique_ptr<T> create(Args &&...args) {
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
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
// so the residual is O(#devices). Do NOT reuse this with an unbounded key
// space.
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
  // Heap-allocated and never destroyed, rather than a plain `static Storage s`.
  // A function-local static whose type has a non-trivial destructor makes the
  // compiler emit `__cxa_atexit(dtor, &s, &__dso_handle)`. `__dso_handle` is
  // hidden-visibility and per-DSO, so the compiler assumes it is in the same
  // linked output and reaches it with a direct PC32 -- no GOT entry, no stub,
  // nothing a linker can redirect. When this bitcode is JIT-linked into the EP
  // process that symbol instead resolves to the host library's copy, at
  // whatever distance ASLR chose; past the 2 GB a PC32 reaches, the fixup
  // fails and the entire runtime module fails to materialize, taking every
  // symbol in it down with it. Leaking the store is cheap: entries are
  // weak_ptrs keyed by device id, and the values are owned by the sessions
  // holding the shared_ptrs, so only the map and mutex shell outlive main.
  static Storage &storage() {
    static Storage *s = new Storage();
    return *s;
  }
};

#endif // HIPDNN_EP_OP_STATE_H
