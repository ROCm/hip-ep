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

struct RuntimeState;

// Base of every per-op state struct. `deletor` is wired at construction to a
// function that destroys the concrete type, so cleanup can free every slot
// without knowing its type. Slots reference nothing in other slots, so
// construction / teardown order never matters.
struct OpState {
  void (*deletor)(OpState *) = nullptr;
};

// Allocate a concrete state T (which must derive from OpState) and wire its
// deletor. Uses a throwing `new` (as the rest of the runtime does): the nothrow
// placement form pulls in `operator new(size_t, nothrow_t)` / `std::nothrow`,
// which are not resolvable when the runtime bitcode is JIT-linked into the EP
// process, so every model's global_ctors would fail to materialize.
template <class T> T *make_op_state() {
  T *st = new T();
  st->deletor = [](OpState *p) { delete static_cast<T *>(p); };
  return st;
}

// Opaque slot accessor (defined in op_state.cpp where RuntimeState is
// complete). Returns nullptr for an out-of-range slot or an unconstructed one.
extern "C" void *hipdnn_ep_op_state_get(RuntimeState *state, int slot);

// Reach a slot's state inside an operator's runtime entry.
template <class T> T *op_state(RuntimeState *state, int slot) {
  return static_cast<T *>(hipdnn_ep_op_state_get(state, slot));
}

// Opt-in sharing for like operators (e.g. an autotuned algorithm table that is
// identical for a device + library version across every session in a process).
// The value lives while some session holds the returned shared_ptr and is
// freed when the last reference is dropped. Sharing is a single line inside a
// constructor, decoupled from slot assignment.
template <class Key, class Val> class WeakStore {
public:
  template <class Factory>
  std::shared_ptr<Val> get_or_create(const Key &key, Factory factory) {
    std::lock_guard<std::mutex> guard(mutex_);
    auto it = entries_.find(key);
    if (it != entries_.end()) {
      if (std::shared_ptr<Val> live = it->second.lock())
        return live;
    }
    std::shared_ptr<Val> created = factory();
    entries_[key] = created;
    return created;
  }

private:
  std::mutex mutex_;
  std::map<Key, std::weak_ptr<Val>> entries_;
};

#endif // HIPDNN_EP_OP_STATE_H
