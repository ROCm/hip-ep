/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "module_registry.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>

namespace hipdnn_ep {

namespace {

// Construct-On-First-Use: avoids static-init ordering fiasco between TUs.
std::vector<const OpModuleSpec *> &spec_table() {
  static std::vector<const OpModuleSpec *> table;
  return table;
}

std::mutex &spec_table_mutex() {
  static std::mutex m;
  return m;
}

} // namespace

int register_op_module(const OpModuleSpec *spec) {
  if (!spec || !spec->name || !spec->init_fn || !spec->destroy_fn) {
    std::fprintf(stderr, "hipdnn_ep::register_op_module: invalid spec "
                         "(name / init_fn / destroy_fn are required)\n");
    std::abort();
  }
  std::lock_guard<std::mutex> lock(spec_table_mutex());
  auto &table = spec_table();
  for (size_t i = 0; i < table.size(); ++i) {
    if (std::strcmp(table[i]->name, spec->name) == 0) {
      std::fprintf(stderr,
                   "hipdnn_ep::register_op_module: duplicate name '%s' "
                   "(existing slot %zu)\n",
                   spec->name, i);
      std::abort();
    }
  }
  int slot_id = static_cast<int>(table.size());
  table.push_back(spec);
  return slot_id;
}

const OpModuleSpec *get_op_module_spec(int slot_id) {
  std::lock_guard<std::mutex> lock(spec_table_mutex());
  const auto &table = spec_table();
  if (slot_id < 0 || static_cast<size_t>(slot_id) >= table.size())
    return nullptr;
  return table[slot_id];
}

int op_module_count() {
  std::lock_guard<std::mutex> lock(spec_table_mutex());
  return static_cast<int>(spec_table().size());
}

// SlotEntry caches the spec's fn-pointers so begin_compute / destroy /
// dump never re-enter the mutex-guarded process-global spec table.
struct SlotEntry {
  void *state_ptr = nullptr;
  OpBeginComputeFn begin_compute_fn = nullptr;
  OpEndComputeFn end_compute_fn = nullptr;
  OpMemBytesFn mem_bytes_fn = nullptr;
  OpDestroyFn destroy_fn = nullptr;
  const char *name = nullptr;
};

struct ModuleRegistry {
  std::vector<SlotEntry> slots;
};

ModuleRegistry *module_registry_create() { return new ModuleRegistry; }

void module_registry_destroy(ModuleRegistry *reg) {
  if (!reg)
    return;
  // Reverse registration order in case a later-registered module ever
  // references an earlier one in its destructor.
  for (size_t i = reg->slots.size(); i-- > 0;) {
    SlotEntry &slot = reg->slots[i];
    if (slot.state_ptr && slot.destroy_fn) {
      slot.destroy_fn(slot.state_ptr);
    }
    slot.state_ptr = nullptr;
  }
  delete reg;
}

void module_registry_begin_compute(ModuleRegistry *reg, RuntimeState *state) {
  if (!reg)
    return;
  for (auto &slot : reg->slots) {
    if (slot.state_ptr && slot.begin_compute_fn) {
      slot.begin_compute_fn(slot.state_ptr, state);
    }
  }
}

void module_registry_dump(ModuleRegistry *reg) {
  // Always print the header so the user can confirm the env var was picked
  // up even when no module has been populated.
  std::fprintf(stderr, "[HIPDNN_EP_DUMP_STATE] registered op-module slots:\n");
  if (!reg) {
    std::fprintf(stderr, "[HIPDNN_EP_DUMP_STATE]   (registry is null)\n");
    return;
  }
  size_t total = 0;
  size_t known = 0;
  for (size_t i = 0; i < reg->slots.size(); ++i) {
    const SlotEntry &slot = reg->slots[i];
    if (!slot.state_ptr)
      continue;
    if (slot.mem_bytes_fn) {
      size_t b = slot.mem_bytes_fn(slot.state_ptr);
      total += b;
      ++known;
      std::fprintf(stderr,
                   "[HIPDNN_EP_DUMP_STATE]   slot=%zu name='%s' "
                   "mem_bytes=%zu\n",
                   i, slot.name ? slot.name : "?", b);
    } else {
      std::fprintf(stderr,
                   "[HIPDNN_EP_DUMP_STATE]   slot=%zu name='%s' "
                   "mem_bytes=?  (define `size_t mem_bytes() const` on the "
                   "state type to report)\n",
                   i, slot.name ? slot.name : "?");
    }
  }
  std::fprintf(stderr,
               "[HIPDNN_EP_DUMP_STATE] total reported = %zu bytes across "
               "%zu module(s) with mem_bytes()\n",
               total, known);
}

void *op_module_get(ModuleRegistry *reg, RuntimeState *state, int slot_id) {
  if (!reg || slot_id < 0)
    return nullptr;

  // Hot path: bounds compare + load + null branch.
  if (static_cast<size_t>(slot_id) < reg->slots.size()) {
    void *p = reg->slots[slot_id].state_ptr;
    if (p)
      return p;
  }

  // Cold path: first access for this (session, slot_id).
  if (static_cast<size_t>(slot_id) >= reg->slots.size()) {
    reg->slots.resize(static_cast<size_t>(slot_id) + 1, SlotEntry{});
  }
  const OpModuleSpec *spec = get_op_module_spec(slot_id);
  if (!spec || !spec->init_fn)
    return nullptr;
  void *p = spec->init_fn(state);
  if (!p)
    return nullptr;

  SlotEntry &slot = reg->slots[slot_id];
  slot.state_ptr = p;
  slot.begin_compute_fn = spec->begin_compute_fn;
  slot.end_compute_fn = spec->end_compute_fn;
  slot.mem_bytes_fn = spec->mem_bytes_fn;
  slot.destroy_fn = spec->destroy_fn;
  slot.name = spec->name;
  return p;
}

} // namespace hipdnn_ep
