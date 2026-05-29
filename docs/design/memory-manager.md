# Memory Manager (MM) Architecture

## Overview

The Memory Manager is a GPU memory allocator specialized for LLM inference.
It manages three allocation classes — KV cache blocks, activation tensors, and
generic buffers — with separate budget pools and allocation strategies for each.

MM is compiled to LLVM bitcode and merged into `runtime.bc` via `llvm-link`,
so model DLLs call MM functions directly with zero FFI overhead.

## File Layout

```
include/mm/
  mm_api.h          Public C API (init, alloc, free, query, metrics)
  mm_types.h        Enums (MemoryClass, Lifetime, KvFormat) and structs (AllocHints, AllocInfo, KvBlockDesc)
  mm_config.h       Config struct + config_default()
  mm_kv.h           KV block API (kv_alloc_block, kv_free_block, kv_fork_block)

lib/MemoryManager/
  mm_core.cpp       Singleton coordinator — dispatches to subsystems, tracks global metrics
  mm_hal.h/cpp      Hardware abstraction layer — wraps hipMalloc/hipFree
  mm_handle_table.h/cpp   Opaque 64-bit handle → AllocInfo registry
  mm_activation.h/cpp     Segregated free-list arena for activation tensors (8 size classes)
  mm_kv_manager.h/cpp     Block manager for paged KV cache with refcounted CoW fork
  mm_config.cpp     Default config factory
```

## Layer Responsibilities

```
┌─────────────────────────────────────┐
│          mm::alloc / mm::free       │  Public API (mm_api.h)
├─────────────────────────────────────┤
│            mm_core.cpp              │  Dispatch + budget + metrics
├──────────┬──────────┬───────────────┤
│ Activation│ KV Mgr  │  HandleTable  │  Subsystems
│  Arena    │         │               │
├──────────┴──────────┴───────────────┤
│              RocmHal                │  hipMalloc / hipFree
└─────────────────────────────────────┘
```

- **mm_core**: routes allocations by `MemoryClass`, enforces total budget,
  aggregates metrics from subsystems.
- **ActivationArena**: 8-class segregated free-list (1 KB → 4 MB + fallback).
  Freed buffers return to the pool for reuse — no `hipFree` until shutdown.
- **KvManager**: maps `kv_block_t` handles to physical GPU blocks.
  `kv_fork_block` increments a refcount (CoW); physical memory freed only
  when refcount → 0.
- **HandleTable**: maps opaque `handle_t` values to `AllocInfo` metadata
  (pointer, size, class, lifetime, device).
- **RocmHal**: thin wrapper over HIP runtime. Translates `hipError_t` to
  `mm::Status`.

## Call Tree: Allocation

```
mm::alloc(size, hints, stream)
 ├─ lock g_mutex
 ├─ if MemoryClass::Activation → ActivationArena::alloc
 │    ├─ find size class (binary search on upper bounds)
 │    ├─ pop from free_list if available (zero hipMalloc)
 │    └─ else: budget CAS check → hal->malloc
 ├─ else → hal->malloc (generic / weight)
 ├─ HandleTable::insert(ptr, metadata)
 └─ update atomic counters (total_allocated, peak)
```

## Call Tree: KV Block Allocation

```
mm::kv_alloc_block(KvBlockDesc, stream)
 ├─ lock g_mutex
 ├─ KvManager::alloc_block
 │    ├─ compute_block_bytes(desc)
 │    │    └─ format_size * num_kv_heads * head_dim * num_layers * 2 * block_size_tokens
 │    ├─ budget check (total + block ≤ kv_budget)
 │    ├─ hal->malloc
 │    └─ physical_[ptr] = { bytes, desc, refcount=1 }
 └─ return handle
```

## Call Tree: Free

```
mm::free(handle, stream)
 ├─ lock g_mutex
 ├─ HandleTable::lookup → AllocInfo
 ├─ if KvCache → KvManager::free_block
 │    ├─ --refcount
 │    └─ if 0 → hal->free + erase physical entry
 ├─ if Activation → ActivationArena::release
 │    └─ push ptr back to size-class free_list (reuse)
 ├─ else → hal->free
 └─ HandleTable::erase
```

## Budget Initialization

```
init():
  free_bytes ← hipMemGetInfo()
  kv_budget  = min(kv_cache_fraction * free_bytes, kv_cache_max_bytes)
  activation_budget = free_bytes - kv_budget
```

Default: 90% of free GPU memory to KV cache, remainder to activations.

## Build Integration

All 6 MM `.cpp` files are compiled to LLVM bitcode with Clang (`-emit-llvm`)
and merged into `runtime.bc` by `llvm-link` (see `lib/Runtime/CMakeLists.txt`).
Model DLLs embed `runtime.bc` at link time — no separate `mm.lib` needed.

## Worked Example: KV Cache Allocation

1. **Model loads** → `inference_init` runs → `RuntimeState` created, `ensure_mm_initialized()` called.

2. **First inference**:
   - Constants blob allocated via `mm::alloc` with `MemoryClass::Weight` / `Lifetime::Static`.
   - Workspace allocated via `mm::alloc` with `MemoryClass::Activation` / `Lifetime::Session`.
   - KV cache blocks allocated via `mm::kv_alloc_block` — each block is a fixed-size
     region of GPU memory managed by the block manager.

3. **Steady-state decode**:
   - KV blocks reused across inferences (no per-call hipMalloc/hipFree).
   - Block manager tracks reference counts for copy-on-write fork support.
   - Workspace grows on demand (1.5x amortization), never shrinks.
