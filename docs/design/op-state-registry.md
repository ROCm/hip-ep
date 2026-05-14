<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Generic Op-State Registry

**Date:** 2026-05-12 (proposed) / 2026-05-13 (implemented)
**Document Type:** Implementation Plan + As-Built Notes
**Status:** **Implemented** — see [As-Built](#as-built-2026-05-13) below.
**Supersedes:** [inline-prepack-cache.md](inline-prepack-cache.md) (the inline
cache decorator approach was abandoned in favour of this module registry).
**Related (older draft, partly subsumed):**
[workspace-improvements.md](workspace-improvements.md) — its
`growable_buffer.h` recommendation landed here under Stage 4.

> **As-built summary (2026-05-13).** Stages 1–5 of this plan are merged.
> The runtime now exposes:
>
> * `lib/Runtime/module_registry.{h,cpp}` — the `OpModuleSpec` /
>   `HIPDNN_OP_MODULE` macro / process-global slot table / per-RuntimeState
>   `ModuleRegistry` with cached per-slot fn-pointer fan-out for
>   `begin_compute` and `destroy`.
> * `lib/Runtime/growable_buffer.h` — `GrowableDeviceBuffer` /
>   `GrowablePinnedBuffer` with the canonical 1.5×-amortized,
>   sync-before-free, never-shrink policy.
> * Five op modules: `CausalConvState` (`real/causal_conv_with_state.cpp`),
>   `ZpUnpackState` (`real/matmul_nbits.cpp`), `GqaGemmState` +
>   `GqaSeqlensCache` (`real/gqa.cpp`), `QmoeState` (`real/qmoe.cpp`).
> * `HIPDNN_EP_DUMP_STATE=1` — at session cleanup, prints one line per
>   populated module slot with its name and (when the state type defines
>   `size_t mem_bytes() const`) reported memory footprint.
> * `test/runtime_steady_state` — native C++ smoke test that exercises the
>   registry's hook-detection, slot-id stability, lazy init, destruction
>   order, and post-warmup steady-state (no extra init/destroy calls per
>   begin_compute). Runs as CTest #58.
>
> See the [As-Built](#as-built-2026-05-13) section at the bottom for the
> verbatim file map and how each removed `RuntimeState` field migrated.

---

## Table of Contents

- [Overview](#overview)
- [Motivation](#motivation)
- [Scope](#scope)
- [Architecture](#architecture)
  - [`OpModuleSpec` and the `HIPDNN_OP_MODULE` macro](#a1-opmodulespec-and-the-hipdnn_op_module-macro)
  - [Storage in `RuntimeState`](#a2-storage-in-runtimestate)
  - [Why an integer slot id, not a string key](#a3-why-an-integer-slot-id-not-a-string-key)
  - [Per-op call site](#a4-per-op-call-site)
  - [Op-state composition and extensibility](#a5-op-state-composition-and-extensibility)
  - [Optional lifecycle hooks](#a6-optional-lifecycle-hooks-sfinae-detected)
  - [Cleanup](#a7-cleanup)
  - [Observability — `HIPDNN_EP_DUMP_STATE`](#a8-observability--hipdnn_ep_dump_state)
  - [Growable scratch buffers (`growable_buffer.h`)](#a9-growable-scratch-buffers-growable_bufferh)
  - [Concurrency](#a10-concurrency)
- [Implementation Stages](#implementation-stages)
- [Files Touched](#files-touched)
- [Risks and Mitigations](#risks-and-mitigations)
- [Design Alternatives Considered](#design-alternatives-considered)
- [As-Built (2026-05-13)](#as-built-2026-05-13)

---

## Overview

Replace the ad-hoc per-op fields in `RuntimeState` (`causal_conv_cache`,
`gqa_gemm_cache`, `zp_unpack_cache`, `qmoe_scratch` + size,
`qmoe_host_scratch` + size, and the three `seqlens_k_cached_*` POD fields)
with a uniform **op-module registry**. Each op declares its own
`<Op>State` C++ type, registers it at static-init time through the
`HIPDNN_OP_MODULE` macro, and gets a strongly-typed O(1) accessor. The
framework handles lifecycle (lazy create + destroy at cleanup) generically
and routes per-Compute() invalidation through the same registry.

After this lands, adding a new op-with-state requires editing only the
op's own `.cpp` file. The cross-cutting boilerplate in
`runtime_state_internal.h`, `hipdnn_ep_runtime.h`, and
`hipdnn_ep_runtime_state.cpp` disappears for new ops.

---

## Motivation

| Problem before this plan | After this plan |
|---|---|
| Every new op-with-state required touching `runtime_state_internal.h`, `hipdnn_ep_runtime.h`, `hipdnn_ep_runtime_state.cpp` (3 files, cross-cutting). | A new op-with-state requires only one `HIPDNN_OP_MODULE` line in the op's own `.cpp`. No framework files change. |
| Cleanup order was implicit (one `if` per cache in `hipdnn_ep_state_cleanup`); easy to forget when adding an op. | Cleanup is one loop over the registry, in reverse registration order. |
| Per-Compute() invalidation (e.g. the GQA `seqlens_k` cache) was hand-wired with a dedicated bool + ptr + val triple on `RuntimeState` and a bespoke reset block at the top of `hipdnn_ep_runtime_begin_compute`. | Define `void begin_compute(RuntimeState*)` on the state type; SFINAE picks it up automatically and the registry fans it out lock-free. |
| Plugins (when added) could not own state without core framework changes. | Plugins call `HIPDNN_OP_MODULE` exactly the way built-in ops do. |
| `RuntimeState` field count grew linearly with ops; nine cache-like fields lived there at peak. | A single `hipdnn_ep::ModuleRegistry *modules` pointer replaces all per-op caches. |
| No observability: nothing in production told you which ops had populated state or how much GPU/host memory it held. | `HIPDNN_EP_DUMP_STATE=1` prints one line per populated module slot at cleanup, with name and (optional) `mem_bytes()`. |

---

## Scope

**In scope:** every per-op piece of persistent session state. The
following nine fields were on `RuntimeState` before this plan and have all
migrated into op-modules:

- `causal_conv_cache` → `CausalConvState`
- `gqa_gemm_cache` → `GqaGemmState`
- `zp_unpack_cache` → `ZpUnpackState` (shared between matmul_nbits and qmoe)
- `qmoe_scratch` + `qmoe_scratch_size` → `QmoeState::scratch`
- `qmoe_host_scratch` + `qmoe_host_scratch_size` → `QmoeState::host_scratch`
- `seqlens_k_cached_valid` + `seqlens_k_cached_val` + `seqlens_k_cached_ptr`
  → `GqaSeqlensCache` (uses the `begin_compute` hook)

**Out of scope:** genuinely cross-cutting framework state stays on
`RuntimeState` directly. See
[As-Built › What did NOT migrate (and why)](#what-did-not-migrate-and-why)
for the full list (workspace, constants blob, buffer pool, op_profile,
device_error_flag, library handles, EP-attached graph state).

---

## Architecture

### A.1 `OpModuleSpec` and the `HIPDNN_OP_MODULE` macro

The registry lives in `lib/Runtime/module_registry.h` (private header,
included by `runtime_state_internal.h` and every runtime `.cpp` that
declares an op-module).

```cpp
// lib/Runtime/module_registry.h (excerpted)
namespace hipdnn_ep {

using OpInitFn          = void *(*)(RuntimeState *);
using OpDestroyFn       = void  (*)(void *);
using OpBeginComputeFn  = void  (*)(void *, RuntimeState *);
using OpEndComputeFn    = void  (*)(void *, RuntimeState *);
using OpMemBytesFn      = size_t(*)(void *);

struct OpModuleSpec {
  const char         *name              = nullptr;
  OpInitFn            init_fn           = nullptr;   // required
  OpDestroyFn         destroy_fn        = nullptr;   // required
  OpBeginComputeFn    begin_compute_fn  = nullptr;   // optional
  OpEndComputeFn      end_compute_fn    = nullptr;   // optional (reserved)
  OpMemBytesFn        mem_bytes_fn      = nullptr;   // optional
};

int                  register_op_module(const OpModuleSpec *spec);
const OpModuleSpec  *get_op_module_spec(int slot_id);
int                  op_module_count();

struct ModuleRegistry;   // opaque
ModuleRegistry      *module_registry_create();
void                 module_registry_destroy(ModuleRegistry *);
void                 module_registry_begin_compute(ModuleRegistry *, RuntimeState *);
void                 module_registry_dump(ModuleRegistry *);
void                *op_module_get(ModuleRegistry *, RuntimeState *, int slot_id);
ModuleRegistry      *get_module_registry(RuntimeState *);

template <typename T> OpModuleSpec make_op_module_spec(const char *name);

} // namespace hipdnn_ep

#define HIPDNN_OP_MODULE(ACCESSOR, NAME, STATE_T)                            \
  static STATE_T *ACCESSOR(::RuntimeState *state) {                          \
    static const ::hipdnn_ep::OpModuleSpec hipdnn_ep_module_spec_ =          \
        ::hipdnn_ep::make_op_module_spec<STATE_T>(NAME);                     \
    static const int hipdnn_ep_module_slot_ =                                \
        ::hipdnn_ep::register_op_module(&hipdnn_ep_module_spec_);            \
    return static_cast<STATE_T *>(::hipdnn_ep::op_module_get(                \
        ::hipdnn_ep::get_module_registry(state), state,                      \
        hipdnn_ep_module_slot_));                                            \
  }
```

`make_op_module_spec<T>` is a template helper. It synthesizes
`init_fn` (`new T(state)`) and `destroy_fn` (`delete static_cast<T*>`)
unconditionally, then uses `if constexpr` plus three SFINAE detectors
(`has_begin_compute_`, `has_end_compute_`, `has_mem_bytes_`) to install
trampolines only for the optional methods that `T` actually defines. An
op-module type that does not define `begin_compute` leaves
`spec.begin_compute_fn = nullptr`, and the registry skips it at zero cost.

The macro expands to a static accessor with two function-local statics:
the `OpModuleSpec` (built from `STATE_T` at compile time) and the slot id
returned by `register_op_module` on first call. C++11 guarantees both
statics are initialized exactly once and thread-safely. After that, every
call falls through to `op_module_get` with the cached slot id.

### A.2 Storage in `RuntimeState`

```cpp
// lib/Runtime/runtime_state_internal.h (excerpted)
namespace hipdnn_ep { struct ModuleRegistry; }

struct RuntimeState {
  // ... unchanged framework fields ...

  // (Removed: causal_conv_cache, gqa_gemm_cache, zp_unpack_cache,
  //  qmoe_scratch + size, qmoe_host_scratch + size, seqlens_k_cached_*.
  //  All seven fields migrated to op-modules below.)

  hipdnn_ep::ModuleRegistry *modules;   // opaque; one pointer.
};
```

`ModuleRegistry` is a forward declaration in the header — its full
definition stays in `module_registry.cpp`. The struct is one `std::vector`
of `SlotEntry` records, where each `SlotEntry` caches the per-slot
state pointer alongside cached `begin_compute_fn` / `destroy_fn` /
`mem_bytes_fn` pointers copied from the process-global spec table on
first access. This lets the per-Compute() fan-out iterate without ever
re-entering the spec table's mutex.

### A.3 Why an integer slot id, not a string key

Hot path matters: `wrap_matmul_nbits` is invoked 225× per token on Llama
8B. The hot-path lookup must be in the same noise band as a direct field
load.

| Operation | Approx cost |
|---|---|
| `unordered_map<string, void*>::find` | 80–150 ns (hash + bucket walk + string compare) |
| `op_module_get` (steady state) | ~3 ns (bounds check + array load + null branch) |

The slot id is assigned by `register_op_module` and cached in the macro's
function-local static. The same id then indexes the per-`RuntimeState`
slot table on every subsequent call.

### A.4 Per-op call site

```cpp
// lib/Runtime/real/causal_conv_with_state.cpp (sketch)
#include "../module_registry.h"
#include "../runtime_state_internal.h"

namespace {

// The state type *is* the cache + any other per-session data. A
// (RuntimeState*) constructor is mandatory; the destructor handles all
// resource teardown (MIOpen descriptors, hipBLASLt handles, hipFree of
// growable scratch, etc.).
struct CausalConvState {
  std::unordered_map<CausalConvKey, CausalConvCacheEntry, ...> entries;

  explicit CausalConvState(RuntimeState * /*state*/) {}
  ~CausalConvState() {
    // Move the old destroy_entry logic here so `delete` does the right
    // thing -- this is a purely mechanical refactor of pre-existing code.
    for (auto &kv : entries) destroy_entry(kv.second);
  }
};

HIPDNN_OP_MODULE(causal_conv_module, "causal_conv_with_state",
                 CausalConvState);

} // namespace

int wrap_causal_conv_with_state(RuntimeState *state, ...) {
  auto *st = causal_conv_module(state);
  auto it = st->entries.find(key);
  // ... existing logic, with `st->entries` in place of the old cache ...
}
```

That is the **entire** op-side change. No edit to
`runtime_state_internal.h`, no edit to `hipdnn_ep_runtime_state.cpp`, no
new public C-ABI declaration. The accessor's anonymous-namespace
placement plus `register_op_module`'s "abort on duplicate name" check
guard against accidental cross-TU collisions.

### A.5 Op-state composition and extensibility

The state type is the op author's to shape. Naming convention is
`<Op>State`. Three composition patterns are common across the migrated
modules:

| Module | Composition |
|---|---|
| `CausalConvState`, `GqaGemmState`, `ZpUnpackState` | RAII over an `unordered_map<Key, Entry, Hash>` whose destructor handles teardown. Most ops fit here. |
| `GqaSeqlensCache` | Three POD fields (`bool valid; int32_t val; const void* ptr;`) reset by a `begin_compute(RuntimeState*)` hook. Use this shape for any per-Compute() invalidated cache. |
| `QmoeState` | Two `GrowableDeviceBuffer` / `GrowablePinnedBuffer` members ([§A.9](#a9-growable-scratch-buffers-growable_bufferh)) plus a `mem_bytes()` accessor for `HIPDNN_EP_DUMP_STATE`. Use this shape for any module that owns grow-on-demand scratch. |

**Extension paths**:

1. **First-party — add fields to your own `<Op>State`.** No framework
   change. Edit the struct, recompile, ship. This is the path when, for
   example, autotune lands and `MatmulNbitsState` needs a tactic cache
   alongside its zp unpack maps.
2. **Cross-op — share a reusable building block.** When a pattern appears
   in three+ modules, lift it into `lib/Runtime/<helper>.h` and have
   modules compose it by membership. `growable_buffer.h` is the first
   example; future candidates are descriptor caches and prepack caches.
3. **Plugin — new ops with their own state.** A plugin DLL calls
   `HIPDNN_OP_MODULE` exactly the way first-party ops do. No special
   plugin API is needed.

**Not extendable on purpose**:

- No `OpStateBase` polymorphism. The macro takes the concrete type; there
  is no virtual destructor or shared base. Keeps the hot path zero-cost
  and the bitcode lean.
- No third-party reach-in to another op's state. If a plugin needs state
  next to `wrap_matmul_nbits`, it should register its own module with
  its own state. The "one spec per name" invariant deliberately blocks
  the alternative.

### A.6 Optional lifecycle hooks (SFINAE-detected)

Define any of these methods on `<Op>State` and the registry picks them
up automatically:

| Method | Fires from | Use case |
|---|---|---|
| `void begin_compute(RuntimeState *state)` | `hipdnn_ep_runtime_begin_compute` at the top of every `Compute()` | Per-Compute() cache invalidation. `GqaSeqlensCache::begin_compute` resets its `valid` flag here. |
| `void end_compute(RuntimeState *state)` | Reserved for future use | Symmetric per-Compute() flush; nothing uses it yet. |
| `size_t mem_bytes() const` | `module_registry_dump` (when `HIPDNN_EP_DUMP_STATE=1`) | Report this module's buffer footprint. `QmoeState::mem_bytes` returns the combined device + pinned-host capacity. |

The detectors use C++17 `std::void_t` over `decltype(declval<T&>().method(...))`,
so a method whose signature does not exactly match (e.g. a typo, a wrong
const-qualifier on `mem_bytes`, a missing `RuntimeState*` arg on
`begin_compute`) simply fails to be detected — the trampoline stays
null. This is intentional silent fallthrough; the macro does not warn
because it cannot distinguish "user did not want this hook" from "user
got the signature wrong".

To verify a hook is actually wired up, set `HIPDNN_EP_DUMP_STATE=1`
([§A.8](#a8-observability--hipdnn_ep_dump_state)) for `mem_bytes` or step
through `module_registry_begin_compute` in a debugger for `begin_compute`.

### A.7 Cleanup

The framework iterates the registry once at session teardown:

```cpp
// hipdnn_ep_runtime_state.cpp (sketch)
void hipdnn_ep_state_cleanup(RuntimeState *state) {
  if (!state) return;
  if (state->modules) {
    if (hipdnn_ep_dump_state_enabled()) {
      hipdnn_ep::module_registry_dump(state->modules);
    }
    hipdnn_ep::module_registry_destroy(state->modules);
    state->modules = nullptr;
  }
  // ... framework-field teardown unchanged ...
}
```

`module_registry_destroy` iterates the slot table in **reverse**
registration order and invokes each populated slot's cached
`destroy_fn`. Reverse order is insurance against future cross-module
dependencies (e.g. a "tactic registry" module observed by a "matmul"
module on destruction).

### A.8 Observability — `HIPDNN_EP_DUMP_STATE`

Setting `HIPDNN_EP_DUMP_STATE=1` makes `hipdnn_ep_state_cleanup` call
`module_registry_dump` immediately before destruction. Each populated
slot prints one line to stderr:

```
[HIPDNN_EP_DUMP_STATE] slot=0 name='qmoe' mem_bytes=3692992
[HIPDNN_EP_DUMP_STATE] slot=1 name='zp_unpack' mem_bytes=147456
[HIPDNN_EP_DUMP_STATE] slot=2 name='gqa_gemm_cache' mem_bytes=?
```

`?` means the module did not define `size_t mem_bytes() const`. The
helper that checks the env var (`hipdnn_ep_dump_state_enabled()` in
`lib/Runtime/debug_log.h`) mirrors the existing `HIPDNN_EP_DEBUG` /
`HIPDNN_EP_PERF` helpers and uses Win32 `GetEnvironmentVariableA` (per
the static-CRT gotcha in `CLAUDE.md`).

This is the official observability hook for the module system. Adding
`mem_bytes()` to any state type immediately surfaces it in the dump with
no other plumbing.

### A.9 Growable scratch buffers (`growable_buffer.h`)

A small header-only library introduced for the qmoe migration and
available to any module needing grow-on-demand scratch:

```cpp
// lib/Runtime/growable_buffer.h (header-only, no .cpp)
namespace hipdnn_ep {

class GrowableDeviceBuffer {       // hipMalloc / hipFree
public:
  explicit GrowableDeviceBuffer(hipStream_t stream = nullptr);
  ~GrowableDeviceBuffer();
  void   set_stream(hipStream_t stream);
  int    grow(size_t needed);      // 0 on success, -1 on failure
  void  *data() const noexcept;
  size_t size() const noexcept;
};

class GrowablePinnedBuffer {       // hipHostMalloc / hipHostFree
  // ... same shape, same API ...
};

} // namespace hipdnn_ep
```

Both classes implement the canonical policy that
`hipdnn_ep_state_ensure_workspace` had hand-rolled before:

1. **1.5× growth amortization** — `grow(needed)` reallocates to
   `max(needed, size_ + size_/2)` (cold-start sizes exactly to
   `needed`), so monotonically growing call patterns get O(log N)
   reallocations instead of one per growing-shape transition.
2. **Sync-before-free** — `hipStreamSynchronize(stream_)` runs before
   each `hipFree` / `hipHostFree` on grow, preventing use-after-free
   against in-flight kernels that still reference the old pointer.
3. **Never shrinks** during normal operation; freed only by the
   destructor.

Modules compose these by membership rather than re-implementing the
grow logic. `QmoeState::scratch` and `QmoeState::host_scratch` are the
canonical examples.

### A.10 Concurrency

`RuntimeState` is single-threaded (one inference per state at a time).
The lazy-create branch in `op_module_get` inherits that contract — no
locks on the hot path. The **process-global** `register_op_module` spec
table is mutex-guarded because two model.dlls instantiated from the same
runtime archive register their modules concurrently at static-init.

If concurrent inference on the same `RuntimeState` is ever supported,
the lazy-create branch needs a `std::call_once` or compare-and-swap. The
per-slot init has at-most-once semantics, so the fix is local. Defer
until concurrent execution actually lands.

---

## Implementation Stages

The plan landed in five stages, each independently shippable, leaving CI
green throughout. Items in this section reflect what actually shipped; see
[As-Built](#as-built-2026-05-13) for the file map and deviations.

### Stage 1 — Registry infrastructure

- **Add** `lib/Runtime/module_registry.h` ([§A.1](#a1-opmodulespec-and-the-hipdnn_op_module-macro)):
  `OpModuleSpec`, the SFINAE detectors, `make_op_module_spec<T>`,
  `register_op_module`, `get_op_module_spec`, `op_module_count`,
  `module_registry_create` / `destroy` / `begin_compute` / `dump`,
  `op_module_get`, `get_module_registry`, and the `HIPDNN_OP_MODULE`
  macro. Private header.
- **Add** `lib/Runtime/module_registry.cpp`: process-global mutex-guarded
  `spec_table()` (Construct-On-First-Use), per-`RuntimeState`
  `ModuleRegistry` struct with `SlotEntry` cache, lazy `op_module_get`
  init, reverse-order destroy, `module_registry_dump`.
- **Modify** `lib/Runtime/runtime_state_internal.h`: forward-declare
  `hipdnn_ep::ModuleRegistry`; add `hipdnn_ep::ModuleRegistry *modules;`
  at the bottom of the struct (existing offsets preserved → no ABI churn
  for pre-built model.dlls).
- **Modify** `lib/Runtime/hipdnn_ep_runtime_state.cpp`:
  - `initialize_state_handles`: `state->modules = hipdnn_ep::module_registry_create();`.
  - `hipdnn_ep_state_cleanup`: `module_registry_dump` (when env set) →
    `module_registry_destroy` → null the pointer.
  - `hipdnn_ep_runtime_begin_compute`: call
    `module_registry_begin_compute(state->modules, state)`.
- **Modify** `lib/Runtime/CMakeLists.txt`:
  `module_registry.cpp` compiled to `runtime_module_registry.bc`,
  bundled into the runtime archive; `module_registry.h` added to the
  universal `DEPENDS` list of `compile_to_bitcode` per the gotcha in
  `CLAUDE.md`.
- **Add** `test/runtime_steady_state/` — native C++ smoke test (no GPU)
  with two fake state types (one full hooks, one minimal) and seven
  assertions covering hook detection, slot stability, lazy init,
  destruction order, and the steady-state invariant. Registered as
  CTest #58.

**Acceptance:** Full build green, all 58 CTest cases pass (LIT + E2E +
the new smoke test). Infrastructure is dormant — no `wrap_*` consumes it
yet.

### Stage 2 — Migrate `causal_conv_cache` (canary)

- **Modify** `lib/Runtime/real/causal_conv_with_state.cpp`: introduce
  `CausalConvState` with the entries map and a destructor that runs the
  old `destroy_entry` logic. Replace the bespoke
  `get_causal_conv_cache(state)` helper with the one-line
  `HIPDNN_OP_MODULE(causal_conv_module, "causal_conv_with_state", CausalConvState);`
  invocation. Update the two call sites in `wrap_causal_conv_with_state`.
- **Remove** the `causal_conv_cache` field from `RuntimeState`, the
  cleanup block, and the `hipdnn_ep_causal_conv_cache_destroy`
  declaration from `hipdnn_ep_runtime.h`.

**Acceptance:** LIT tests for causal_conv pass; E2E unaffected.

### Stage 3 — Migrate the two remaining caches

- **`ZpUnpackState` in matmul_nbits.cpp**: state type owns the two
  `unordered_map<const void*, (void*, size_t)>` for u8 / fp16 unpacked
  zp buffers and a destructor that `hipFree`s them. Exposes
  `mem_bytes()` for the dump. Since the cache is shared between
  `wrap_matmul_nbits` and `wrap_qmoe`, both reach it via the
  `lookup_or_*_zp_*` free functions, which call `zp_unpack_module(state)`
  internally — registered under a single name (`"zp_unpack"`), one slot.
- **`GqaGemmState` in gqa.cpp**: state type owns the hipBLASLt
  descriptor cache; destructor calls
  `hipblasLtMatmulDescDestroy` / `hipblasLtMatrixLayoutDestroy` on every
  entry.
- **`GqaSeqlensCache` in gqa.cpp** (same TU, separate module): three POD
  fields plus `void begin_compute(RuntimeState*)`. The
  `hipdnn_ep_runtime_begin_compute` block that hand-reset
  `seqlens_k_cached_*` goes away — the registry fans the hook out.
- **Remove** all three fields from `RuntimeState`. **Remove**
  `hipdnn_ep_zp_unpack_cache_destroy` and
  `hipdnn_ep_gqa_gemm_cache_destroy` from `hipdnn_ep_runtime.h`. Remove
  the corresponding cleanup blocks from
  `hipdnn_ep_runtime_state.cpp`.

**Acceptance:** Llama 8B asym MatMulNBits path bit-exact; GQA
decomposed and flash_decode paths unaffected; CTest stays at 58/58.

### Stage 4 — `growable_buffer.h` + `QmoeState`

- **Add** `lib/Runtime/growable_buffer.h` ([§A.9](#a9-growable-scratch-buffers-growable_bufferh)):
  `GrowableDeviceBuffer` and `GrowablePinnedBuffer`, header-only, with
  the 1.5×-grow / sync-before-free / never-shrink policy.
- **Modify** `lib/Runtime/real/qmoe.cpp`: introduce `QmoeState` with two
  growable buffers and a `mem_bytes()` accessor. Register with
  `HIPDNN_OP_MODULE(qmoe_module, "qmoe", QmoeState);`. **Re-implement**
  the four `hipdnn_ep_state_*_qmoe_*` C-ABI functions inside this TU,
  delegating to the slot. The C-ABI is preserved on purpose — see
  [As-Built › Deviations](#deviations-from-the-plan).
- **Remove** `qmoe_scratch`, `qmoe_scratch_size`, `qmoe_host_scratch`,
  and `qmoe_host_scratch_size` from `RuntimeState`. **Remove** the four
  function bodies from `hipdnn_ep_runtime_state.cpp` (they live in
  `qmoe.cpp` now).
- **Defer** the shared `workspace` field migration. The workspace is
  multi-op (GQA, MatMul GEMM, MIOpen conv) and genuinely framework-tier;
  routing every consumer through a module slot would add a hot-path
  indirection with no architectural win.

**Acceptance:** gpt-oss-20b QMoE decode passes; no leak across 5
session create / destroy cycles measured by `hipMemGetInfo`.

### Stage 5 — Observability + documentation

- **Add** `hipdnn_ep_dump_state_enabled()` to `lib/Runtime/debug_log.h`
  (uses `GetEnvironmentVariableA` to match the existing PERF/DEBUG
  helpers).
- **Wire** `module_registry_dump` in `hipdnn_ep_state_cleanup` under the
  `HIPDNN_EP_DUMP_STATE=1` guard.
- **Update** `CLAUDE.md`: new "Runtime module system (op-state
  registry)" section; "Adding a New Operator" mentions op-modules as the
  default mechanism; existing gotchas referencing the now-deleted fields
  rewritten.
- **Mark** [inline-prepack-cache.md](inline-prepack-cache.md) as
  Superseded and [workspace-improvements.md](workspace-improvements.md)
  as Partly subsumed, with explicit "what landed / what did not" lists.

**Acceptance:** docs build cleanly; verified end-to-end on the qmoe E2E
test that `HIPDNN_EP_DUMP_STATE=1` produces the expected stderr line.
All 58 CTest cases pass.

---

## Files Touched

| File | Stage | Change |
|---|---|---|
| `lib/Runtime/module_registry.h` | 1 | NEW (private header) |
| `lib/Runtime/module_registry.cpp` | 1 | NEW |
| `lib/Runtime/growable_buffer.h` | 4 | NEW (header-only) |
| `lib/Runtime/debug_log.h` | 5 | Add `hipdnn_ep_dump_state_enabled()` |
| `lib/Runtime/runtime_state_internal.h` | 1, 2-4 | Add `modules` pointer; remove seven per-op fields |
| `lib/Runtime/hipdnn_ep_runtime_state.cpp` | 1, 2-4, 5 | Create / destroy / begin_compute registry; remove per-op blocks |
| `lib/Runtime/hipdnn_ep_runtime.h` | 2-4 | Remove three typed `*_cache_destroy` decls; keep the four `hipdnn_ep_state_*_qmoe_*` decls |
| `lib/Runtime/CMakeLists.txt` | 1, 4 | Add new sources and headers to `DEPENDS` |
| `lib/Runtime/real/causal_conv_with_state.cpp` | 2 | `CausalConvState` |
| `lib/Runtime/real/matmul_nbits.cpp` | 3 | `ZpUnpackState` (shared by qmoe) |
| `lib/Runtime/real/gqa.cpp` | 3 | `GqaGemmState` + `GqaSeqlensCache` |
| `lib/Runtime/real/qmoe.cpp` | 4 | `QmoeState`; relocated C-ABI qmoe getters |
| `test/runtime_steady_state/` | 1 | NEW (native C++ smoke test, CTest #58) |
| `CMakeLists.txt` (root) | 1 | Add `test/runtime_steady_state` subdir |
| `CLAUDE.md` | 5 | "Runtime module system" section + gotcha rewrites |
| `docs/design/inline-prepack-cache.md` | 5 | Mark Superseded |
| `docs/design/workspace-improvements.md` | 5 | Mark Partly subsumed |

---

## Risks and Mitigations

Captured before implementation; outcomes recorded after.

| Risk | Outcome |
|---|---|
| Static-init ordering: spec table not initialized before first registration. | Construct-On-First-Use (function-local static `std::vector` inside `spec_table()`) eliminates the risk. Verified by the order-independent test in `test/runtime_steady_state/`. |
| Slot-lookup overhead measurable on hot path. | Measured: indistinguishable from the previous direct-field load. No TPS regression on any model in the perf table. |
| Stale bitcode in `%TEMP%/morphizen_mlir_*` after the migration. | Documented gotcha in `CLAUDE.md`; manual `del %TEMP%\morphizen_mlir_*` between stages. |
| `DEPENDS` list missed → header edits don't trigger bitcode rebuild. | `module_registry.h` and `growable_buffer.h` were added to the universal `DEPENDS` list of `compile_to_bitcode` at the top of `lib/Runtime/CMakeLists.txt`. Re-verified after every stage. |
| Plugin or future op forgets to register before its first session creation. | The macro registers at static-init via a function-local static; the slot id is captured the first time the accessor is invoked. Subsequent sessions reuse the slot. There is no "before first session" requirement. |
| Cross-CRT bitcode boundary breaks STL-typed registry. | Mitigated by keeping `RuntimeState`'s new field as an opaque pointer and the registry's storage inside `module_registry.cpp` only. Bitcode `wrap_*` TUs see only the function-pointer `OpModuleSpec` and free functions. |

---

## Design Alternatives Considered

Three storage-and-lookup designs were evaluated. The matrix below
summarizes the comparison; the chosen design is option **(C)** — integer
slot id under the hood, with the `HIPDNN_OP_MODULE` macro
([§A.1](#a1-opmodulespec-and-the-hipdnn_op_module-macro)) borrowing the
self-describing-key ergonomics from option (B).

| Dimension | (A) Typed pointer per op (pre-plan) | (B) `unordered_map<string_view, unique_ptr<OpStateBase>>` | **(C) Integer-id array (chosen)** |
|---|---|---|---|
| Adding a new op-with-state | Edit 3 files | `get_or_create<T>("name")` | One `HIPDNN_OP_MODULE` line |
| Hot-path lookup cost | ~3 ns (direct field) | ~60–150 ns (hash + bucket walk + string compare) | ~3 ns (array index) |
| Self-describing keys (logs / dump) | No | Yes | Yes (name lives in `OpModuleSpec`) |
| Polymorphism in state types | None | `OpStateBase` virtual destructor required | None — concrete C++ type via SFINAE hooks |
| Plugin support | No | Yes | Yes |
| `RuntimeState` stays C-ABI-friendly | Yes | No (STL types in struct) | Yes (opaque pointer) |
| Per-Compute() invalidation hooks | Hand-wired per cache | Polymorphic virtual call | SFINAE-detected, fan-out lock-free |
| Public surface in `hipdnn_ep_runtime.h` | Leaks `*_cache_destroy` decls | Clean | Clean |

### Why not (B)?

(B) is a perfectly defensible design and would be the natural fit in a
codebase where `RuntimeState` could freely hold STL types. Two factors
pushed HipDNN toward (C):

1. **Bitcode + static-CRT cross-module surface.** `RuntimeState` lives in
   `runtime_state_internal.h`, included by both the host-compiled runtime
   and the bitcode that gets linked into every compiled `model.dll` (per
   the static-CRT gotcha documented in `CLAUDE.md`). Putting
   `unordered_map<string_view, unique_ptr<OpStateBase>>` directly in
   `RuntimeState` widens that surface considerably — every consumer
   pulls in `<unordered_map>`, `<memory>`, `<string_view>`, plus
   `OpStateBase`'s vtable. Keeping the new field as a single opaque
   pointer (`hipdnn_ep::ModuleRegistry *modules;`) — with all STL hidden
   inside `module_registry.cpp` — mirrors the existing `op_profile`
   field and avoids that risk.

2. **Hot-path overhead.** At Llama 8B decode (~475 op calls per token at
   32 tok/s), a 100 ns string lookup adds ~1.5 ms/sec to decode (≈0.15%).
   Small in absolute terms, but free with (C).

The macro reclaims (B)'s real ergonomic win — names visible at the call
site — without paying either cost.

### Why not stay with (A)?

(A) is what HipDNN did before this plan. Its O(3 ns) field-access wins
on raw speed but loses every other dimension: cross-cutting edits per
new op, public-ABI leakage of typed cleanup decls, no plugin support,
hand-wired per-Compute() invalidation. The chosen design matches (A)'s
hot-path performance while removing all of those costs.

---

## As-Built (2026-05-13)

This section documents the **actual implementation** that landed against
the plan above. The plan stages are referenced by id; deviations and
shortcuts taken in the final code are called out explicitly.

### File map

| Path | Role | Plan stage |
|------|------|------------|
| `lib/Runtime/module_registry.h` | `OpModuleSpec`, function-pointer types, `make_op_module_spec<T>` template with SFINAE hook detection, `HIPDNN_OP_MODULE` macro, all public free functions. | 1 |
| `lib/Runtime/module_registry.cpp` | Process-global `spec_table()` (mutex-guarded vector), per-`RuntimeState` `ModuleRegistry` struct with `SlotEntry` cache, `op_module_get` cold/hot paths, `module_registry_dump`. | 1 |
| `lib/Runtime/runtime_state_internal.h` | Added `hipdnn_ep::ModuleRegistry *modules;` at the bottom. Removed seven op-specific fields (`gqa_gemm_cache`, `causal_conv_cache`, `zp_unpack_cache`, `qmoe_scratch` + size, `qmoe_host_scratch` + size, three `seqlens_k_cached_*`). | 1, 3, 4 |
| `lib/Runtime/hipdnn_ep_runtime_state.cpp` | Calls `module_registry_create()` in init, `module_registry_destroy()` (guarded by `module_registry_dump()` when `HIPDNN_EP_DUMP_STATE=1`) in cleanup, `module_registry_begin_compute()` in `hipdnn_ep_runtime_begin_compute()`. | 1, 5 |
| `lib/Runtime/hipdnn_ep_runtime.h` | Removed `hipdnn_ep_*_cache_destroy` decls; the C-ABI `hipdnn_ep_state_*_qmoe_*` getters are still declared but their implementations moved to `real/qmoe.cpp`. | 2, 3, 4 |
| `lib/Runtime/growable_buffer.h` | `GrowableDeviceBuffer` + `GrowablePinnedBuffer` with 1.5× growth, sync-before-free, header-only RAII. | 4 |
| `lib/Runtime/debug_log.h` | `hipdnn_ep_dump_state_enabled()` (Win32 `GetEnvironmentVariableA` path matches existing PERF/DEBUG helpers). | 5 |
| `lib/Runtime/CMakeLists.txt` | `module_registry.h` and `growable_buffer.h` added to the universal `DEPENDS` list of `compile_to_bitcode`; `module_registry.cpp` compiled to `runtime_module_registry.bc` and bundled into the runtime archive. | 1, 4 |
| `lib/Runtime/real/causal_conv_with_state.cpp` | `CausalConvState` (RAII over `unordered_map<CausalConvKey, CausalConvCacheEntry>` with MIOpen-descriptor teardown). | 2 |
| `lib/Runtime/real/matmul_nbits.cpp` | `ZpUnpackState` (RAII over two `unordered_map<const void*, (void*, size_t)>` for u8 / fp16 unpacked zp buffers, with `mem_bytes()`). Shared between `wrap_matmul_nbits` and `wrap_qmoe` via `lookup_or_*_zp_*` helpers. | 3 |
| `lib/Runtime/real/gqa.cpp` | `GqaGemmState` (RAII over hipBLASLt descriptor cache) + `GqaSeqlensCache` (begin_compute-invalidated per-Compute() cache of `seqlens_k`). | 3 |
| `lib/Runtime/real/qmoe.cpp` | `QmoeState` (two growable buffers, `mem_bytes()` reports combined footprint). Also hosts the four C-ABI `hipdnn_ep_state_*_qmoe_*` definitions that delegate to its slot. | 4 |
| `test/runtime_steady_state/runtime_steady_state.cpp` | Native C++ smoke test with two fake state types (`FullHooksState` + `MinimalState`), seven assertions; CTest #58. | 1 |
| `test/runtime_steady_state/CMakeLists.txt` | Builds `module_registry.cpp` natively (no bitcode) into the test executable; registers as CTest target. | 1 |

### What landed exactly as planned

* `HIPDNN_OP_MODULE(accessor, "name", StateT)` macro with two function-local
  statics (spec + slot id), C++11-guaranteed thread-safe initialization,
  steady-state hot path of "one bounds check + one load + one null branch".
* SFINAE-detected optional hooks: `begin_compute(RuntimeState*)`,
  `end_compute(RuntimeState*)`, `mem_bytes() const`. Required: a
  `(RuntimeState*)` constructor and a destructor (both compiler-supplied
  if not user-defined).
* `module_registry_begin_compute()` fan-out: iterates a tiny vector of
  cached fn-pointer / state-pointer pairs in registration order; modules
  without a `begin_compute` hook are skipped at zero cost (cached fn
  pointer is null).
* `module_registry_destroy()` runs in **reverse** registration order
  (insurance against future cross-module dependencies).

### Deviations from the plan

* **No `growable_buffer.cpp` split.** The plan considered a `.cpp` for the
  buffer helpers; in practice both classes inline cleanly into bitcode and
  a header-only template-free pair is simpler. No correctness or perf
  difference.
* **`workspace` field stayed in `RuntimeState`.** Stage 4 of the plan
  also considered migrating the shared workspace to `growable_buffer.h`.
  Skipped on purpose: the workspace is a true framework-level resource
  shared across many ops (GQA pipeline, MatMul GEMM, MIOpen conv), not
  an op-specific cache. Routing every consumer through a
  `WorkspaceState` module would add a slot indirection on the hot path
  without any architectural win. The hand-rolled grow logic in
  `hipdnn_ep_state_ensure_workspace` is small and stable.
* **Zp unpack cache is shared (`zp_unpack` module name, single slot).**
  Plan suggested per-op modules. The unpack cache is genuinely shared
  between `wrap_matmul_nbits` and `wrap_qmoe` (the asym AWQ unpack
  helpers in `zp_unpack_cache.h` are called from both), so it lives in
  matmul_nbits.cpp's TU and both ops reach it through the
  `lookup_or_*_zp_*` free functions. Splitting would have required
  caching the same `(zp_packed, dst, size)` tuple in two places.
* **`hipdnn_ep_state_*_qmoe_*` C-ABI preserved.** Plan considered
  deleting these and having `wrap_qmoe` reach into the QmoeState module
  directly. Kept the C-ABI because removing it would have rippled into
  every bitcode TU that included `hipdnn_ep_runtime.h` and the cost
  saving (one indirect call) is invisible on the qmoe hot path.

### Steady-state invariant (verified)

For a session whose Compute() loop drives a model with the migrated ops
(`causal_conv_with_state`, `gqa`, `matmul_nbits` asym, `qmoe`):

* **Per call:** `op_module_get` returns from the hot path
  (bounds check + load + null check) for every populated slot.
  No `hipMalloc`, no allocator, no string operations.
* **Per Compute() entry:** `module_registry_begin_compute` walks N slots
  (N = number of registered modules, currently ≤ 5) and calls each one's
  cached `begin_compute_fn`. Only `GqaSeqlensCache::begin_compute` does
  work today (resets two POD fields). No allocations, no syncs.
* **Per cleanup:** modules destruct in reverse registration order; each
  destructor frees the resources it owns; `module_registry_destroy()`
  frees the registry itself. No globals leaked.

The CTest #58 (`RuntimeSteadyState`) asserts these invariants on fake
state types with explicit counters for `begin_compute` calls per
`Compute()` entry.

### How to add a new op-module

1. Write `MyOpState` with a `(RuntimeState*)` constructor and any state
   members. Add a destructor if the members aren't RAII.
2. Optionally add `begin_compute(RuntimeState*)` for per-Compute()
   invalidation, and / or `mem_bytes() const` for `HIPDNN_EP_DUMP_STATE`.
3. Register: `HIPDNN_OP_MODULE(my_op_module, "my_op", MyOpState);` inside
   an anonymous namespace in the op's `.cpp`.
4. Use: `auto *s = my_op_module(state); s->cache.find(...);` inside the
   `wrap_*` body. No other files change. No CMake edits needed.

If the new module wants growable device or pinned-host scratch, prefer
composition with `hipdnn_ep::GrowableDeviceBuffer` /
`GrowablePinnedBuffer` from `growable_buffer.h` over hand-rolling the
grow logic.

### What did NOT migrate (and why)

| Symbol | Reason kept on `RuntimeState` |
|---|---|
| `workspace` / `workspace_size` | Genuinely framework-level, multi-op-shared. |
| `gpu_constants_blob`, `gpu_constants`, `num_constants`, `shared_constants_*` | Loaded once at `inference_init`; not op-specific. |
| `pool_base`, `buffer_offsets`, `num_buffers`, `pool_size` | Compiled-graph buffer pool. Compiler-emitted layout. |
| `op_profile` | Cross-op profiling state; would gain nothing from being a module. |
| `device_error_flag` | One int. Lifecycle pinned to state init / cleanup. |
| `hipdnn_handle`, `hipdnn_graph_registry` | Owned by the EP, attached after `inference_init` — not amenable to module lifecycle. |
| `stream`, `miopen_handle`, `hipblas_handle` | Library handles created in init, used by every module. |
