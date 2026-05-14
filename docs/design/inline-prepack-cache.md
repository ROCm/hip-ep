<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Inline Prepack Cache for `wrap_*` Functions

**Date:** 2026-05-12
**Document Type:** Implementation Plan (historical)
**Status:** **Superseded by [op-state-registry.md](op-state-registry.md)** (2026-05-13).

> ## Why this is obsolete
>
> The inline cache decorator approach proposed here was abandoned once the
> op-module registry landed. The registry covers the same use cases
> (per-op persistent state, no `RuntimeState` bloat, no public-ABI surface
> leakage) with a smaller diff and explicit per-Compute() invalidation
> hooks.
>
> **None of the symbols described below — `PrepackCache`,
> `hipdnn_ep_prepack_or_get`, `hipdnn_ep_prepack_cache_destroy`,
> `state->prepack_cache`, the `KIND_ZP_*` convention — exists in the
> codebase.** The two `zp_unpack_cache` use cases that motivated this
> plan migrated through a different route:
>
> 1. The two `unordered_map<const void*, (void*, size_t)>` caches are now
>    members of `ZpUnpackState` in `lib/Runtime/real/matmul_nbits.cpp`.
> 2. The op-module registry — not a JIT prepack helper — owns their
>    lifecycle.
> 3. The shared lookup helpers (`lookup_or_unpack_zp_u8`,
>    `lookup_or_convert_zp_fp16`) stayed as free functions in the same TU
>    and now reach the state through `zp_unpack_module(state)`.
>
> The "JIT prepack" abstraction was over-engineered for one consumer
> (the asym AWQ unpack), so it never shipped. If a future op genuinely
> needs JIT-prepacked GPU buffers with `(source_ptr, kind)` keying, build
> it as a new op-module on top of `module_registry.h` rather than a
> framework-tier prepack helper.
>
> See [op-state-registry.md › As-Built › Deviations](op-state-registry.md#deviations-from-the-plan)
> for the matching deviation note.
>
> This document is retained for historical context only; do **not**
> implement anything in it.

**Related:** [op-state-registry.md](op-state-registry.md)

---

## Table of Contents

- [Overview](#overview)
- [Motivation](#motivation)
- [Scope](#scope)
- [Architecture](#architecture)
  - [Cache types](#b1-cache-types)
  - [Storage in RuntimeState](#b2-storage-in-runtimestate)
  - [The hot-path function](#b3-the-hot-path-function)
  - [Op-side usage](#b4-op-side-usage-pattern)
  - [Cache key conventions](#b5-cache-key-conventions)
- [Implementation Stages](#implementation-stages)
- [Files Touched](#files-touched)
- [Risks and Mitigations](#risks-and-mitigations)
- [Estimate](#estimate)

---

## Overview

Generalize the `lookup_or_unpack_zp_u8` / `lookup_or_convert_zp_fp16` pattern
in `lib/Runtime/real/matmul_nbits.cpp` into a single reusable helper. Any
`wrap_*` function can call:

```cpp
const void *packed = hipdnn_ep_prepack_or_get(
    state, source_ptr, kind_id,
    plan_fn, execute_fn, plan_args);
```

…and receive a cached, on-GPU, prepacked buffer. The first call performs
plan + alloc + execute; subsequent calls hit the cache. The cache is owned by
`RuntimeState` and is freed at session cleanup.

This is a JIT (just-in-time) prepack pattern: the prepack happens on the
first compute call rather than at compile time or session-init time. It is
appropriate when the target format depends on data only seen at runtime, or
when first-call latency is acceptable.

---

## Motivation

| Problem today | After this plan |
|---|---|
| Each prepack-style cache is bespoke (`zp_u8` and `zp_fp16` are already two separate maps with copy-pasted plumbing) | One generic helper covers all of them |
| Adding a new prepack (e.g., MatMulNBits B-tile, QMoE expert weights) requires writing yet another near-identical cache | Just supply `plan_fn` + `execute_fn` — one helper call from `wrap_*` |
| Op-specific cleanup decl (`hipdnn_ep_zp_unpack_cache_destroy`) leaks into the public ABI of `hipdnn_ep_runtime.h` | Generic cache is owned by the framework; no per-op cleanup symbol |

---

## Scope

**In scope:**
- A generic, pointer-keyed prepack cache in `RuntimeState`.
- A single helper `hipdnn_ep_prepack_or_get` that op authors call from `wrap_*`.
- Migration of the two existing `zp_unpack_cache` lookups to the new helper.

**Out of scope:**
- Compile-time host-side prepacking (this plan is pure runtime JIT).
- Per-format tactic selection (stays in `wrap_*`; the op author knows which
  `kind_id` to pass for which tactic).
- The op-state registry from [op-state-registry.md](op-state-registry.md) —
  this plan does not depend on that one and can land independently.

---

## Architecture

### B.1 Cache types

The cache is a pointer-keyed `unordered_map`:

```cpp
// lib/Runtime/prepack_cache.cpp (private)

struct PrepackKey {
    const void *source_ptr;
    int64_t     kind_id;
    bool operator==(const PrepackKey &o) const {
        return source_ptr == o.source_ptr && kind_id == o.kind_id;
    }
};
struct PrepackKeyHash {
    size_t operator()(const PrepackKey &k) const {
        size_t h = 0;
        hash_combine_val(h, reinterpret_cast<uintptr_t>(k.source_ptr));
        hash_combine_val(h, k.kind_id);
        return h;
    }
};
struct PrepackEntry {
    void  *packed_buffer;
    size_t packed_bytes;
};
struct PrepackCache {
    std::unordered_map<PrepackKey, PrepackEntry, PrepackKeyHash> entries;
    std::mutex mu;   // matches the existing zp_unpack_cache locking style
};
```

The shared `hash_combine_val` helper from `lib/Runtime/real/cache_utils.h` is
reused.

### B.2 Storage in RuntimeState

A single new opaque field replaces today's `zp_unpack_cache`:

```c
// runtime_state_internal.h
struct RuntimeState {
    // ... existing fields ...

    // PrepackCache* (opaque). Created lazily on first lookup; freed in
    // hipdnn_ep_state_cleanup via hipdnn_ep_prepack_cache_destroy.
    void *prepack_cache;
};
```

### B.3 The hot-path function

Public C ABI in a new private header:

```c
// lib/Runtime/prepack_cache.h

#ifdef __cplusplus
extern "C" {
#endif

// Op-supplied: tells the framework how big the destination must be.
// Returns 0 on success, non-zero on error. Sets *out_packed_bytes.
typedef int (*HipdnnPrepackPlanFn)(const void *plan_args,
                                    size_t *out_packed_bytes);

// Op-supplied: launches the GPU kernel that writes prepacked data into
// `dest`. `stream` is the RuntimeState's HIP stream; `source` is the input
// buffer (caller-owned, stable). Returns 0 on success, non-zero on error.
typedef int (*HipdnnPrepackExecuteFn)(void *stream, const void *source,
                                       void *dest, const void *plan_args);

// Lookup-or-create. On hit returns the cached destination buffer. On miss:
// calls plan_fn -> hipMalloc -> execute_fn -> caches and returns.
//
// Returns NULL on hipMalloc / plan / execute failure (with stderr message).
//
// Key is (source_ptr, kind_id). Use distinct kind_ids when the same source
// can be packed into multiple formats (e.g., u8 vs fp16 unpack of zp).
const void *hipdnn_ep_prepack_or_get(
    struct RuntimeState *state,
    const void *source_ptr,
    int64_t kind_id,
    HipdnnPrepackPlanFn plan_fn,
    HipdnnPrepackExecuteFn execute_fn,
    const void *plan_args);

// Called from hipdnn_ep_state_cleanup. Not for op authors.
void hipdnn_ep_prepack_cache_destroy(void *cache);

#ifdef __cplusplus
}
#endif
```

Implementation sketch:

```cpp
const void *hipdnn_ep_prepack_or_get(
    RuntimeState *state, const void *source_ptr, int64_t kind_id,
    HipdnnPrepackPlanFn plan_fn, HipdnnPrepackExecuteFn execute_fn,
    const void *plan_args) {

    if (!state || !source_ptr || !plan_fn || !execute_fn) return nullptr;

    auto *cache = static_cast<PrepackCache *>(state->prepack_cache);
    if (!cache) {
        cache = new PrepackCache;
        state->prepack_cache = cache;
    }

    PrepackKey key{source_ptr, kind_id};

    std::lock_guard lock(cache->mu);
    auto it = cache->entries.find(key);
    if (it != cache->entries.end()) return it->second.packed_buffer;

    // Miss: plan + alloc + execute.
    size_t packed_bytes = 0;
    if (plan_fn(plan_args, &packed_bytes) != 0 || packed_bytes == 0) {
        return nullptr;
    }
    void *dst = nullptr;
    if (hipMalloc(&dst, packed_bytes) != hipSuccess) {
        fprintf(stderr,
                "hipdnn_ep_prepack_or_get: hipMalloc(%zu) failed\n",
                packed_bytes);
        return nullptr;
    }
    if (execute_fn(state->stream, source_ptr, dst, plan_args) != 0) {
        hipFree(dst);
        return nullptr;
    }

    cache->entries.emplace(key, PrepackEntry{dst, packed_bytes});
    return dst;
}
```

### B.4 Op-side usage pattern

Replacing the existing `lookup_or_unpack_zp_u8` call site in
`lib/Runtime/real/matmul_nbits.cpp`:

```cpp
namespace {
// kind_ids for this op — file-local, never escape.
constexpr int64_t KIND_ZP_U8   = 0x100001;
constexpr int64_t KIND_ZP_FP16 = 0x100002;

struct ZpPlanArgs {
    int N;
    int groups_k;
};

int plan_zp_u8(const void *args, size_t *out_bytes) {
    auto *a = static_cast<const ZpPlanArgs *>(args);
    *out_bytes = static_cast<size_t>(a->N) * static_cast<size_t>(a->groups_k);
    return 0;
}
int execute_zp_u8(void *stream, const void *src, void *dst,
                  const void *args) {
    auto *a = static_cast<const ZpPlanArgs *>(args);
    hip_matmul_nbits_unpack_zp_u8(stream, src, dst, a->N, a->groups_k);
    return 0;
}

int plan_zp_fp16(const void *args, size_t *out_bytes) {
    auto *a = static_cast<const ZpPlanArgs *>(args);
    *out_bytes = static_cast<size_t>(a->N) *
                 static_cast<size_t>(a->groups_k) * sizeof(__fp16);
    return 0;
}
int execute_zp_fp16(void *stream, const void *src, void *dst,
                    const void *args) {
    auto *a = static_cast<const ZpPlanArgs *>(args);
    hip_matmul_nbits_convert_zp_fp16(stream, src, dst, a->N, a->groups_k);
    return 0;
}
} // namespace

// Inside wrap_matmul_nbits, the asym path:
const ZpPlanArgs args{static_cast<int>(N), ngk};
pre_zp_u8 = hipdnn_ep_prepack_or_get(state, zero_points, KIND_ZP_U8,
                                      plan_zp_u8, execute_zp_u8, &args);
if (!pre_zp_u8) return -1;

if (wmma_data_format && M > 1) {
    pre_zp_fp16 = hipdnn_ep_prepack_or_get(state, zero_points, KIND_ZP_FP16,
                                            plan_zp_fp16, execute_zp_fp16,
                                            &args);
    if (!pre_zp_fp16) return -1;
}
```

The op author writes ~10 lines of boilerplate (plan + execute pair plus
kind_id constants). The framework owns cache, allocation, mutex, and cleanup.

### B.5 Cache key conventions

The cache key is `(source_ptr, kind_id)`.

- **`source_ptr`** is the GPU pointer of the unpacked / un-prepacked source
  buffer. For HipDNN today, source pointers are stable for session lifetime
  because they point into `gpu_constants_blob`. This stability is the
  precondition that makes pointer-keyed caching correct.
- **`kind_id`** disambiguates multiple prepack formats over the same source
  (e.g., the same zero_points blob unpacked into u8 vs converted into fp16).

**Convention for `kind_id` values** (no central registry needed):

- High byte identifies the op family: `0x10` = matmul_nbits, `0x20` = qmoe,
  `0x30` = future ops.
- Low bytes are op-local enumerations.

This convention makes collisions visible by inspection and avoids the cost of
a central allocation authority. If a collision ever occurs, both ops will
share entries and produce wrong results — the convention's hygiene is the
defense.

---

## Implementation Stages

Each stage is independently shippable, leaves CI green, and does not change
observable behavior.

### Stage 1 — Add the generic cache (~1 day)

- **Add** `lib/Runtime/prepack_cache.h` with the public C ABI.
- **Add** `lib/Runtime/prepack_cache.cpp` with `PrepackKey` / `PrepackEntry`
  / `PrepackCache` types, the `hipdnn_ep_prepack_or_get` body, and
  `hipdnn_ep_prepack_cache_destroy`.
- **Modify** `lib/Runtime/runtime_state_internal.h`: add `void *prepack_cache;`.
- **Modify** `lib/Runtime/hipdnn_ep_runtime_state.cpp`:
  - Zero-init `prepack_cache` in `initialize_state_handles`.
  - Call `hipdnn_ep_prepack_cache_destroy(state->prepack_cache)` from
    `hipdnn_ep_state_cleanup`.
- **Modify** `lib/Runtime/CMakeLists.txt`: add the new source. Update the
  `compile_to_bitcode` `DEPENDS` list per the build gotcha rule documented in
  `CLAUDE.md` — every header included by a runtime `.cpp` must be in
  `DEPENDS` or stale bitcode results.

**Acceptance:** Full build green, all tests pass, the helper is dormant
(nothing calls it yet).

### Stage 2 — Migrate `lookup_or_unpack_zp_u8` (~0.5 day)

- **Modify** `lib/Runtime/real/matmul_nbits.cpp`:
  - Add `KIND_ZP_U8`, `ZpPlanArgs`, `plan_zp_u8`, `execute_zp_u8` per
    [§B.4](#b4-op-side-usage-pattern).
  - Replace the call site of `lookup_or_unpack_zp_u8` in
    `wrap_matmul_nbits` with the generic helper.
- **Modify** `lib/Runtime/real/qmoe.cpp` similarly — `qmoe.cpp` also
  consumes `lookup_or_unpack_zp_u8` (per the existing
  `zp_unpack_cache.h` contract).

**Acceptance:**
- Llama 8B asym MatMulNBits decode is bit-exact with the previous
  implementation.
- gpt-oss-20b QMoE path passes.
- Decode TPS for both within ±2% of baseline.
- `hipMemGetInfo` shows no leak across the cleanup path (run a model 5×,
  measure delta).

### Stage 3 — Migrate `lookup_or_convert_zp_fp16` (~0.5 day)

Same shape as Stage 2 but for the fp16 variant. The WMMA path
(`batch==1 && K%32==0 && M>=16`) is the only consumer; exercised by Llama 8B
prefill at certain shapes.

**Acceptance:** Same as Stage 2; WMMA path verified.

### Stage 4 — Decommission the bespoke `zp_unpack_cache` (~0.5 day)

- **Delete** `lib/Runtime/real/zp_unpack_cache.h` (no longer included anywhere).
- **Delete** the `ZpUnpackCache` struct, `lookup_or_unpack_zp_u8`,
  `lookup_or_convert_zp_fp16`, and `hipdnn_ep_zp_unpack_cache_destroy`
  definitions from `lib/Runtime/real/matmul_nbits.cpp`.
- **Delete** `state->zp_unpack_cache` field from `runtime_state_internal.h`.
- **Delete** `hipdnn_ep_zp_unpack_cache_destroy` declaration from
  `hipdnn_ep_runtime.h`.
- **Delete** the `if (state->zp_unpack_cache)` block from
  `hipdnn_ep_state_cleanup` and the corresponding nullptr init in
  `initialize_state_handles`.
- **Update** `lib/Runtime/CMakeLists.txt` `DEPENDS` lists for both
  `matmul_nbits.cpp` and `qmoe.cpp`.

**Acceptance:**
- `rg "zp_unpack_cache"` returns no matches anywhere in the repo.
- All tests pass; decode TPS unchanged.

### Stage 5 — Documentation (~0.25 day)

- **Update** `CLAUDE.md`: under "Adding a New Operator" or "Architecture",
  add a section "JIT prepack from `wrap_*`" with a 5-line code template
  showing the canonical usage.

**Acceptance:** docs build cleanly; the example compiles when copy-pasted
into a new op skeleton.

---

## Files Touched

| File | Stage | Change |
|---|---|---|
| `lib/Runtime/prepack_cache.h` | 1 | NEW |
| `lib/Runtime/prepack_cache.cpp` | 1 | NEW |
| `lib/Runtime/runtime_state_internal.h` | 1, 4 | add field; remove `zp_unpack_cache` |
| `lib/Runtime/hipdnn_ep_runtime_state.cpp` | 1, 4 | add cleanup call; remove old block |
| `lib/Runtime/hipdnn_ep_runtime.h` | 4 | remove `hipdnn_ep_zp_unpack_cache_destroy` decl |
| `lib/Runtime/CMakeLists.txt` | 1, 4 | add new sources; update DEPENDS lists |
| `lib/Runtime/real/matmul_nbits.cpp` | 2, 3, 4 | migrate then delete bespoke cache |
| `lib/Runtime/real/qmoe.cpp` | 2, 3 | migrate (also calls the zp helpers) |
| `lib/Runtime/real/zp_unpack_cache.h` | 4 | DELETE |
| `CLAUDE.md` | 5 | document the pattern |

---

## Risks and Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Stale model DLL in `%TEMP%` references old runtime symbols | High | Already documented in `CLAUDE.md`; delete `%TEMP%/morphizen_mlir_*` between stages |
| Bitcode `DEPENDS` list missed → header edits don't trigger rebuild | Medium | Re-read `lib/Runtime/CMakeLists.txt` `compile_to_bitcode` calls before each stage |
| Pointer aliasing: a different source pointer compares equal because `hipMalloc` reuses freed memory | Low (constants are stable today) | Document the constraint: "JIT prepack assumes source pointer is stable for session lifetime"; that holds for `gpu_constants_blob` pointers |
| `kind_id` collision between two ops | Low | Convention in [§B.5](#b5-cache-key-conventions): top byte is op family |
| Concurrency regression if RuntimeState ever goes multi-threaded | Low | `std::mutex` in `PrepackCache` matches existing `ZpUnpackCache` lock — no regression |

---

## Estimate

**Total: ~2.75 days of focused work**, plus ~0.5 day for testing and rollout
= **~3.25 days end-to-end**.

---

## Relationship to Op-State Registry Plan

This plan is independent of [op-state-registry.md](op-state-registry.md).
Either may land first.

If the op-state registry lands first, the `prepack_cache` field added by
this plan can later be moved into the registry as a follow-up — that
follow-up is optional and not part of this plan's acceptance criteria.

If this plan lands first, the registry plan's Stage 3 (zp_unpack_cache
migration) becomes a no-op since `zp_unpack_cache` no longer exists.
