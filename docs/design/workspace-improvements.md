<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Workspace Lifecycle Improvements

**Date:** 2026-05-13
**Document Type:** Implementation Plan (partially historical)
**Status:** **Partly subsumed by [op-state-registry.md](op-state-registry.md)** (2026-05-13).

> ## Section-by-section status
>
> This doc has four architecture pieces (A.1 – A.4). Only the
> `growable_buffer.h` half of A.1 reached the codebase, and even that
> landed under a different name and shape than described below. Treat
> the rest as historical exploration.
>
> | Section | Status | Where it lives now |
> |---|---|---|
> | A.1 Pinned host workspace **(helper)** | **Implemented** | `lib/Runtime/growable_buffer.h::GrowablePinnedBuffer`; used by `QmoeState::host_scratch` in `real/qmoe.cpp`. |
> | A.1 Pinned host workspace **(`hipdnn_ep_state_*_pinned_workspace` C-ABI)** | **NOT implemented** | The four functions described below were never added. `qmoe_host_scratch` migrated through an op-module slot instead, preserving the existing `hipdnn_ep_state_*_qmoe_host_scratch` C-ABI. |
> | A.2 Workspace arena helper | **NOT implemented** | `lib/Runtime/workspace_arena.h` does not exist. Wrappers still compute sub-buffer offsets manually. |
> | A.3 Advisory size hint in model metadata | **NOT implemented** | `workspace_size_hint` / `pinned_workspace_size_hint` were never added to `schemas/model_metadata.fbs`. No codegen pass computes a workspace upper bound. |
> | A.4 Optional phase-transition checkpoint hook | **NOT implemented** | `hipdnn_ep_state_workspace_release_oversize` does not exist. |
>
> **What also did NOT happen** (and shouldn't, going forward):
> migrating the shared `workspace` field itself to the new helper. The
> workspace is a true framework-level multi-op resource; modularizing it
> would add a slot indirection on the hot path without an architectural
> win. The existing `hipdnn_ep_state_ensure_workspace` stays — see the
> "workspace stayed on `RuntimeState`" deviation in
> [op-state-registry.md › As-Built › Deviations](op-state-registry.md#deviations-from-the-plan).
>
> Retained here for historical context — do **not** implement A.2 / A.3 /
> A.4 from this doc without first confirming the design against
> `op-state-registry.md` and the current state of `growable_buffer.h`.

**Related:** [op-state-registry.md](op-state-registry.md), [inline-prepack-cache.md](inline-prepack-cache.md)

---

## Table of Contents

- [Overview](#overview)
- [What the current design gets right](#what-the-current-design-gets-right)
- [Limits worth fixing](#limits-worth-fixing)
- [Architecture](#architecture)
  - [Pinned host workspace as a peer facility](#a1-pinned-host-workspace-as-a-peer-facility)
  - [Workspace arena helper](#a2-workspace-arena-helper)
  - [Advisory size hint in model metadata](#a3-advisory-size-hint-in-model-metadata)
  - [Optional phase-transition checkpoint hook](#a4-optional-phase-transition-checkpoint-hook)
- [Out of Scope](#out-of-scope)
- [Implementation Stages](#implementation-stages)
- [Files Touched](#files-touched)
- [Risks and Mitigations](#risks-and-mitigations)
- [Estimate](#estimate)
- [Relationship to Other Plans](#relationship-to-other-plans)

---

## Overview

The runtime has one **session-lifetime device workspace** today
(`hipdnn_ep_state_ensure_workspace` / `hipdnn_ep_state_get_workspace` in
`lib/Runtime/hipdnn_ep_runtime_state.cpp`). It is grown on demand with 1.5×
amortization, never shrinks, syncs the stream before reallocation, and is
shared across all ops on the same stream within and across `Compute()` calls.

This design is correct in shape — keeping a single buffer alive through the
inference lifecycle eliminates per-call `hipMalloc` / `hipFree` storms that
otherwise dominate small-shape decode. The improvements proposed in this
document keep that core unchanged and address three secondary gaps:

1. The pinned-host counterpart is hand-rolled per-op (`qmoe_host_scratch`).
2. Wrappers manually compute byte offsets into the workspace.
3. Workspace grows reactively during the first `Compute()` calls, making TTFT
   nondeterministic relative to the same-shape steady state.

Each improvement is independent and shippable on its own.

---

## What the current design gets right

These properties are intentional and must be preserved through any change:

- **Single device buffer**, `hipMalloc`'d once and reused across all
  workspace-using ops on the stream.
- **1.5× growth**, so a generation issues O(log N) reallocations rather than
  one per growing-shape transition.
- **Never shrinks** during normal operation. Prefill→decode workspace
  reuse is the common case and freeing only to re-`hipMalloc` would waste
  cycles.
- **Sync-before-free** on growth, preventing use-after-free against in-flight
  async kernels that still reference the old pointer.
- **Cross-Compute reuse**, so generation-loop iterations pay zero
  workspace-management cost in steady state.

---

## Limits worth fixing

| Symptom today | Root cause |
|---|---|
| `qmoe_host_scratch` re-implements the entire grow-on-demand-never-shrink dance with `hipHostMalloc` because the shared workspace is device-only | No peer pinned-host facility; ops that need pinned host scratch for async D2H/H2D each reinvent it |
| `wrap_causal_conv_with_state`, `wrap_qmoe`, `wrap_gqa` each compute sub-buffer offsets by hand (`off_a = 0`, `off_b = off_a + sz_a`, …), with manual 64-byte alignment, then call `ensure_workspace(total)` and pointer-arithmetic out of the base | No abstraction for "carve N typed sub-buffers out of the workspace"; bug class includes mis-aligned offsets, miscomputed totals, and forgetting to update offset arithmetic when adding a sub-buffer |
| First `Compute()` after session creation reallocates the workspace one or more times as the heaviest ops widen it; TTFT is nondeterministic and ~1–2 ms slower than the second-call steady state | Workspace grows only reactively at the first request; the codegen knows the max shape but doesn't tell the runtime |

---

## Architecture

Four orthogonal additions, in priority order. Each can ship without the
others.

### A.1 Pinned host workspace as a peer facility

Mirror the device-workspace API for pinned host memory:

```c
void  *hipdnn_ep_state_get_pinned_workspace(RuntimeState *state);
size_t hipdnn_ep_state_get_pinned_workspace_size(RuntimeState *state);
int    hipdnn_ep_state_ensure_pinned_workspace(RuntimeState *state,
                                               size_t needed_size);
```

Implementation parallels `hipdnn_ep_state_ensure_workspace`:

- `hipHostMalloc(..., hipHostMallocDefault)` instead of `hipMalloc`.
- Same 1.5× growth, never-shrink, sync-before-realloc rules.
- Stored on `RuntimeState` as `void *pinned_workspace; size_t pinned_workspace_size;`.
- Freed in `hipdnn_ep_state_cleanup` via `hipHostFree`.

This is the **only** new public ABI surface in this plan, and it is purely
additive (older model.dlls never call it; newer model.dlls fall back to
device workspace if the runtime does not export it).

**Migration:** `wrap_qmoe`'s `qmoe_host_scratch` collapses to "use pinned
workspace." The `qmoe_host_scratch` field, its accessors, and its cleanup
block come out of `RuntimeState` and `hipdnn_ep_state_cleanup`.

> **Why a peer facility, not a single combined scratch:** device and host
> pinned memory have different alignment, allocator latency profiles, and
> use patterns. Combining them would force the larger of the two
> requirements onto both, wasting memory. Two facilities, same shape.

### A.2 Workspace arena helper

A header-only RAII helper in `lib/Runtime/workspace_arena.h` that handles
sub-buffer allocation, alignment, and total sizing in one place:

```cpp
class WorkspaceArena {
public:
  // Captures state* and the starting offset (= 0).
  explicit WorkspaceArena(RuntimeState *state);

  // Reserve a sub-buffer in the arena. Increments the running total.
  // Returns nullptr until commit() is called.
  void *reserve(size_t bytes, size_t align = 64);

  // Commit phase: ensures the underlying workspace is at least the
  // running total, then resolves all reserved sub-buffer pointers.
  // Must be called once after all reserve() calls. Returns 0 on success.
  int commit();

  // Returns the resolved pointer for a previously reserved slot.
  // Only valid after commit().
  void *get(int slot_index) const;

  size_t total_bytes() const;

private:
  RuntimeState *state_;
  std::vector<size_t> offsets_;
  size_t running_total_;
  bool committed_;
  char *base_;
};
```

Or, in a one-pass form callers tend to prefer:

```cpp
WorkspaceArena ws(state);
size_t off_virtual = ws.reserve(virtual_size, /*align=*/64);
size_t off_sigmoid = ws.reserve(sigmoid_size);
size_t off_conv_ws = ws.reserve(conv_workspace_size);
if (ws.commit() != 0)
  return -1;
void *virtual_buf  = ws.at(off_virtual);
void *sigmoid_buf  = sigmoid_size ? ws.at(off_sigmoid) : nullptr;
void *conv_ws_buf  = conv_workspace_size ? ws.at(off_conv_ws) : nullptr;
```

Same memory, same growth, same single `ensure_workspace` call — but
alignment, running-total tracking, and pointer resolution are encapsulated.
Eliminates a real bug class (mis-computed offsets, alignment forgotten on a
new sub-buffer, total wrong by one slot).

**This is cosmetic from the runtime's perspective** — no new device
allocation strategy, no new ABI. Adopting it is purely a wrapper-side
refactor.

### A.3 Advisory size hint in model metadata

The codegen knows every `(op, shape)` tuple that will run during inference
and could compute the maximum workspace requirement at compile time. Embed
that as an optional **hint** in the FlatBuffers metadata blob
(`schemas/model_metadata.fbs`):

```fbs
table HipModelMetaInfo {
  version: int32 = 1;
  constants_filename: string;
  constants: [ConstantInfo];
  input_count: int64;
  output_count: int64;
  inputs: [TensorInfo];
  outputs: [TensorInfo];
  workspace_size_hint: int64 = 0;          // NEW: 0 = no hint, advisory only
  pinned_workspace_size_hint: int64 = 0;   // NEW: 0 = no hint, advisory only
}
```

At session init (immediately after constants are loaded in
`hipdnn_ep_state_init_with_fs`), the runtime reads the hints and pre-grows
the workspaces:

```cpp
if (auto hint = meta->workspace_size_hint(); hint > 0)
  hipdnn_ep_state_ensure_workspace(state, static_cast<size_t>(hint));
if (auto hint = meta->pinned_workspace_size_hint(); hint > 0)
  hipdnn_ep_state_ensure_pinned_workspace(state, static_cast<size_t>(hint));
```

**Critical property: the hint is advisory, not contractual.**

- A model.dll built without the hint emits 0 (or omits the field). The
  runtime sees 0 and behaves exactly as today (lazy reactive growth). **Old
  model.dlls keep working.**
- A model.dll built with a hint, run against an older runtime that doesn't
  read the field, falls back to lazy growth. **Old runtimes keep working.**
- A model.dll with a too-small hint (codegen bug, wrapper added new
  workspace requirement after compile time, etc.) still produces correct
  results — `ensure_workspace` grows reactively from there.

This sidesteps the codegen↔runtime ABI coupling concern raised in the
op-state-registry discussion: the hint is **not** part of the wrapper ABI.
The runtime never depends on it being present, present-and-correct, or
present-and-large-enough. It is a one-way performance hint.

The only real cost is in the codegen: walking the lowered IR after
bufferization to sum each op's known workspace footprint. The op-side
"how much workspace do I need for shape X?" function is op-specific and
must be authored once per workspace-using op family. That is bounded work
(today: causal_conv, gqa, gemm, matmul) and grows only with new
workspace-consuming ops.

> **Why this isn't full compile-time `op_init` emission:** that approach
> would require codegen to emit named init calls per op into
> `inference_init`, with backwards-compat semantics on missing symbols
> across runtime/model.dll generation skew. The hint approach achieves the
> "predictable TTFT" benefit with strictly less coupling.

### A.4 Optional phase-transition checkpoint hook

For the rare workload where prefill workspace dwarfs decode workspace and
the session runs many decode tokens after prefill, expose an opt-in hook
that lets the EP signal "leaving large-workspace territory":

```c
// EP-side, called at most once per phase transition.
void hipdnn_ep_state_workspace_release_oversize(RuntimeState *state);
```

Implementation: if `workspace_size > peak_recent / 2`, free and reallocate
to `peak_recent`; otherwise no-op. `peak_recent` is a high-water tracked
since the last call.

**Default behavior unchanged.** EPs that don't call this don't pay any
runtime cost. The hook is also pure runtime — no codegen knowledge, no
per-Compute scheduling.

This is **lower priority** than A.1–A.3 and should land only if memory
pressure on long-decode workloads is actually measured (so far it hasn't
been). Listed here to avoid re-litigating the design later.

---

## Out of Scope

These were considered and rejected:

| Idea | Rejection reason |
|---|---|
| Tiered workspaces (small/medium/large pools) | Single-stream sequential reuse already wins; tiered pools fragment memory for no measurable benefit |
| Cross-session shared workspace (analogous to constants blob sharing) | Concurrent sessions have independent streams; safe shared access would require cross-session barriers — wrong threading model |
| `hipMallocManaged` / unified memory for workspace | Hot-path scratch hitting the paging layer is a perf cliff |
| Per-op private workspaces | Loses the cross-op reuse that makes the current single-buffer design work |
| Periodic shrink heuristic on idle | Premature; no measured memory pressure today, and shrink-then-grow defeats the cache-warmth of `hipMalloc` reuse |
| Compile-time-emitted `op_init`/`op_cleanup` calls per op | Couples codegen to runtime ABI in a way A.3's advisory hint avoids; benefit is only marginally larger; cost is significantly larger |
| Unifying workspace with the inference memory pool (`pool_base`) | Different lifetimes (pool buffers are per-op-call output buffers, workspace is per-op-call temporary scratch); merging them would require tracking finer lifecycle boundaries |

---

## Implementation Stages

Each stage is independently shippable, leaves CI green, and does not change
observable model output.

### Stage 1 — Pinned host workspace facility (~1 day)

- **Modify** `lib/Runtime/runtime_state_internal.h`: add
  `void *pinned_workspace; size_t pinned_workspace_size;`. Document
  parallel to existing `workspace` fields.
- **Modify** `lib/Runtime/hipdnn_ep_runtime.h`: add the three new
  `pinned_workspace` accessors / `ensure` declarations.
- **Modify** `lib/Runtime/hipdnn_ep_runtime_state.cpp`:
  - Zero-init in `initialize_state_handles`.
  - Implement `hipdnn_ep_state_get_pinned_workspace`,
    `hipdnn_ep_state_get_pinned_workspace_size`, and
    `hipdnn_ep_state_ensure_pinned_workspace` mirroring the existing
    device-workspace functions but with `hipHostMalloc` /
    `hipHostFree`.
  - Add `hipHostFree(state->pinned_workspace)` to
    `hipdnn_ep_state_cleanup`.
- **Modify** `lib/Runtime/CMakeLists.txt`: update the `compile_to_bitcode`
  `DEPENDS` list per the build gotcha in `CLAUDE.md`.

**Acceptance:** Full build green, all tests pass, the facility is dormant
(nothing calls it yet).

### Stage 2 — Migrate `qmoe_host_scratch` to the pinned workspace (~0.5 day)

- **Modify** `lib/Runtime/real/qmoe.cpp`: replace
  `hipdnn_ep_state_ensure_qmoe_host_scratch` /
  `hipdnn_ep_state_get_qmoe_host_scratch` calls with the pinned-workspace
  equivalents.
- **Delete** `qmoe_host_scratch` and `qmoe_host_scratch_size` fields from
  `RuntimeState` (`runtime_state_internal.h`).
- **Delete** the `qmoe_host_scratch`-specific `ensure` / `get` functions
  from `hipdnn_ep_runtime_state.cpp` and their declarations from
  `hipdnn_ep_runtime.h`.
- **Delete** the `qmoe_host_scratch` cleanup block from
  `hipdnn_ep_state_cleanup`.

**Acceptance:**
- gpt-oss-20b QMoE decode passes.
- Decode TPS within ±2% of baseline.
- `hipMemGetInfo` shows no leak across cleanup (run a model 5×, measure
  delta).
- `rg "qmoe_host_scratch"` returns no matches anywhere.

### Stage 3 — Workspace arena helper (~1 day)

- **Add** `lib/Runtime/workspace_arena.h` with the helper class.
- **Modify** `lib/Runtime/real/causal_conv_with_state.cpp`: replace the
  hand-rolled offset arithmetic in the conv-path workspace setup
  (`virtual_buf` / `sigmoid_buf` / `conv_workspace`) with the arena
  helper. Behavior unchanged.
- **Modify** `lib/Runtime/real/qmoe.cpp`: replace the device-side
  sub-buffer offset computation similarly. Behavior unchanged.
- **Modify** `lib/Runtime/real/gqa.cpp`: replace the workspace
  partitioning (Q/K split, rope temps, flash partials) similarly.
  Behavior unchanged.

**Acceptance:**
- All affected models bit-exact with previous output (Llama 1B/8B,
  gpt-oss-20b, causal-conv-using models).
- TPS within ±1% of baseline (cosmetic refactor; no perf delta expected).
- Static analysis clean (no new warnings).

### Stage 4 — Advisory size hint (~2–3 days)

- **Modify** `schemas/model_metadata.fbs`: add the two new optional fields
  with default 0. Regenerate `model_metadata_generated.h`.
- **Modify** the codegen pipeline (`lib/Dialect/Transforms/`) — likely a
  new pass that runs after bufferization and before
  `GenerateInterface.cpp` emits the metadata blob:
  - Walk every workspace-using op in the lowered IR.
  - Each such op contributes a per-shape workspace estimate via a
    table-driven helper (`WorkspaceEstimate.cpp`) keyed on the op name.
  - The pass takes the max across the graph and writes
    `workspace_size_hint` into the metadata blob.
  - Pinned-workspace estimate is computed similarly (qmoe is the only
    consumer today).
- **Modify** `lib/Runtime/hipdnn_ep_runtime_state.cpp`: in
  `hipdnn_ep_state_init_with_fs`, after the constants load completes,
  read the two hint fields and call the corresponding `ensure_*` if the
  hint is non-zero. **Defensive**: ignore wildly large hints (e.g., > 1
  GB) with a warning.
- **Add** a new bool `workspace_pre_grew` to `RuntimeState` for diagnostic
  purposes only — `[workspace]` log line in `RUNTIME_DEBUG_LOG` reports
  whether the workspace was pre-grown by hint vs. lazy first-call growth.

**Acceptance:**
- Old model.dlls (compiled before this change) still produce correct
  output and run with lazy growth. **No regression.**
- New model.dlls report `[workspace] pre-grew to X bytes via hint` on the
  first session creation log.
- First-`Compute()` time on Llama-8B at L=128 is within 0.5 ms of
  steady-state same-shape `Compute()` time. (Today, first call is
  ~1–2 ms slower due to reallocation.)
- TPS unchanged.
- A deliberately-too-small hint (test-only override) still produces
  correct output via lazy growth.

### Stage 5 — Documentation (~0.25 day)

- **Update** `CLAUDE.md`:
  - Under "Critical Build Gotchas", document the FlatBuffers schema
    version implications of adding the hint fields and the regenerate
    step.
  - Under "Adding a New Operator", add a 5-line code template for using
    `WorkspaceArena` and (if applicable) the pinned workspace.
- **Update** `docs/design/compiler-runtime-contract.md` to note that
  workspace-size hints are advisory, not part of the contract.

**Acceptance:** docs build cleanly; the example compiles when copy-pasted
into a new op skeleton.

### Stage 6 (optional, defer) — Phase-transition release hook (~0.5 day)

Only land if memory pressure on long-decode workloads is measured.
Implementation per [§A.4](#a4-optional-phase-transition-checkpoint-hook).

---

## Files Touched

| File | Stage | Change |
|---|---|---|
| `lib/Runtime/runtime_state_internal.h` | 1, 2 | Add `pinned_workspace` fields; remove `qmoe_host_scratch` fields |
| `lib/Runtime/hipdnn_ep_runtime.h` | 1, 2 | Add 3 pinned-workspace decls; remove qmoe-host-scratch decls |
| `lib/Runtime/hipdnn_ep_runtime_state.cpp` | 1, 2, 4 | Implement pinned-workspace; remove qmoe-host-scratch impl; read hint at init |
| `lib/Runtime/workspace_arena.h` | 3 | NEW |
| `lib/Runtime/real/qmoe.cpp` | 2, 3 | Migrate to pinned workspace; arena |
| `lib/Runtime/real/causal_conv_with_state.cpp` | 3 | Arena |
| `lib/Runtime/real/gqa.cpp` | 3 | Arena |
| `lib/Runtime/CMakeLists.txt` | 1, 3 | DEPENDS list updates |
| `schemas/model_metadata.fbs` | 4 | Add `workspace_size_hint`, `pinned_workspace_size_hint` |
| `lib/Dialect/Transforms/` (new pass) | 4 | Compute workspace estimate, write into metadata |
| `lib/Dialect/Transforms/CMakeLists.txt` | 4 | Register new pass |
| `lib/Dialect/Transforms/Pipelines.cpp` | 4 | Insert new pass after bufferization |
| `CLAUDE.md` | 5 | Document the pattern |
| `docs/design/compiler-runtime-contract.md` | 5 | Note advisory nature of hint |

---

## Risks and Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Pinned-workspace allocation hits `hipHostMalloc` failure on memory-pressured systems where regular `hipMalloc` would have succeeded | Low | Ensure-function returns `-1` on failure; callers already check the return code. Document that pinned host allocation can fail before device allocation does. |
| Stale model.dll in `%TEMP%` references runtime symbols that exist only in the new runtime | High | Already documented in `CLAUDE.md`; delete `%TEMP%/morphizen_mlir_*` between stages 1 and 2. |
| Workspace arena ergonomics worse than the hand-rolled offset arithmetic in some callers (e.g., when offsets are reused for both compute and a later `present_state` extraction) | Low | Arena exposes both `at(slot_index)` and a raw `offset_of(slot_index)` for callers that need to do additional pointer math. |
| Codegen workspace estimator under-predicts for some op (e.g., a conv whose Find returns a larger workspace than `GetWorkSpaceSize` reported) | Medium | Hint is advisory; lazy growth handles under-prediction correctly. Acceptance test in Stage 4 includes a deliberately-too-small hint. |
| FlatBuffers schema change breaks existing model.dlls | Low | Adding optional fields with defaults is backwards-compatible by FlatBuffers design; verified by Stage 4 acceptance test on a pre-change DLL. |
| Bitcode `DEPENDS` list missed → header edits don't trigger rebuild | Medium | Re-read `lib/Runtime/CMakeLists.txt` `compile_to_bitcode` calls before each stage (per the `CLAUDE.md` gotcha). |

---

## Estimate

**Total: ~5 days of focused work**, plus ~1 day for testing and rollout
= **~6 days end-to-end**.

| Stage | Effort |
|---|---|
| 1. Pinned host workspace facility | 1 day |
| 2. Migrate qmoe_host_scratch | 0.5 day |
| 3. Workspace arena helper + 3 wrappers | 1 day |
| 4. Advisory size hint (codegen pass + runtime read) | 2–3 days |
| 5. Documentation | 0.25 day |
| 6. (Deferred) Phase-transition hook | 0.5 day |

Stages 1–3 together (~2.5 days) deliver most of the maintenance win and
remove `qmoe_host_scratch`. Stage 4 is the largest single piece and the
only one touching the codegen; if codegen bandwidth is constrained it can
be deferred without blocking 1–3.

---

## Relationship to Other Plans

This plan is **independent of** [op-state-registry.md](op-state-registry.md)
and [inline-prepack-cache.md](inline-prepack-cache.md). Any of the three
may land first.

If the op-state registry lands first, the new `pinned_workspace` field
added by Stage 1 of this plan can later be migrated into the registry as
a follow-up — that follow-up is optional and not part of this plan's
acceptance criteria. The workspace facilities are facility-tier
(stream, handles, pool) and arguably belong on `RuntimeState` directly
rather than in the op-state registry; the choice is a follow-up
discussion, not a precondition.

If the prepack cache plan lands first, no interaction with this plan.

The advisory-hint design in [§A.3](#a3-advisory-size-hint-in-model-metadata)
deliberately mirrors the pattern that
[op-state-registry.md](op-state-registry.md) uses for runtime ↔ codegen
backwards compatibility: extend the metadata schema with optional fields,
treat their presence as a hint not a contract, and ensure both directions
of version skew (old DLL / new runtime, new DLL / old runtime) keep
working.
