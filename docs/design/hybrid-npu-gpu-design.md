<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Hybrid NPU + GPU Execution for the MorphiZen HIP Execution Provider

**Date:** 2026-08-13
**Document Type:** Architecture
**Status:** Draft
**Audience:** Engineers familiar with hybrid-llm and/or hip-ep. Assumes ORT EP concepts,
DynamicDispatch, XRT, and `RyzenMM` are known.
**Supersedes:** the 2026-08-03 revision, which designed the *inverse* integration (HIP kernels
consumed by hybrid-llm). See [Direction](#direction-npu-into-hip-ep-not-hip-into-hybrid-llm).
**Execution plan:** [hybrid-npu-gpu-tasks.md](hybrid-npu-gpu-tasks.md) — phases, gates, verification.

---

## Overview

hip-ep runs whole ONNX graphs on an AMD GPU: `GetCapability` claims the graph, the MLIR pipeline
compiles it to a single artifact, and ORT calls `inference_compute` once per `Run`. Both prefill and
decode are served by that one path.

This design adds a second execution target. **Prefill runs on the XDNA NPU; decode continues on the
GPU**, inside one ORT session, one execution provider, and one memory manager. The NPU capability is
drawn from hybrid-llm's DynamicDispatch integration as reference, not as a dependency.

Target is a Strix-class APU (`gfx1150` iGPU + XDNA NPU). First model is Llama-3.2-1B, then
Llama-3.1-8B. **Accuracy is the acceptance criterion** — no phase is gated on a throughput number.

**Zero copy is a requirement, not an optimization.** No tensor is copied when execution crosses
between NPU and GPU. This constrains the memory model, the dtype policy, and the KV handoff, and it
removes fallbacks that would otherwise be available.

---

## Contents

**Part I — Motivation**
[Problem](#problem) ·
[Concept](#concept) ·
[Direction](#direction-npu-into-hip-ep-not-hip-into-hybrid-llm) ·
[hip-ep vs the RyzenAI EP](#hip-ep-vs-the-ryzenai-ep)

**Part II — Decisions**
[Decision record](#decision-record)

**Part III — Architecture**
[Why this is tractable](#why-this-is-tractable) ·
[Architecture](#architecture) ·
[Compile time](#flow-compile-time) ·
[Session setup](#flow-session-setup) ·
[Prefill call](#flow-a-prefill-call-ttft) ·
[Decode call](#flow-a-decode-call-tps) ·
[Execution model](#execution-model) ·
[Op mapping](#op-mapping) ·
[GPU islands](#gpu-islands) ·
[Shape contract](#shape-contract) ·
[Data type policy](#data-type-policy)

**Part IV — Memory**
[Memory classes today](#memory-classes-today) ·
[Ownership: three candidates](#ownership-three-candidates) ·
[The NPU pool](#the-npu-pool) ·
[Registered-memory rule](#the-registered-memory-rule) ·
[Zero-copy requirement](#zero-copy-requirement)

**Part V — State and packaging**
[KV cache](#kv-cache-ownership-and-maintenance) ·
[Weight lifecycle](#weight-lifecycle) ·
[Packaging and the shim](#packaging-and-the-shim-boundary) ·
[Configuration](#configuration-surface)

**Part VI — Execution of the work**
[Performance expectations](#performance-expectations) ·
[Approach](#approach) ·
[Testing](#testing) ·
[Risks and stop conditions](#risks-and-stop-conditions) ·
[Open questions](#open-questions) ·
[Where the code will live](#where-the-code-will-live)

---

# Part I — Motivation

## Problem

Hybrid NPU + GPU inference works today in hybrid-llm: RyzenAI EP drives the NPU through
DynamicDispatch, DirectML serves the GPU half. **DirectML support is going away**, which removes the
decode half of that arrangement. NPU-only decode is the weaker configuration, so the capability needs
a new GPU foundation.

hip-ep is the natural host: a mature whole-graph GPU path (MLIR → HIP → MIOpen / hipBLASLt / custom
kernels) in production across Llama, Mistral, Phi, Qwen, Gemma, gpt-oss, DeepSeek, and Whisper.
Relocating the NPU capability there also widens it — hybrid-llm covers the models its preformatting
pipeline handles; hip-ep covers all of the above.

No NPU expert is on this work. DynamicDispatch is undocumented and hybrid-llm's kernels are the only
working reference, so API recovery is an explicit up-front deliverable rather than something absorbed
while building. hybrid-llm itself is **read-only** and must keep working.

## Concept

Prefill is compute-bound (all prompt tokens in parallel, high arithmetic intensity). Decode is
memory-bandwidth-bound (full weight set read per token). Route each phase to the accelerator that
fits.

```
prompt ──► PREFILL (NPU) ──► KV cache ──► DECODE (GPU) ──► tokens
           compute-bound      shared        bandwidth-bound
           improves TTFT      buffer        existing path
```

Two hard constraints follow.

**One memory manager.** Prefill produces the KV cache; decode reads and appends to it every step. It
is the largest tensor in the system. Two sessions or two allocators fracture ownership of it, and
copying it at the transition can cost more than the hybrid arrangement saves.

**Zero copy.** One allocation, visible to CPU, GPU, and NPU, written in place by whichever accelerator
is running.

Note what is *not* claimed: GPU-only (today's hip-ep) is not broken. Hybrid improves TTFT and total
power because each accelerator idles during the phase it is worse at.

## Direction: NPU into hip-ep, not HIP into hybrid-llm

Both directions were specified; the earlier revision of this document designed the inverse — keep
hybrid-llm's EP, per-node dispatch, and `RyzenMM`, replacing only its DirectML kernels with HIP.

| | **A: HIP into hybrid-llm** | **B: NPU into hip-ep** ← chosen |
|---|---|---|
| Host EP | RyzenAI EP | MorphiZen HIP EP |
| Dispatch granularity | per ONNX node | whole graph, one fused node |
| Tensors crossing accelerators | every inter-node tensor | graph boundary only (inputs, outputs, KV) |
| Input model | preformatted (`GQO` / `SSMLP` / `MatMulNBitsBf`) | stock ONNX |
| Model coverage | hybrid-llm's pipeline | everything hip-ep supports |
| Changes to hybrid-llm | substantial | none |
| GPU decode path | new (HIP into DML's place) | existing, untouched |

The decisive difference is **granularity** — see below. Option B also leaves the GPU path in place, so
any decode regression is unambiguously a bug in this work rather than a cost of it.

## hip-ep vs the RyzenAI EP

| | **RyzenAI EP** (hybrid-llm) | **MorphiZen HIP EP** |
|---|---|---|
| Graph claim | individual `com.ryzenai` nodes | the **entire graph**, as one fused node |
| Dispatch unit | one ONNX node per `Compute()` | one `Compute()` for the whole model |
| Backend choice | per node, per call, from its own input shape | per call, once, for the whole graph |
| Input model | preformatted by a Python optimizer | stock `com.microsoft` ONNX |
| Intermediates | ORT-visible `OrtValue`s between nodes | internal to the artifact; ORT never sees them |
| Memory manager | `RyzenMM` (CPU + XRT + Direct3D 12) | HIP EP allocator + pooled GPU arena |
| GPU half | DirectML | HIP / MIOpen / hipBLASLt / custom kernels |

**`RyzenMM` exists because that design is per-node.** If every tensor between two nodes might be
produced by the NPU and consumed by the GPU, every tensor must be visible to both at all times.

hip-ep does not have that problem: intermediates live inside one compiled artifact and are never
returned to ORT. Only **graph-boundary** tensors — inputs, outputs, KV cache — cross between
accelerators. That is a small enumerable set, which is what makes zero copy achievable by registering
a handful of buffers rather than by building a tri-platform allocator.

Note this argument is about *granularity*, not about `RyzenMM` being unnecessary as a component. Its
suitability as hip-ep's host-accessible allocator is an open question — see
[Ownership](#ownership-three-candidates).

---

# Part II — Decisions

## Decision record

Settled before drafting, recorded so implementing agents do not relitigate them.

| # | Decision | Consequence |
|---|---|---|
| 1 | NPU is a **second MLIR lowering target**, not a runtime branch and not a per-node custom-op layer | [Execution model](#execution-model) |
| 2 | The **whole prefill graph** targets the NPU; the GPU is idle during prefill except for islands | [GPU islands](#gpu-islands) |
| 3 | **One memory manager owns every host-accessible allocation.** Which component — HIP, XRT, or `RyzenMM` — is **OPEN**, resolved by a Phase 0 spike | [Ownership](#ownership-three-candidates) |
| 3a | **Zero copy is required.** No tensor is copied at the NPU/GPU boundary. A design that cannot achieve it is not shippable | [Zero-copy requirement](#zero-copy-requirement) |
| 4 | bf16 is **NPU-internal only**. The graph, the KV cache, and the GPU path stay fp16 | [Data type policy](#data-type-policy) |
| 4a | The fp16/bf16 conversion must be **device-side and fused** — never a host-side pass | [The dtype boundary](#zero-copy-and-the-dtype-boundary) |
| 5 | Input is **stock OGA-exported ONNX**. No Python graph optimizer is ported | [Why this is tractable](#why-this-is-tractable) |
| 6 | DynamicDispatch is reached through a **C-ABI shim DLL**, loaded at runtime | [Packaging](#packaging-and-the-shim-boundary) |
| 7 | Sequence lengths are **bucketed and padded** for the NPU | [Shape contract](#shape-contract) |
| 8 | NPU weights **stream in and are released**; NPU and GPU weight residency never peak together | [Weight lifecycle](#weight-lifecycle) |
| 9 | Any NPU rejection **falls back to GPU prefill**, with a strict env var that turns rejection into an abort for CI | [Configuration](#configuration-surface) |
| 10 | v1 is **pure ORT + IOBinding**; OGA is a later phase | [Testing](#testing) |
| 11 | hybrid-llm is **read-only reference**. Nothing there may be changed or broken | [Problem](#problem) |
| 12 | **MLP fusion is in v1 scope**, not deferred. Attention is already fused at the dialect level; the MLP is not | [Performance](#performance-expectations) |

> **Decision 3 was reopened on 2026-08-13.** It previously read "HIP owns every physical allocation;
> XRT imports HIP pointers; `RyzenMM` is not used." `RyzenMM` is now a live candidate, because it
> already solves the multi-view registration and sub-pointer lookup this design otherwise builds, and
> it is the only candidate proven against XDNA.

---

# Part III — Architecture

## Why this is tractable

The two projects have opposite execution models, and bridging them naively means either rebuilding
per-node dispatch inside hip-ep or porting the graph optimizer. Neither is necessary, because:

> **The HIP dialect is already at DynamicDispatch's granularity.**

hip-ep's ONNX→HIP converters already turn stock `com.microsoft` operators into `hip.matmul_nbits`,
`hip.gqa`, `hip.rope`, `hip.rms_norm`, `hip.skip_rms_norm`, and `hip.qmoe`. Those are the fused
abstractions the NPU wants.

**No preformatted model is needed.** The fused operators in hybrid-llm's optimized graph are *its ONNX
custom ops*, not DD operators. DD's inventory:

```
llm_ops/      bmm  elwmul  mladfadd  mladfmatmulbias  mladfmharope
              mladfrmsnorm  maskedsoftmax  silu                     ← primitives
transformer/  flash_mha  chunk_flash_mha  RoPE  RMSNorm  ssmlp
              MatMulQ  BatchedMatMul  general_activation             ← fused
```

**There is no `gqo` operator in DD at all.** `GQO` is an 8,000-line custom-op kernel that calls
`mladfmharope` → `flash_mha` → `mladfmatmulbias`. The preformatting exists to give a per-node EP a
dispatch unit worth dispatching; it is not an NPU requirement. Since hip-ep compiles the whole graph
into one unit, its dispatch granularity costs nothing to make finer, so it can drive those same DD
operators directly from a plan.

Fusion state per block:

| Block | hip-ep dialect | Fused? |
|---|---|---|
| Attention (+ rope + KV append) | `hip.gqa` | Yes — one op |
| Quantized matmul | `hip.matmul_nbits` | Yes |
| Norm + residual | `hip.skip_rms_norm` | Yes |
| MoE | `hip.qmoe` | Yes (out of v1 scope) |
| MLP | 3× `hip.matmul_nbits` + `hip.silu` + `hip.mul` | **No** — see [Performance](#performance-expectations) |

What must genuinely be recovered from hybrid-llm is narrower than its ~18.5k lines under `npu/`: the
exact DD call sequence per operator, the weight layouts, and the shape-bucketing rules. The rest is
ORT marshaling, `RyzenMM` staging, and dtype plumbing this design does elsewhere or differently.

## Architecture

```
ORT session
  └── MorphiZen EP  (one EP, one allocator, one graph claim)
        │
        ├── GetCapability → MLIR compile
        │     ├── convert-hip-to-llvm ────────────► GPU artifact   (existing, unchanged)
        │     └── convert-hip-to-npu-plan ───────► NPU plan        (new lowering target)
        │
        └── MlirCustomOp::Compute
              │  seq_len > 1 && npu_plan valid && bucket supported ?
              │
              ├── yes ──► NpuPlanRunner  (EP-side C++)
              │             ├── morphizen_npu_shim.dll   (C ABI, runtime-loaded)
              │             │     └── DynamicDispatch → XRT → NPU
              │             └── GPU islands for unmapped ops
              │
              └── no  ──► inference_compute(state, inputs)   (existing GPU path)
```

Four properties follow. There is **one EP and one graph claim** — ORT's view is unchanged. The **GPU
path is untouched**. **Fallback is free**, because both artifacts come from the same claimed graph, so
declining the NPU runs the artifact that already exists with no partial state to unwind. And the **NPU
plan is interpreted by EP-side C++**, not generated code — forced, because the GPU artifact is
self-contained bitcode linked by `lld-link` and cannot link a shim depending on XRT and a foreign C
runtime.

## Flow: compile time

```
GetCapability → claim whole graph
  │
  ▼
convert-onnx-to-hip                        (existing: stock ops → hip.* dialect)
one-shot-bufferize, dealloc, pool-allocs   (existing)
  │
  ├───────────────────────► convert-hip-to-llvm ──► LLVM IR ──► lld-link ──► GPU artifact
  │
  └───────────────────────► hip-fuse-mlp-for-npu   (new: 5 dialect ops → 1 fused MLP op)
                                    │
                                    ▼
                            convert-hip-to-npu-plan (new)
                                    │
                                    ▼
                            NPU plan (FlatBuffers, cached beside the artifact)
                              ├─ ordered operator invocations
                              ├─ per entry: kind, attributes, operand/result bindings
                              ├─ bindings ∈ {graph input i, graph output i,
                              │              NPU pool offset, constant id}
                              ├─ required NPU pool size
                              └─ the sequence bucket it was built for
```

Both artifacts describe the same claimed graph, so their input and output ordering is identical —
that is what makes them interchangeable at `Compute` time.

## Flow: session setup

```
InferenceSession created
  │
  ├─ EP registers its allocator with ORT
  │
  ├─ Caller allocates through it (IOBinding / OGA):
  │     graph inputs, graph outputs, KV cache      → page-aligned, host-accessible
  │     → inserted into the memory registry
  │     → registered with XRT (bind_bo)
  │
  ├─ Shim loaded (LoadLibrary); DD operators instantiated once per plan entry
  │     → shape-dependent setup per bucket, not per call
  │     → xrt::run objects built once via create_run(), reused across calls
  │
  ├─ NPU pool allocated + registered, grow-on-demand
  │
  └─ Weights formatted into DD layout (cached on disk, keyed like the artifact cache)
        → transposed, block scales rearranged, gate/up concatenated for the fused MLP
```

Instantiating a DD operator is expensive, so it happens once. `create_run()` returning an `xrt::run`
is what makes this clean: the run is built at setup and merely executed per call.

## Flow: a prefill call (TTFT)

```
ORT Run → MlirCustomOp::Compute
  │
  ├─ NPU selected?  seq_len > 1
  │                 AND an NPU plan exists
  │                 AND seq_len fits a supported bucket
  │                 AND NPU init succeeded
  │                 AND every boundary tensor resolves in the registry
  │
  ▼ yes
NpuPlanRunner::run(plan, inputs, outputs)
  │
  ├─ pad seq_len to the bucket; mask padding so it cannot reach attention
  │    denominators or normalization statistics
  │
  ├─ for each plan entry, in order:
  │     ├─ resolve every binding through the memory registry
  │     │     unregistered pointer → HARD ERROR (never a copy)
  │     ├─ GPU island (e.g. embedding Gather)?
  │     │     → run that op through the existing GPU path, same registered buffers
  │     └─ otherwise → shim call → DD execute() on the prebuilt xrt::run
  │
  ├─ NPU attention writes K/V into the shared KV buffer at [0, prompt_len),
  │    in decode's layout
  │
  └─ release streamed NPU weights (ordered behind the work that reads them)

  ▼ no  (any condition fails)
inference_compute(state, inputs)          ← existing GPU path, unchanged
```

## Flow: a decode call (TPS)

```
ORT Run → MlirCustomOp::Compute
  │
  ├─ seq_len == 1  →  NPU not selected
  ▼
inference_compute(state, inputs)          ← byte-identical to today
  ├─ hip.gqa reads the KV buffer NPU prefill wrote
  ├─ appends the new token's K/V in place at offset past_len
  └─ past_len derived from seqlens_k
```

Nothing is copied at the transition: same allocation, same dtype, same layout, same conventions. The
handoff mechanism is "nothing happens," which is the point.

Because dispatch is decided **per call** rather than per artifact, a session can prefill on NPU,
decode on GPU, and re-prefill on NPU after a conversation turn, with no artifact reload and no state
machine.

## Execution model

**What the target emits.** A pass consumes the bufferized HIP dialect and emits the NPU plan described
above. The plan is **data, not code** — serialized alongside the GPU artifact and cached with it.

**What runs it.** An EP-side interpreter walks the plan, resolves each binding to a registered
pointer, and calls through the shim's C ABI. It holds per-operator DD state, instantiated once at
session setup, because instantiation is expensive and shape-dependent setup is bucketed.

**Dispatch.** `MlirCustomOp::Compute` chooses per call, on the predicate above. The first thing that
can be built and tested is therefore "the NPU is never chosen," which must be byte-identical to today.

**Why not the alternatives.** Branching inside the runtime operator wrappers would drag XRT and DD
into the JIT-linked model artifact — a self-contained bitcode module with a static CRT, the worst
possible place for a foreign toolchain. Porting the per-node custom-op model would require porting the
graph optimizer and would duplicate a stack that must keep working. Two sessions would fracture the KV
cache across allocators, which the single-memory-manager requirement rules out.

## Op mapping

Derived from reading hybrid-llm's NPU kernels. **Every row must be confirmed against the real DD
headers in Phase 0** — template arguments and call ordering in particular. This is a research index,
not an API contract.

| HIP dialect op | DynamicDispatch target | Notes |
|---|---|---|
| `hip.matmul_nbits` | `mladfmatmulbias<uint16_t, int8_t, uint16_t, uint16_t>`, tagged `"bfloat16" / "int4" / "bfloat16"` | Dominant by count and time. Weight layout differs from the GPU path |
| `hip.gqa` | `transformer::flash_mha`, or `bmm` + `maskedsoftmax` + `bmm` decomposed | Layout and mask conventions are the risk. Decomposing materializes the score matrix in DDR — avoid |
| MLP block (fused) | `transformer::ssmlp` | `general_activation + rmsnorm0 + matmul_gateup + matmul + rmsnorm1`. Requires **concatenated gate/up weights** |
| `hip.rope` | `mladfmharope` / `transformer::RoPE` | Cos/sin cache needs a DD-specific packed layout |
| `hip.rms_norm` | `mladfrmsnorm` | |
| `hip.skip_rms_norm` | `mladfrmsnorm` + `mladfadd` | Or absorbed by `ssmlp`'s trailing norm |
| `hip.mul` | `transformer::ElwMul` / `mladfelwmul` | |
| `hip.add` | `mladfadd` | |
| SiLU / GELU | `silu` / `general_activation` | |
| `hip.qmoe` | per-expert `mladfmatmulbias` | Out of v1 scope; recorded so the gap is known |

Two operator gaps in the wider model set have no counterpart and are out of scope for Llama: logit
softcapping and interleaved rotary embedding. Recorded deliberately — other model families need them.

## GPU islands

"The whole prefill graph runs on the NPU" is an approximation. hybrid-llm registers NPU kernels for
fourteen operators and has none for `Gather`, `Cast`, `Sub`, or `ReduceSum`. For Llama the practical
consequence is the **embedding lookup**, plus whatever small type or shape operators survive
conversion.

Unmapped operators run on the GPU as **islands** inside the prefill plan. Three rules keep this from
degenerating into the per-node shared-memory problem:

- Islands are **enumerated at compile time**, not discovered at run time. The plan records which
  entries are islands and how many. An unexpected count means conversion changed.
- Island boundary tensors come from the **NPU pool**, so they satisfy the registered-memory rule by
  construction.
- The expected count is small. **If it is not small, the design assumption is wrong and the phase
  should stop rather than proceed** — an island between every pair of operators is the per-node
  architecture in disguise, with none of its benefits.

An alternative worth evaluating during bring-up: shrink the claimed subgraph so the embedding lookup
sits outside it and ORT runs it on CPU. That trades an island for a graph-claim change.

## Shape contract

DD operates on bucketed shapes; hip-ep compiles a specialization per observed shape. The NPU path
adopts **bucketing**, because DD requires it and because it keeps one plan valid across many prompt
lengths. The plan is built for a bucket, prompts are padded up to it, and the padding is masked.

Three consequences:

**Padding must be provably inert.** A masked-out position leaking into a normalization statistic or an
attention denominator is a small accuracy drift rather than an obvious failure — expensive to find
later. Per-operator tests must include a case where the padded region holds a **poison value** rather
than zeros.

**A prompt longer than the largest bucket has no plan** and falls back to GPU prefill. Acceptable for
v1; chunked prefill later fixes it by making one bucket the chunk size and iterating.

**The bucket ceiling joins the plan cache key.** The existing artifact key is deliberately narrow — it
does not include the runtime version, which is why stale artifacts must be deleted manually. The NPU
plan key must additionally cover the bucket set, the DD configuration, and the NPU binary identity, or
a configuration change silently reuses an incompatible plan.

## Data type policy

The NPU computes in bf16; the GPU path is fp16. The boundary sits **inside the NPU operators**.

Stock OGA-exported Llama ONNX is fp16 throughout, and hybrid-llm's default hybrid mode also keeps the
graph fp16, converting at the kernel boundary. Three things follow.

**The kernel ABI problem disappears.** hip-ep's kernel launchers select element type by *byte width*,
accepting 2 for half and 4 for float. bf16 is also two bytes and therefore unrepresentable in that
ABI. Making bf16 a graph-level type would mean replacing byte-width dispatch with an explicit type
enumeration across every launcher, every runtime wrapper, and the MLIR lowering that derives width
from the memref element type — then adding bf16 instantiations of attention, quantized matmul, and
MoE kernels.

**The KV cache stays fp16.** This matters more. Decode appends into the buffer prefill wrote, so it
cannot hold two dtypes; a bf16 NPU write would require converting the whole cache at the transition
and a bf16 variant of every GQA kernel.

**The cost is a conversion per NPU operator boundary during prefill.** Real, bounded, measurable, and
confined to prefill. If it proves material, moving bf16 to the graph boundary remains available — but
as a measured decision after correctness, not an assumed one.

How that conversion is implemented is governed by
[the dtype boundary](#zero-copy-and-the-dtype-boundary), not by this section.

---

# Part IV — Memory

## Memory classes today

Five classes exist. The property that matters is whether the pages are host-accessible, since that
determines whether XRT can register them.

| Class | Call | Holds | NPU-visible |
|---|---|---|---|
| EP ORT allocator | `hipHostMalloc(Mapped\|Coherent)`, size-class pooled | graph inputs/outputs, **KV cache** | **Yes** |
| Host scratch | `hipHostMalloc(Mapped)` | tiny host-staged shape scalars | **Yes** |
| **NPU pool** *(new)* | `hipHostMalloc(Mapped)`, grow-on-demand | plan intermediates, island boundary tensors | **Yes** |
| Session GPU pool | `hipMalloc` | all GPU graph activations | No |
| Constants blob | `hipMalloc` | GPU weights | No |

Two allocator details matter, from `morphizen/ort-bridge/src/morphizen-hip-gpu-allocator.cpp`:

**Allocations ≤ 16 MB are pooled by size class; larger ones are exact-size and returned straight to
the driver on free.** A KV buffer is normally well over 16 MB, so it is *not* pooled — one
`hipHostMalloc` for the session, freed once. Registration therefore happens once and the pointer is
stable for the session.

**Each `Alloc` is its own `hipHostMalloc`; the allocator does not sub-allocate.** Exact-match lookup
would suffice for allocator pointers. Interval lookup is still required, because the **NPU pool and
host scratch are single large buffers carved into offsets**, and because ORT can hand out a pointer
into the middle of a registered range.

`hipHostMalloc` returns page-aligned memory, satisfying DD's 4 KiB binding requirement for free.

> **Doc drift.** `CLAUDE.md` states the allocator uses `hipHostMallocMapped | hipHostMallocNonCoherent`
> and attaches measured perf numbers. Both copies of the source use `hipHostMallocCoherent`; the
> `NonCoherent` change is not in the tree. Treat `Coherent` as current behaviour.

## Ownership: three candidates

The design rests on one unverified assumption — that a single allocation can be made simultaneously
visible to HIP and to XRT. **Which component owns it is open** (Decision 3), resolved by a Phase 0
spike. All three satisfy zero copy; they differ in ownership, dependency surface, and how much
bookkeeping we build versus inherit.

| | **1: HIP owns** | **2: XRT owns** | **3: `RyzenMM` owns** |
|---|---|---|---|
| Allocation | `hipHostMalloc(Mapped)` from the EP allocator | `xrt::bo` host-only + `map()` | `RyzenMM` |
| Other viewer imports via | `xrt::ext::bo(dev, host_ptr, size, access)` | `hipHostRegister(Mapped)` + `hipHostGetDevicePointer` | `hipHostRegister(Mapped)`; XRT view built in |
| Registration bookkeeping | new memory registry | new memory registry | inherited |
| Proven with XDNA | unverified | unverified | **yes, in production** |
| Dependency surface | none added | none added | third-party allocator in the ORT allocation path |

`RyzenMM`'s NPU view is not exotic — `Platform_XRT.cpp` calls
`xrt::ext::bo(device, host_ptr, size, access_mode)`, plain host-pointer registration of memory it
allocated itself, and `UnmanagedBuffers.cpp` provides the sub-pointer offset lookup. Its *Direct3D 12*
view is the complicated half, and this design has no Direct3D 12 — which is why candidate 1 is
plausible at all.

### The `RyzenMM` candidate

Worth evaluating as hip-ep's **primary host-accessible memory manager**, not merely as a component
removed. It already solves what this design otherwise builds: one allocation with multiple accelerator
views, sub-pointer/interval lookup, and — critically — it is **proven against XDNA on this hardware**,
which is exactly the risk candidates 1 and 2 carry.

Adopting it would mean:

- `RyzenMM` backs the three host-accessible classes (ORT allocator, NPU pool, host scratch); HIP
  imports those pointers via `hipHostRegister(Mapped)`.
- The planned memory registry collapses into a wrapper over `RyzenMM`'s existing lookup rather
  than a new subsystem.
- The session GPU pool and constants blob stay on `hipMalloc` — `RyzenMM` does not manage device
  memory, so hip-ep retains two allocators either way.

Costs to weigh:

- A third-party allocator sits in ORT's allocation path, which today is hip-ep's own (size-class
  pooled). Its alignment, lifetime, and pooling behaviour must satisfy ORT's contract.
- It carries Direct3D 12 machinery this design has no use for — dead surface plus a dependency.
- Same packaging constraints as DD (protobuf pin, dynamic CRT, non-redistributable source), so it sits
  behind the same shim, which must then expose allocation.
- It couples hip-ep's memory management to a hybrid-llm component, cutting against treating that
  project as read-only reference rather than a dependency.

**Decision rule.** If HIP-owned registration works, prefer it — fewest dependencies, hip-ep keeps
ownership of its allocation policy. If it does not, `RyzenMM` is the stronger fallback than XRT-owned,
being the only candidate already proven against XDNA and the only one that removes the registry from
our build. If **none** works there is no zero-copy path; escalate rather than degrade.

## The NPU pool

Do not attempt to make the session GPU pool NPU-visible. It is `hipMalloc` device memory by design,
`PoolAllocs` packs the whole graph's transients into it, and changing that would ripple through the
memory planner for one execution target.

Instead the plan gets **its own pool**, from the same host-mapped family as the EP allocator,
registered once at session setup:

```
hipdnn_ep_get_npu_pool_base(state, needed_size)
    → host-mapped alloc + XRT registration, grow-on-demand, never shrinks
```

This mirrors `hipdnn_ep_get_host_scratch_base` exactly, including the grow policy (stream-synchronize,
free, reallocate, return the new base) and the hygiene contract: zero allocations per inference at a
steady-state shape, reuse across runs, growth only when a shape change demands it.

Every buffer the plan touches comes from one of the three registered sources — never from the GPU pool
or the constants blob.

## The registered-memory rule

> A buffer may be handed to the NPU **only because of which allocator produced it**, never because an
> access to it happens to work.

The development target hides this design's most likely bug. On `gfx1150`, `hipMalloc` returns
UMA-mapped memory that happens to be host-accessible; on `gfx1151` it returns true device memory. This
repository has already been bitten by that asymmetry — it is why the host-scalar materialization pass
exists.

So if the plan is handed a GPU-pool buffer, **it will work on the development target and fail on the
next one.** No amount of accuracy testing on `gfx1150` surfaces it.

NPU visibility is therefore asserted on **where the buffer came from**, not on whether an access
faults:

- A registry maps every registered base pointer to its XRT buffer object and extent. Only the EP
  allocator, the NPU pool, and the host scratch insert into it.
- Every buffer is resolved through it before binding. **An unregistered pointer is a hard error** —
  not a fallback, not a warning, never a copy.
- Resolution is an **interval** lookup (ORT hands out sub-range pointers) and is **memoized**, so
  steady-state calls are cache hits.

If the only way to obtain a bindable pointer is through a registry that refuses to invent one, a copy
cannot appear without someone writing it deliberately.

## Zero-copy requirement

Stating it precisely matters, because "no copies" is not literally achievable and an imprecise version
is unenforceable.

> **Definition.** No tensor is copied, staged, or re-laid-out as a consequence of execution moving
> between the NPU and the GPU. Every activation, every KV entry, and every graph input and output is
> read and written in place, in one physical allocation, by whichever accelerator is running.

**In scope — must be zero copy:** graph inputs and outputs for both paths; all prefill activations,
including tensors either side of a GPU island; the KV cache in both directions; the transition from
the last prefill operator to the first decode operator; any fp16/bf16 conversion.

**Out of scope — not per-inference data movement:** one-time weight formatting at session setup; host
writes that fill a buffer before the first inference (e.g. zero-initializing KV); data movement inside
one accelerator's own execution that the operator would perform regardless.

The distinguishing test: **would this movement disappear if a single accelerator ran the whole graph?**
If yes, it is a boundary copy and forbidden.

### Preconditions on the caller

Zero copy is not achievable unilaterally. hip-ep's existing marshaling copies host-to-device when an
input's memory type is not GPU, and device-to-host when ORT hands back a host buffer. Both are correct
today and both violate the requirement here.

So the requirement carries a caller contract: **graph inputs, outputs, and the KV cache must be bound
through this EP's allocator**, via IOBinding with device memory or OGA's equivalent. A session binding
plain host tensors cannot be zero copy, and the honest response is to say so.

Concretely: when a tensor arrives in unregistered memory, the EP declines the NPU for that call and
falls back to the GPU — the existing, correct, copying behaviour. Under strict mode it aborts naming
the tensor. What it must never do is copy the tensor into registered memory and proceed, producing a
system that reports success while violating its own requirement.

### Zero-copy and the dtype boundary

The interaction most likely to be missed, and the one place the reference implementation is **not** a
safe pattern to follow.

hybrid-llm's default hybrid mode performs the fp16↔bf16 conversion as a **host-side pass over the
buffer** — a CPU loop reading fp16 and writing bf16 into another allocation. Ported as-is that is a
full copy of every activation at every operator boundary during prefill, defeating the requirement
while leaving every other part of this design looking correct.

The rule is therefore stricter than "the graph stays fp16":

- The conversion must be **device-side**. No host loop over tensor data, ever.
- It must be **fused into the consuming or producing operator** where supported, so no additional
  buffer traversal occurs at all.
- Where fusion is unavailable, a device-side conversion into an NPU-pool buffer is the fallback — one
  extra traversal, bounded and confined to prefill.
- A host-side conversion is a **defect**, not a slow path.

If fusion proves unavailable for the operators that matter, the extra traversal becomes the dominant
argument for moving bf16 to the graph boundary after all, on measured evidence.

### The known exception

`mladfmatmulbias::format_output()` depads the kernel's padded-N stride back to the real N, row by row
— DD's own comment cites a vocab of 120818 padded to 120832. That is a restride copy of the
**logits**, the largest output tensor, and it is not optional: it falls out of the kernel padding N.

Preferred resolution is to allocate the logits at padded N and let the consumer stride over it. If
unworkable, it becomes a documented single-tensor exception rather than a counter-flagged defect.

Notably `transformer::ssmlp` declares both `need_format_input()` and `need_format_output()` as
`false`, so the fused MLP path has no such staging — another argument for fusing early.

### Enforcement

Enforced structurally, because a copy is easy to introduce and invisible in output comparison.

The [registered-memory rule](#the-registered-memory-rule) is the primary mechanism. On top of it, the
runtime maintains **boundary-copy counters** — one per category (input staging, output staging, KV
repack, dtype conversion) — always compiled in, cheap, and asserted zero by the tests. A test that
merely compares numbers cannot detect a copy; a test that asserts the counter is zero can.

---

# Part V — State and packaging

## KV cache: ownership and maintenance

### Who allocates it

**Not hip-ep, and not the NPU path. The caller does — through the EP's allocator.**

Under **IOBinding**, the application creates `OrtValue`s on the EP's device memory info
(`device_type="gpu", vendor_id=0x1002`); ORT routes that to the EP allocator. Under **OGA**,
`DefaultKeyValueCache::Add()` creates one `OrtValue` per (layer, K/V) at `max_length` capacity and
binds it as both the `past_key_values.N.key` **input** and the `present.N.key` **output**.

Shape is `[batch, kv_heads, max_length, head_dim]`, dtype fp16.

### How the identity is maintained

With `past_present_share_buffer = true`, past and present are the **same buffer** for the session:

1. The `present.*` output's dynamic sequence dim is materialized **in-graph** by `hip.alloc_output`
   from `memref.dim %past_key` — the past input buffer's actual extent.
2. Under shared-buffer mode that extent already *is* the `max_length` capacity.
3. So `GetOutput` is asked for capacity and returns the pre-bound `OrtValue`.
4. Therefore `past_key == present_key` as pointers, and GQA appends **in place**.

There is deliberately **no EP-side override** for `present.*` shapes — the allocator callback passes
the in-graph shape to `GetOutput` verbatim, which keeps it model-agnostic.

### How validity is tracked

Not by the buffer — by `seqlens_k`. The convention is `seqlens_k[b] = number of PAST tokens`, and the
runtime derives `total_seq = seqlens_k[b] + 1`, `past_len = total_seq - sq`. Non-causal attention is
exempt from the `+1` and uses `total_seq = skv`, `past_len = 0`.

The region **beyond** `past_len + sq` is semantically undefined and deliberately **not zeroed** — the
pool does not zero on alloc or release, so the tail can hold bytes from a previous consumer. Nothing
may depend on it being zero.

### What changes when the NPU is involved

Allocation does not change at all. The same buffer additionally gets **registered with XRT** at first
use and resolved through the registry.

```
session setup   caller allocates KV via the EP allocator
                registry insert + XRT registration      → one buffer, three viewers

prefill (NPU)   flash_mha writes K/V into [0, prompt_len)      ── same buffer
                seqlens_k set to prompt_len - 1  (the +1 convention)

decode  (GPU)   hip.gqa reads [0, past_len), appends at past_len ── same buffer
                past_len grows by 1 per step

transition      nothing happens. no copy, no conversion, no repack.
```

### Three things that must agree, and are not known to

- **Tensor layout.** hip-ep's attention runtime uses a specific head/sequence ordering; DD's operators
  have their own. A mismatch means a transpose no phase budgets for.
- **The valid-length convention.** A mismatch of one produces attention that is subtly wrong rather
  than obviously broken.
- **The undefined tail.** Any NPU-side assumption that it is zero reads garbage.

Designated an explicit Phase 0 investigation, because it is answerable by reading two sets of headers
and one runtime file.

**A repack is not an acceptable resolution** — it is a boundary copy of the largest tensor in the
system. If layouts disagree, the **NPU side must conform**: a different DD variant, a different
binding, or arranging the NPU attention to write what decode already expects. Only if no NPU-side
arrangement works does the question escalate, and the next candidate is changing what the GPU side
expects, not inserting a copy.

The valid-length convention and the tail are cheaper to reconcile, being conventions rather than
layouts: satisfied by what the NPU path writes into `seqlens_k` and by not depending on the tail.

### A padding trap

Prefill pads the sequence to a bucket. If the NPU writes padded positions into KV they land in
`[prompt_len, bucket)` — the undefined tail, so harmless. **But only if decode's append offset comes
from the true prompt length, not the bucket.** Derive `past_len` from the bucket and decode skips a
gap of garbage positions and attends subtly wrong. Cheap to get right, expensive to diagnose later.

## Weight lifecycle

DD wants quantized weights in its own preformatted layout: transposed, block scales laid out to match,
optionally pre-packed, and — for `transformer::ssmlp` — with **gate and up concatenated** into one
`matmul_gateup` tensor. The GPU path wants them in the layout hip-ep's externalized constants blob
already uses. These differ, and the naive outcome is both resident simultaneously — noise on a 1B
model, several gigabytes on an 8B one.

The plan is to **stream NPU weights and release them**, mirroring hybrid-llm: read ahead into host
memory, load ahead onto the device, free after the last prefill operator. The point is not peak memory
in the abstract; it is that NPU and GPU weight residency **never peak together**, because prefill and
decode do not overlap.

Two details decide whether this is straightforward:

**Formatting must be cached across sessions.** Transposing and repacking an 8B model's weights per
session would dominate setup. The result is a pure function of the source weights and the DD
configuration, so it can be written next to the artifact cache and keyed the same way.

**Release must be ordered behind the work that reads it.** Freeing a weight buffer while the NPU still
consumes it is an asynchronous-lifetime bug that reproduces intermittently. The release point must be
a synchronization boundary, not a best-effort hint.

## Packaging and the shim boundary

DD cannot be linked into this repository directly. Three separate conflicts say so:

| Conflict | hip-ep | DynamicDispatch / hybrid-llm |
|---|---|---|
| Protobuf | pinned to a recent major version | pinned to 3.21.5 |
| C runtime | static release (`/MT`), forced by prebuilt LLVM and ORT | dynamic (`/MD`) |
| Source availability | MIT, license headers enforced per file | internal; not redistributable into an MIT tree |

All three collapse at a **DLL boundary with a pure C ABI**. The shim links the prebuilt DD already
present in the hybrid-llm install tree, builds with whatever that requires, and exports a small C
interface with no C++ types, no ORT types, no protobuf, and no XRT types crossing it. hip-ep loads it
at run time by name and never links it. DD's own headers are MIT, but their public signatures carry
`xrt::bo` and `std::map<std::string, std::any>` — precisely what the C ABI exists to contain.

That boundary also carries the licensing answer: the shim's *source* is first-party and lives here
under the same MIT terms, while the library it links is neither vendored nor redistributed. And it
makes the feature optional — a build without the shim, or a host without an NPU, loads nothing and
behaves exactly as today.

The ABI must be narrow enough to be stable and wide enough to avoid chatty round trips: init and
teardown, operator creation keyed by kind and configuration, buffer registration and release,
per-operator execution taking already-registered handles, and error reporting as return codes with a
retrievable message. **Exceptions must not cross it** — the same rule hip-ep already applies to its
output-allocator callback, for the same reason.

Note that XRT is fetched from an internal artifact server by hybrid-llm's build. That build must
therefore be run, with credentials, before this one — making "the hybrid-llm install tree is present
and current" a prerequisite of the NPU build rather than an assumption of it.

## Configuration surface

| Name | Kind | Default | Meaning |
|---|---|---|---|
| `morphizen_prefill_backend` | EP provider option | `gpu` | `gpu` or `npu`. `npu` enables plan emission and per-call dispatch |
| `morphizen_npu_max_seq_length` | EP provider option | model-dependent | Bucket ceiling; prompts above it fall back to GPU prefill |
| `HIPDNN_EP_NPU_STRICT` | environment | unset | Any NPU rejection — missing shim, failed init, unmapped op, unsupported bucket, unregistered pointer — aborts with a diagnostic instead of falling back |
| `HIPDNN_EP_NPU_SHIM_PATH` | environment | alongside the EP DLL | Explicit shim location |
| `HIPDNN_EP_NPU_DEBUG` | environment | unset | Per-plan-entry trace of operator kind, shapes, and which registered buffer each pointer resolved to |
| `HIPDNN_EP_NPU_ALLOW_COPY` | environment | unset | Bring-up only. Permits a boundary copy instead of declining, logging loudly. Violates [zero copy](#zero-copy-requirement); no phase may complete while set; mutually exclusive with strict mode |

The strict flag exists for the same reason `HIPDNN_EP_STRICT` does. Silent fallback is correct for
production and catastrophic for testing. **Every accuracy test claiming NPU coverage must set it.**

A related trap inherited from this repository's experience: `HIPDNN_EP_NPU_DEBUG` must not be enabled
in any process that will later measure throughput. Debug flags here are latched into a static on first
read and cannot be turned off afterwards, so a transient enable to capture a trace permanently
degrades the rest of that process.

---

# Part VI — Execution of the work

## Performance expectations

"We don't use the preformatted model" sounds like giving up the performance it was built for. Two
different benefits are at play.

**The ORT-boundary benefit — we already exceed it.** Most of what fusing into `SSMLP`/`GQO` buys the
RyzenAI EP is escaping its own per-node dispatch, where every node boundary is an ORT-materialized
tensor. hip-ep fuses the entire graph into one node, so there are zero internal ORT boundaries. They
still pay one per layer between `SSMLP` and `GQO`; we pay none.

**The NPU-kernel benefit — mostly already there, MLP is the gap.** Attention, the expensive and
DDR-hostile part of prefill, is *already* a single dialect op mapping to a single fused DD call.
Decomposing it into `bmm + maskedsoftmax + bmm` would materialize the full score matrix in DDR — tens
of MB per layer at prefill lengths, round-tripped twice. We are not exposed to that.

The MLP is where we are coarser. `transformer::ssmlp` absorbs five DD dispatches into one, with gate
and up as a single `matmul_gateup`, plus an optional trailing norm that can eat the next layer's input
norm.

**Why the gap costs less here.** Fusion payoff is dominated by per-dispatch overhead and elementwise
DDR round-trips, and both amortize against compute. At `M=1` they are most of the runtime; at prefill
`M` they are a fraction of three large GEMMs. **We only use the NPU for prefill** — the regime where
fusion matters least. hybrid-llm runs decode on the NPU too, which is exactly where unfused MLP would
bleed.

Rough sizing for Llama-1B at a 512-token chunk: the unfused MLP round-trips gate/up/product
activations through DDR on the order of tens of MB per layer, against a couple of ms of matmul compute
per layer. That reads like a low-tens-of-percent penalty on the MLP, and the MLP is roughly
three-quarters of the parameter FLOPs — so **on the order of 10–20% slower prefill, not 2×**. Treat
that as reasoning, not measurement.

**Consequence (Decision 12):** MLP fusion is in v1 scope. It needs an MLIR pattern-match pass
(`rms_norm → matmul_nbits ×2 → silu → mul → matmul_nbits` → one plan entry) and gate/up weight
concatenation during formatting, which `matmul_gateup` and `cal_shuffled_gateup_size()` confirm is
mandatory.

## Approach

Six principles, in priority order. They determine the sequencing.

**1. Answer the memory question first.** Everything rests on it. Three candidates, tested as throwaway
spikes before production code. Early signal is positive — DD exposes `bind_bo(void*, size, read_only)`,
`create_bo(void* use_ptr, …)`, `set_bo(…)`, and `execute(vector<xrt::bo>&, …)` that bypasses `Tensor`
staging entirely, so accepting externally-owned buffers is first-class. Unverified is whether XDNA
accepts a *HIP-pinned* pointer specifically.

**2. Do not let the GPU path move.** First milestone is "NPU never selected," byte-identical to today.
Everything after is additive, and fallback is free.

**3. Make silence impossible.** A silent NPU decline falls back to GPU and produces *correct results* —
a CPU-reference test then passes perfectly while the NPU never ran. This has repeatedly happened on
the GPU path here. Three countermeasures, built in from the start: a **strict mode** turning any
rejection into an abort, mandatory in every NPU test; tests **assert dispatch** via a positive signal
rather than inferring it from output; tests **assert boundary-copy counters are zero**, since a copy
does not change results.

**4. Depth before breadth.** Take quantized matmul alone through plan emission → dispatch →
interpretation → registration → numeric validation. That establishes the pattern every later operator
follows; the rest is repetition rather than discovery. **Attention last** — it carries the KV layout
and valid-length conventions, and everything else validates without it.

**5. Contain the foreign toolchain at one boundary.** See [Packaging](#packaging-and-the-shim-boundary).

**6. Assume the dev machine hides the worst bug.** See
[the registered-memory rule](#the-registered-memory-rule).

### Stage shape

| Stage | Purpose |
|---|---|
| Spikes | Memory ownership, toolchain coexistence, DD API recovery, KV layout. Throwaway. Gating |
| Foundations | Shim + frozen C ABI, plus a mock shim so downstream is testable without an NPU |
| Memory | NPU pool, registry (or `RyzenMM` wrapper), copy counters |
| First operator | Plan emission, dispatch, interpretation — end to end on quantized matmul only |
| Coverage | Remaining operators, attention last; then MLP fusion |
| Integration | GPU islands, weight streaming, Llama-1B end to end, then Llama-8B |
| Beyond | Chunked prefill, OGA, multi-turn re-prefill, then perf and power characterization |

The gates that matter are between spikes and foundations — the memory answer changes the design — and
before operator coverage, where "the NPU runs one operator correctly, end to end, in a real session"
is the first evidence the architecture is real.

**Accuracy is the acceptance criterion.** Perf work follows correctness — partly because a fast wrong
answer is worthless, partly because this repository has a documented history of chasing perf
regressions that were measurement artifacts.

## Testing

**Per-operator, against a CPU reference.** hip-ep already has a per-operator numeric suite that builds
a single-op ONNX model, runs it through the EP, and compares against ORT CPU. Extending it with an NPU
backend selection gives per-operator coverage across the existing shape matrix at almost no harness
cost, and it is the development loop for operator coverage. Cases must include padded buckets with
**poisoned padding**.

**Whole-graph prefill.** Prefill logits and the resulting KV cache against a CPU reference. First test
exercising plan emission, islands, and the memory path together.

**The phase transition.** Prefill on NPU then decode on GPU, against an all-GPU run of the same
prompt. The only test that can catch a KV layout or valid-length mismatch — it should exist *before*
the end-to-end test.

**End to end.** Greedy token equality against a CPU reference, then perplexity.

**Zero copy.** Every test above asserts the boundary-copy counters are zero. Separate from the numeric
assertion and catching a different failure class: a system that copies produces perfectly correct
numbers, so output comparison can never detect it. It belongs in the same tests so it cannot be
skipped independently.

Three harness rules are non-negotiable, the first two from hard-won experience here: every NPU test
**sets the strict flag**; tests **assert on dispatch, not infer it**; tests **assert the copy counters
are zero**.

## Risks and stop conditions

Three findings would invalidate the approach rather than complicate it.

**No memory-ownership candidate works.** With a copy fallback this was a performance risk; as a
requirement it is a viability risk. Escalate and reconsider the architecture; do not ship a version
that copies the KV cache per transition.

**Toolchains cannot coexist in one process.** HIP + ORT + XRT + a foreign CRT is untested, and this
repository has already been damaged once by cross-runtime heap corruption in its JIT path. The shim
boundary is the mitigation; the Phase 0 linkage smoke test is the check, early, because a failure here
is architectural.

**The island count explodes.** Then whole-graph offload has degenerated into per-node dispatch with
none of its benefits. Detected by the compile-time assertion; the response is to stop, not proceed.

Remaining risks, serious but not existential:

**A boundary copy can hide indefinitely.** Copies do not change results, so accuracy testing cannot
find them and a reviewer reading one file cannot either. Mitigated by making the registry the only
source of bindable pointers *and* asserting counters — either alone is insufficient.

**The dtype conversion is the most likely way zero copy is lost**, because the reference implementation
does it host-side and a faithful port silently introduces a full copy per operator boundary.

**KV layout disagreement.** A repack is forbidden; the NPU side must conform.

**Weight formatting cost.** Mitigated by caching, but the cache key must be right or it silently
serves stale layouts.

**Bucket padding leaking into reductions.** Small accuracy drift rather than visible failure.
Mitigated by poisoned-padding cases.

**The prerequisite build chain.** XRT arrives from an internal artifact server via hybrid-llm's build,
which must run with credentials before this repository's NPU build can configure. Check on day one.

## Open questions

- Which memory-ownership candidate works, and is `RyzenMM` preferable to a first-party registry if
  more than one does? (Gating.)
- Do the two attention implementations agree on KV layout, the valid-length convention, and the
  undefined tail? If layouts disagree, can the NPU side be arranged to write what decode expects,
  given that a repack is forbidden?
- Can the fp16/bf16 conversion be fused into the DD operators, or does it need a separate device-side
  pass? Determines whether Decision 4 holds or must be revisited on measured evidence.
- Does `flash_mha` require **packed QKV**? If so we inherit QKV weight concatenation even in the
  unfused v1, since attention is one DD call regardless of how many MatMulNBits feed it. Stock OGA
  Llama has separate q/k/v projections.
- How many GPU islands does a converted Llama prefill graph actually contain, and is shrinking the
  claimed subgraph to exclude the embedding lookup cleaner than islanding it?
- Does OGA bind its KV cache and activations through the EP allocator in every case?
- What bucket ladder should the NPU path expose, and does a single bucket plus chunked prefill dominate
  a multi-bucket scheme once chunking exists?
- Should the plan be emitted for every compiled graph, or only when the provider option requests it?
- Does anything in hybrid-llm's session-level NPU state — power mode, transaction caching, context
  teardown — need an equivalent here?

## Where the code will live

No files are copied from either source project. DD is consumed as a prebuilt artifact; hybrid-llm's
`npu/` directory is read as the specification for driving it. Everything below is new first-party code.

**Shim** — a separate top-level target, built with DD's toolchain settings:

```
npu-shim/include/morphizen_npu_shim.h     ← the frozen C ABI (the ONLY file both sides see)
npu-shim/src/shim_api.cpp                 ← entry points, exception → error code
npu-shim/src/shim_{matmul,attention,mlp,elementwise}.cpp
npu-shim/src/shim_memory.cpp              ← buffer registration
npu-shim/src/shim_weights.cpp             ← initialize_const_params, export_const_params
npu-shim/mock/shim_mock.cpp               ← same ABI, no NPU, for CI on any machine
```

**MLIR** — following the `lib/Conversion/HipToLLVM/` convention:

```
include/hip/Conversion/HipToNpuPlan/HipToNpuPlan.h
lib/Conversion/HipToNpuPlan/{HipToNpuPlan,PlanBuilder}.cpp
lib/Conversion/HipToNpuPlan/{MatMulNBits,Gqa,Mlp,Elementwise}Plan.cpp
lib/Dialect/Transforms/FuseMlpForNpu.cpp
schemas/npu_plan.fbs
```

plus registration in `include/hip/Dialect/Transforms/Passes.td` and wiring in `Pipelines.cpp`.

**EP-side runtime** — outside `lib/Runtime/`, which is bitcode-compiled and cannot reach a shim DLL:

```
lib/Npu/NpuPlanRunner.cpp        ← the plan interpreter
lib/Npu/NpuShimLoader.cpp        ← LoadLibrary + symbol table
lib/Npu/NpuMemoryRegistry.cpp    ← memory registry, interval lookup, memoization
lib/Npu/NpuPool.cpp              ← NPU-visible pool, grow-on-demand
lib/Npu/NpuWeightCache.cpp       ← formatted-weight cache + streaming
lib/Npu/NpuCopyCounters.h        ← always-compiled boundary counters
```

with `MlirCustomOp.cpp` modified for the `seq_len > 1` branch.

**Tests and build:** `test/lit/Conversion/hip-to-npu-plan/`, GPU-free
`test/runtime/test_npu_{registry,plan_runner}.cpp` against the mock shim,
`test/numeric/tests/test_npu_*.py`, and `cmake/npu.cmake` locating the hybrid-llm install behind a
`BUILD_NPU` option.

`morphizen_npu_shim.h` is the **only** file visible from both sides. Everything on the DD side is
prebuilt and unvendored; everything on the hip-ep side is new first-party MIT code. That is what
confines the protobuf, CRT, and licensing conflicts to a single boundary.

---

## Related Documents

- [hybrid-npu-gpu-tasks.md](hybrid-npu-gpu-tasks.md) — the executable work breakdown for this design
- [morphizen-ep-integration.md](morphizen-ep-integration.md) — how the EP claims graphs and invokes artifacts
- [compiler-runtime-contract.md](compiler-runtime-contract.md) — the contract the NPU target must satisfy alongside the LLVM target
- [pool-allocs-memory-planning.md](pool-allocs-memory-planning.md) — why the GPU activation pool cannot be NPU-visible
- [output-allocator-design.md](output-allocator-design.md) — the output contract that stays unchanged, and how `present.*` gets its shape
- [constant-handling-design.md](constant-handling-design.md) — constant externalization, the starting point for NPU weight formatting
- [custom_kernel_design.md](custom_kernel_design.md) — the kernel launcher ABI whose byte-width type dispatch shapes the dtype policy
