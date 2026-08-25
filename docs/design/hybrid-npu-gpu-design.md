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

Target is a Strix-class APU (`gfx1151` iGPU + XDNA NPU). First model is Llama-3.2-1B, then
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

**Dispatch needs a host-visible sequence length.** Choosing a bucket and filling the attention param
buffer both require knowing the sequence length on the host, but `total_seq_len` and `seqlens_k` are
device tensors, and this repository forbids a bare host read of device memory. The GPU path already
solves this: `read_seqlens_k_for_dispatch` performs a synchronized readback cached per `Compute`. The
NPU predicate reuses that value rather than adding a second read. Note the KV-append step needs no such
read at all — it derives `past_len` from `seqlens_k` on the device.

**NPU dispatch is serialized.** ORT may call `Compute` concurrently, but per-operator DD state and the
`xrt::run` objects built at session setup are reused across calls and are not reentrant. One NPU plan
executes at a time, under a lock held for the plan rather than per entry. Concurrent callers either wait
or take the GPU path; they must never share an `xrt::run`.

**The plan must be restartable.** Fallback is only free if a partially executed plan leaves nothing
behind that a retry would double-apply. Overwriting a graph output twice is harmless, but applying the
KV append twice is not. So the write offset must be derived from `seqlens_k`, which the plan does not
mutate — never from a running counter incremented as entries execute. Any future entry that mutates
persistent state has to preserve this property or the fallback guarantee is void.

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
| `hip.gqa` | **decomposes — see [below](#hipgqa-is-a-decomposition-not-a-call)** | Not a single call. `hip.gqa` performs RoPE, the KV-cache append *and* attention; DD's attention operators do only the attention math |
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

### `hip.gqa` is a decomposition, not a call

The one place our dialect is *more* fused than DD. `hip.gqa` implements the full
`com.microsoft.GroupQueryAttention` spec — RoPE, the KV-cache append, and attention, in one operation
producing `output`, `present_key` and `present_value`. DD's `chunk_flash_mha` computes attention and
nothing else: it declares no in-place support, and K/V are read-only inputs for the current call.

One `hip.gqa` therefore becomes five plan entries:

| # | Step | Target | Notes |
|---|---|---|---|
| 1 | RoPE on Q | `mladfmharope` | Cos/sin cache needs DD's packed layout |
| 2 | RoPE on K | `mladfmharope` | |
| 3 | **KV-cache append** | **GPU step** — `hip_gqa_kv_cache_append` | No DD operator exists for this. See below |
| 4 | K‖V staging pack | interpreter-issued device copy | Required because DD binds the buffer wholesale; see [the known exceptions](#the-known-exceptions) |
| 5 | Attention | `transformer::chunk_flash_mha` | Chosen over `flash_mha`: the only variant able to express a sequence shorter than its bucket, and it packs only K‖V rather than Q‖K‖V. Decomposing further into `bmm` + `maskedsoftmax` + `bmm` would materialize the score matrix in DDR — avoid |

**The KV append is a GPU step, deliberately.** DynamicDispatch has no cache-append operator, so this
work has to be assigned somewhere, and we already own a correct implementation. Reusing it costs one
kernel launch per layer and buys three things: it already **transposes BSHD to the cache's BNSD**, it
already handles the shared-buffer in-place case, and it derives `past_len` **from `seqlens_k` on the
device**, so it needs no host readback and cannot disagree with the GPU path about the write offset.
Its `kv_dtype` argument also carries the INT8-cache path for free.

The alternative — writing an NPU-side append so prefill never touches the GPU — was rejected for v1: it
reimplements tested code against an operator set that does not offer it, for no correctness gain.

Consequence to accept: prefill is not *purely* NPU. It is NPU compute with an O(layers) GPU step, which
is why the island rule below is stated in terms of structure rather than count.

## Operator admissibility

Matching an operation by name is not sufficient. Our dialect operations carry optional features their
DD counterparts do not implement, and emitting a plan entry that silently ignores one computes a
different function — strictly worse than declining, because the numbers look plausible.

Plan emission therefore applies an **admissibility check per operation**, and a rejection is a normal
outcome rather than an error. `hip.gqa` is the clearest case: `chunk_flash_mha` supports sliding window
and nothing else, so any instance carrying `attention_bias`, `head_sink` (smooth softmax),
`k_scale`/`v_scale` (KV-cache quantization), `output_qk`, or an unsupported rotary variant is
inadmissible.

Three rules:

- **Reject on anything unrecognized, never ignore it.** An attribute the emitter does not understand is
  a rejection, not a default. This is what makes the check safe as DD and the dialect both evolve.
- **Record the reason in the plan or the log.** "No NPU plan" with no explanation is the silent-failure
  mode this design exists to avoid; under strict mode a rejection names the operation and the feature.
- **Rejection granularity is the graph, not the operation, unless the operation can be an island.** An
  inadmissible operation with a GPU implementation becomes an island; one without means no plan.

## GPU islands

"The whole prefill graph runs on the NPU" is an approximation, and the KV append above makes that
explicit. hybrid-llm registers NPU kernels for fourteen operators and has none for `Gather`, `Cast`,
`Sub`, or `ReduceSum`. For Llama the practical consequences are the **embedding lookup**, the
**per-layer KV append**, and whatever small type or shape operators survive conversion.

Unmapped operations run on the GPU as **islands** inside the prefill plan. Three rules keep this from
degenerating into the per-node shared-memory problem:

- Islands are **enumerated at compile time**, not discovered at run time. The plan records which
  entries are islands. An unexpected *kind* of island means conversion changed.
- Island boundary tensors come from the **NPU pool**, so they satisfy the registered-memory rule by
  construction.
- **The test is structure, not count.** Islands that are architectural and scale with depth — one KV
  append per layer, one embedding lookup per graph, so `O(layers)` — are expected and fine. Islands
  that scale with the operation count, appearing between arbitrary pairs of operations, mean
  whole-graph offload has degenerated into per-node dispatch with none of its benefits. **That** is the
  stop condition, and it is about the pattern rather than the number: 33 structured islands on a 16-layer
  model is healthy; 33 unstructured ones is not.

An alternative worth evaluating during bring-up: shrink the claimed subgraph so the embedding lookup
sits outside it and ORT runs it on CPU. That trades an island for a graph-claim change.

### Which processor runs an island

"Island" has meant "GPU" above, but on a UMA part that is a choice rather than a constraint: island
boundary tensors live in the NPU pool, which is host-mapped and therefore directly CPU-addressable, so
a host-executed island copies nothing either. The right target differs by island kind.

**Small type and shape operations are CPU candidates.** `Cast`, `Sub` and `ReduceSum` on small tensors
carry no arithmetic for a kernel launch to amortize, and `hip-materialize-host-scalars` already
establishes host handling of small shape work. The embedding lookup is better served by the graph-claim
change above than by an island of either kind.

**The KV append stays on the GPU**, and not for throughput. Prefill writes the cache through the same
kernel decode uses, so the two cannot disagree about layout or write offset; a host implementation
would duplicate the BSHD→BNSD transform and the `past_len` derivation, then need them kept in agreement
by review. It would also require `seqlens_k` on the host — available, since bucket selection already
reads it once per `Compute`, but a coupling the device-side derivation does not have.

A CPU island is **not** the [prohibited host-side conversion](#zero-copy-and-the-dtype-boundary): an
island performs work the graph requires, in place, whereas that pass adds a whole traversal per
operator boundary. The two are close enough to be confused, so a proposal should say which it is, and
should account for the cache traffic a large host write leaves behind for the next device read.

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

Two constraints discovered by the T0.4 comparison shape this more than expected.

**The bucket ladder is discovered, not chosen.** Supported shapes come from transaction binaries loaded
at run time from an external library; `get_supported_shapes()` must be called against the opened device
to enumerate them, and they cannot be read from source. This inverts the dependency the paragraph above
assumes: the plan cache key includes the bucket set, but the bucket set is not knowable until a device
is open. So plan emission must either take the ladder as an input resolved at session setup, or emit for
a ladder declared in configuration and validate it against the device before first use. The second is
preferable — it keeps compilation independent of the host.

**`chunk_flash_mha` requires `S_q == S_k`.** Fine for v1, which prefills a fresh prompt in one shot, so
query and key lengths are equal. It does *not* hold for multi-turn re-prefill or chunked prefill, where
the query is one chunk while the key spans the whole context so far. Both cases are served by the
operator's param buffer — `query_start_pos_id`, `num_query_data`, `kv_start_pos_id`, `num_kv_data`,
`is_first_key_chunk`, `is_last_key_chunk` — rather than by the tensor extents, so the mechanism exists;
v1 simply does not exercise it. Any plan emitted with `S_q != S_k` before that path is built must be
rejected, not padded into shape.

**Partial-chunk K/V tails must be zeroed** before the call. hybrid-llm does this explicitly and its
comments state that stale tail data breaks the online softmax. This is a correctness requirement, not a
hygiene preference, and it is the one place the design deliberately writes into a buffer region it
otherwise treats as undefined.

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

Whether this design's most likely bug is visible at all depends on the part it runs on. On `gfx1150`,
`hipMalloc` returns UMA-mapped memory that happens to be host-accessible; on `gfx1151` it returns true
device memory. This repository has already been bitten by that asymmetry — it is why the host-scalar
materialization pass exists.

So if the plan is handed a GPU-pool buffer, **whether that is caught is a property of the target, not
of the test.** The `gfx1151` host is the favourable case because it faults instead of silently
tolerating the violation; no amount of accuracy testing on a `gfx1150` part would surface it.

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

### The known exceptions

**Attention input packing.** DD's attention operators require their inputs contiguous in one buffer
and bind it wholesale — `flash_mha::create_run` passes `input.at(0).bo` while ignoring
`NPUBufferSpan`'s `offset` and `size` fields. So the current chunk's K and V must be staged into a
packed buffer rather than read in place out of the persistent KV cache. This is bounded to the
**current chunk**, not the cache: order of 1 MB per layer for Llama-1B at a 512-token chunk, once per
prefill. It is categorically smaller than the full-cache repack this design forbids, and it is counted
separately. Whether allocating each layer's K and V adjacently would satisfy the packing requirement
by construction is an open question recorded in T0.4.

**Logit depadding.** `mladfmatmulbias::format_output()` depads the kernel's padded-N stride back to the real N, row by row
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

**Know the limit of that instrument.** The counters see copies this repository performs. They cannot
see one performed *below* them — inside XRT's buffer synchronization, for instance. A counter reading
zero therefore proves that hip-ep introduced no copy, not that no bytes moved. Closing that remaining
gap is a Phase 0 measurement rather than an assertion; see
[the open question on `sync()`](#open-questions).

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

prefill (NPU)   RoPE (NPU) → KV append (GPU step) writes K/V     ── same buffer
                  into [0, prompt_len), transposing BSHD→BNSD
                → K‖V staged for the current chunk
                → chunk_flash_mha reads the staged K/V
                seqlens_k carries prompt_len - 1  (the +1 convention)

decode  (GPU)   hip.gqa reads [0, past_len), appends at past_len ── same buffer
                past_len grows by 1 per step

transition      nothing happens. no copy, no conversion, no repack.
```

Note what the append step buys beyond correctness: because prefill writes the cache through the *same
kernel* decode uses, the two phases cannot disagree about layout, write offset, or the `seqlens_k`
convention. The agreement is structural rather than maintained by review.

### Three things that had to agree — resolved by T0.4

All three were open when this design was written and all three are now settled by source comparison.
Recorded because the answers are load-bearing, not merely reassuring.

| | Question | Answer |
|---|---|---|
| Tensor layout | Do the two attention implementations use the same head/sequence ordering? | **Yes — BNSD on both sides**, and hybrid-llm's persistent cache too. DD calls it BMK (`heads, seq, head_dim`) but the memory order is identical |
| Valid-length convention | Is the meaning of `seqlens_k` the same? | **Yes, exactly.** Both derive `total_seq = seqlens_k + 1`, arrived at independently |
| Undefined tail | Does either assume the region past the valid length is zero? | **Neither does.** Both rely on masking and bounds, not on physical cleanliness |

**No repack is required, and the phase transition really is "nothing happens."** A repack would have
been a boundary copy of the largest tensor in the system and was never an acceptable resolution; the
question is now moot. Since the KV append reuses the GPU path's own kernel, the layout agreement is
enforced by construction rather than by convention.

What T0.4 did surface, in place of the layout risk, is that DD's attention operators bind their input
buffer wholesale and require K‖V contiguous, so the *current chunk* must be staged. That is bounded and
counted separately — see [the known exceptions](#the-known-exceptions).

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

**Cos/sin caches are formatted the same way.** `mladfmharope` wants its rotary tables in a DD-specific
packed layout, and in our path those tables arrive as ordinary graph initializers. They belong in the
same formatting-and-caching pipeline as the quantized weights even though they are not weights — the
distinction that matters is "constant input needing a DD-specific layout," not what the tensor means.
Easy to overlook precisely because the section title says weights.

Three details decide whether this is straightforward:

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

## Build and test topology

This work is developed and **built on a machine with no NPU**, and the resulting binaries are copied to
a Strix Halo host to run. That differs from [the repository's remote workflow](../remote-dev-workflow.md),
which keeps one clone on the remote and builds there. The divergence is workable because most of this
project is compiler-side and verifiable against a mock, but it introduces three hazards that the
build-on-target model does not have.

### The GPU architecture must be overridden, every time

`build.py` detects the architecture of the **local** GPU. On a build host that is not Strix, detection
returns the wrong target, and per this repository's standing rule a mismatched architecture **builds
successfully and fails only when a kernel launches** — an access violation inside `hipLaunchKernel`, far
from its cause. Every build intended for the remote must therefore pass `--hip_arch gfx1151` explicitly
and must not rely on detection.

This is not a background concern for the NPU path specifically, for two reasons. The KV-cache append is a
**HIP kernel**, so the attention decomposition puts an architecture-sensitive kernel on the critical path
of NPU prefill. And the memory argument underlying the registered-memory rule is itself
architecture-specific: `gfx1151` is the architecture that exposes the aliasing bug `gfx1150` masks, so
building for the wrong target can silently remove the very failure mode the tests exist to provoke.

### What crosses the wire

Because the copy is manual, the set has to be written down; an incomplete copy is the most likely way this
loop produces a confusing failure. The shim is its own artifact, built with DD's toolchain settings rather
than the EP's, so it is easy to rebuild one side and ship the other.

| Item | Copied | Note |
|---|---|---|
| EP provider library | Yes | |
| **Shim DLL** | Yes | Separate toolchain (`/MD`, protobuf 3.21.5); rebuilt and shipped independently of the EP |
| Runtime bitcode and custom-kernel library | Yes | Architecture-sensitive — see above |
| Test binaries and test data | Yes | |
| DD runtime, XRT, `xclbin` | **No** | Resident on the remote as a prerequisite |
| Cached model artifacts and NPU plans | **No — invalidate instead** | Copying a stale cache is worse than having none |

### Version skew is now a first-class failure mode

The shim's ABI version check was introduced to catch genuine interface drift. Under manual copying it
acquires a second and more frequent job: catching a **partially updated install**, where a new EP loads an
old shim or vice versa. It must therefore fail loudly and name both versions, not merely refuse to load.

The same reasoning extends the plan cache key. This repository's existing artifact key is deliberately
narrow and does not include the runtime version, which is why stale artifacts already have to be deleted
by hand. A plan cache is worse, because a plan encodes operator selections and buffer bindings that a
rebuilt shim may no longer honour. The key must cover the **shim ABI version** and **NPU binary identity**
alongside the bucket set, so a stale remote cache is rejected rather than replayed.

### What the local machine can and cannot prove

The remote workflow doc already warns that test passes on a non-`gfx1151` box are not authoritative,
because they do not exercise the runtime path that fails in production. Manual copying makes each remote
verification expensive, which pushes in the opposite direction — toward proving as much as possible
locally. Both pressures are satisfied by being explicit about which claims each side can support:

| Locally, no NPU, fast loop | Only on the remote |
|---|---|
| Plan emission and admissibility rejections (LIT) | Registration and zero-copy behaviour |
| Registry and copy-counter unit tests | Any NPU numeric result |
| Interpreter driven against the mock shim | The phase transition and end-to-end runs |
| Structural claims: entry ordering, bindings, decline paths | Every performance claim |

The dividing line is that local tests may prove **structure** — that the right entries are emitted in the
right order with the right bindings, and that the wrong graph is declined — while **correctness and
performance are remote-only**. Passing locally is never evidence a phase is complete.

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

**The mock shim is the primary development vehicle, not a stopgap.** Under
[this project's build topology](#build-and-test-topology) every hardware test is a manual round trip, so
the loop only stays workable if the mock carries most of the iteration. That sets a requirement on the
mock beyond "it links": it must record the call sequence it received, so a test can assert the interpreter
issued the *right* operations in the right order with the right bindings, against the right buffers. A mock
that merely returns success turns the local loop into a compilation check.

Two consequences worth designing for rather than discovering. Remote verification should be **batched** —
accumulate local confidence, then take one trip covering several tasks, because per-trip cost is fixed and
high. And each phase's gate must state plainly **which side satisfies it**: a gate whose evidence is
remote-only cannot be closed by a green local run, and the copy-counter and numeric gates are exactly
those.

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

**Attention input staging.** *Retired risk, replaced by a smaller one.* KV layout disagreement was the
headline risk here; T0.4 resolved it — layouts agree and no repack is needed. What remains is that DD's
attention operators bind their buffer wholesale, so the current chunk's K‖V must be staged. Bounded to
the chunk rather than the cache, and counted.

**The attention decomposition is where the operator count hides.** One `hip.gqa` becomes five entries
including a GPU step, so attention is a materially larger task than the op-mapping table's single row
suggested. Sequencing it last is correct; budgeting it as one operator is not.

**Weight formatting cost.** Mitigated by caching, but the cache key must be right or it silently
serves stale layouts.

**Bucket padding leaking into reductions.** Small accuracy drift rather than visible failure.
Mitigated by poisoned-padding cases.

**The prerequisite build chain.** XRT arrives from an internal artifact server via hybrid-llm's build,
which must run with credentials before this repository's NPU build can configure. Check on day one.

**Architecture mismatch from building off-target.** The build host is not Strix, detection picks the local
GPU, and the failure surfaces as an access violation at kernel launch rather than at build time. Mitigated
only by passing `--hip_arch gfx1151` on every build — and the consequence is not merely a crash: building
for `gfx1150` would mask the aliasing bug the registered-memory rule exists to catch, so a green run on the
wrong target is actively misleading. See [the build topology](#build-and-test-topology).

**Partial manual copy.** The EP, the shim, and the kernel library are separate artifacts on separate
toolchains, shipped by hand. A half-updated remote install can present as a plausible numeric or dispatch
bug. Mitigated by a loud ABI-version check naming both sides, by invalidating rather than copying plan
caches, and by treating the documented copy set as a checklist.

## Open questions

- Which memory-ownership candidate works, and is `RyzenMM` preferable to a first-party registry if
  more than one does? (Gating.)
- Can the fp16/bf16 conversion be fused into the DD operators, or does it need a separate device-side
  pass? Determines whether Decision 4 holds or must be revisited on measured evidence.
- Can each layer's K and V be allocated **adjacently as one buffer**, so `chunk_flash_mha`'s K‖V
  packing requirement is satisfied by construction and the staging copy disappears? This is the one
  idea that would make attention genuinely copy-free. It depends on whether a sub-range can be bound
  at an offset, which the eager path does not currently do.
- Can fusion RT's `sub_bo` path bind at an offset? If so, the wholesale-BO restriction lifts and pool
  offsets become directly bindable — which also decides whether the NPU pool can reuse PoolAllocs'
  256-byte-aligned offsets or needs its own packing.
- **Do per-layer islands cost more power than the arrangement saves?** The premise is that each
  accelerator idles during the phase it is worse at, but Decision 2 concedes the GPU is idle during
  prefill *except for islands*, and the KV append makes that one wake per layer. A GPU re-entered
  `O(layers)` times never reaches a deep idle state, so part of the claimed power benefit is spent by
  the island structure itself. The CPU is awake regardless — it runs the interpreter — so
  host-executed islands would add no wake-ups. Whether any of this is measurable is unknown; it
  belongs with perf and power characterization. This is the one argument that would move the KV append
  off the GPU despite the structural agreement that currently keeps it there, so it should be measured
  before that trade is considered rather than after.
- **Does `xrt::bo::sync()` move bytes on an imported buffer?** The registered-memory rule secures
  *addressability*; it says nothing about **cache maintenance**. XRT's normal lifecycle syncs a buffer
  to and from the device around each run, and on a discrete part those syncs are DMA transfers. On a
  UMA part with an imported host allocation they should be cache operations or no-ops — but if the
  implementation copies, then **zero copy is silently false**: the values all still compare equal, and
  the copy is invisible to the boundary-copy counters because it happens inside XRT, below the level
  the EP instruments. This is the one failure mode in this design that would survive every test written
  for it. `bind_bo`'s cost being measured does not cover it, since that is a one-time registration cost
  while `sync` recurs per inference. T0.1's spike now times both and reports implied throughput; a
  figure near DRAM bandwidth is the signature of a copy. If it copies, the finding bears on Decision 3
  — which ownership candidate is viable — rather than on performance.
- What is the actual supported-shape ladder? Not answerable from source; requires calling
  `get_supported_shapes()` against an opened device.
- ~~Is the target `gfx1150` or `gfx1151`?~~ **Resolved: `gfx1151`, and this is the favourable answer.**
  Per [the remote workflow](../remote-dev-workflow.md), the authoritative build-and-test host is a
  Strix Halo part, and `gfx1151` is precisely the architecture that *exposes* the UMA aliasing bug which
  `gfx1150` masks by returning host-accessible memory from `hipMalloc`. The registered-memory rule exists
  to catch that class of bug, so the test host can actually fail on a violation rather than silently
  tolerating it. A registration spike that passes on `gfx1150` would prove much less.
- How many GPU islands does a converted Llama prefill graph actually contain beyond the structural
  ones, and is shrinking the claimed subgraph to exclude the embedding lookup cleaner than islanding
  it?
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
