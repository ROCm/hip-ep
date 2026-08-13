<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# NPU Prefill Backend — Execution Plan

**Date:** 2026-08-06
**Document Type:** Work breakdown
**Status:** Draft
**Design:** [hybrid-npu-gpu-design.md](hybrid-npu-gpu-design.md) — read it first. This file assumes its decisions and does not repeat their rationale.

---

## How to use this document

Each task has an ID, an owner slot, explicit dependencies, the files it is expected to touch, an exit gate, and a verification command. A task is done when its gate is demonstrated, not when its code compiles.

Tasks marked **[GATE]** block everything downstream. Tasks sharing a `∥` group can run concurrently. The plan assumes two to three agents working in parallel, with hard serialization at the gates.

Nothing in `D:\develop\git\gitenterprise\hybrid-llm` may be modified. It is read-only reference and must keep working.

---

## Contents

- [Prerequisites](#prerequisites)
- [Conventions every agent must follow](#conventions-every-agent-must-follow)
- [Reference index — where to look things up](#reference-index--where-to-look-things-up)
- [Phase 0 — Spikes](#phase-0--spikes)
- [Phase 1 — Shim and C ABI](#phase-1--shim-and-c-abi)
- [Phase 2 — NPU memory class](#phase-2--npu-memory-class)
- [Phase 3 — Plan emission](#phase-3--plan-emission)
- [Phase 4 — Dispatch and interpretation](#phase-4--dispatch-and-interpretation)
- [Phase 5 — Operator coverage](#phase-5--operator-coverage)
- [Phase 6 — GPU islands](#phase-6--gpu-islands)
- [Phase 7 — Weight streaming](#phase-7--weight-streaming)
- [Phase 8 — End to end](#phase-8--end-to-end)
- [Phase 9 — Beyond v1](#phase-9--beyond-v1)
- [Parallelism map](#parallelism-map)
- [Definition of done](#definition-of-done)

---

## Prerequisites

Verify all four on day one. A failure here blocks Phase 0 entirely and is worth discovering before any planning effort is spent.

| # | Check | How |
|---|---|---|
| P1 | Target host has an XDNA NPU and a supported iGPU in the same machine | Device manager shows a Ryzen AI / XDNA accelerator; `amdgpu-arch` reports `gfx1150` |
| P2 | The hybrid LLM build has been run with credentials, so XRT is present | `hybrid-llm/install/xrt_package/xrt/share/cmake/XRT/xrt-config.cmake` exists |
| P3 | Prebuilt DynamicDispatch is present | `hybrid-llm/install/lib/dyn_dispatch_core.lib` and `hybrid-llm/install/bin/dyn_dispatch_core.dll` exist |
| P4 | NPU binaries are present | `hybrid-llm/DynamicDispatch/xclbin/` is populated, and the variant matching the target silicon is identified |

XRT is fetched from an internal artifact server by `hybrid-llm/build.py` (see its `ensure_xrt`). If the download fails with an authentication error, that must be resolved before anything else; there is no offline path.

This repository builds with `python build.py` and is unaffected by any of the above until Phase 1.

---

## Conventions every agent must follow

These are repository rules, not suggestions. Violating them will fail review.

- **Documentation is part of the change.** Any behaviour documented in `CLAUDE.md` or `docs/` that a change affects must be updated in the same session. New gotchas discovered during a task go into `CLAUDE.md` before the task is closed.
- **IR snippets are mandatory for compiler code.** Any pass or rewrite in `lib/Conversion/`, `lib/Dialect/`, `lib/Compiler/`, `include/hip/`, or `backend-mlir-compiler/` carries a `Before:` / `After:` MLIR snippet in its comment block, kept in sync with the code.
- **No hardware names in code comments.** Describe behaviour generically. Architecture-specific facts belong in `CLAUDE.md` or commit messages.
- **No model names in generic compiler code.** Describe the IR pattern, not the model that happens to trigger it.
- **Prefer `llvm::seq`** over index-counting loops in compiler code. Not in `lib/Runtime/`, kernels, or tools.
- **Comment the non-obvious.** Workarounds, spec quirks, and correctness invariants get a short *why*. Never narrate what the code does.
- **Never introduce a boundary copy.** Zero copy is a requirement. If a tensor is not in registered memory, decline the NPU for that call — do not copy it into registered memory, do not stage it, do not repack it. A copy produces correct output and is therefore invisible to every test except the counter assertion, which is exactly why the rule has to be followed rather than discovered.
- **Revert completely on failure.** If an approach does not work, remove it entirely rather than leaving experimental code in the tree. If a multi-layer fix fails once, revert and reassess.
- **Run the linter.** `lintrunner -a` before finishing.
- **Delete stale artifacts after runtime changes.** `del %TEMP%\morphizen_mlir_*` — the artifact cache is keyed on the ONNX hash, not the runtime version.

---

## Reference index — where to look things up

Reading, not porting. Cite these rather than rediscovering them.

**In the hybrid LLM project (read-only):**

| Topic | Path |
|---|---|
| NPU operator implementations | `src/onnx_custom_ops/hybrid_llm/npu/` |
| Quantized matmul — the cleanest DD example | `npu/matmulnbits.cpp` |
| Attention — largest, most convention-laden | `npu/gqo.cpp` |
| Two-phase init, session config, cast attributes | `npu/npu_op.{hpp,cpp}` |
| Weight streaming (read-ahead / load-ahead) | `npu/jit_node.hpp`, `npu/jit_node_impl.hpp` |
| Grow-on-demand NPU scratch, buffer rebinding | `npu/shared_buffer.cpp` |
| bf16 conversion helpers | `npu/common.cpp`, `npu/npu_utils.hpp` |
| Host-pointer to buffer-object binding, 4 KiB check | `DynamicDispatch/include/ops/op_interface.hpp` |
| RyzenMM's XRT view — the mechanism being replaced | `build/_deps/ryzen_mm-src/src/library/Platform_XRT.cpp` |
| Pointer registry with sub-pointer offset lookup | `build/_deps/ryzen_mm-src/src/library/UnmanagedBuffers.cpp` |
| Backend dispatch by sequence dimension | `src/onnx_custom_ops/hybrid_llm/operators/hybrid_kernel.cpp` |
| Memory architecture narrative | `docs/ai_context/memory-allocation.md` |
| DD and XRT build notes | `docs/ai_context/dynamic-dispatch.md`, `docs/ai_context/xrt.md` |

**In this repository:**

| Topic | Path |
|---|---|
| EP allocator to be reused as the NPU-visible source | `morphizen/ort-bridge/src/morphizen-hip-gpu-allocator.cpp` |
| Host-mapped scratch — the pattern the NPU pool mirrors | `lib/Runtime/hipdnn_ep_runtime_state.cpp`, `hipdnn_ep_get_host_scratch_base` |
| Tensor marshaling, memory-type detection, dispatch point | `backend-mlir-compiler/custom-op-mlir/src/MlirCustomOp.cpp` |
| Pass pipeline where the NPU target is scheduled | `lib/Dialect/Transforms/Pipelines.cpp` |
| The peer lowering target | `lib/Conversion/HipToLLVM/` |
| ONNX-to-HIP converters producing the fused ops | `lib/Conversion/OnnxToHip/` |
| Attention runtime — KV conventions | `lib/Runtime/real/gqa.cpp` |
| Per-operator numeric harness to extend | `test/numeric/` |

---

## Phase 0 — Spikes

All four are throwaway. Do not attempt to productionize any of them. Their deliverable is a written finding appended to this file plus, where noted, a standalone program retained under `spikes/`.

### T0.1 **[GATE]** — Zero-copy registration, both directions  ∥A

Prove that one physical allocation can be read and written in place by CPU, GPU, and NPU. Zero copy is a requirement, so this task must establish that **at least one** of the three ownership candidates works; it is not complete after trying only the first. This task also *decides* Decision 3, which the design leaves open.

*Candidate A — HIP owns.* Allocate with `hipHostMalloc(..., hipHostMallocMapped)`. Register the pointer as an XRT buffer object using the same call `RyzenMM` uses. Preferred if it works: fewest dependencies, and hip-ep keeps ownership of its allocation policy.

*Candidate B — XRT owns.* Allocate `xrt::bo` host-only, `map()` it, and import the mapping with `hipHostRegister(..., hipHostRegisterMapped)`.

*Candidate C — `RyzenMM` owns.* `RyzenMM` allocates and supplies the XRT view; HIP imports via `hipHostRegister(..., hipHostRegisterMapped)`. Stronger fallback than B: it is the only candidate already proven against XDNA in production, and it replaces T2.2's registry with a wrapper over `UnmanagedBuffers.cpp`'s existing sub-pointer lookup. Cost is a third-party allocator in ORT's allocation path plus unused Direct3D 12 surface — record both so the trade is explicit.

Note that DD already exposes the entry points candidate A needs — `bind_bo(void*, size, read_only)`, `create_bo(void* use_ptr, ...)`, `set_bo(...)`, and an `execute(vector<xrt::bo>&, ...)` that bypasses `Tensor` staging — so accepting externally-owned buffers is first-class. What is unverified is whether XDNA accepts a pointer the **HIP driver has already pinned**. Test that specific combination, not merely that the API exists.

For whichever direction is being tested: write a recognizable pattern from the CPU and read it on the NPU; write from the NPU and read from both the CPU and a HIP kernel. **Confirm the addresses alias rather than copy** — verify that a write through one view is observable through the others at the same address, not merely that the values agree. Measure registration cost.

Also determine: does the 4 KiB alignment requirement hold for every allocation the EP allocator produces, including its size-class-rounded ones? Does registration survive the pointer being freed and reissued from the allocator's freelist?

**Gate:** one candidate demonstrably works, with aliasing proven rather than inferred, and Decision 3 is recorded with the chosen owner. If **none** works, stop and escalate — there is no zero-copy path, the requirement is unmet, and the response is architectural rather than a fallback. Do not proceed into Phase 1 on the assumption that a boundary copy will be acceptable.
**Files:** `spikes/npu/registration/`
**Verification:** the spike program runs on the target host and prints observed values matching written values in all four directions, plus an address-aliasing check.

### T0.2 **[GATE]** — Linkage and runtime smoke  ∥B

Prove three toolchains coexist. Build a minimal DLL against the prebuilt DynamicDispatch with its own runtime settings, exporting one C function that runs one trivial DD operator. Load it from a process that has already loaded HIP, ORT, and this repository's EP DLL. Allocate through the EP allocator, free through it, and run the DD operator, in that order and repeatedly.

Watch specifically for heap corruption across the runtime boundary — this repository has been damaged by exactly that before. Run under the application verifier or an equivalent heap check.

**Gate:** no crash, no heap corruption, over a sustained loop.
**Files:** `spikes/npu/linkage/`

### T0.3 — DynamicDispatch API recovery  ∥A (after T0.1)

Recover the exact call sequence for the quantized matmul operator: class and template arguments, construction attributes, the constant-parameter initialization layout, the shape-setting call, buffer binding, execution, and any required synchronization. Crib from `npu/matmulnbits.cpp`, ignoring its ONNX and RyzenMM scaffolding.

Write a standalone program that runs one quantized matmul on the NPU against known inputs and compares to a CPU reference. **Retain this program** — it is the reference for every subsequent operator and the fastest way to isolate a numeric problem away from the EP.

While in the headers, answer two structural questions that change v1's weight-formatting scope and are cheap to settle now rather than during Phase 5:

- **Does `transformer::flash_mha` require packed QKV?** If it does, QKV weight concatenation is inherited even in the unfused v1 — attention is one DD call regardless of how many `MatMulNBits` feed it, and stock OGA Llama exports separate q/k/v projections.
- **Confirm `transformer::ssmlp`'s concatenated gate/up requirement** and the layout `cal_shuffled_gateup_size()` implies, since T5.6 depends on it. Also confirm `need_format_input()` / `need_format_output()` are both false there, which is what keeps the fused MLP path free of staging copies.

**Gate:** the standalone program matches a CPU reference within tolerance, the exact weight layout expected by the operator is written down, and both structural questions above are answered in writing.
**Files:** `spikes/npu/dd-matmul/`, findings appended here.

### T0.4 — KV layout and convention comparison  ∥B

Read this repository's attention runtime and DynamicDispatch's attention operators. Answer, in writing: do they agree on tensor layout; do they agree on how valid sequence length is expressed; and does either assume anything about the region beyond the valid length?

This requires no hardware and no build. It is pure comparison, and it determines how much work Phase 5's attention task carries.

**A repack is not an available answer.** The KV cache is the largest tensor in the system and copying it at the phase transition violates the zero-copy requirement. If the layouts disagree, the recommendation must be an NPU-side arrangement that writes what decode already expects — a different DD operator variant, different binding, or different attention decomposition.

**Gate:** a written comparison with an explicit recommendation — layouts match, or here is the NPU-side arrangement that makes them match. If no NPU-side arrangement exists, escalate rather than proposing a copy.

---

## Phase 1 — Shim and C ABI

### T1.1 **[GATE]** — Freeze the C ABI

Specify the interface before implementing it, so Phases 2, 3, and 4 can proceed against it in parallel. It must cover initialization and teardown, operator creation keyed by kind and configuration, buffer registration and release, execution against already-registered handles, and error reporting.

Rules: no C++ types, no ORT types, no protobuf, no XRT types cross it. Exceptions must not propagate across it — catch everything and return a code, exactly as this repository's output-allocator callback does. Version the header so a mismatched shim is rejected with a diagnostic rather than crashing.

**Gate:** header reviewed and merged. Downstream phases may then begin.
**Files:** `include/hip/Npu/npu_shim_abi.h`

### T1.2 — Implement the shim  ∥A

Implement the ABI against the prebuilt DynamicDispatch. Build as a separate CMake target that is not part of the default build and that this repository never links — only loads.

**Files:** `npu-shim/`, `cmake/deps.cmake` (optional dependency discovery)
**Verification:** loads and initializes on the target host; rejects a version mismatch cleanly.

### T1.3 — Mock shim  ∥B

Implement the same ABI with CPU reference computation and no NPU. This is what makes every downstream phase testable on machines without an NPU, and it is the oracle for numeric comparison.

**Gate:** the mock passes the same interface conformance test as the real shim.
**Files:** `npu-shim/mock/`

### T1.4 — Loader

Runtime loading by name, with the search order documented, honouring the path override. Absence is not an error unless strict mode is set.

**Files:** `lib/Runtime/npu/npu_shim_loader.cpp`

---

## Phase 2 — NPU memory class

Depends on T0.1 and T1.1.

### T2.1 — NPU pool

A grow-on-demand host-mapped pool for plan intermediates, mirroring the existing host-scratch accessor including its grow policy: synchronize, free, reallocate, return the new base; never shrink; freed at session cleanup.

Honour the memory-hygiene contract the repository already holds itself to — zero allocations per inference at a steady-state shape, reuse across runs, growth only on shape change.

**Files:** `lib/Runtime/hipdnn_ep_runtime_state.cpp`, `lib/Runtime/runtime_state_internal.h`, `lib/Runtime/hipdnn_ep_runtime.h`

### T2.2 **[GATE]** — Memory registry

The registry mapping registered base pointers to buffer objects and extents, with interval lookup for sub-pointers and memoization of resolution results.

The gate is negative, not positive: **a pointer from the GPU activation pool must be rejected as a hard error.** Write that test first and watch it fail before the registry exists.

Do not add a fallback that copies an unregistered buffer into a registered one. That path hides the exact bug this registry exists to catch.

**Files:** `lib/Runtime/npu/npu_buffer_registry.cpp`
**Verification:** a GPU-free unit test under `test/runtime/`, run via `ctest --test-dir install/build -C RelWithDebInfo -R NpuBufferRegistry`

### T2.3 — Allocator registration hook

Register EP-allocator and NPU-pool allocations into the registry at allocation, deregister at free. Respect the allocator's freelist: a recycled pointer must not be registered twice, and must still resolve.

**Files:** `morphizen/ort-bridge/src/morphizen-hip-gpu-allocator.cpp`

### T2.4 **[GATE]** — Boundary-copy counters

Always-compiled counters, one per category: input staging, output staging, KV repack, dtype conversion. Incremented at every site capable of moving tensor data across the NPU/GPU boundary, readable from the test harness, reset per inference.

These exist because a boundary copy produces perfectly correct numbers and is therefore invisible to every other test. This is the only positive signal that distinguishes "zero copy" from "copies, correctly."

Add the counters *before* the paths they count exist, so that any path added later is added against a visible contract rather than retrofitted into one.

**Gate:** a test asserts all counters are zero across a full GPU-only inference, establishing the baseline that the NPU phases must preserve.
**Files:** `lib/Runtime/npu/npu_copy_counters.cpp`, `lib/Runtime/hipdnn_ep_runtime.h`
**Verification:** `ctest --test-dir install/build -C RelWithDebInfo -R NpuCopyCounters`

### T2.5 — Caller-contract enforcement

When the NPU path is selected and an input, output, or KV tensor arrives in unregistered memory, decline the NPU for that call and fall back to the GPU path; under strict mode, abort with a diagnostic naming the tensor.

Do not copy the tensor into registered memory. That path defeats both the zero-copy requirement and the registered-memory rule, and it converts a loud configuration error into a silent performance loss.

**Files:** `backend-mlir-compiler/custom-op-mlir/src/MlirCustomOp.cpp`

---

## Phase 3 — Plan emission

Depends on T1.1. Can run alongside Phase 2.

### T3.1 — Plan representation and serialization

Define the plan: ordered entries, each with operator kind, attributes, and operand/result bindings drawn from graph input index, graph output index, NPU pool offset, or constant identifier. Plus required pool size and the bucket it targets.

**Files:** `include/hip/Npu/NpuPlan.h`, `lib/Runtime/npu/npu_plan.cpp`

### T3.2 **[GATE]** — Lowering target, quantized matmul only

The new pass consuming bufferized HIP dialect and emitting a plan. Scheduled in the pipeline as a peer of the LLVM target. One operator only — everything else causes plan emission to decline, which must be a clean decline that leaves the GPU artifact intact.

Carries a `Before:` / `After:` IR snippet per repository convention.

**Files:** `lib/Conversion/HipToNpu/`, `lib/Dialect/Transforms/Pipelines.cpp`, `include/hip/Dialect/Transforms/Passes.td`
**Verification:** LIT test under `test/lit/Conversion/hip-to-npu/`, run via `ctest --test-dir install/build -C RelWithDebInfo -R MorphizenMLIRLitTests`

### T3.3 — Plan caching

Serialize the plan alongside the GPU artifact. The cache key must cover the ONNX hash, the bucket set, the DynamicDispatch configuration, and the NPU binary identity. A narrower key silently serves incompatible plans after a configuration change.

**Files:** `lib/Compiler/CompilerDriver.cpp`

### T3.4 — Decline paths

Every reason a plan cannot be emitted — unmapped operator, unsupported bucket, unsupported dtype — produces a diagnostic naming the cause, declines cleanly, and aborts instead when strict mode is set.

---

## Phase 4 — Dispatch and interpretation

Depends on T2.2, T3.2, T1.4.

### T4.1 **[GATE]** — Null dispatch

Wire the per-call decision in `Compute` with the NPU never selected. Behaviour must be byte-identical to today across the existing test suite.

This is deliberately a no-op change and should review as one. It is also the cheapest possible proof that the dispatch point is correctly placed.

**Files:** `backend-mlir-compiler/custom-op-mlir/src/MlirCustomOp.cpp`
**Verification:** `ctest --test-dir install/build -C RelWithDebInfo` and `pytest test/python/test_llama1b.py -v -s` both unchanged.

### T4.2 — Plan interpreter

Walk plan entries, resolve bindings through the registry, call the shim. Hold DD operator state across calls — created once at session setup, not per invocation.

**Files:** `lib/Runtime/npu/npu_plan_runner.cpp`

### T4.3 — Enable dispatch

Select NPU when the sequence dimension exceeds one, a plan exists, the bucket is supported, and initialization succeeded. Otherwise the GPU path. Add the provider options and environment variables from the design's configuration section.

### T4.4 **[GATE]** — First operator end to end

A single-operator quantized-matmul ONNX model, running on the NPU inside a real ORT session, matching a CPU reference, under strict mode, with a dispatch assertion proving the NPU actually ran **and the boundary-copy counters at zero**.

This is the milestone that says the architecture works. All three assertions are load-bearing: the numeric one proves it computed, the dispatch one proves the NPU did it, and the counter one proves it did so without copying.

**Verification:** `pytest test/numeric --backend ort_ep -k matmul_nbits` with the NPU backend selected and strict mode set.

---

## Phase 5 — Operator coverage

Depends on T4.4. Each task is independent and follows the same shape: extract the DD sequence from the reference implementation, add the plan-emission mapping, add the interpreter case, add a LIT test for emission and a numeric test for correctness.

Attention is last because it carries the conventions and because everything else can be validated without it.

| ID | Operator | Notes | ∥ |
|---|---|---|---|
| T5.1 | Rotary embedding | Cos/sin cache needs a DD-specific packed layout — treat the layout as a sub-task | A |
| T5.2 | Normalization (`rms_norm`, `skip_rms_norm`) | Skip variant may need an added elementwise operator | B |
| T5.3 | Elementwise multiply and add | | B |
| T5.4 | Activation (SiLU, GELU) | Confirm which form survives conversion for the target model | A |
| T5.5 **[GATE]** | Attention | Apply T0.4's recommendation. Prefer adapting the NPU side over touching the GPU path | — |
| T5.6 | MLP fusion into `transformer::ssmlp` | v1 scope per Decision 12. Depends on T5.2–T5.4 as its numerics reference, not on T5.5 | — |

**T5.6 is in v1, not deferred.** The MLP is the one block where hip-ep's dialect is coarser than DD's fused operator: attention is already one op mapping to one DD call, but the MLP arrives as `rms_norm → matmul_nbits ×2 → silu → mul → matmul_nbits`. The task is an MLIR pattern-match pass (`lib/Dialect/Transforms/FuseMlpForNpu.cpp`) collapsing that chain into a single plan entry, plus gate/up weight concatenation during formatting — which `matmul_gateup` makes mandatory rather than optional, so it cannot be added later without redoing the weight cache. Build the unfused path first and keep it as the numerics reference; the fused entry must match it before it replaces it.

Every task in this phase carries two additional obligations beyond its own correctness.

**Poisoned padding.** Each must include a padded-bucket case with the padding filled with a poison value, per the design's shape contract. Zero-filled padding will not detect a leak into a reduction or an attention denominator.

**Device-side, fused dtype conversion.** The fp16/bf16 conversion at each operator boundary must be fused into the operator, or failing that performed device-side into an NPU-pool buffer. A host-side pass over tensor data is a defect, not a slow path — it is a full copy of every activation and it defeats the zero-copy requirement while leaving results correct. Port the *intent* of the reference implementation's cast handling, not its mechanism. Each task asserts the dtype-conversion copy counter reflects only what its chosen mechanism justifies.

---

## Phase 6 — GPU islands

Depends on Phase 5.

### T6.1 — Island representation

A plan entry that runs a HIP operator through the existing GPU path, with boundary tensors allocated from the NPU pool.

### T6.2 **[GATE]** — Island enumeration and assertion

Islands are enumerated at compile time and their count recorded in the plan. An unexpectedly high count is reported loudly.

**Stop condition:** if a converted Llama prefill graph produces many islands rather than a handful, the whole-graph assumption is wrong. Stop and escalate rather than proceeding — a graph with an island between every operator pair is the per-node architecture without its benefits.

### T6.3 — Embedding lookup

The concrete first island. Evaluate the alternative of shrinking the claimed subgraph so ORT runs the embedding on CPU, and record which is cleaner.

---

## Phase 7 — Weight streaming

Depends on Phase 5. Can run alongside Phase 6.

### T7.1 — Weight formatting

Transform quantized weights from the externalized constants layout into DynamicDispatch's, using the layout recorded by T0.3.

### T7.2 — Formatting cache

Cache the formatted result next to the artifact cache, keyed the same way. Without this, session setup on a larger model is dominated by repacking.

### T7.3 — Read-ahead and load-ahead

Port the streaming lifecycle from the reference implementation.

### T7.4 **[GATE]** — Ordered release

Release after the last prefill operator, behind a synchronization boundary rather than a best-effort hint. Verify that NPU and GPU weight residency do not peak together by observing process memory across a prefill-then-decode run.

---

## Phase 8 — End to end

### T8.1 **[GATE]** — Llama-3.2-1B

Prefill on NPU, decode on GPU, one session, strict mode. Prefill logits and KV cache match a CPU reference; the phase transition matches an all-GPU run; greedy tokens match a CPU reference; **boundary-copy counters are zero across the whole generation**.

**Verification:** a new `test/python/test_llama1b_npu.py` mirroring the existing per-model test structure, with strict mode set, a dispatch assertion, and a copy-counter assertion.

### T8.2 — Phase-transition test

Explicitly compare NPU-prefill-then-GPU-decode against all-GPU for the same prompt. This is the only test that catches a KV convention mismatch, and it belongs before the end-to-end test rather than after.

It is also the sharpest zero-copy check in the suite: the KV cache is the largest tensor crossing the boundary, so if anything repacks or stages it, this is where the counter fires.

### T8.3 — Llama-3.1-8B

Same, at scale. This is where weight streaming stops being optional.

### T8.4 — Perplexity

On the assembled model, both sizes.

### T8.5 — Documentation

Update `CLAUDE.md` with the NPU build prerequisites, the configuration surface, every gotcha discovered along the way, and the registered-memory rule. Reconcile this document and the design document with what was actually built.

---

## Phase 9 — Beyond v1

Not scheduled. Recorded so the gaps are known rather than rediscovered.

| ID | Item | Why it waits |
|---|---|---|
| T9.1 | Further fusion beyond `ssmlp` — absorbing a layer's trailing norm into the next layer's input norm | MLP fusion itself is T5.6, in v1. This is the incremental remainder |
| T9.2 | Chunked prefill | Removes the bucket-ceiling fallback for long prompts |
| T9.3 | OGA integration | v1 is pure ORT |
| T9.4 | Multi-turn re-prefill | Per-call dispatch already permits it; needs testing |
| T9.5 | Mixture-of-experts on NPU | Out of v1 model scope |
| T9.6 | Throughput and power characterization | Accuracy first; no numeric gate exists |
| T9.7 | Logit softcapping, interleaved rotary embedding | Needed by other model families, absent on both sides |
| T9.8 | Native bf16 across the boundary | Only if the per-operator conversion cost proves material |

---

## Parallelism map

```
P1..P4 prerequisites
   │
   ├── T0.1 [GATE] ──┬── T0.3 ────────────┐         (agent A)
   │                 │                    │
   ├── T0.2 [GATE] ──┴── T0.4 ────────────┤         (agent B)
   │                                      │
   └──────────────► T1.1 [GATE] ◄─────────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
      T1.2/T1.3       T2.1/T2.2       T3.1/T3.2       (three agents)
      T1.4            T2.3/T2.5       T3.3/T3.4
                      T2.4 [GATE]
        │               │               │
        └───────────────┴───────┬───────┘
                                │
                          T4.1 [GATE]  →  T4.2 → T4.3 → T4.4 [GATE]
                                │
                ┌───────────────┴───────────────┐
                │                               │
        T5.1/T5.4 (agent A)             T5.2/T5.3 (agent B)
                └───────────────┬───────────────┘
                                │
                          T5.5 [GATE]  attention
                                │
                ┌───────────────┴───────────────┐
              Phase 6 islands            Phase 7 weights
                └───────────────┬───────────────┘
                                │
                          Phase 8 end to end
```

Three serialization points are absolute. Nothing starts before T0.1 and T0.2 answer. Nothing implements against the ABI before T1.1 freezes it. Nothing in Phase 5 starts before T4.4 proves one operator works end to end.

---

## Definition of done

- Llama-3.2-1B and Llama-3.1-8B both prefill on the NPU and decode on the GPU, in one ORT session, through one execution provider, with every allocation made by this repository's memory manager.
- **Zero copy holds end to end.** No activation, no graph input or output, and no KV cache entry is copied, staged, or repacked when execution crosses between the NPU and the GPU. The boundary-copy counters read zero across a full generation, and no debug copy path is active.
- No reference to `RyzenMM` exists anywhere in the resulting system.
- Greedy generation matches a CPU reference; perplexity is within tolerance.
- Every accuracy test claiming NPU coverage sets strict mode, asserts on dispatch, and asserts the copy counters are zero — so it can pass neither by silently falling back nor by silently copying.
- The registered-memory rule is enforced by a hard error, and a test proves a GPU-pool pointer is rejected.
- Decode throughput and numerics are unchanged from before this work.
- The hybrid LLM project is unmodified and still works.
- `CLAUDE.md`, this document, and the design document reflect what was built.
