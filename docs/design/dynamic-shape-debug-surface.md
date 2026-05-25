<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Dynamic-shape Debug Surface

**Date:** 2026-05-22
**Document Type:** Design
**Status:** Draft (one env var wired as a partial-coverage gate)
**Related:** [compiler-runtime-contract.md](compiler-runtime-contract.md), [morphizen-ep-integration.md](morphizen-ep-integration.md)

---

## Table of Contents

- [Overview](#overview)
- [Wired today](#wired-today)
  - [`HIPDNN_EP_DEBUG_SHAPES` (alias `HIPDNN_EP_TRACE_SHAPES`)](#hipdnn_ep_debug_shapes-alias-hipdnn_ep_trace_shapes)
  - [`HIPDNN_EP_TRACE_SLOTS`](#hipdnn_ep_trace_slots)
  - [Always-on sanity checks](#always-on-sanity-checks)
- [Deferred: `HIPDNN_EP_VALIDATE_SHAPES`](#deferred-hipdnn_ep_validate_shapes)
  - [What users see today](#what-users-see-today)
  - [Intended behavior](#intended-behavior)
  - [What it would take to wire](#what-it-would-take-to-wire)
  - [Why it is deferred (low-priority)](#why-it-is-deferred-low-priority)
  - [What would trigger reviving it](#what-would-trigger-reviving-it)
- [Related Documents](#related-documents)

---

## Overview

The data-dependent dynamic-output-shape feature (see
[compiler-runtime-contract.md § Runtime slot mechanics](compiler-runtime-contract.md#runtime-slot-mechanics))
spans three layers — the compiler's `output_dim_specs` attribute on every
`hip.*` op, the EP-side `DimSpecResolver` that runs against per-`Compute()`
input shapes/values, and the in-DLL runtime slot ABI
(`publish_dim` / `read_dim` / `publish_buffer` / `read_buffer`). When an output
shape comes out wrong, the question is almost always *which* layer lied. The
debug surface is designed to localize that in one log read.

All gates are **zero-overhead when off** — each helper in
`lib/Runtime/debug_log.h` uses a `static const bool` cache + branch-prediction,
read once on first call, never again.

---

## Wired today

### `HIPDNN_EP_DEBUG_SHAPES` (alias `HIPDNN_EP_TRACE_SHAPES`)

| Aspect | Detail |
|---|---|
| **Where it fires** | EP side (`backend-mlir-compiler/custom-op-mlir/src/MlirCustomOp.cpp`, `lib/Runtime/DimSpecResolver.cpp`). |
| **At session open** | Dumps every output's full DimSpec tree once (`[CTor] Output[i]: …`). This is the *declared* compiler output for the model. |
| **Per `Compute()`** | Dumps the resolved shape pre- and post-Compute (`[EP] Output[i] pre-/post-compute resolved shape=[…]`), and traces each DimSpec leaf as the resolver walks it (`[Resolver] InputValueI64(…)` / `[Resolver] Add(…)`). |
| **Use** | When an output dim resolves to a surprising value — gives you `(declared spec) → (per-leaf resolution) → (final dim)` in one log, so you can tell whether the compiler emitted the wrong spec or the resolver did the wrong arithmetic. |
| **Smoke test** | `test/numeric/tests/test_debug_surface.py::test_debug_shapes_traces_resolution`. |

### `HIPDNN_EP_TRACE_SLOTS`

| Aspect | Detail |
|---|---|
| **Where it fires** | Runtime side (`lib/Runtime/real/hipdnn_ep_runtime_state.cpp`), inside the model.dll. |
| **What it logs** | Every `publish_dim` / `read_dim` / `publish_buffer` / `read_buffer` call:<br/>`[Slots] publish_dim(7) = 12   <- wrap_nonzero`<br/>`[Slots] read_dim(7)    = 12   <- wrap_shape consumer` |
| **Use** | Confirm that a producer wrap fires BEFORE its consumer, or localize a read-before-publish abort by name without rebuilding. |
| **Smoke test** | `test/numeric/tests/test_debug_surface.py::test_trace_slots_logs_slot_publishes`. |

### Always-on sanity checks

These have **no env var** and **cannot be disabled** — they're correctness
guards, not diagnostics. Defined in `hipdnn_ep_state_read_dim` and
`hipdnn_ep_state_read_buffer` (`lib/Runtime/real/hipdnn_ep_runtime_state.cpp`),
they `LOG(FATAL)` (which is `fprintf(stderr) + std::abort` in the runtime) on:

- **Slot id out of range.** Model metadata is inconsistent with the wrapper
  that called it — almost always a stale `morphizen_mlir_*` cache after a
  runtime edit, a new wrapper added without its `getRuntimeFuncSpecs()` entry,
  or hand-rolled bitcode skew. Run `del %TEMP%\morphizen_mlir_*` after every
  runtime change.
- **Read-before-publish.** The producing Category-C wrap either didn't run, or
  ran but bailed before `publish_dim` / `publish_buffer`, or the lowering
  ordered the read ahead of the write (ComposeDimSpecs ordering bug). The abort
  message names the offending slot id; the crash-handler stack trace names the
  consumer.

---

## Deferred: `HIPDNN_EP_VALIDATE_SHAPES`

### What users see today

| Aspect | Status |
|---|---|
| Env var | Recognised — `hipdnn_ep_validate_shapes_enabled()` in `lib/Runtime/debug_log.h` returns `true` when set. |
| Behavior when set | **None.** The gate is currently unreferenced — no call site reads it. Setting it produces no extra logging and no extra work. |
| Test coverage | The gate is recognised by `test/numeric/tests/test_debug_surface.py` only insofar as the harness scrubs it from the environment before non-VALIDATE tests run, to keep the surface clean once it gets wired. There is no test that exercises the validate path itself. |
| Tested as documented in CLAUDE.md | Yes — the "Dynamic-shape Debugging" table lists it as "gate exists; wiring deferred". |

### Intended behavior

After every `Compute()` that produces at least one dynamic-shape output, the EP
would:

1. Materialize an in-memory ONNX subgraph from the fused `onnxruntime::Model *`
   the EP already holds (`MlirCustomOp::MlirCustomOp` constructor — see
   `backend-mlir-compiler/custom-op-mlir/src/MlirCustomOp.cpp` line 546).
2. Run that subgraph on the ORT CPU EP with the same inputs the current
   `Compute()` was called with.
3. Compare **shape only** (not values) against the EP's `resolved_shapes[i]`
   for every output.
4. On divergence, `LOG(ERROR)` (not `LOG(FATAL)` — this is a diagnostic, not a
   correctness gate) with a one-line `actual=[a,b,c] expected=[a,b',c']`
   message.

The natural slot-in is right after the existing post-Compute
`debug_shapes_enabled()` block in `MlirCustomOp::Compute()` (around
`resolved_shapes[i]` finalization), where the resolved shape is already in
scope.

### What it would take to wire

Concretely, the work is **not large** — the underlying `Model *` and the
resolved shapes are already available — but it spans a few moving parts that
all have to land together:

1. **Cache an ORT CPU `InferenceSession` per fused MlirCustomOp**, built once
   in the constructor from the fused `Model *`. The model already carries the
   subgraph proto; building a session is a one-time cost.
2. **Per-Compute() input marshaling.** Wrap each input `OrtKernelContext`
   tensor into an `Ort::Value` bound to the CPU session — including respecting
   the `input_index_map_` that maps compiler order to ORT kernel context order.
3. **Per-Compute() run + compare.** Run the CPU session, walk
   `session.GetOutputCount()`, compare each output's `GetTensorTypeAndShapeInfo()
   .GetShape()` against `resolved_shapes[i]`. Use the same
   `output_index_map_` mapping in reverse.
4. **Diagnostic message format.** Include the per-leaf DimSpec resolution that
   `HIPDNN_EP_DEBUG_SHAPES` already traces, so the divergence is debuggable
   without re-running.
5. **Test.** Add a test under `test/numeric/tests/test_debug_surface.py` that
   forces a divergence (e.g. a synthetic model whose DimSpec is hand-edited via
   FB JSON to lie) and asserts the `LOG(ERROR)` line appears.

Approximate scope: ~150–250 lines of EP-side code, one test, no compiler or
runtime changes (the gate already exists, and the underlying model + shape
data are already on the EP side).

### Why it is deferred (low-priority)

The validator's design intent — *catch a wrong DimSpec or wrong resolver in
CI before it manifests as a kernel crash* — is **already covered well enough
by the existing numeric tests**, for the cost-benefit ratio to come out
negative right now:

1. **End-to-end correctness coverage exists.** The full numeric suite under
   `test/numeric/tests/` runs every wrap with concrete inputs and compares
   actual output values (not just shapes) against the ORT CPU reference. A
   wrong DimSpec on a dynamic-shape output manifests as one of:
   - an EP-side fatal — wrong dim resolves to a negative or insanely-large
     value, `ctx.GetOutput` aborts;
   - a runtime fatal — wrong dim causes `read_before_publish` or an OOB slot
     read (always-on sanity check fires);
   - a numeric divergence — wrong shape means wrong data written into the
     allocated output, the numeric test's `assert_allclose` fails.
   In all three cases the *existing* `pytest test/numeric/` run already
   surfaces the bug. Shape-only validation would just give us a slightly
   nicer error message for the third case.
2. **The interesting failures aren't shape-only.** When we have seen
   compose-DimSpecs or resolver bugs in practice (Qwen embedding
   `Shape(hip.transpose(hip.nonzero))` chain, ABI-format mismatch between
   compiler and runtime in early stage-2 work), the failure mode was a
   runtime abort or a numeric mismatch, not a silent shape-only divergence.
   We have not yet hit the failure class this validator is purpose-built for.
3. **The wired tracers cover the diagnostic angle.** When debugging a wrong
   resolved shape, `HIPDNN_EP_DEBUG_SHAPES` already prints
   `(declared spec) → (per-leaf resolution) → (final dim)` AND
   `HIPDNN_EP_TRACE_SLOTS` already prints `(publishing wrap, value)` for every
   slot. Adding `(CPU reference shape)` to that picture is incremental — it's
   another data point on a problem the developer is already looking at.
4. **Cost is non-trivial.** Building, holding, and per-Compute()-running a
   second ORT session for every fused MlirCustomOp doubles the framework
   overhead in CI: model construction, input marshaling, output materialization
   on the CPU side, every Compute(). The doc string in `debug_log.h` already
   warns of a "10-100× per-Compute() slowdown", which is an estimate but
   directionally correct.
5. **Higher-impact work exists.** The dynamic-shape feature still has a known
   framework gap — `ShapeToHip` runs before DimSpec composition, so
   rank-preserving HIP ops like `hip.transpose` propagate no `output_dim_specs`
   on dynamic dims (this is exactly why
   `test_nonzero_qwen_embedding.py::test_qwen_embedding_dynamic_shape_*` is
   `xfail`). Closing that gap unlocks real-world models. Building a CPU-shape
   validator that would only ever assert against models the EP cannot yet
   compile is the wrong order of operations.

### What would trigger reviving it

The deferred validator goes from "low-priority" to "build it" when **any** of
the following happens:

1. **A new DimSpec node kind ships.** The current set
   (`Static`, `InputDim`, `InputValueI64`, `RuntimeSlot`, plus the
   `Add` / `Mul` / `Min` / `Max` / `Div` / `Mod` operators) is enough for
   today's models. The first time we add a new node kind (say,
   `InputValueI64Sum` for ranged inputs, or a `Where`-style conditional), the
   resolver and ComposeDimSpecs pass both grow new code paths that need an
   independent ground-truth comparison. A shape-only CPU validator is exactly
   the right tool: it doesn't care which kind we added, just whether the
   resolved shape matches what ORT CPU computes.
2. **A real silent-shape-divergence bug is reported.** If we ever see a case
   where the EP produces wrong-shaped output with no fatal and no numeric
   divergence (because the wrong shape happens to be a numerical subset of the
   right shape, or because the consumer happens to ignore the extra rows), the
   only catch-all guard is shape-only CPU validation.
3. **A new compiler optimization touches `output_dim_specs`.** Any future pass
   that rewrites or fuses ops in the HIP dialect after DimSpec composition has
   a real risk of dropping or mutating a DimSpec. A per-Compute() shape check
   would catch the regression in a single CI run.
4. **The Qwen embedding `Shape(hip.transpose(hip.nonzero))` xfail is closed.**
   When `ShapeToHip` learns to propagate DimSpecs through rank-preserving HIP
   ops, the new code path is exactly the kind of thing where shape-only
   validation pays off — the composition is non-trivial and the failure mode
   (wrong dim ordering) is shape-only by construction.

When any of those happens, the work in
[§ What it would take to wire](#what-it-would-take-to-wire) becomes the right
next thing.

---

## Related Documents

- [compiler-runtime-contract.md](compiler-runtime-contract.md) — DimSpec node
  kinds, runtime slot ABI, pipeline-ordering invariants for the dyn-shape
  codegen.
- [morphizen-ep-integration.md](morphizen-ep-integration.md) — EP DLL
  contracts; `inference_init` public interface that the validator would
  re-enter.
- `lib/Runtime/debug_log.h` — gate definitions.
- `test/numeric/tests/test_debug_surface.py` — smoke tests for the wired
  tracers; placeholder env-scrub for the deferred validator.
