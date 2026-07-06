<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Debug-only CPU fallback (ORT CPU EP Invoke) — implementation plan

This document describes the **development / debug** path that lets selected `wrap_*`
runtime ops fall back to **ORT CPU EP** alignment checks via a Quark-style
`Ort::Op::Create` + `Ort::Op::Invoke` + CPUGate stack. **Not part of the default
Release product**; production inference remains **all-GPU**. Performance on the debug
path is not a goal.

---

## 1. Goals and non-goals

### Goals

- In Debug / RelWithDebInfo builds (controlled by a **CMake option**), named ops in
  `wrap_*` may take:
  **stream sync → input D2H → EP callback → output H2D**,
  to compare against **ORT CPU EP** and isolate accuracy gaps.
- **`model.dll` (runtime bitcode)**: **branch + sync + memcpy + C ABI callback only**;
  does **not** link ONNX Runtime or embed the Gate.
- **`onnxruntime_morphizen_ep.dll`**: **CPUGate** (embedded `cpugate_shell.onnx` shell
  + `com.amd.morphizen.cpu.CpuGate` custom op) borrows **`OrtKernelContext*`**; lazily
  caches **`Ort::Op`** per `(op, axis, dtypes, …)` and runs **`Ort::Op::Invoke`** on
  the gate thread (first op: **Gather**). Does **not** use MorphiZen `OpInvoker` /
  `model_proto_serialize_as_string`. `InferenceState` calls `set_cpu_fallback` on init
  and clears on destroy.

### Non-goals

- Release builds **exclude** this path by default (option OFF + optional `NDEBUG`).
- Not all ops on day one; GPU fp16 vs CPU fp32 need not be **bit-identical** (separate
  thresholds).

---

## 2. Architecture

```
[model.dll — wrap_*]
  if (debug fallback compiled in && env lists this op)
        hipStreamSynchronize(stream)
        D2H → fill C descriptor → state->cpu_fallback.invoke(user, &desc)
        H2D → return
  else
        existing GPU path (unchanged)

[EP — MorphiZen]
  after inference_init: get_method("hipdnn_ep_set_cpu_fallback") → Gate Manager → set_cpu_fallback
  invoke: Ort::Value(borrow host) → CPUGate Ort::Op::Invoke (ORT CPU kernels only)
  on Session destroy: stop gate thread, release Manager, set_cpu_fallback(null)
```

**Environment variables** (names may drift; code wins):

- `HIPDNN_EP_DEBUG_CPU_FALLBACK_OPS`: comma-separated list (first op: `Gather`).
- Optional: `HIPDNN_EP_DEBUG_CPU_FALLBACK=1` master switch.

**Reading env from `model.dll`**: use **`hip_get_env` / `GetEnvironmentVariableA`**
(static CRT — not `std::getenv`). See `include/hip/debug_log.h`.

**Design principle — ORT CPU kernels only.** Every CPU fallback path must run
through `Ort::Op::Create` + `Ort::Op::Invoke` on the CPUGate thread so results
match **ORT CPU EP**. No hand-rolled reference implementations (they may align
with the GPU kernel but not with ORT and would invalidate accuracy bisection).

**All ops (including Gather)** use `HipdnnCpuFbGenericDesc` → `HipdnnCpuFbGenericHostDesc`
→ `invoke_generic_cpu` → CPUGate `Ort::Op::Invoke` on the gate thread. Gather-specific
logic in the EP is limited to int32→int64 index widen and ONNX output-rank correction
(`gather_onnx_output_shape` in `cpu_fallback_bridge.cpp`).

---

## 3. Phased delivery

### Phase 0 — Build

| Item | Notes |
|------|-------|
| CMake | `HIPDNN_EP_DEBUG_CPU_FALLBACK`, default **OFF**; CI/Release stay OFF. |
| EP | When ON, link ORT C++ API; compile Gate + Manager. |
| Runtime / bitcode | When ON, `lib/Runtime` enables branches; when OFF, callback null. |

### Phase 1 — C ABI and `RuntimeState`

1. Pure C **descriptor**: `HipdnnCpuFbGenericDesc` / `HipdnnCpuFbGenericHostDesc`
   (`op_name`, IO tensors, attrs).
2. `RuntimeState`: `cpu_fallback_user`, `cpu_fallback_invoke`.
3. Export `hipdnn_ep_set_cpu_fallback`; `initialize_state` clears it.
4. `GenerateInterface.cpp` export list matches other runtime exports.

### Phase 2 — EP CPUGate + `Ort::Op`

1. Pattern from `3rd-party/Quark/.../execution_provider_cpugate.cpp`: `Manager`, Gate
   `CustomOp`, background thread `Session::Run` blocks to hold **`OrtKernelContext*`**.
2. Gate registered **inside this EP** (no `onnxruntime_providers_ryzenai.dll`).
3. `Invoke` serialized; pending-invoke queue dispatches on gate thread.
4. **`Ort::Op` cache** keyed by `(domain, op_type, opset, types, attrs)`.
5. **InferenceState**: `set_cpu_fallback` after successful init; clear on destroy.

### Phase 3 — First vertical slice: **Gather**

- **`wrap_gather` only**: macro + env hit → `invoke`; else GPU kernel.
- Flow: `hipStreamSynchronize` → host staging vectors → D2H → `desc` → `invoke` → H2D.
- EP: CPUGate + `Ort::Op::Invoke` only (ORT CPU EP alignment).
- Acceptance: match Python single-op ORT CPU or small ONNX session within cosine /
  max_abs thresholds.

### Phase 4 — Extension

- Table: `wrap_*` → op_id → ONNX name/domain/opset/attrs; add `Where`, `LayerNorm`, etc.
- Per-op regression tests.
- On `Invoke` failure: **`LOG(ERROR)` + non-zero return**; `wrap_*` falls back to GPU.
  **No `LOG(FATAL)`** in the callback (would abort the outer ORT `Run`).

### Phase 5 — Documentation

- This file + one-line index in `CLAUDE.md` (debug-only, not Release).

---

## 4. Suggested PR split

| PR | Content |
|----|---------|
| PR1 | CMake option + `RuntimeState` + C ABI + empty callback smoke. |
| PR2 | EP CPUGate + `Ort::Op::Invoke` smoke without `wrap_*`. |
| PR3 | **`wrap_gather` end-to-end** + minimal comparison test. |
| PR4 | Docs + more ops as needed. |

---

## 5. Risks and mitigations

| Risk | Mitigation |
|------|------------|
| ORT upgrade changes `Invoke` / Context | Keep minimal Invoke smoke in debug CI. |
| Gate depends on ORT internals | Document; debug-only; re-smoke on ORT bump. |
| fp16 GPU vs fp32 CPU | Document thresholds; fallback uses storage dtype on ORT CPU. |
| Nested `Session::Run` / `OpInvoker` | Do not use `OpInvoker` in EP. All ops (including large Gather) use CPUGate + gate-thread `Ort::Op::Invoke` only. No `CHECK`/`FATAL` in callback. |
| `HipdnnCpuFbGenericHostDesc` / `sizeof` mismatch | Rebuild EP + model.dll; delete `%TEMP%\morphizen_mlir_*` cache. |
| MorphiZen MLIR serialize ≠ ONNX protobuf | Do not feed MLIR bytes to `Ort::Session`. Use CPUGate shell + `Ort::Op::Create`. |
| ORT Gather output rank vs MLIR memref rank | Prefer ONNX formula rank when element count matches (`gather_onnx_output_shape`). |
| `hip-onnx-runner` random `input_ids` | Runner bounds `input_ids` to `[0, --token-vocab)` by default. |
| Negative shape dims | `wrap_gather` skips CPU fallback; EP validates `product(shape) == num_elements`. |
| Large fp16 embedding D2H staging OOM | Fallback returns error → `wrap_*` uses GPU; no non-ORT reference. |

---

## 6. Consistency summary

- **CPUGate / ORT stay out of `model.dll`**; only env-gated callback from `wrap_*`.
- **Shipping product**: all GPU; feature **off in Release** by default.
- **Phase 3 first op**: `Gather` (`wrap_gather` ↔ ONNX `Gather`).

---

## Revision history

| Date | Notes |
|------|-------|
| 2026-06-12 | Initial plan from design discussion. |
| 2026-06-12 | Gather descriptor element counts + shape validation; no `CHECK`/`FATAL` in callback. |
| 2026-06-15 | Dropped `OpInvoker`+`Ort::OpAttr`; embedded per-variant Gather ONNX + `Session::Run`. |
| 2026-06-15 | `indices_element_size_bytes`; int32→int64 widen; ONNX output rank for `CreateTensor`. |
| 2026-06-15 | `Session::Run` with preallocated outputs instead of IoBinding for Gather session. |
| 2026-06-25 | CPUGate + `Ort::Op::Invoke`; `cpugate_shell.onnx`; removed per-variant Gather embed. |
| 2026-06-25 | Unified Gather onto generic descriptor + CPUGate; removed per-op gather paths. |
| 2026-06-25 | Large Gather: generic D2H + `ort_session` worker `Session::Run` (embedded ONNX in `kSessionSpecs`); small ops stay CPUGate `Invoke`. **Superseded** — large Gather now uses CPUGate `Invoke` like other ops. |
| 2026-06-29 | Quark `RunOnKernelConstruction`: queue `Ort::Op::Create` on `pending_inits_`, run in `on_kernel_constructed` during shell Session ctor (not after). |
| 2026-06-25 | Removed debug-only `ort_session` worker, `GatherInvokeProbe`, and A/B/C context experiment (`HIPDNN_EP_DEBUG_CPU_FB_CTX_TEST`). |
