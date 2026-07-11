# HIP-graph decode capture — findings & current status

Reference notes for the Qwen3.5-35B-A3B (MoE, GQA) decode-throughput / HIP-graph-capture
work on the AMDGPU (MorphiZen / hipgpu) execution provider. **Latest, corrected status only.**
Model dir: `D:\Qwen3.5-35B-A3B-fp16-ve-fp16-int4-text-gs32-dml`.

Last updated: 2026-07-11.

---

## 1. Goal & the core lever

Decode is launch/dispatch-overhead-bound (~1,100 dispatches/token, ~45% of span in
per-kernel barriers). The proven big lever is **GPU-graph capture of the decode step**
(OGA PR #2070: **+20–45%** on this exact Qwen3.5 family with the CUDA EP; DML PR #1305:
token-gen ~10× on Phi-3.5). Capture requires a **static, capture-clean decode** (fixed
shapes, stable buffer addresses, **no host readbacks / stream syncs mid-step**).

---

## 2. Confirmed baseline (REAL decode path)

Measured via `benchmark_multimodal` (through `run_onnx.py`) with `HIPDNN_EP_PERF=1`,
parsing the op-profile:

- **Decode: exactly 3 `readback_scalar` per token** (+ occasional `readback_i32`).
- Prefill: 199 `readback_scalar` (one-time).
- Decode throughput: **~38 tps** baseline.

`readback_scalar` = `hipdnn_ep_readback_scalar` = **D2H copy + `hipStreamSynchronize`**
(see `lib/Runtime/real/memory.cpp`, `lib/Conversion/OnnxToHip/ReadbackScalar.h`). Emitted
whenever a converter needs a host scalar for shape arithmetic (Range bound, dynamic
Reshape `-1`, Expand extent, Pad entry, Loop trip count). **Folds to a constant only when
the value is a compile-time constant** (`arith.constant` / `onnx.Constant`).

---

## 3. THE decisive finding (what blocks capture and why)

**The 3 readbacks are compiled into the EP artifact and execute every decode step
regardless of the runtime input shapes.** A mid-capture `hipStreamSynchronize` is **fatal**
on HIP/Windows (hard crash `0xC0000409`, not a soft error like CUDA) — so the decode is
**not capture-clean**, and capture cannot even be *attempted* until the readbacks are gone.
(Verified via the S1 capture probe: enabling it crashes inside `compute_fn_`.)

### Route A (OGA static-mask config) does NOT work — proven by experiment
Hypothesis was: enable OGA's static-mask handling for AMDGPU (share-buffer is already on)
→ fixed shapes → readbacks fold. **Refuted.** Patched
`onnxruntime-genai/src/models/position_inputs.cpp` `ShouldUseStaticMaskHandling()` to enable
static mask for AMDGPU, rebuilt OGA, swapped the DLL, measured on real 35B decode:
- **`readback_scalar` still = 3/token** (no fold).
- **tps regressed 38 → 5.9** (config `max_length=262144` ⇒ enormous static mask; AMDGPU
  falls back to CPU mask updates each step).

**Why it can't work:** the EP compiles the ONNX graph **symbolically once**; the
`readback_scalar` ops are baked in. Feeding fixed-size inputs at runtime does not make the
scalars compile-time constants, so nothing folds. Readback elimination is **unavoidable
EP-compiler work** — it cannot be bypassed via OGA/runtime config.

### Why DML/CUDA EPs don't have this problem
They never lower shape scalars to host readbacks — shapes come from ORT shape inference /
on-device. The MorphiZen EP chose `hip.readback_scalar` (host D2H+sync), which is inherently
capture-incompatible.

---

## 4. The real remaining work (EP compiler) — two options

To make decode capture-clean (`readback_scalar` decode count → 0, output byte-identical),
one of:

1. **Static-dim decode compile** — compile the decode with the relevant dims fixed so the 3
   scalars become compile-time constants and fold. (Original "S3".) Requires the EP to
   shape-specialize the decode artifact.
2. **Change the shape-op lowering** so those scalars are consumed **on-device** (no host
   D2H+sync) — mirror how DML/CUDA stay capture-clean. More robust; generalizes beyond decode.

Either is genuinely multi-day EP work. After it lands: implement HIP capture/replay
(stable buffers, in-place recurrent/conv state per OGA #2070) and wire it (see §6).

---

## 5. Model facts (this model is capture-CAPABLE)

- Type `qwen3_5_moe`: standard **GQA decoder** (`past_key_values.%d.key/value`,
  `num_attention_heads=16`, `num_key_value_heads=2`, `head_size=256`, 40 layers) + MoE.
  (The `qwen.py` builder also has a GatedDeltaNet linear-attn/conv-state hybrid path, but
  this exported decoder uses standard KV — no conv/recurrent state inputs in the config.)
- `past_present_share_buffer: true` AND `num_beams==1` ⇒ `IsPastPresentShareBufferEnabled`
  is **true** on AMDGPU ⇒ KV is already a **fixed max buffer**.
- **DML runs the IDENTICAL model + config capture-clean** (`genai_config.json.dml` is the
  same). ⇒ the readbacks are a **MorphiZen-EP artifact, not a model property**.
- `text_seq1.onnx` exists alongside `text.onnx` (a seq=1 decode-specialized graph) — not
  currently used by the EP config (`filename: text.onnx`); worth investigating for option 1.

---

## 6. OGA integration facts (for the eventual capture wiring)

- AMDGPU is a **plugin EP V2** (`SessionOptionsAppendExecutionProvider_V2`, `amdgpu-ep.dll`,
  options via `ep.amdgpu.*`). ⇒ can hook ORT's **provider-agnostic graph-capture framework**
  (onnxruntime PR #28002: ORT core orchestrates warmup→capture→replay via the `OrtEp` C API,
  capture-safe dispatch skips `hipStreamSynchronize`/device-set during capture).
- `onnxruntime-genai/src/amdgpu/session_options.cpp` `AppendExecutionProvider` currently
  **ignores `disable_graph_capture`** and never enables capture — no wiring at all. The CUDA
  reference to mirror is `src/cuda/session_options.cpp` + `src/config.cpp` (PR #2070):
  enable `enable_cuda_graph`-equivalent, force OFF for vision/embedding, ON for decoder only.
- Capture preconditions enforced by OGA: `past_present_share_buffer` (else `kv_cache.cpp:331`
  throws), stable addresses (recurrent/conv in-place copy, not pointer swap), single
  generator reused via `RewindTo(0)` (PR #2002), `num_beams==1`.

---

## 7. Tooling / environment facts (hard-won)

- **`hip-onnx-runner` is UNUSABLE for this model.** It feeds random inputs; being MoE, some
  expert gets 0 routed tokens ⇒ divide-by-zero (`0xC0000094`) in expert dispatch, during
  "Running inference...". `-f` free-dims and `-p` positive-only do NOT fix it. So it cannot
  be used for readback counting, crash analysis, or clean IR dumps of this model.
  **Corollary:** the earlier "S2 static-specialization 0xC0000094" analysis was confounded
  by this — baseline (no flags) crashes too. Use the REAL path for everything.
- **Readback / perf measurement:** REAL path only —
  `run_onnx.py --benchmark benchmark_multimodal.py ... -v` with `HIPDNN_EP_PERF=1`; grep
  `[PERF]  readback_scalar`. Harness: `tools/perf-report/measure_decode.py` (note its
  `readback` mode uses `hip-onnx-runner` and is therefore unreliable for THIS model; its
  `tps` paired-A/B mode via benchmark is fine).
- **IR dump:** `HIPDNN_EP_IR_DUMP_PATH` (+`_SINGLE`/`_AFTER_ONLY`/`_TREE`). Buffered
  single-file dump does NOT flush if the process crashes (so runner dumps are lost). Real-path
  full-IR dump is impractically slow (per-pass print on ~40 MB module × 3 sessions).
- **Readback site count:** compile-time there are **212** readback sites in the decoder
  graph, all `loc(unknown)` (import drops op names). Only 3 execute per decode step.
  Pinpointing the 3 needs **runtime op-ID tagging** (unique attr on each `ReadbackScalarOp`,
  plumbed through HipToLLVM lowering to the runtime call, logged per execution). Not yet done.
- **OGA IS buildable locally** (recurring blocker resolved):
  `python build.py --use_dml --config Release --parallel --skip_tests --skip_examples`
  from `onnxruntime-genai/`. `build.py` **auto-downloads** a matching DirectML ORT (no
  `ORT_HOME` needed; internet required). Needs `requests`+`packaging` in the build python.
  Core `onnxruntime-genai.dll` compiles fine; the C **examples** target fails on
  `onnxruntime_cxx_api.h` include path ⇒ always pass `--skip_examples`.
  - Gotcha: CMake picks the "best" Python for the binding (built **cp310** here, venv is
    **cp314**); `--cmake_extra_defines Python_EXECUTABLE=...` only takes on a *clean*
    reconfigure. **But** the core `onnxruntime-genai.dll` is Python-independent, so you can
    swap just that DLL into the venv (`site-packages/onnxruntime_genai/`) and keep the venv's
    cp314 `.pyd` — ABI matches for same 0.14.0 version / body-only changes.
- Build/run env: EP built via `ninja -C build/onnx-hipdnn-ep` (after `vcvars64.bat`), THEROCK
  runtime at `therock-keep/bin`, venv at `ci-fresh/venv314`, ORT runtime =
  `onnxruntime_directml` **1.25.1**, OGA = `onnxruntime_genai_directml` **0.14.0**.

---

## 8. Committed artifacts (branch `perf/s0-measurement-harness` in `onnx-hipdnn-ep`)

- **S0** (`616023a1`): `tools/perf-report/measure_decode.py` measurement harness.
- **S1** (`0330d6d0`): `HIPDNN_EP_CAPTURE_PROBE` capture-safety probe in
  `InferenceState::compute` (off by default). Proved decode not capture-clean (crashes).
- **S2** (`d4852f94`): gated `HIPDNN_EP_DECODE_SEQ1`/`_BATCH1` import-time dim binding +
  diagnosis. **NOTE:** its "runtime divide-by-zero requires both static dims" conclusion is
  **partly invalidated** — the crash is the `hip-onnx-runner` MoE-random-input artifact (§7),
  not a real static-decode bug. Re-verify any static-compile crash on the REAL path.

No OGA changes are committed (the static-mask experiment patch was reverted; venv DLL restored).

---

## 8b. Step A / Step B implementation progress (branch perf/s0-measurement-harness)

Env flag `HIPDNN_EP_DECODE_SKIP_SYNC` (off by default) + a per-Compute decode
hint (`RuntimeState::decode_hint`, `decode_seqlens_k`) set by the EP
(`MlirCustomOp::Compute` detects a decoder single-token step: >5 inputs, an
`inputs_embeds` rank-3 hidden>=512 seq==1). All changes greedy-token-identity
verified (SHA256 `08DA1990...` unchanged) and pushed.

Decode capture blockers ELIMINATED (each a device->host sync removed in decode,
host-sourced or capture-guarded):
1. `readback_scalar` x3/token (Step A) -- host-read resident scalar.
2. `readback_i32` -- host-read resident scalar.
3. `cumsum` axis -- host-read constant.
4. GQA `seqlens_k` (`read_seqlens_k_for_dispatch`) -- host-sourced from
   `attention_mask.shape[1]-1` (batch==1 unpadded).
5. `hipdnn_ep_stream_sync` end-of-compute sync -- skipped when
   `hipStreamIsCapturing` (capture-only guard, not decode-gated).
6. `slice` starts/ends/axes/steps -- host-read without sync.

Capture-probe progress (HIPDNN_EP_CAPTURE_PROBE, decode-scoped, warm step 4):
during-capture errors 2397 -> 1; first failing op moved
`hip_matmul_nbits`(cold pool `hipMalloc`) -> `hip_gqa_kv_cache_append` ->
`hip_elementwise_equal` (current).

CURRENT WALL: `hip_elementwise_equal` (a scalar i64==i64 control check feeding a
`Where`) fails with "previous error during capture". The illegal op is NOT in
the ~200 lines before it and is NOT a logged sync/alloc -- intervening
`matmul_nbits` launches don't check `hipGetLastError`, so the invalidating op is
undetected until the Equal. Most likely a silent **allocation** during capture
(pool/output/intermediate `hipMalloc`), i.e. the allocation-stabilization piece,
not another simple sync.

## 8c. The two hard remaining pieces (multi-day/multi-week)

1. **Allocation stabilization**: no `hipMalloc` during capture. Requires all
   pools fully pre-grown (warmup), output allocator reuse, and per-op workspaces
   (matmul zp cache, GQA/qmoe scratch) warm/max-sized. This is the current wall.
2. **Static decode shapes (THE crux, original S3)**: a captured graph bakes in
   the current `total_sequence_length`. Correct replay across steps requires
   fixed launch dims (max-buffer) + the varying length read on-device from a
   scalar (`seqlens_k`), so the SAME graph is valid for every token. GQA fused
   decode already does this; ALL decode ops must. Without it, a captured decode
   is only valid for one sequence length.
3. Then the S4 capture/replay machinery: warmup -> capture -> instantiate ->
   replay on matching shapes, eager fallback, stable buffer addresses.

## 9. Bottom line

- The static-KV precondition is already met; the model is capture-capable; DML proves it.
- **The one hard blocker is EP-side: the 3 decode `readback_scalar` host-syncs baked into the
  compiled artifact.** They are NOT removable via OGA/runtime config (proven). They must be
  eliminated in the EP compiler (option 1 or 2, §4) before any capture path can work.
- Everything downstream (HIP capture/replay + OGA decoder-only wiring via the plugin-EP
  framework) is well-understood and lower-risk once the readbacks are gone.
