<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Running the ORCA 2-bit model (build → run → benchmark)

End-to-end recipe to build the EP and run the ORCA 2-bit (W2A8) decode benchmark
on an AMD APU (validated on **Strix Point / gfx1150**; the same steps apply to
**Strix Halo / gfx1151** — the GPU arch is auto-detected). Windows host.

> **TL;DR of the non-obvious bits:** (1) init the MorphiZen submodule over HTTPS;
> (2) if the default in-process JIT crashes at load, switch the artifact format to
> **NATIVE**; (3) put the *matching* `onnxruntime.dll` next to the tools so a stale
> system copy isn't picked up; (4) for max/stable perf on an APU, use Ultimate
> Performance but **let the CPU idle** (don't force it busy).

---

## 0. Prerequisites

- **Visual Studio 2022** Build Tools (MSVC). A "Desktop development with C++"
  workload is enough. See the ATL note in step 2 if the LLVM build fails.
- **conda** with the project env:
  ```bash
  conda env create -f environment.yml   # one-time
  conda activate hipdnn-ep
  ```
  This provides `cmake`, `ninja`, `sccache`, `lit`, and a Python with
  `onnxruntime` (must match the ORT version pinned in `cmake/deps.txt`).
- The **ORCA 2-bit model files** in one directory (`<MODEL_DIR>`):
  - `orca_2bit_embeddings_w4a32.quant.onnx`
  - `orca_2bit_1.onnx`      (decode, seq=1)
  - `orca_2bit_128.onnx`    (prefill, seq=128)
  - `orca_2bit_lm_head_fp16.onnx`
  - `orca_2bit_gqa_fp16.data`  (external weight data)

---

## 1. Clone + submodules

The `3rd-party/morphizen` submodule is declared with an SSH URL. If you clone
over HTTPS, add a one-time rewrite so the submodule resolves over HTTPS too:

```bash
git config --global url."https://github.com/".insteadOf "git@github.com:"
git submodule update --init --recursive
```

---

## 2. Build

`build.py` auto-detects the GPU arch (gfx1150/gfx1151), downloads the TheRock
ROCm SDK + LLVM, and builds/install into `../install/`.

From an **"x64 Native Tools Command Prompt for VS 2022"** (so `cl.exe`/`link.exe`
are on PATH) with the conda env active:

```bash
python build.py --cmake_generator Ninja --skip_tests
```

Artifacts land in `<workspace>/install/bin/` (EP = `hipgpu.dll`,
`hip-compiler.dll`, `custom_kernels_<arch>.dll`, tools) and the TheRock SDK in
`<workspace>/build/<repo>/_therock/`.

> **ATL / DIA build fix.** If the from-source LLVM build fails with
> `fatal error C1083: Cannot open include file: 'atlbase.h'`, your VS install
> lacks the ATL component (used only by LLVM's DIA/PDB support, which the EP does
> not need). Either install the "C++ ATL" VS component, **or** disable it in
> [`cmake/deps.cmake`](../../cmake/deps.cmake) — in the from-source LLVM block
> add:
> ```cmake
> set(LLVM_ENABLE_DIA_SDK OFF CACHE BOOL "" FORCE)
> ```

---

## 3. Runtime setup

### 3a. Use the matching onnxruntime.dll

The EP requests the ORT C-API version it was built against (`cmake/deps.txt`,
e.g. 1.25.x). A stale `onnxruntime.dll` elsewhere on the DLL search path (e.g. a
Windows-ML copy in `System32`) can be loaded instead and crash the EP with
`The requested API version [N] is not available`. Copy the built ORT DLLs next
to the tools so the adjacent (correct) copy wins:

```bash
cp build/<repo>/_deps/onnxruntime-src/lib/onnxruntime.dll                 install/bin/
cp build/<repo>/_deps/onnxruntime-src/lib/onnxruntime_providers_shared.dll install/bin/
```

### 3b. Environment for running

```bash
export THEROCK_DIST="<workspace>/build/<repo>/_therock"
export PATH="<workspace>/install/bin:$THEROCK_DIST/bin:$PATH"
```

- **Default (LLVM_IR / in-process JIT):** the above is sufficient.
- **NATIVE artifact (see step 4):** the per-model `lld-link` step also needs the
  MSVC `LIB` env — run from a VS "x64 Native Tools" prompt (or `call vcvars64.bat`).

---

## 4. Artifact format — LLVM_IR (default) vs NATIVE

The EP defaults to **LLVM_IR** (OS-portable bitcode, JIT-loaded in-process). Try
this first.

If session creation **crashes right after** `Emitted LLVM bitcode to ...`
(access violation `0xC0000005`), the in-process JIT is failing on this
box/driver. Switch to the **NATIVE** artifact format (compile each model to a
`.dll` linked with `lld-link`, loaded via `LoadLibrary`). The provider option
does not currently thread through the plugin-EP path, so set the default in
[`backend-mlir-compiler/level-1-pass/src/pass_main.cpp`](../../backend-mlir-compiler/level-1-pass/src/pass_main.cpp)
(`load_config`):

```cpp
std::string artifact_format_str =
    ctx->get_provider_option("artifact_format", "NATIVE");   // was "LLVM_IR"
```

Rebuild (step 2, incremental — LLVM stays cached). NATIVE requires the MSVC `LIB`
env at run time (step 3b).

---

## 5. Max / stable performance (APU power + thermal)

Decode is memory-bandwidth-bound, so clocks matter little, but these give the
best and most repeatable numbers:

```bash
# Windows: Ultimate Performance plan + full CPU clock + no PCIe link power-down
powercfg /setactive a17f6449-3b46-49aa-953f-efba0de7891a   # Ultimate Performance
powercfg /setacvalueindex <SCHEME> SUB_PROCESSOR PROCTHROTTLEMIN 100
powercfg /setacvalueindex <SCHEME> SUB_PCIEXPRESS ASPM 0
powercfg /setactive <SCHEME>
```

> **APU thermal tip (measured):** do **NOT** disable CPU idle
> (`SUB_PROCESSOR IDLEDISABLE 1`). CPU and GPU share one power/thermal budget on
> an APU; keeping the CPU busy while it merely waits on the GPU steals the GPU's
> budget and *lowers* throughput (~7% in testing). Let the CPU idle.

Also warm up before timing: the M=1 GEMV autotune warms over the first ~2 tokens
(per process), so discard the first couple of steps and run the decode loop
back-to-back (no sleeps) so the GPU stays boosted.

---

## 6. Run the benchmark

Point the timing script at your model dir and the EP DLL. Using the in-tree
`test/` runner or your own script, the essential setup is:

```python
import onnxruntime as ort
ep_dll   = r"<workspace>\install\bin\hipgpu.dll"
model_dir = r"<MODEL_DIR>"
ort.register_execution_provider_library("MorphiZenExecutionProvider", ep_dll)
from onnxruntime.capi._pybind_state import get_ep_devices
devices = [d for d in get_ep_devices() if d.ep_name == "MorphiZenExecutionProvider"]
so = ort.SessionOptions()
so.add_session_config_entry("session.disable_aot_function_inlining", "1")
so.add_provider_for_devices(devices, {})
sess = ort.InferenceSession(model_path, sess_options=so)
```

Load the 4 sub-models (embeddings → decode body → lm_head), feed the KV cache
across steps, and time the decode loop (drop the first ~2 steps for autotune
warmup). Report ms/step and tok/s.

**Verify GPU dispatch** (not a silent CPU fallback): run once with
`HIPDNN_EP_DEBUG=1` and confirm `[REAL] wrap_*` lines appear.

**Operator profiling:** run with `HIPDNN_EP_PERF=1` for a per-op GPU-time
breakdown. Note this synchronizes after every op, so it inflates absolute times
and lowers throughput while profiling — use the **% breakdown**, and cite the
throughput from the `HIPDNN_EP_PERF=0` run.

---

## 7. Expected results & interpretation

- Decode is **memory-bandwidth-bound** on `matmul_nbits` (the weight GEMVs) —
  ~68% of GPU time; weights are read once per token, so throughput ≈
  `achievable_BW / bytes_per_token` (~4.4 GB/token for this model).
- Measure achievable BW with a copy/read microbenchmark; the roofline ceiling is
  `achievable_BW / 4.4 GB`. Reference: Strix Point (~88 GB/s) → ~19–20 tok/s
  ceiling; Strix Halo (~212 GB/s) → proportionally higher.
- Known optimization gap: the wide-output/short-reduction shapes
  (**gate/up_proj**, **lm_head**) underutilize BW; a split-K GEMV kernel is the
  main remaining lever.
