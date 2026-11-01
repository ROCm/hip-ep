<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# TurboQuant external patches and local files

This inventory separates required external source changes from local build and
measurement artifacts. Paths below use `WORK` for the isolated
`hip-ep-turboquant` directory and `LOCAL` for `_local/turboquant` in the original
Windows `hip-ep` checkout. The latter is local tooling outside the submitted
source change, even though its directory is inside the original checkout.

## Required patches and changes outside HIP-EP

Ring PR #1366 is already merged and incorporated into this TurboQuant branch.
Multi-turn use through **ONNX Runtime GenAI v0.17.0** additionally requires the
[companion patch](../../tools/turboquant/patches/onnxruntime-genai-0.17-amdgpu-append.patch).
The full patch below targets stock v0.17.0. Current HIP-EP CI already applies GenAI #2691, which supplies AMDGPU continuous decoding; that stack needs only [the two-file pending-token patch](../../tools/turboquant/patches/onnxruntime-genai-0.17-pending-append.patch). Do not apply the full patch twice. The full patch has four target files in the separate GenAI repository:

| GenAI file | Change |
| --- | --- |
| `src/generator/generators.cpp` | Admit AMDGPU in `SupportsContinuousDecoding`, allowing a second prompt on the same generator. |
| `src/decoding/standard_decoding_strategy.h` | Declare the `PrepareForAppend` override. |
| `src/decoding/standard_decoding_strategy.cpp` | Evaluate a selected but not yet cached token before appending the next prompt; skip this when logits/KV were already computed. |
| `test/cpp/engine/request_lifecycle_tests.cpp` | Add the AMDGPU continuation-capability assertion. |

The `PrepareForAppend` change belongs to the shared standard decoding strategy,
so its scope is broader than AMDGPU. It needs upstream GenAI review and wider
backend testing; the upstream C++ suite was not run. The patched Windows build
passed our small GenAI lifecycle tests and Qwen baseline/Ring/TurboQuant append
tests. Rewind behavior is unchanged.

**External runtime/setup files:**

- Rebuilt `onnxruntime-genai.dll`, deployed in an isolated
  `install/genai_append/onnxruntime_genai/` package selected with `PYTHONPATH`.
  A file comparison confirmed this DLL is the only package difference from
  stock 0.17 (excluding Python bytecode caches); Python bindings were reused.
- Per-run `genai_config.json`: HIP-EP provider/config paths, shared cache and
  context capacity; TurboQuant additionally sets `kvCacheQuantizationBits=4`,
  `kv_cache_codec=tq4_mse_qr_v1` and `kv_cache_codec_file`.
- `reference/codec/codec.bin` is the required frozen codec payload.
  The build/setup generator creates `codec.bin`, `constants.npz` and `manifest.json`;
  no BIN/NPZ assets are tracked. Older `fixtures.npz` is a local diagnostic artifact.
- Separate HIP-EP and GenAI build/install trees, a Python venv, downloaded
  dependencies, model hard links, build/profiling helpers and generated test
  receipts/logs are local setup artifacts. They are not extra upstream patches.

No ONNX Runtime core patch, ONNX schema change, GenAI cache-allocation/layout
patch, model-weight change or production ONNX graph rewrite is required by this
implementation. The original model files and stock GenAI package were preserved;
small ONNX graphs were generated only for integration tests.

[External-file audit and full helper inventory](#historical-local-file-locations)
lists exact source files and local artifact locations. Comparing the isolated
GenAI source tree with its v0.17.0 archive found exactly the four changes above,
with no added or missing source files. This audits the current Windows setup;
it does not claim a fresh machine-wide or lx01 audit.

## Current PR1366-aligned local deployment

The current upstream branch is `WORK/hip-ep-upstream`; its build tree is
`WORK/build/upstream` and deployed HIP-EP is `WORK/install/turboquant_upstream131`.
It uses the existing Ring interpreter and ORT 1.31 SDK/runtime. The isolated
GenAI package is `WORK/install/genai_append131`; only its core DLL differs from
Ring's package. The retained companion DLL was built with the 1.30 SDK and is
validated against the 1.31 runtime. No ORT core or new GenAI source compatibility
patch was added for this alignment.

`WORK/torch-reference-venv` adds Transformers 5.22.1 for state-aware BF16 append
and reuses Ring's NumPy 2.4.3 dependency paths. It does not patch PyTorch or the
model implementation. BF16 weights are in the user's Hugging Face cache.
The current model remains `WORK/models/qwen-original`; generated codec files
remain under `WORK/reference/codec`. Current receipts, logs, private model hard
links and raw logits are under `WORK/results/upstream-pr1366`. Judge requests
and raw responses are private `LOCAL/judge-requests` / `LOCAL/judge-responses`
artifacts. Experimental results and reports remain local pending author review
and a separate publication decision.

The detailed inventory below records the earlier profiling deployment; its
frozen installations are preserved and are not the new current build.

## Historical local file locations

| Location | Files or purpose | Required to run TurboQuant? |
| --- | --- | --- |
| `WORK/deps/genai-append-src/` | The four patched GenAI source/test files listed above; remaining release files match the archive. | Build input for GenAI append support. |
| `WORK/build/genai-append/` | CMake/Ninja outputs and rebuilt `onnxruntime-genai.dll`. | Build artifact; deploy the DLL. |
| `WORK/install/genai_append/onnxruntime_genai/onnxruntime-genai.dll` | The only changed file in the isolated copied Python package. | Yes, for this GenAI 0.17 multi-turn setup. |
| `WORK/install/{existing,ring_q_resident,turboquant,turboquant_append}/` | Separate HIP-EP compiler/runtime/kernel installations and provider configuration; frozen comparison installs preserved during append work. | Use the desired compatible installation; comparison copies are optional. |
| `WORK/results/append/{qwen,qwen-ring,qwen-baseline}/model/genai_config.json` | Generated per-variant settings, capacity 4608, shared cache, provider paths and optional TQ codec options. | Equivalent application configuration is required. |
| `WORK/results/append/integration/` and earlier per-run result trees | Generated small test ONNX graphs/configurations, receipts, logits arrays and logs. | Tests only. |
| `WORK/reference/codec/codec.bin` | Frozen rotation and centroids with versioned header/checksum. | Yes. |
| `WORK/reference/codec/{constants.npz,fixtures.npz,manifest.json}` | Reference arrays, fixtures and codec provenance. | Validation/export only. |
| `WORK/reference/standalone/` | Snapshot of the original standalone TurboQuant implementation, including source provenance; imported reference, not another deployment patch. | Reference/export only. |
| `WORK/models/qwen-original/` | Downloaded model and original configuration. Per-run directories hard-link immutable model files and write separate configurations. | Model input; weights/production graph unchanged. |
| `WORK/venv/pyvenv.cfg`, `WORK/venv/Lib/site-packages/`, `WORK/venv/Scripts/` | Isolated Python environment, with existing system ROCm/Torch packages made visible using `include-system-site-packages=true`. Stock GenAI remains here. | Reproducible environment, not a source patch. |
| `WORK/deps/`, `WORK/downloads/` | Downloaded SDKs and build dependencies; GenAI extensions source pinned to `0353c93d966d74706724e41ddab63f580eebbbf0`. | Build setup; not additional feature patches. |
| `WORK/build/`, `WORK/results/` | Compiler objects, temporary response files, memory-counter DLL, model compilation outputs, diagnostic data and benchmark receipts. | Generated artifacts, grouped rather than listing every object/log. |
| `WORK/scripts/` | Exact helper filenames listed below. | Build/profiling/reproduction helpers, not upstream runtime patches. |
| `LOCAL/`, including `pr-ready/` | Local implementation, audit, build and publication helpers; staged reports, PR text and publication metadata. Exact top-level inventory is in the linked JSON. | Development/publication only. |

The build uses process-local `PATH`, `AMDGPU_EP_PATH`, `THEROCK_DIST` and
`PYTHONPATH` selections. Windows long-path helpers create a temporary build-drive
alias and temporary response files. These are local build accommodations.

## Reproduce the external GenAI change

Apply `tools/turboquant/patches/onnxruntime-genai-0.17-amdgpu-append.patch` from this
PR to a matching GenAI v0.17.0 source tree, checking with `git apply --check`
first. Build against the matching ORT SDK (1.30.0 in the measured environment).
The tested Windows build used Release/Ninja, `USE_CUDA=OFF`, `USE_DML=OFF`,
`ENABLE_PYTHON=OFF`, `ENABLE_TESTS=OFF`, `ENABLE_MODEL_BENCHMARK=OFF`, and
`ENABLE_TELEMETRY=OFF`. Its existing cp313 GenAI 0.17 C-API binding was reused in
an isolated package with the rebuilt core DLL. This binding reuse is the tested
local configuration, not a claim of arbitrary cross-version binary compatibility.

Select that package with `PYTHONPATH`, select the compatible HIP-EP installation,
and run the integration and multi-turn checks described in
[the test README](../../tools/turboquant/README.md).

## Exact helper-file inventory

[Machine-readable audit](turboquant-external-files.json) records all top-level
files in `WORK/scripts` and `LOCAL` at audit time, plus GenAI source/package
comparisons and hashes. It includes historical development helpers, not just
the commands required to reproduce the final build. Generated dependency,
environment, binary and model trees are grouped above. This is not a record of
every file ever touched on either machine.

Files in `WORK/scripts/`:

- `scripts/build-comparison.cmd`
- `scripts/build_comparison.py`
- `scripts/check_long_windows.py`
- `scripts/check_profile_controls.py`
- `scripts/compile_memory.cmd`
- `scripts/complete_comparison.py`
- `scripts/consolidate_comparison.py`
- `scripts/diagnose_windows_encoder.py`
- `scripts/export_constants.py`
- `scripts/extend_windows_profile.py`
- `scripts/fetch_windows_llvm.py`
- `scripts/fetch_windows_ort.py`
- `scripts/fetch_windows_small_deps.py`
- `scripts/finalize_pr_report.py`
- `scripts/finalize_windows_profile.py`
- `scripts/finish_windows_profile.py`
- `scripts/hydrate_windows_dependency.py`
- `scripts/hydrate_windows_llvm.py`
- `scripts/integration_probe_windows.py`
- `scripts/record_windows_setup.py`
- `scripts/short_msvc_output.py`
- `scripts/summarize_windows_profile.py`
- `scripts/validate-pr-package.cmd`
- `scripts/validate_pr_package.py`
- `scripts/validate_ring_windows.py`
- `scripts/validate_windows_memory.py`
- `scripts/validate_windows_memory_v2.py`
- `scripts/windows-dev.cmd`
- `scripts/windows_build_paths.cmake`
- `scripts/windows_build_short_path.py`
- `scripts/windows_dev.py`
- `scripts/windows_gpu_memory.py`
- `scripts/windows_memory.py`
- `scripts/windows_memory_counter.cpp`
- `scripts/windows_memory_observer.py`
- `scripts/windows_profile.py`
- `scripts/windows_profile_campaign.py`
- `scripts/windows_profile_run.py`
- `scripts/windows_qwen_4k.py`
- `scripts/windows_synthetic.py`

## Generated codec build outputs

See [generation instructions](../../tools/turboquant/README.md#generate-codec-assets-during-build-or-test-setup). The opt-in `HIP_EP_GENERATE_TQ_CODEC` CMake option generates the codec assets and installs the runtime codec plus manifest. The table above records local artifact locations, not additional checked-in binaries.
