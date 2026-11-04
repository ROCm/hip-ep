<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# TurboQuant setup and validation

This is an experimental, explicit-opt-in MSE-only QR codec for the Qwen
24-query-head / 4-KV-head / dimension-256 geometry. Read
[`turboquant-kv-cache.md`](../../docs/design/turboquant-kv-cache.md) and the
[known limitations](../../docs/design/turboquant-kv-cache.md#validation-and-known-limitations) before use.
The strict encoder gate and some full-model comparisons currently fail; this
is not a production-quality accuracy claim.

## Enable TurboQuant on the tested Windows setup

This walkthrough covers Radeon 8060S (`gfx1151`), batch one, Qwen3.8-27B with
24 query heads, four KV heads, head dimension 256, FP16 activations/cache carrier,
and external RoPE. INT4 **weights** alone do not enable compressed KV. The
current environment uses Python 3.13, ORT 1.31.0, NumPy 2.4.3, GenAI 0.17.0
with the append companion, and TheRock 7.15.26333, aligned with Ring PR1366.
For the recorded three-configuration timing and memory campaign, use the
[performance report](../../docs/design/turboquant-current-performance.md) and
[manual performance command](../../test/long_context/README.md#performance-only-campaign-windows).
It includes kernels through 256K, single-turn model runs through 64K, and eight
4K appends, without warm-up. Accuracy and judge findings are separate.

Older performance tables used ORT 1.30 and are not current-main measurements.
Other configurations need separate validation. See the
[Windows build prerequisites](../../docs/quick_start.md) for MSVC, Ninja, SDK and
dependency setup. This walkthrough is not a new Linux qualification.

### 1. Select source, environment and paths

Use the upstream `users/rajeevp/turboquant-kv4` branch. Ring PR #1366 is
already merged into main; this branch adds the packed cache implementation. Work in an isolated directory so existing benchmark
installations remain unchanged. The commands below are **PowerShell** commands
run from an x64 MSVC developer environment with CMake and Ninja on `PATH`.
Choose an already prepared Python environment with the matching ORT/GenAI and
ROCm packages. The Windows test scripts expect `_rocm_sdk_core` and
`_rocm_sdk_devel` under that interpreter's base `Lib/site-packages` directory.

```powershell
$WorkRoot = 'C:/work/hip-ep-turboquant' # Change to your workspace; source is <work>/hip-ep.
$Repo = "$WorkRoot/hip-ep"
$Python = "$WorkRoot/venv/Scripts/python.exe"
$HipBuild = "$WorkRoot/build/turboquant"
$HipInstall = "$WorkRoot/install/turboquant"
$OrtSdk = "$WorkRoot/deps/onnxruntime-sdk/onnxruntime-win-x64-1.31.0"
$BasePython = & $Python -c 'import sys; print(sys.base_prefix)'
$RocmDevel = "$BasePython/Lib/site-packages/_rocm_sdk_devel"
$RocmCore = "$BasePython/Lib/site-packages/_rocm_sdk_core"
Set-Location $Repo
$env:OMP_NUM_THREADS = "2"
$env:OPENBLAS_NUM_THREADS = "2"
& $Python -c 'import numpy, torch, onnxruntime, onnxruntime_genai; print(numpy.__version__, numpy.__file__); print(torch.__version__, onnxruntime.__version__, onnxruntime.__file__, onnxruntime_genai.__version__)'
```

Use the compatible PyTorch already supplied with the ROCm environment; the codec
generator only uses its CPU operations. Do not upgrade a working ROCm/ORT/GenAI
stack merely to run this command. Stop on a failed command before continuing.

Use the established Ring Python dependency paths, including its NumPy/OpenBLAS
build, instead of inheriting a different Conda NumPy or upgrading dependencies.
Verify imported paths as well as versions. Required setup packages include
SciPy, pytest, ONNX and psutil. Do not use `KMP_DUPLICATE_LIB_OK` to bypass an
OpenMP conflict. HIP-EP builds against the ORT 1.31 SDK and uses the 1.31 runtime.
The retained GenAI companion has separate SDK provenance described in step 3.

### 2. Build Ring-KV plus TurboQuant and generate the codec

```powershell
& $Python build.py --hip_arch gfx1151 --config Release --cmake_generator Ninja `
  --build_dir $HipBuild --install_dir $HipInstall --therock_dist $RocmDevel `
  --cmake_extra_defines HIP_EP_ENABLE_RING_ATTENTION=ON `
  --cmake_extra_defines HIP_EP_ENABLE_TURBOQUANT=ON `
  --cmake_extra_defines ENABLE_ROCMLIRTRITON=OFF `
  --cmake_extra_defines "FETCHCONTENT_SOURCE_DIR_ONNXRUNTIME=$OrtSdk" `
  --cmake_extra_defines HIP_EP_GENERATE_TQ_CODEC=ON `
  --cmake_extra_defines "Python3_EXECUTABLE=$Python" `
  --cmake_extra_defines "HIP_EP_TQ_CODEC_DIR=$WorkRoot/reference/codec"
```

`build.py` configures, builds, installs and runs its default checks. Supply any
local dependency prefixes as described in the Windows quick start. The codec
target produces `reference/codec/{codec.bin,constants.npz,manifest.json}` and
installs `codec.bin`/`manifest.json` under
`$HipInstall/share/hip-ep/turboquant`. Generation does **not** enable quantization
by itself; step 4 supplies the runtime options. Both build options default to ON.
Ordinary inference still uses upstream GQA. Set `HIPDNN_EP_USE_RING_ATTENTION=1`
before process startup for dense Ring; packed-cache options explicitly select
Ring prefill plus TurboQuant decode/append.

After updating from the earlier `wrap_group_query_attention_v3` build, regenerate
compiled model/session artifacts with this build. The v4 packed-cache ABI now
includes main's optional host sequence length; old compiled artifacts are not
binary compatible. Codec files and the packed row format are unchanged.

With `HIP_EP_ENABLE_RING_ATTENTION=OFF`, TurboQuant append is unavailable.
In that build, and in architecture slices where Ring is disabled, `hip_tq4_prefill_workspace` returns zero and `hip_tq4_prefill` returns
`hipErrorNotSupported` without accessing GPU memory. These exports must remain
linkable even when `ring_gqa.hip` is omitted. To check such a library (with the
ROCm runtime on the loader path), run:

```powershell
& $Python "$Repo/tools/turboquant/check_unavailable_append.py" --library "$HipInstall/bin/custom_kernels_gfx11-generic.dll" --dll-dir "$RocmCore/bin"
```

Use a library built with Ring unavailable for this check, not the enabled
`gfx1151` library produced above.

For an already configured build, the exact CMake-only generation commands and
hash checks are in [Generate codec assets](#generate-codec-assets-during-build-or-test-setup).

### 3. Build the GenAI companion patch for multi-turn append

Single-turn use can use stock GenAI 0.17, but repeated prompt append on AMDGPU
requires this patch. Start from a separate **v0.17.0** source tree and a matching
ORT 1.30.0 SDK containing headers and libraries. Use an SDK prepared through the
quick start or the matching official ORT release archive; the Python wheel alone
is not the C++ SDK. Set `$GenaiOrtSdk` to its extracted root. This retained
companion is loaded with the Ring ORT 1.31 runtime; only the GenAI core DLL
differs from the original Ring Python package. Its release C API is validated
by the append integration test. Stock v0.17 experimental ModelPackage headers
prevent a straightforward rebuild with the 1.31 SDK, so do not silently treat
these SDKs as interchangeable or claim the companion was rebuilt with 1.31.

```powershell
$GenaiSource = "$WorkRoot/deps/genai-append-src"
$GenaiBuild = "$WorkRoot/build/genai-append"
$GenaiPackageRoot = "$WorkRoot/install/genai_append"
$GenaiOrtSdk = "$WorkRoot/deps/onnxruntime-sdk/onnxruntime-win-x64-1.30.0"
# Clone only if this source directory does not already exist.
git clone --branch v0.17.0 --depth 1 https://github.com/microsoft/onnxruntime-genai.git $GenaiSource
$Patch = "$Repo/tools/turboquant/patches/onnxruntime-genai-0.17-amdgpu-append.patch"
git -C $GenaiSource apply --check $Patch
git -C $GenaiSource apply $Patch
cmake -S $GenaiSource -B $GenaiBuild -G Ninja -DCMAKE_BUILD_TYPE=Release `
  "-DORT_HOME=$GenaiOrtSdk" -DUSE_CUDA=OFF -DUSE_DML=OFF -DENABLE_PYTHON=OFF `
  -DENABLE_TESTS=OFF -DENABLE_MODEL_BENCHMARK=OFF -DENABLE_TELEMETRY=OFF
cmake --build $GenaiBuild --target onnxruntime-genai --parallel 12
```

CMake fetches its pinned dependencies, including ONNX Runtime Extensions.
The tested setup reused the existing **cp313 GenAI 0.17** Python binding and
replaced only the core DLL in a separate package. With that same version/ABI:

```powershell
# Run before pointing PYTHONPATH at a patched package.
$StockPackage = & $Python -c 'import importlib.util,pathlib; print(pathlib.Path(importlib.util.find_spec("onnxruntime_genai").origin).parent)'
New-Item -ItemType Directory -Force $GenaiPackageRoot | Out-Null
Copy-Item -LiteralPath $StockPackage -Destination $GenaiPackageRoot -Recurse
Copy-Item -LiteralPath "$GenaiBuild/onnxruntime-genai.dll" `
  -Destination "$GenaiPackageRoot/onnxruntime_genai/onnxruntime-genai.dll" -Force
$env:PYTHONPATH = $GenaiPackageRoot
& $Python -c 'import onnxruntime_genai as og; print(og.__file__)'
```

Verify that the printed path points into `install/genai_append`. For another
Python/GenAI ABI, build a matching Python package instead of swapping a DLL into
an unrelated binding. The shared strategy patch still needs broader backend
qualification; these instructions do not claim the upstream C++ test suite passed.

### 4. Configure the model and runtime

Use a separate copy of the model configuration and preserve the model's other
keys, graph paths and token metadata. Merge these settings into
`genai_config.json` (replace the example absolute paths with your installed files):

```json
{
  "model": {
    "decoder": {
      "session_options": {
        "session.disable_cpu_ep_fallback": "1",
        "provider_options": [{"AMDGPU": {
          "profile": "hip",
          "config_file": "C:/work/hip-ep-turboquant/install/turboquant/bin/morphizen_config.json",
          "kvCacheQuantizationBits": "4",
          "kv_cache_codec": "tq4_mse_qr_v1",
          "kv_cache_codec_file": "C:/work/hip-ep-turboquant/install/turboquant/share/hip-ep/turboquant/codec.bin"
        }}]
      }
    },
    "embedding": {
      "session_options": {
        "provider_options": [{"AMDGPU": {
          "profile": "hip",
          "config_file": "C:/work/hip-ep-turboquant/install/turboquant/bin/morphizen_config.json"
        }}]
      }
    }
  },
  "search": {"past_present_share_buffer": true}
}
```

All three decoder KV options are required together. Keep the cache carrier FP16
and shared past/present buffers; GenAI allocates width 66 for the packed 256-wide
KV rows. Do not convert model weights, rewrite the production ONNX graph or
manually allocate a second dense history. For the multi-turn test below, the
harness writes this configuration itself, using the generated work-directory
codec and capacity 4608. Ordinary applications must choose a capacity that fits
their history, prompt and requested output within the documented limits.

Set the runtime environment in the same PowerShell session before launching:

```powershell
$env:AMDGPU_EP_PATH = "$HipInstall/bin/hipgpu.dll"
$env:THEROCK_DIST = $RocmDevel
$env:PATH = "$HipInstall/bin;$RocmCore/bin;$RocmDevel/bin;$RocmDevel/lib/llvm/bin;$env:PATH"
$env:HIPDNN_EP_STRICT = '1'
$env:OMP_NUM_THREADS = '2'
$env:OPENBLAS_NUM_THREADS = '2'
$env:TQ_WORK_ROOT = $WorkRoot
$env:TQ_INTEGRATION_INSTALL = $HipInstall
$env:TQ_INTEGRATION_RESULTS = "$WorkRoot/results/integration"
$env:TQ_APPEND_RESULTS = "$WorkRoot/results/append"
$env:TQ_MULTITURN_DENSE_CONTROL = '0'
```

These are process-local settings. Keep the codec fixed for the lifetime of any
packed cache. To return to dense FP16 KV, start a new generator/cache and remove
all three KV options; leaving the codec options with quantization disabled is
rejected. Rewind, beams and batch expansion are outside the qualified scope.

### 5. Check enablement and then run the model

```powershell
& $Python -m pytest tools/turboquant/test_reference.py tools/turboquant/test_generate_codec.py -q
& $Python tools/turboquant/check_genai_windows.py
```

The second command needs no downloaded model: it generates a small ONNX graph
and checks packed cache allocation, repeated append/decode, unchanged old rows,
and invalid-codec rejection. Inspect `results/integration/receipt.json` and
`run.log`; the receipt must report `status: passed`.

For full-model checks, continue with the [manual test guide](../../test/long_context/README.md).

## Generate codec assets during build or test setup

Only generator source is checked in. `generate_codec.py` creates `codec.bin`,
`constants.npz` and `manifest.json` on CPU using the standalone shared QR seed-42
rotation and 16-centroid Gaussian Lloyd-Max recipe (source revision
`a6218a9b99ecabd3028da670fb0f85bb1114e552`). Install NumPy, SciPy and PyTorch
in the selected Python environment; no GPU or model weights are required.

For an existing HIP-EP CMake build, configure with
`-DHIP_EP_GENERATE_TQ_CODEC=ON -DPython3_EXECUTABLE=/path/to/python`, then run:

```console
cmake --build <build-dir> --target turboquant-codec
cmake --install <build-dir> --component turboquant-codec
```

When enabled, the normal build also generates the assets. They default to
`<build-dir>/turboquant`; `HIP_EP_TQ_CODEC_DIR` can select another output directory.
Installation puts the runtime `codec.bin` and its manifest under
`<install-prefix>/share/hip-ep/turboquant`. Set `kv_cache_codec_file` to that
installed file. `constants.npz` stays in the build tree for tests. The option
defaults to OFF, so normal HIP-EP builds gain no Torch/SciPy/NumPy dependency.

For the GPU checks' external work layout, generate directly into its codec
directory (or set `HIP_EP_TQ_CODEC_DIR` to that same absolute path):

```console
python tools/turboquant/generate_codec.py --output-dir <work>/reference/codec
python -m pytest tools/turboquant/test_reference.py tools/turboquant/test_generate_codec.py -q
```

The manifest records the seed, package versions, generator hash and output hashes.
The generator refuses to replace different existing constants; use a new output
directory for a new codec. The current Windows environment reproduces the
historically measured runtime codec SHA-256 exactly:
`a2a312f645ba0a7970bda404fb4f1e1f15a01c2dd18c257b6a4c83397e806a3c`.
Pass `--expected-sha256 <hash>` to enforce an archived codec before writing files.
QR results can differ across platforms/library versions: preserve and compare
codec hashes when reproducing measurements, and never mix an existing packed
cache with another codec. Historical benchmark receipts retain their original
hashes; a new codec needs its own numerical validation.

`write_codec.py <constants.npz> <codec.bin>` remains available to serialize
already-exported constants without regenerating them. The older standalone
`fixtures.npz` is a local diagnostic artifact; the current tests generate their
input tensors and do not need it.

## Tests and measurements

Use the same installed build for all three paths. The commands below serve
separate purposes; the long model campaign runs only when explicitly invoked.

| Command / file | Purpose |
| --- | --- |
| `python -m pytest tools/turboquant/test_reference.py tools/turboquant/test_generate_codec.py -q` | CPU codec mathematics, packing and generated-asset provenance. |
| `check_hip.py` | Strict GPU encoder bytes, decode oracle, bounds and allocation guards. The known centroid-boundary failure remains a failure. |
| `check_hip_extended.py` | Guarded decode checks through 64K and paired Ring comparisons. Add `--long` for 128K, 128K+128 and 256K. |
| `check_append.py` | Mixed packed-history/FP16-tail attention against FP64, chunk boundaries and unchanged old cache rows. |
| `check_genai_windows.py` | Small generated-model integration, repeated append/decode and invalid-codec rejection. |
| `check_unavailable_append.py` | Export/admission checks for a build without the Ring append implementation; see step 2. |
| [Manual model tests](../../test/long_context/README.md) | One runner for matched-history and free-running conversations, optional BF16 reference and blinded judge packets. |
| [Performance campaign](../../test/long_context/README.md#performance-only-campaign-windows) | One runner for kernel, single-turn model and eight-turn timing/memory tables. |

For direct HIP checks set `TQ_WORK_ROOT` to the work directory,
`TQ_TEST_LIBRARY` and `TQ_RING_LIBRARY` to the installed custom-kernel library,
and `TQ_HIP_RUNTIME` to the HIP runtime. Initialize the SDK DLL search directories
on Windows. `hip_test_utils.py` supplies shared ctypes bindings and guarded
allocations; `reference.py` supplies the independent CPU oracle.

`check_append.py` compares new packed bytes with the frozen encoder selected by
`TQ_ENCODER_BASELINE_LIBRARY`. Its default is `<work>/install/turboquant`.
Select an independently preserved library explicitly when validating a new
encoder. `TQ_APPEND_RESULTS` selects the result directory. Passing that regression
does not imply strict CPU/GPU encoder byte parity.

For the full model use `amd/Qwen3.8-27B_gpuep_1.0_gpu_128`, revision
`9b37cf56a001720ba7ce98ff80ea651b02e5931b`, in `<work>/models/qwen-original`.
Verify the downloaded model files and retain their hashes; runners hard-link
weights into fresh case directories on the same volume. Keep generated models,
codec files, logits, answers and receipts outside Git.

### Troubleshooting

| Symptom | Check |
| --- | --- |
| AMDGPU continuous decoding rejected | Imported GenAI package/DLL and the companion patch in step 3. |
| Missing/invalid codec | All three decoder options, absolute codec path and manifest/hash. |
| Unsupported geometry/device | gfx1151, Ring build, batch one, H24/G4/D256, FP16 and shared buffers. |
| Wrong DLL or missing dependency | `AMDGPU_EP_PATH`, SDK DLL search directories and receipt library hashes. |
| Strict encoder failure at 4K | Preserve the recorded discrepancy; do not weaken the assertion. |

## External dependencies

HIP-EP needs no ONNX Runtime core patch, ONNX schema change, cache allocator
patch, weight conversion or production graph rewrite for this feature.
The GenAI append companion is the only external source change:

| GenAI source | Purpose |
| --- | --- |
| `src/generator/generators.cpp` | Admit AMDGPU continuous decoding on stock v0.17.0. |
| `src/decoding/standard_decoding_strategy.h` and `.cpp` | Flush a selected token whose KV has not yet been computed before appending a prompt. |
| `test/cpp/engine/request_lifecycle_tests.cpp` | Assert AMDGPU continuation capability. |

For a GenAI source tree that already includes AMDGPU continuous decoding (such
as the CI stack using #2691), apply only
[`onnxruntime-genai-0.17-pending-append.patch`](patches/onnxruntime-genai-0.17-pending-append.patch).
Use `git apply --check` against the actual dependency revision. Do not apply
both patches. The shared decoding-strategy change needs broader backend review;
local integration checks do not replace GenAI's upstream C++ suite.

The retained Windows companion reuses matching cp313 GenAI 0.17 bindings and
replaces only its core DLL. That companion was built with the ORT 1.30 SDK and
validated with the Ring ORT 1.31 runtime. HIP-EP itself uses the 1.31 SDK.
Record these separately rather than implying SDK equivalence. Per-application
`genai_config.json`, DLL search paths, downloaded SDKs, environments and generated
codec/model artifacts are setup inputs, not additional source patches.
