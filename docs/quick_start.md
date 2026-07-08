<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Quick Start Guide

This guide builds onnx-hipdnn-ep on Windows. The C++ dependencies
(LLVM/MLIR/LLD, FlatBuffers, Protobuf, ONNX Runtime) and the TheRock ROCm SDK
are resolved automatically by CMake (see step 2): they are reused from
a prefix when one is available, and otherwise built or downloaded from source.
ONNX Runtime is built explicitly only if you need to modify it or build OGA
(step 1 is otherwise optional).

## Prerequisites

| Tool | Purpose |
|------|---------|
| **CMake** 3.20+ | Build system |
| **Ninja** | Build generator (faster than MSVC, required) |
| **MSVC 2022** | C++ compiler (Visual Studio Build Tools or full IDE) |
| **Python 3** | LLVM/MLIR tools runtime |
| **sccache** | Compiler cache (significantly speeds up rebuilds) |
| **`gh` CLI** | Authenticating access to the private ROCm repositories (`gh auth login`) |
| **Git** | Clone the repo; from-source deps are fetched via git |

**<span style="color:red">IMPORTANT</span> -- MSVC Environment Setup:**

You **must** launch Git Bash from inside a "**x64** Native Tools Command Prompt for VS
XXXX" (where XXXX is your VS version: 2019, 2022, 2026, etc.).
This is required so that `cl.exe`, `link.exe`, and the MSVC headers/libraries
are visible to the build system. All commands in this guide assume this
environment.

```bash
# Verify MSVC environment is inherited
echo $INCLUDE  # Should contain paths under "Microsoft Visual Studio"
```

**Install prerequisites (PowerShell or cmd):**

```powershell
# 1. Visual Studio 2022 Build Tools — provides MSVC (cl.exe / link.exe), the
#    Windows SDK, and CMake (bundled in the "Desktop development with C++"
#    workload). Skip this step if you already have a full VS 2022 IDE.
winget install Microsoft.VisualStudio.2022.BuildTools --override "--quiet --wait --norestart --add Microsoft.VisualStudio.Workload.NativeDesktop --add Microsoft.VisualStudio.Component.VC.CMake.Project --includeRecommended"

# 2. Ninja, Python 3, sccache, GitHub CLI, Git (provides Git Bash)
winget install Ninja-build.Ninja
winget install Python.Python.3.14
winget install Mozilla.sccache
winget install GitHub.cli
winget install Git.Git

# 3. Authenticate gh for access to the private ROCm repositories.
gh auth login
```

> **Note**: `winget` is included in Windows 10 1809+ and Windows 11. If it is not
> available, install [App Installer from the Microsoft Store](https://apps.microsoft.com/detail/9NBLGGH4NNS1).

**Verify the install** (run inside Git Bash launched from "x64 Native Tools Command
Prompt for VS 2022"; see "MSVC Environment Setup" above):

```bash
cl.exe 2>&1 | head -1     # Microsoft (R) C/C++ Optimizing Compiler ...
cmake --version           # cmake version 3.20+
ninja --version
python --version
sccache --version
gh --version
```

## Directory Layout

All paths are relative to the directory that contains the project source
(`<workspace>/`). Clone the repository so that the source, build, and
local directories are siblings:

```
<workspace>/
├── <repo>/             # project source — this is $PWD for all commands below
├── build/
│   └── <repo>/         # cmake build output  ../build/$(basename $PWD)
└── local/              # install/staging prefix  ../local
    ├── bin/
    ├── include/
    └── lib/cmake/onnxruntime/
```

## One-Time Setup

### 0. Clone the repository

Pick a workspace directory and clone the project so all later commands can
use `..` to reference sibling directories:

```bash
cd <workspace>
git clone https://github.com/ROCm/onnx-hipdnn-ep.git
cd onnx-hipdnn-ep
```

All subsequent commands run from the `onnx-hipdnn-ep/` project root unless
explicitly noted.

### 1. Build ONNX Runtime (optional)

This step is **optional**: build ONNX Runtime from source only if you need to
modify ORT itself. Otherwise skip it -- the EP's CMake auto-downloads the pinned
ONNX Runtime release (step 2) and uses that. When built from source, ORT is
built standalone -- not as a subdirectory of this project -- so its bundled
FlatBuffers does not conflict with ours.

> **Note**: the OGA build (step 3) needs a local ORT for its `ORT_HOME`. If you
> plan to run OGA, build ORT here (or stage the release zip into `ORT_HOME`).

**Clone ONNX Runtime** (if not already cloned):

```bash
cd ..  # Go to workspace directory (parent of project root)
# Recommend a release branch such as rel-1.25.1
git clone -b rel-1.25.1 https://github.com/Microsoft/onnxruntime.git
cd onnxruntime
```

**Build and install ONNX Runtime**:

```bash
# Build ONNX Runtime using build.bat (do NOT set CMAKE_INSTALL_PREFIX during build)
# This ensures ONNX Runtime uses its own FlatBuffers version, avoiding conflicts with ours.
# --build_wheel additionally produces a Python wheel under
# ../build/onnxruntime/Release/dist/, used only by the optional Python wheels
# step (step 4); it is not required to build OGA itself.
./build.bat --config Release --build_shared_lib --parallel --compile_no_warning_as_error --skip_submodule_sync --build_dir ../build/onnxruntime --skip_tests --disable_memleak_checker --use_dml --build_wheel

# Install to local (set prefix at install time, not during configuration)
mkdir -p ../local
LOCAL_DIR=$(cd ../local && pwd)
cmake --install ../build/onnxruntime/Release --prefix "$LOCAL_DIR"
```

**Verify installation**:

```bash
ls $LOCAL_DIR/lib/cmake/onnxruntime/
# Should see onnxruntime-config.cmake and related files

ls ../build/onnxruntime/Release/dist/onnxruntime_directml-*.whl
# Should see one wheel matching your Python version, e.g.
# onnxruntime_directml-1.25.1-cp314-cp314-win_amd64.whl
```

### 2. Build onnx-hipdnn-ep

`build.py` is the cross-platform build driver (the same one used on Linux
and in CI): it sets up the build, resolves every dependency
via `cmake/deps.cmake` (TheRock SDK + GPU arch auto-detected for real builds),
and runs the cmake configure/build/install plus the LIT tests. A fresh tree needs
no manual dependency setup; the cold from-source LLVM build is the long pole
(multi-hour).

On Windows the default generator is "Visual Studio 17 2022" (it locates MSVC on
its own, so no special prompt is needed). To use Ninja instead, run from an
"x64 Native Tools Command Prompt for VS" and pass `--cmake_generator Ninja`.

Run from the project root:

```bash
mkdir -p ../local
LOCAL_DIR=$(cd ../local && pwd)
# For new users we recommend the HIP shipped by TheRock; unset a pre-installed
# HIP so it does not interfere (skip if you intend to use a pre-installed HIP).
unset HIP_PATH

python build.py --install_dir "$LOCAL_DIR" --cmake_prefix_path "$LOCAL_DIR"
#   --mock                    mock runtime (no GPU/HIP/TheRock)
#   --hip_arch gfx1151        target GPU; auto-detected by default. Set it for a
#                             cross-machine build+run -- the *target* GPU's arch
#                             (e.g. from `offload-arch.exe` on that machine)
#   --therock_dist <path>     reuse a local TheRock SDK (else auto-downloaded)
#   --cmake_generator Ninja   use Ninja (run from an x64 Native Tools prompt)
#   --config RelWithDebInfo   build type (default Release)
#   --skip_tests              skip the LIT tests (run by default after install)
#   --clean                   remove build/ and install/
```

`--cmake_prefix_path "$LOCAL_DIR"` lets the build reuse an ONNX Runtime you
installed in step 1; omit it and ORT is auto-downloaded. The hipgpu EP, HIP
tools and LIT tests are built and installed into `../local/`.

### 3. Build OGA (onnxruntime-genai)

OGA (ONNX Runtime GenAI) provides the generative-pipeline runtime used by
`model_benchmark.exe` (and the Python `onnxruntime_genai` module) to drive
prefill + decode token generation. Build it after step 1 (ORT) so it links
against the same ORT version.

**Prerequisites**: a local ORT install for `ORT_HOME` (build it in step 1, or
stage the release zip). `--build_wheel` is only needed if you also want the ORT
Python wheel (step 4); it is not required to build OGA's C++ artifacts. Step 2
is only needed later to *run* `model_benchmark` against hipgpu EP, not to
build OGA itself.

Run all commands below from the `onnx-hipdnn-ep/` project root.

#### 3a. Prepare ORT_HOME

OGA's CMake expects an `ORT_HOME/` directory layout with `include/` and `lib/`
subdirectories. Stage one from the ORT install under `$LOCAL_DIR`:

```bash
mkdir -p ../build/oga-ort-home/include ../build/oga-ort-home/lib
ORT_HOME=$(cd ../build/oga-ort-home && pwd)
cp -r "$LOCAL_DIR/include/onnxruntime/"* "$ORT_HOME/include/"
cp "$LOCAL_DIR/lib/"onnxruntime*.lib "$ORT_HOME/lib/"
cp "$LOCAL_DIR/bin/onnxruntime.dll" "$ORT_HOME/lib/"
cp "$LOCAL_DIR/bin/onnxruntime_providers_shared.dll" "$ORT_HOME/lib/"
```

#### 3b. Clone OGA

```bash
cd ..  # Go to workspace directory (sibling of onnx-hipdnn-ep)
git clone -b v0.14.0 https://github.com/microsoft/onnxruntime-genai.git
cd onnxruntime-genai
git submodule update --init --recursive

# Apply the AMDGPU integration PR on top of the upstream tag. pull/<n>.patch is
# a format-patch series (it renames src/morphizen_ep -> src/amdgpu mid-series),
# so apply it with `git am` -- `git apply` flattens the series and fails on the
# rename whose pre-image only exists after an earlier commit.
curl -fsSL https://github.com/microsoft/onnxruntime-genai/pull/2194.patch -o /tmp/oga-2194.patch
git am --3way --whitespace=nowarn /tmp/oga-2194.patch
```

> **Note**: the upstream tag + PR list are pinned in CI via `OGA_VERSION` and
> `OGA_PR_PATCHES` in
> [`.github/workflows/windows-build.yml`](../.github/workflows/windows-build.yml);
> match those for byte-for-byte reproducibility.

#### 3c. Build OGA

> **Note**: when executing below python command,
> you may meet error of "ModuleNotFoundError: No module named ..."
> in this case, you need manually install the missing module such as:
```bash
pip install util requests
```

```bash
python build.py \
  --config Release \
  --cmake_generator Ninja \
  --use_dml \
  --ort_home "$ORT_HOME" \
  --skip_tests --skip_examples \
  --parallel \
  --build_dir ../build/onnxruntime-genai \
  --cmake_extra_defines \
    CMAKE_C_COMPILER_LAUNCHER=sccache \
    CMAKE_CXX_COMPILER_LAUNCHER=sccache
```

This produces three artifacts under `../build/onnxruntime-genai/Release/`:

| Artifact | Path | Purpose |
|----------|------|---------|
| `model_benchmark.exe` | `benchmark/c/model_benchmark.exe` | End-to-end LLM benchmark (used in [OGA End-to-End Benchmarking](#oga-end-to-end-benchmarking-with-model_benchmark)) |
| `onnxruntime-genai.dll` | `onnxruntime-genai.dll` | Runtime DLL (loaded by `model_benchmark.exe` and the Python module) |
| `onnxruntime_genai_directml-*.whl` | `wheel/onnxruntime_genai_directml-*.whl` | Python wheel (see [Install Python Wheels](#4-install-python-wheels-optional)) |

#### 3d. Install OGA artifacts into local

```bash
cp ../build/onnxruntime-genai/Release/benchmark/c/model_benchmark.exe "$LOCAL_DIR/bin/"
cp ../build/onnxruntime-genai/Release/onnxruntime-genai.dll "$LOCAL_DIR/bin/"
```

After this, `$LOCAL_DIR/bin/` has the OGA artifacts (`model_benchmark.exe`
and `onnxruntime-genai.dll`) used by [OGA End-to-End Benchmarking](#oga-end-to-end-benchmarking-with-model_benchmark)
below. `onnxruntime_perf_test.exe` is staged separately in
[Latency Benchmarking](#latency-benchmarking-with-onnxruntime_perf_test).

### 4. Install Python Wheels (Optional)

If you also want to drive ORT / OGA from Python (e.g. to write your own
generation script instead of using `model_benchmark.exe`), install the wheels.
The hipgpu EP wheel is built by default by `build.py` (pass `--skip_wheel` to
opt out, or build just it with `cmake --build <build> --target wheel`) at
`../build/onnx-hipdnn-ep/python/dist/`.

**Install** -- give ORT / EP / OGA as local `.whl` files (so pip uses your custom
builds, not the stock PyPI ones), and let `--extra-index-url` resolve ROCm:

```bash
cd ../onnx-hipdnn-ep  # Back to the project root
pip install \
  ../build/onnxruntime/Release/dist/onnxruntime_directml-*.whl \
  ../build/onnx-hipdnn-ep/python/dist/onnxruntime_ep_hip-*.whl \
  ../build/onnxruntime-genai/Release/wheel/onnxruntime_genai_directml-*.whl \
  --extra-index-url https://repo.amd.com/rocm/whl/gfx1151/
```
> **Note**: the ORT whl may be under `../build/onnxruntime/Release/Release/dist/`.
> Replace `gfx1151` with your GPU arch. Install the EP wheel AFTER onnxruntime:
> it ships its own native files (EP plugin `hipgpu.dll`, hip-compiler, custom
> kernels) straight into `onnxruntime/capi/` next to `onnxruntime.dll`. The ROCm
> runtime DLLs (amdhip64/MIOpen/hipBLASLt) come from the `rocm[devel]` wheel
> (expanded next). The wheel does NOT bundle the AMD GPU umbrella
> (`amdgpu-ep.dll` + `hip-backend.dll`); driving OGA through the umbrella needs
> those supplied separately (CI injects them via `HIP_WHEEL_EXTRA_DLLS`).

**Expand ROCm devel** -- the EP's JIT linker needs the ROCm import libs, which
`rocm[devel]` ships compressed. Expand them once:

```bash
rocm-sdk init   # populates site-packages/_rocm_sdk_devel/lib
```

**Runtime env** -- depends on the artifact mode:

- Default (bitcode): the ROCm runtime DLLs on `PATH`.
- NATIVE artifact (opt-in): the above plus `THEROCK_DIST` and `LIB`, used by the
  lld-link step that builds the per-model `.dll`.

**Use** -- there is no Python API to import; the native files are found by
location. Run a model whose `genai_config.json` selects the EP via
`provider_options`; OGA discovers the colocated EP automatically. This is what
the CI wheel smoke does (`Run OGA wheel smoke (Python)` in
`.github/workflows/windows-build.yml`):

```bash
# site-packages root + the colocated capi dir
SP=$(python -c "import onnxruntime, os; print(os.path.dirname(os.path.dirname(onnxruntime.__file__)))")
CAPI="$SP/onnxruntime/capi"

# Default (bitcode): the ROCm runtime DLLs + colocated capi on PATH.
# Replace gfx1151 with your GPU arch.
export PATH="$CAPI:$SP/_rocm_sdk_core/bin:$SP/_rocm_sdk_libraries_gfx1151/bin:$PATH"

# Run OGA's own end-to-end benchmark from the OGA source cloned in step 3
python onnxruntime-genai/benchmark/python/benchmark_e2e.py \
  -i /path/to/model_dir -l 128 -g 128 -r 5 -w 1 -b 1 -m -1 -v
```

`benchmark_e2e.py` runs with the default `-e follow_config`, so the model's
`genai_config.json` selects the EP via `provider_options`. With the upstream OGA
(v0.14.0 + PR2194, DeviceType AMDGPU) this is the AMD GPU umbrella
(`provider_options [{ "AMDGPU": {"profile": "llm"} }]`), which loads
`amdgpu-ep.dll` and needs the umbrella DLLs colocated (see
`.github/workflows/windows-build.yml`); the default wheel ships only the hipgpu
chain. For plain ORT (direct hipgpu, no OGA), register the colocated plugin via
`ort.register_execution_provider_library("hipgpu", "$CAPI/hipgpu.dll")`.

## Testing & Benchmarking

### Model Inference with hip-onnx-runner

`hip-onnx-runner` runs a single ONNX model through hipgpu EP and reports
timing. It is built automatically when `BUILD_HIP_TOOLS=ON`.

```bash
# first cd to your onnx-hipdnn-ep directory
# Default (bitcode) needs the ROCm runtime DLLs on PATH; TheRock's bin is under
# the build dir (or your own -DTHEROCK_DIST path). NATIVE artifacts additionally
# use THEROCK_DIST + LIB for the lld-link step.
export PATH="$(cd ../build/$(basename $PWD)/_therock/bin && pwd):$LOCAL_DIR/bin:$PATH"
```

> **Note**: `hip-onnx-runner.exe` feeds the model random input by default. For an
> LLM the `input_ids` must be within range (`< vocab size`), not random, so
> generate a valid input directory first and pass it with `-i`:
>
> ```bash
> python tools/hip-onnx-runner/gen_hip_onnx_runner_inputs.py -o gen_inputs /path/to/your_test_model.onnx
> ```

```bash
# Run with hipgpu EP (default), on your model directly (dynamic shape)
$LOCAL_DIR/bin/hip-onnx-runner.exe -m /path/to/model.onnx -i gen_inputs

# Resolve symbolic input dims at runtime (the EP still compiles the dynamic graph)
$LOCAL_DIR/bin/hip-onnx-runner.exe -m /path/to/model.onnx -i gen_inputs -f sequence_length:128

# Run with CPU only (no EP)
$LOCAL_DIR/bin/hip-onnx-runner.exe -m /path/to/model.onnx -n -i gen_inputs

# Dump outputs for comparison
$LOCAL_DIR/bin/hip-onnx-runner.exe -m /path/to/model.onnx -i gen_inputs -d 2

# L2-norm compare EP vs CPU outputs
$LOCAL_DIR/bin/hip-onnx-runner.exe -L ep_o_dump,cpu_o_dump
```

**Key flags:**

| Flag | Description |
|------|-------------|
| `-m` | Path to `.onnx` model |
| `-n` | CPU only, skip EP registration |
| `-d 0\|1\|2\|3` | Dump level: 0=off, 1=inputs, 2=outputs, 3=both |
| `-i <dir>` | Load inputs from directory instead of random |
| `-f <name>:<val>` | Resolve a symbolic input dim at runtime (repeatable/comma-separated); EP still compiles the dynamic graph, unmatched symbolic dims default to 1 |
| `-L dir1,dir2` | L2-norm comparison of two output directories |

### Latency Benchmarking with onnxruntime_perf_test

Use `onnxruntime_perf_test` to benchmark inference latency. The examples below
compare hipgpu EP (AMD GPU via HIP) against DML EP.

**Setup:**

`onnxruntime_perf_test.exe` is not installed by
default. Copy it into `../local/bin` before running:

```bash
cp ../build/onnxruntime/Release/Release/onnxruntime_perf_test.exe $LOCAL_DIR/bin/

# Default (bitcode) needs the ROCm runtime DLLs on PATH; TheRock's bin is under
# the build dir (or your own -DTHEROCK_DIST path). NATIVE artifacts additionally
# use THEROCK_DIST + LIB for the lld-link step.
export PATH="$(cd ../build/$(basename $PWD)/_therock/bin && pwd):$PATH"
cd $LOCAL_DIR/bin
```

**hipgpu EP:**

```bash
./onnxruntime_perf_test.exe \
  --plugin_ep_libs "hipgpu|hipgpu.dll" \
  --plugin_eps "hipgpu" \
  -t 60 -c 1 -s -I \
  /path/to/model.onnx
```

**DML EP:**

```bash
./onnxruntime_perf_test.exe \
  -e dml \
  -C "ep.dml.disable_graph_fusion|1" \
  -t 60 -c 1 -s -I \
  /path/to/model.onnx
```

**Key flags:**

| Flag | Description |
|------|-------------|
| `-t 60` | Run for 60 seconds |
| `-c 1` | 1 concurrent thread |
| `-s` | Show per-iteration latency statistics |
| `-I` | Use sequential inputs (do not randomize) |

### OGA End-to-End Benchmarking with model_benchmark

`model_benchmark` benchmarks the full generative pipeline (prefill + decode
token generation). It was built and installed into `$LOCAL_DIR/bin/` in
[step 3](#3-build-oga-onnxruntime-genai). Point `-i` at an OGA model directory
(the one containing `genai_config.json`); make sure its tokenizer files are
present there too.

The EP is selected by the model's `genai_config.json` `provider_options` and
auto-discovered next to `onnxruntime-genai.dll` -- do NOT pass `--ep_library`
(upstream `model_benchmark` rejects it). With the upstream OGA (v0.14.0 +
PR2194) the EP is the AMD GPU umbrella (`provider_options [{ "AMDGPU":
{"profile": "llm"} }]`), so `amdgpu-ep.dll` must sit next to the OGA DLLs (see
`.github/workflows/windows-build.yml`).

```bash
# Auto-generated prompt (512 tokens)
./model_benchmark.exe \
  -i /path/to/oga_model_dir \
  -l 512 -g 128 \
  -r 5 -w 1

# Prompt from file
./model_benchmark.exe \
  -i /path/to/oga_model_dir \
  --prompt_file /path/to/prompt.txt -g 128 \
  -r 5 -w 1
```

**Key flags:**

| Flag | Description |
|------|-------------|
| `-i <path>` | Path to OGA model directory (with `genai_config.json`) |
| `-l <n>` | The number of auto-generated prompts (exclusive with `--prompt_file`) |
| `--prompt_file <path>` | Load prompt text from a file (exclusive with `-l`) |
| `-g <n>` | Max number of tokens to generate |
| `-r <n>` | Number of benchmark repetitions |
| `-w <n>` | Number of warmup runs |

## Linux

Linux uses Docker for build and a separate path for runtime. See
[quick_start_linux.md](quick_start_linux.md).

## ABI Note

The dependencies are compiled with the **Release `/MT`** runtime
(`MultiThreaded` static CRT). Using `-DCMAKE_BUILD_TYPE=Debug` or
`-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDebug` will produce runtime library
mismatch linker errors.

## Updating the C++ Dependencies

Dependency versions are pinned in [`cmake/deps.txt`](../cmake/deps.txt) -- the
single source of truth for LLVM/MLIR/LLD, protobuf, flatbuffers, ONNX Runtime
and TheRock. To move to a newer version, edit the relevant line there; the next
configure picks it up (rebuilding from source, or re-resolving from your prefix).

## Troubleshooting

### CMake builds LLVM from source unexpectedly (slow cold configure)

**Symptom:** configure starts cloning/building `llvm-project` (multi-hour).

**Cause:** no LLVM/MLIR was found on `CMAKE_PREFIX_PATH`, so
[`cmake/deps.cmake`](../cmake/deps.cmake) falls back to a from-source build.

**Solution:** this is expected on a fresh tree with no prefix. To avoid it,
install a prebuilt LLVM/MLIR/LLD into a prefix and pass
`-DCMAKE_PREFIX_PATH=<prefix>`, and reuse the same build dir so the source build
is cached across reconfigures.

### Dependency build cannot find a compiler

**Symptom:** the dependency builds fail with `cl.exe`/`ninja` not found
or a CMake compiler-detection error.

**Solution:** Run configure/build from a Visual Studio Developer shell (so
`cl.exe`, `ninja` and `git` are on PATH). The LLVM build also needs several GB
of free disk and is long on a cold `sccache`.

### sccache not found

**Symptom:** CMake configure error: `Could not find compiler launcher sccache`.

**Solution:** Install sccache (see Prerequisites) or remove the
`CMAKE_C_COMPILER_LAUNCHER` and `CMAKE_CXX_COMPILER_LAUNCHER` options from the
configure command.

### Runtime library mismatch

**Symptom:** Linker errors mentioning `/MT` vs `/MTd` or `/MD`.

**Solution:** Use exactly `-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded` and
`-DCMAKE_BUILD_TYPE=Release`. Do not mix Debug and Release configurations.

### LIT tests not running (ctest reports instant pass)

**Symptom:** `MorphizenMLIRLitTests` completes in under 1 second and reports
`Passed` without printing any test names.

**Solution:** Ensure `lit` is installed via pip (`pip install lit`) and that
CMake found the correct `lit.exe` during configure. Re-run configure with
`--fresh` after installing lit.
