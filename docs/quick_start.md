<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Quick Start Guide

This guide covers the **pre-built flow**: downloading ready-made LLVM/MLIR/LLD,
FlatBuffers, and Protobuf binaries instead of compiling them from source.

## Prerequisites

| Tool | Purpose |
|------|---------|
| **CMake** 3.20+ | Build system |
| **Ninja** | Build generator (faster than MSVC, required) |
| **MSVC 2022** | C++ compiler (Visual Studio Build Tools or full IDE) |
| **Python 3** | LLVM/MLIR tools runtime |
| **sccache** | Compiler cache (significantly speeds up rebuilds) |
| **`gh` CLI** | Downloading pre-built binaries (`gh auth login` required) |
| **`unzip`** | Extracting downloaded archives (available in Git Bash / MSYS2) |

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

# 2. Ninja, Python 3, sccache, GitHub CLI, Git (provides Git Bash + unzip)
winget install Ninja-build.Ninja
winget install Python.Python.3.14
winget install Mozilla.sccache
winget install GitHub.cli
winget install Git.Git

# 3. Authenticate gh / set up an SSH key with access to the private
#    ROCm/MorphiZen submodule (cloned by `git submodule update`).
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
unzip -v | head -1        # provided by Git for Windows
```

## Directory Layout

All paths are relative to the directory that contains the project source
(`<workspace>/`). Clone the repository so that the source, build, and
prebuilt-local directories are siblings:

```
<workspace>/
├── <repo>/             # project source — this is $PWD for all commands below
├── build/
│   └── <repo>/         # cmake build output  ../build/$(basename $PWD)
└── prebuilt-local/     # pre-built Release binaries  ../prebuilt-local
    ├── bin/
    ├── include/
    └── lib/cmake/{llvm,mlir,lld,flatbuffers,protobuf,...}
```

## One-Time Setup

### 0. Clone the repository

Pick a workspace directory and clone the project so all later commands can
use `..` to reference sibling directories:

```bash
cd <workspace>
git clone https://github.com/ROCm/onnx-hipdnn-ep.git
cd onnx-hipdnn-ep
git submodule update --init --recursive
```

All subsequent commands run from the `onnx-hipdnn-ep/` project root unless
explicitly noted.

### 1. Build ONNX Runtime (Required for BUILD_EP=ON)

Build ONNX Runtime **before** downloading pre-built dependencies to avoid
FlatBuffers version conflicts (ORT bundles its own FlatBuffers).

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
# This ensures ONNX Runtime uses its own FlatBuffers version, avoiding conflicts with prebuilt FlatBuffers.
# --build_wheel additionally produces a Python wheel under
# ../build/onnxruntime/Release/dist/, which is needed for the OGA build below
# and lets you `pip install` ONNX Runtime later (see "Install Python Wheels").
./build.bat --config Release --build_shared_lib --parallel --compile_no_warning_as_error --skip_submodule_sync --build_dir ../build/onnxruntime --skip_tests --disable_memleak_checker --use_dml --build_wheel

# Install to prebuilt-local (set prefix at install time, not during configuration)
mkdir -p ../prebuilt-local
PREBUILT_DIR=$(cd ../prebuilt-local && pwd)
cmake --install ../build/onnxruntime/Release --prefix "$PREBUILT_DIR"
```

**Verify installation**:

```bash
ls $PREBUILT_DIR/lib/cmake/onnxruntime/
# Should see onnxruntime-config.cmake and related files

ls ../build/onnxruntime/Release/dist/onnxruntime_directml-*.whl
# Should see one wheel matching your Python version, e.g.
# onnxruntime_directml-1.25.1-cp314-cp314-win_amd64.whl
```

### 2. Build the C++ Dependencies from Source

You must build LLVM/MLIR/LLD, protobuf and flatbuffers from source yourself (from a Visual Studio Developer shell) and install them into `../prebuilt-local/` — there is no prebuilt download. Use the following pinned versions; the exact cmake invocations are the `Build LLVM/MLIR/LLD from source`, `Build protobuf from source` and `Build flatbuffers from source` steps in [`.github/workflows/windows-build.yml`](../.github/workflows/windows-build.yml), which is the canonical recipe (point each `-DCMAKE_INSTALL_PREFIX` at the shared `../prebuilt-local/`).

| Component | Source | Version |
|---|---|---|
| LLVM / MLIR / LLD | `github.com/llvm/llvm-project` | tag `llvmorg-22.1.0` (`mlir;lld`, X86 only, `/MT`, `LLVM_INSTALL_UTILS=ON` for FileCheck) |
| Protobuf (+ abseil) | `github.com/protocolbuffers/protobuf` | `v34.0` (build with `CMAKE_CXX_STANDARD=17`) |
| FlatBuffers | `github.com/google/flatbuffers` | `v25.12.19` |

The LLVM build is the long pole (multi-hour cold build).

### 3. Build onnx-hipdnn-ep

**Prerequisites**: Complete steps 1-2 (build ONNX Runtime, build the C++ dependencies from source) and install [TheRock SDK](https://repo.amd.com/rocm/tarball/). Recommended version: **TheRock 7.11.0**.

Run from the project root:

```bash
PREBUILT_DIR=$(cd ../prebuilt-local && pwd)
THEROCK_DIST=$(cd ../therock && pwd)
# For new users of this project, we recommend using the HIP provided by therock.
# If you really want to use a pre-installed HIP instead, you can skip this unset.
unset HIP_PATH

cmake -S . -B ../build/$(basename $PWD) \
  -G Ninja \
  -DBUILD_SHARED_LIBS=OFF \
  -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded \
  -DCMAKE_BUILD_TYPE=Release \
  "-DCMAKE_PREFIX_PATH=$PREBUILT_DIR" \
  "-DCMAKE_INSTALL_PREFIX=$PREBUILT_DIR" \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DCMAKE_C_COMPILER_LAUNCHER=sccache \
  -DCMAKE_CXX_COMPILER_LAUNCHER=sccache \
  -DTHEROCK_DIST="$THEROCK_DIST" \
  -DHIP_PLATFORM=amd \
  -DHIP_ARCHITECTURES=gfx1151 \
  -DBUILD_EP=ON \
  -DBUILD_MOCK_RUNTIME=OFF \
  -DBUILD_HIP_TOOLS=ON
```

**Note**: Replace `gfx1151` with your GPU architecture. Detect using: `$THEROCK_DIST/lib/llvm/bin/amdgpu-arch.exe`

**Key options:**

| Option | Value | Notes |
|--------|-------|-------|
| `-G Ninja` | — | Recommended for faster builds; MSVC generator also works |
| `CMAKE_MSVC_RUNTIME_LIBRARY` | `MultiThreaded` | Must match pre-built binaries (`/MT`) |
| `CMAKE_BUILD_TYPE` | `Release` | Pre-built binaries are Release; Debug CRT (`/MTd`) is incompatible |
| `CMAKE_PREFIX_PATH` | `$PREBUILT_DIR` | Where to find dependencies |
| `CMAKE_INSTALL_PREFIX` | `$PREBUILT_DIR` | Where to install |
| `CMAKE_C/CXX_COMPILER_LAUNCHER` | `sccache` | Omit if sccache is not installed |
| `THEROCK_DIST` | Path to TheRock SDK | Required |
| `HIP_PLATFORM` | `amd` | Required |
| `HIP_ARCHITECTURES` | GPU architecture (e.g., `gfx1151`) | Required |
| `BUILD_EP` | `ON` | Build MorphiZen Execution Provider |

#### Build

```bash
cmake --build ../build/$(basename $PWD) --config Release --parallel
cmake --install ../build/$(basename $PWD) --config Release
```

### 4. Build OGA (onnxruntime-genai)

OGA (ONNX Runtime GenAI) provides the generative-pipeline runtime used by
`model_benchmark.exe` (and the Python `onnxruntime_genai` module) to drive
prefill + decode token generation against models prepared via the [ONNX
Model Splitter](../tools/onnx-model-splitter/README.md). Build it after
step 1 (ORT) so it links against the same ORT version.

**Prerequisites**: step 1 (ORT built with `--build_wheel`). Step 3 is only
needed later to *run* `model_benchmark` against MorphiZen EP, not to build
OGA itself.

Run all commands below from the `onnx-hipdnn-ep/` project root.

#### 4a. Prepare ORT_HOME

OGA's CMake expects an `ORT_HOME/` directory layout with `include/` and `lib/`
subdirectories. Stage one from the ORT install under `$PREBUILT_DIR`:

```bash
mkdir -p ../build/oga-ort-home/include ../build/oga-ort-home/lib
ORT_HOME=$(cd ../build/oga-ort-home && pwd)
cp -r "$PREBUILT_DIR/include/onnxruntime/"* "$ORT_HOME/include/"
cp "$PREBUILT_DIR/lib/"onnxruntime*.lib "$ORT_HOME/lib/"
cp "$PREBUILT_DIR/bin/onnxruntime.dll" "$ORT_HOME/lib/"
cp "$PREBUILT_DIR/bin/onnxruntime_providers_shared.dll" "$ORT_HOME/lib/"
```

#### 4b. Clone OGA

```bash
cd ..  # Go to workspace directory (sibling of onnx-hipdnn-ep)
git clone -b feat/oga_hipdnn_experiment https://github.com/AMDmoore/onnxruntime-genai.git
cd onnxruntime-genai
git submodule update --init --recursive
```

> **Note**: CI pins a specific commit (see `OGA_REF` in
> [`.github/workflows/windows-build.yml`](../.github/workflows/windows-build.yml)).
> If you want byte-for-byte reproducibility, check out that commit instead of
> the branch tip.

#### 4c. Build OGA

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
| `onnxruntime_genai_directml-*.whl` | `wheel/onnxruntime_genai_directml-*.whl` | Python wheel (see [Install Python Wheels](#5-install-python-wheels-optional)) |

#### 4d. Install OGA artifacts into prebuilt-local

```bash
cp ../build/onnxruntime-genai/Release/benchmark/c/model_benchmark.exe "$PREBUILT_DIR/bin/"
cp ../build/onnxruntime-genai/Release/onnxruntime-genai.dll "$PREBUILT_DIR/bin/"
```

After this, `$PREBUILT_DIR/bin/` has the OGA artifacts (`model_benchmark.exe`
and `onnxruntime-genai.dll`) used by [OGA End-to-End Benchmarking](#oga-end-to-end-benchmarking-with-model_benchmark)
below. `onnxruntime_perf_test.exe` is staged separately in
[Latency Benchmarking](#latency-benchmarking-with-onnxruntime_perf_test).

### 5. Install Python Wheels (Optional)

If you also want to drive ORT / OGA from Python (e.g. to write your own
generation script instead of using `model_benchmark.exe`), install the two
wheels produced by steps 1 and 4:

```bash
cd ../onnx-hipdnn-ep  # Back to the project root
pip install \
  ../build/onnxruntime/Release/dist/onnxruntime_directml-*.whl \
  ../build/onnxruntime-genai/Release/wheel/onnxruntime_genai_directml-*.whl
```
> **Note**: the first whl file may be in  ../build/onnxruntime/Release/Release/dist/
>  Try this path if you meet error "the file does not exist"

Verify:

```bash
python -c "import onnxruntime as ort; print(ort.__version__)"
python -c "import onnxruntime_genai as og; print(og.__version__)"
```

## Model Preparation

Please See
[tools/onnx-model-splitter/README.md](../tools/onnx-model-splitter/README.md) for full details of this step.
In this document, there is a step by step guide which use Llama-3.1-8B as example.

Use `tools/onnx-model-splitter` to export prefill/decode ONNX models for
benchmarking with ONNX Runtime GenAI. Install Python dependencies first:

```bash
pip install -r tools/onnx-model-splitter/requirements.txt
```

**Step 1 — Generate `genai_config_pipeline.json`:**

```bash
python tools/onnx-model-splitter/genai_config_pipeline_from_folder.py \
  /path/to/model_folder \
  --max-length 16384
```

This reads the model's config (e.g. `config.json`) and produces
`genai_config_pipeline.json` in the model folder.

**Step 2 — Export prefill/decode ONNX:**

```bash
python tools/onnx-model-splitter/export_chunk_model.py \
  --model /path/to/model_folder/model.onnx \
  --output /path/to/output \
  -T /path/to/model_folder/genai_config_pipeline.json
```

This exports `prefill_p512m16384.onnx`, `decode_p512m16384.onnx`,
`genai_config_p512m16384.json`, and shared external weights into the output
directory.

## Testing & Benchmarking

### Model Inference with hip-onnx-runner

`hip-onnx-runner` runs a single ONNX model through MorphiZen EP and reports
timing. It is built automatically when `BUILD_HIP_TOOLS=ON`.

```bash
# first cd to your onnx-hipdnn-ep directory
export THEROCK_DIST=$(cd ../therock && pwd)
export PATH="$THEROCK_DIST/bin:$PREBUILT_DIR/bin:$PATH"

> **Note**: hip-onnx-runner.exe run onnx model with random data as input.
>  But for llm model, the input_ids should be in valid scope (< voab size) and not be random.
>  So for below test, it is necessary to produce valid data file as input.
>  Please run :
>  # python tools/hip-onnx-runner/gen_hip_onnx_runner_inputs.py -o gen_inputs /path/to/your_test_model.onnx
>  to produce dir holding test data and use "-i" option to let hip-onnx-runner.exe use this dir as input

# Run with MorphiZen EP (default), using a fixed-shape model from Model Preparation
$PREBUILT_DIR/bin/hip-onnx-runner.exe -m /path/to/output/prefill_p512m16384.onnx -i gen_inputs

# Run with CPU only (no EP)
$PREBUILT_DIR/bin/hip-onnx-runner.exe -m /path/to/output/prefill_p512m16384.onnx -n -i gen_inputs

# Dump outputs for comparison
$PREBUILT_DIR/bin/hip-onnx-runner.exe -m /path/to/output/prefill_p512m16384.onnx -i gen_inputs -d 2

# L2-norm compare EP vs CPU outputs
$PREBUILT_DIR/bin/hip-onnx-runner.exe -L ep_o_dump,cpu_o_dump
```

**Key flags:**

| Flag | Description |
|------|-------------|
| `-m` | Path to `.onnx` model |
| `-n` | CPU only, skip EP registration |
| `-d 0\|1\|2\|3` | Dump level: 0=off, 1=inputs, 2=outputs, 3=both |
| `-i <dir>` | Load inputs from directory instead of random |
| `-L dir1,dir2` | L2-norm comparison of two output directories |

### Latency Benchmarking with onnxruntime_perf_test

Use `onnxruntime_perf_test` to benchmark inference latency. The examples below
compare MorphiZen EP (AMD GPU via HIP) against DML EP.

**Setup:**

`onnxruntime_perf_test.exe` is not installed by
default. Copy it into `../prebuilt-local/bin` before running:

```bash
cp ../build/onnxruntime/Release/Release/onnxruntime_perf_test.exe $PREBUILT_DIR/bin/

export THEROCK_DIST=$(cd ../therock && pwd)
export PATH="$THEROCK_DIST/bin:$PATH"
cd $PREBUILT_DIR/bin
```

**MorphiZen EP:**

```bash
./onnxruntime_perf_test.exe \
  --plugin_ep_libs "MorphiZenEP|onnxruntime_morphizen_ep.dll" \
  --plugin_eps "MorphiZenEP" \
  --plugin_ep_options "config_file|morphizen_config.json" \
  -t 60 -c 1 -s -I \
  /path/to/output/prefill_p512m16384.onnx
```

**DML EP:**

```bash
./onnxruntime_perf_test.exe \
  -e dml \
  -C "ep.dml.disable_graph_fusion|1" \
  -t 60 -c 1 -s -I \
  /path/to/output/prefill_p512m16384.onnx
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
token generation). It was built and installed into `$PREBUILT_DIR/bin/` in
[step 4](#4-build-oga-onnxruntime-genai).  Before running, please copy the
tokenizer related files from original model directory to /path/to/output

```bash
# Auto-generated prompt (512 tokens)
./model_benchmark.exe \
  -i /path/to/output \
  --ep_library MorphiZenEP onnxruntime_morphizen_ep.dll \
  -l 512 -g 128 \
  -r 5 -w 1

# Prompt from file
./model_benchmark.exe \
  -i /path/to/output \
  --ep_library MorphiZenEP onnxruntime_morphizen_ep.dll \
  --prompt_file /path/to/prompt.txt -g 128 \
  -r 5 -w 1
```

**Key flags:**

| Flag | Description |
|------|-------------|
| `-i <path>` | Path to OGA model directory (with `genai_config.json`) |
| `--ep_library <name> <path>` | Register a custom EP library |
| `-l <n>` | The number of auto-generated prompts (exclusive with `--prompt_file`) |
| `--prompt_file <path>` | Load prompt text from a file (exclusive with `-l`) |
| `-g <n>` | Max number of tokens to generate |
| `-r <n>` | Number of benchmark repetitions |
| `-w <n>` | Number of warmup runs |

## Linux

Linux uses Docker for build and a separate path for runtime. See
[quick_start_linux.md](quick_start_linux.md).

## ABI Note

The pre-built binaries are compiled with the **Release `/MT`** runtime
(`MultiThreaded` static CRT). Using `-DCMAKE_BUILD_TYPE=Debug` or
`-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDebug` will produce runtime library
mismatch linker errors.

## Updating the C++ Dependencies

To move to a newer LLVM/protobuf/flatbuffers, update the `LLVM_TAG` /
`PROTO_TAG` / `FLATBUFFERS_TAG` values in
[`.github/workflows/windows-build.yml`](../.github/workflows/windows-build.yml)
and rebuild the affected dependency into `../prebuilt-local/`.

## Troubleshooting

### CMake cannot find LLVM/MLIR

**Symptom:** `Could not find package LLVM` or similar.

**Solution:** Verify `../prebuilt-local/lib/cmake/llvm/` exists. Rebuild the
dependencies from source (step 2) if the directory is missing.

### Dependency build cannot find a compiler

**Symptom:** the dependency builds fail with `cl.exe`/`ninja` not found
or a CMake compiler-detection error.

**Solution:** Run the script from a Visual Studio Developer shell (so `cl.exe`,
`ninja` and `git` are on PATH). The LLVM build also needs several GB of free
disk and is long on a cold `ccache`.

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
