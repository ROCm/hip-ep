<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->

# Quick Start (Windows)

This guide walks through every step from a fresh clone to a green
end-to-end inference run on a self-generated toy ONNX model, executed
through the **MorphiZen Execution Provider (EP)** on an AMD GPU via
HIP / TheRock. It is written so that anyone -- regardless of prior
experience with `onnx-hipdnn-ep` -- can follow it end-to-end. Every
command below is PowerShell unless explicitly marked as bash.

## What this guide covers

Use this guide when you want to:

- Build `onnx-hipdnn-ep` from a fresh checkout and load the resulting EP
  into ONNX Runtime, running an ONNX model on the AMD GPU.
- Verify that a recent local change still produces a working EP (the
  toy-model recipe at the end is a small smoke test).
- Look up what each one-time setup step does, so you can adapt the
  workflow to a non-default directory layout.

This guide focuses on Windows. Linux and WSL coverage may be added in a
later revision.

## What you will have at the end

After completing the steps below you will have:

1. A Python virtual environment at `<repo>/.venv` containing `cmake`, `onnx`,
   `numpy`, `lit`, and `ninja`.
2. A `prebuilt-local/` directory next to the repo containing pre-built LLVM,
   MLIR, Protobuf, and FlatBuffers (no need to compile them from source), plus
   the `onnxruntime_morphizen_ep.dll` and `onnxruntime_perf_test.exe` produced
   by the local build.
3. A small Conv ONNX model (`examples/quickstart/conv_hybrid.onnx`)
   generated from
   [`examples/quickstart/gen_conv_only.py`](../examples/quickstart/gen_conv_only.py).
4. A latency report from `onnxruntime_perf_test.exe` running that model
   through the MorphiZen plugin EP for 60 seconds, demonstrating that the
   full stack (build -> JIT compile -> GPU init -> custom kernel / MIOpen
   dispatch -> output) works end-to-end.

## System prerequisites

| Component | Version | Why |
|-----------|---------|-----|
| Windows | 11 (or Server 2022) | The scripts assume PowerShell 5.1+ and the Windows 10 SDK installer layout. |
| Visual Studio 2022 | Build Tools or full IDE, "Desktop development with C++" workload | Provides `cl.exe`, `link.exe`, MSVC headers/libs. The scripts read `vcvarsall.bat` from `C:\msvsn2022\VC\Auxiliary\Build\vcvarsall.bat` -- if your VS lives elsewhere, see [Troubleshooting: vcvarsall.bat not found](#vcvarsallbat-not-found). |
| Windows 10 SDK | 10.0.26100.0 (the exact version the scripts patch in) | The scripts patch this version into `INCLUDE`/`LIB`/`PATH` if `vcvarsall` does not pick it up. If your installed version differs, edit the `$sdkVer` line in both scripts -- see note below. |
| Python | 3.10+ | Used both via the `.venv` (build/runtime tools) and indirectly by ONNX Runtime's own build. |
| Git for Windows | latest | Provides Git Bash, required only for the one-time `scripts/setup-prebuilt.sh` invocation. |
| `gh` CLI | latest, authenticated | Downloads the prebuilt LLVM/MLIR/Protobuf/FlatBuffers archives from the `wcy123/llvm-mlir-prebuilt` GitHub releases. Run `gh auth login` if not already done; access may require GitHub authentication. |
| AMD GPU | gfx1100 / gfx1151 (or any TheRock-supported gfx) | The build defaults to `gfx1151`. Detect yours with `<therock>\lib\llvm\bin\amdgpu-arch.exe`. |
| TheRock SDK | 7.11.0 | Provides HIP runtime, MIOpen, hipBLASLt, etc. See [TheRock GitHub releases](https://github.com/ROCm/TheRock/releases) or your AMD distribution channel. |
| sccache | optional but recommended | Significantly speeds up rebuilds. `winget install Mozilla.sccache`. |

The remaining build tools (`cmake`, `ninja`, `onnx`, `numpy`, `lit`) are all
installed via `pip` inside the venv created in step 4.1.b, so you do not need
system-wide installations of those.

If your Windows 10 SDK version is something other than `10.0.26100.0` (check
the contents of `C:\Program Files (x86)\Windows Kits\10\Include\` -- each
sub-directory is one installed SDK version), edit the `$sdkVer` line in
[setup_and_configure.ps1:38](../setup_and_configure.ps1) and
[build_and_install.ps1:24](../build_and_install.ps1) to match. The version
must include the trailing `.0` patch number, e.g. `10.0.22621.0`.

## Workspace layout

All paths in this guide assume the following parent directory:

```
C:\Users\<you>\ROCmEP\OnnxHipDNN\
|-- onnx-hipdnn-ep\               # this repo (clone target)
|   |-- .venv\                    # created in step 4.1.b
|   |-- setup_and_configure.ps1
|   |-- build_and_install.ps1
|   |-- etc\morphizen_config.json # plugin EP config, auto-installed
|   |-- examples\                 # self-contained demo programs (see examples\README.md)
|   |   |-- quickstart\           # toy-model demo referenced by this guide
|   |   |   |-- gen_conv_only.py
|   |   |   |-- README.md
|   |   |   `-- .gitignore
|   |   |-- *.hip.mlir            # HIP-dialect MLIR test programs (legacy flat layout)
|   |   `-- main_*.cpp            # C++ drivers for the MLIR programs
|   |-- scripts\setup-prebuilt.sh
|   `-- ...
|-- build\onnx-hipdnn-ep\         # cmake build output
|-- prebuilt-local\               # install prefix
|   |-- bin\                      # hip-onnx-runner.exe, onnxruntime_morphizen_ep.dll, onnxruntime_perf_test.exe, morphizen_config.json
|   |-- include\
|   `-- lib\cmake\{llvm,mlir,lld,flatbuffers,protobuf,onnxruntime,...}\
|-- therock\                      # TheRock SDK (HIP, MIOpen, hipBLASLt, ...) -- scripts default to therock-7.11.0-clean; rename or set $env:THEROCK_DIST_OVERRIDE
`-- OnnxRuntime\
    `-- onnxruntime\              # ORT source for the DML build
```

The scripts derive their paths from `$PSScriptRoot` (the script's own
location), so the layout above works without any edits. If your layout
differs, set the matching environment variables before dot-sourcing the
scripts:

```powershell
$env:HIPDNN_WORKSPACE_ROOT = "<your parent dir>"     # parent of onnx-hipdnn-ep
$env:HIPDNN_BUILD_DIR      = "<your build dir>"      # cmake build output
$env:HIPDNN_PREBUILT_DIR   = "<your prebuilt dir>"   # cmake install prefix (setup_and_configure.ps1)
$env:HIPDNN_BIN_DIR        = "<your prebuilt>\bin"   # bin directory (build_and_install.ps1)
$env:THEROCK_DIST_OVERRIDE = "<your therock dir>"    # TheRock SDK directory
$env:ORT_PERF_TEST_PATH    = "<...\onnxruntime_perf_test.exe>"  # if your ORT lives elsewhere
```

You can also edit the variable defaults at the top of
[setup_and_configure.ps1](../setup_and_configure.ps1) and
[build_and_install.ps1](../build_and_install.ps1) directly if you prefer.

## One-time setup

The seven sub-steps below run once per machine. After this you only need
[Section 5: Incremental dev loop](#incremental-dev-loop).

### 4.1 Clone the repo and submodules

In a fresh PowerShell (not cmd, not Git Bash):

```powershell
cd C:\Users\<you>\ROCmEP\OnnxHipDNN
git clone https://github.com/ROCm/onnx-hipdnn-ep.git
cd onnx-hipdnn-ep
git submodule update --init --recursive
```

The `--recursive` flag is required because the EP depends on the
[morphizen](https://github.com/ROCm/morphizen) submodule, which itself has
nested submodules.

### 4.1.b Bootstrap the Python venv

The [setup_and_configure.ps1](../setup_and_configure.ps1) script auto-activates
`<repo>\.venv\Scripts\Activate.ps1` if `$env:VIRTUAL_ENV` is not set (see
[setup_and_configure.ps1:19-23](../setup_and_configure.ps1)), so the venv must
exist at exactly that location.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install cmake onnx numpy lit ninja
```

What each package is for:

| Package | Purpose |
|---------|---------|
| `cmake` | Build system. PyPI's `cmake` ships a recent `cmake.exe` so you do not need a system install. |
| `onnx` | Required by [`examples\quickstart\gen_conv_only.py`](../examples/quickstart/gen_conv_only.py) to construct the toy ONNX graph. |
| `numpy` | Required by the same script for weight tensor generation. |
| `lit` | Required by the LIT test suite (`ctest -R MorphizenMLIRLitTests`); without it ctest reports a spurious instant pass. |
| `ninja` | Build generator that [setup_and_configure.ps1:61](../setup_and_configure.ps1) selects via `-G Ninja`. |

If `Activate.ps1` fails with "running scripts is disabled on this system",
loosen the execution policy for your user once:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

### 4.2 Download prebuilt LLVM, MLIR, Protobuf, and FlatBuffers

This is the only step that runs from Git Bash rather than PowerShell.
It uses `gh release download` to fetch three archives from the
`wcy123/llvm-mlir-prebuilt` GitHub repository and extracts them into
`..\prebuilt-local\`. Access may require GitHub authentication.

```bash
bash scripts/setup-prebuilt.sh
```

Assets fetched:

| Release tag | Asset |
|---|---|
| `llvm-22.1.0-release` | `llvm-22.1.0-release-windows-x64.zip` |
| `protobuf-34.0-release` | `protobuf-34.0-release-windows-x64.zip` |
| `flatbuffers-25.12.19-release` | `flatbuffers-25.12.19-release-windows-x64.zip` |

The script writes a sentinel file (`.extracted-<asset>`) after each successful
extraction, so subsequent runs are no-ops unless you delete the sentinel or
the zip archive. If `gh` reports HTTP 401, run `gh auth login` and try again.

### 4.3 Build ONNX Runtime once (DirectML build)

ONNX Runtime is **not** part of the prebuilt assets above; you build it from
source once. The build is required for two artifacts:

1. **`onnxruntime.dll`** and CMake configs in `prebuilt-local\lib\cmake\onnxruntime\`,
   which `setup_and_configure.ps1` consumes during configure.
2. **`onnxruntime_perf_test.exe`**, copied to `prebuilt-local\bin\` by
   [build_and_install.ps1:84-91](../build_and_install.ps1) (silently skipped
   if the ORT build is missing -- everything else still works, but the
   end-to-end demo in [Section 6](#end-to-end-toy-model-walkthrough) cannot
   run without it).

Do this build once; rebuild only when you need a newer ONNX Runtime version.
The build itself takes 15-25 minutes on a modern machine.

```powershell
cd C:\Users\<you>\ROCmEP\OnnxHipDNN
mkdir OnnxRuntime -Force | Out-Null
cd OnnxRuntime
git clone -b rel-1.25.1 https://github.com/Microsoft/onnxruntime.git
cd onnxruntime

.\build.bat --config Release --build_shared_lib --parallel `
    --compile_no_warning_as_error --skip_submodule_sync `
    --build_dir ..\..\build\onnxruntime --skip_tests `
    --disable_memleak_checker --use_dml `
    --cmake_generator Ninja

cmake --install ..\..\build\onnxruntime\Release `
    --prefix C:\Users\<you>\ROCmEP\OnnxHipDNN\prebuilt-local
```

`--cmake_generator Ninja` matches what the GitHub Actions Windows build uses
([.github/workflows/windows-build.yml:171](../.github/workflows/windows-build.yml))
and is significantly faster than the default MSVC generator. If you prefer
MSBuild, omit that flag.

Why `--use_dml`: the MorphiZen EP coexists in the same ORT process with the
DirectML EP (used as a CPU-fallback / comparison baseline in
[DML EP comparison](#dml-ep-comparison)). The DML EP is only built into ORT
when this flag is set.

`onnxruntime_perf_test.exe` and `DirectML.dll` are not installed by ORT's
`cmake --install`, so copy them manually if you plan to run perf_test or use
the DML EP comparison:

```powershell
$prebuilt = "C:\Users\<you>\ROCmEP\OnnxHipDNN\prebuilt-local"
$ortBuild = "..\..\build\onnxruntime\Release\Release"
Copy-Item "$ortBuild\onnxruntime_perf_test.exe" "$prebuilt\bin\" -Force
# DirectML.dll lives under packages\Microsoft.AI.DirectML.*\bin\x64-win\
$dmlDll = Get-ChildItem -Recurse "..\..\build\onnxruntime" `
    -Filter DirectML.dll | Select-Object -First 1
if ($dmlDll) { Copy-Item $dmlDll.FullName "$prebuilt\bin\" -Force }

cd ..\..\onnx-hipdnn-ep
```

(The `build_and_install.ps1` script you run later will also re-copy
`onnxruntime_perf_test.exe` automatically -- see Section 5 step 4 -- so the
manual copy above is only required if you want it available before your
first invocation of that script.)

Verify the install:

```powershell
ls C:\Users\<you>\ROCmEP\OnnxHipDNN\prebuilt-local\lib\cmake\onnxruntime
# Should list onnxruntime-config.cmake and related files.
```

### 4.4 Install the TheRock SDK

Obtain a TheRock 7.11.0 build (from the
[TheRock GitHub releases](https://github.com/ROCm/TheRock/releases) or
your AMD distribution channel) and extract it to:

```
C:\Users\<you>\ROCmEP\OnnxHipDNN\therock\
```

The shipped scripts default to a directory named `therock-7.11.0-clean\`
(see [setup_and_configure.ps1:15](../setup_and_configure.ps1) and
[build_and_install.ps1:8](../build_and_install.ps1)). Two ways to bridge:

- Recommended: rename your extraction to `therock\` to match this guide,
  and edit the `$therock` variable in both scripts to point at it.
- Alternative: keep the `therock-7.11.0-clean\` name to match the script
  defaults verbatim, and mentally substitute it everywhere this guide says
  `therock\`.

After extraction, the directory should contain `bin\amdhip64_7.dll`,
`bin\hipdnn_backend.dll`, `bin\hipdnn_plugins\`, and `lib\llvm\bin\amdgpu-arch.exe`.
If any of those are missing, re-extract the SDK; the extraction may be
incomplete.

### 4.5 Configure CMake

Open a fresh PowerShell, activate the venv, then run
[setup_and_configure.ps1](../setup_and_configure.ps1):

```powershell
cd C:\Users\<you>\ROCmEP\OnnxHipDNN\onnx-hipdnn-ep
.\.venv\Scripts\Activate.ps1
. .\setup_and_configure.ps1
```

The `.` (dot-source) prefix is required so that environment variables the
script sets (`HIP_PATH`, `THEROCK_DIST`, `INCLUDE`, `LIB`, `PATH`) persist in
your shell after the script returns.

What the script does (refer to [setup_and_configure.ps1](../setup_and_configure.ps1)
for the source):

1. Activates `.venv\Scripts\Activate.ps1` if `$env:VIRTUAL_ENV` is not set.
2. Locates `vcvarsall.bat` via `vswhere` (the official VS discovery tool),
   falling back to the `C:\msvsn2022` junction recipe described in
   [Troubleshooting](#vcvarsallbat-not-found) if `vswhere` is unavailable,
   and imports `vcvarsall.bat x64` so MSVC headers/libs are visible.
3. Patches in Windows 10 SDK 10.0.26100.0 if `vcvarsall` did not include it.
4. Sets `HIP_PATH = $therock`, `THEROCK_DIST = $therock`, prepends `<therock>\bin`
   to `PATH` so HIP/MIOpen DLLs load.
5. Wipes the build directory (forces a clean configure).
6. Runs `cmake -S . -B <buildDir> -G Ninja ...` with the flags below.

CMake flags it sets (informational; you do not edit these unless you have a
reason to):

| Flag | Value | Why |
|------|-------|-----|
| `-G Ninja` | -- | Required so the build tracks header dependencies correctly. |
| `-DBUILD_SHARED_LIBS` | OFF | Static link mode; matches prebuilt binary expectations. |
| `-DCMAKE_MSVC_RUNTIME_LIBRARY` | `MultiThreaded` | Matches prebuilt `/MT`. Mismatch -> linker errors at link time. |
| `-DCMAKE_BUILD_TYPE` | `Release` | Prebuilt binaries are Release-only. |
| `-DCMAKE_PREFIX_PATH` | `<prebuilt-local>` | Where CMake finds LLVM, MLIR, Protobuf, FlatBuffers, onnxruntime. |
| `-DCMAKE_INSTALL_PREFIX` | `<prebuilt-local>` | Install destination (so `prebuilt-local\bin` ends up containing your build outputs). |
| `-DTHEROCK_DIST` | `<therock>` | Where CMake finds HIP, MIOpen, hipBLASLt headers and libs. |
| `-DHIP_PLATFORM` | `amd` | Required (vs. `nvidia`). |
| `-DHIP_ARCHITECTURES` | `gfx1151` | Replace with your GPU's gfx (`<therock>\lib\llvm\bin\amdgpu-arch.exe`). |
| `-DBUILD_EP` | `ON` | Build the MorphiZen Execution Provider DLL. |
| `-DBUILD_HIP_TOOLS` | `ON` | Build `hip-onnx-runner.exe`, `hip-compiler.exe`, `hip-test-dll.exe`. |
| `-DBUILD_MOCK_RUNTIME` | `OFF` | Use the real GPU runtime (mock runtime is for CI without GPUs). |
| `-DBUILD_HIPDNN_GRAPH` | `OFF` | Disabled by default in current main. |

A successful configure ends with:

```
=== Configure done. Now run: . .\build_and_install.ps1 ===
```

## Incremental dev loop

After step 4.5 succeeds, the only command you need for day-to-day work is:

```powershell
. .\build_and_install.ps1
```

What [build_and_install.ps1](../build_and_install.ps1) does:

1. Re-applies the env setup defensively (lines 13-31; idempotent so it is
   safe to run multiple times in the same shell).
2. `cmake --build <buildDir> --config Release` followed by
   `cmake --install <buildDir> --config Release`.
3. Force-syncs `hipdnn_backend.dll` and `amdhip64_7.dll` from the TheRock
   SDK into `prebuilt-local\bin` if their sizes differ. This handles SDK
   downgrades cleanly: when switching from TheRock 7.13 back to 7.11, the
   newer DLLs that the loader cached in `prebuilt-local\bin` would still
   shadow the freshly-installed 7.11 ones, which can produce subtle ABI
   mismatches. Comparing sizes (rather than timestamps) ensures the
   correct copy ends up on the loader's search path.
4. Copies `onnxruntime_perf_test.exe` from the ORT build into
   `prebuilt-local\bin` if it can find it (silently skipped otherwise --
   reported as `WARNING: ... not found ... skipping copy`).
5. `cd`s into `prebuilt-local\bin` so subsequent invocations of
   `.\hip-onnx-runner.exe` or `.\onnxruntime_perf_test.exe` work without
   absolute paths.

When to re-run `setup_and_configure.ps1` versus just `build_and_install.ps1`:

| Situation | Re-run |
|-----------|--------|
| Edited a `.cpp` / `.cc` / `.h` / `.hpp` / `.hip` file | `build_and_install.ps1` |
| Edited a `CMakeLists.txt` / `.cmake` / added a new source file | `build_and_install.ps1` (CMake auto-reconfigures) |
| Edited the cmake flags in `setup_and_configure.ps1` itself | `setup_and_configure.ps1` (forces a fresh build dir) |
| Switched git branches or pulled with submodule changes | `setup_and_configure.ps1` (avoids stale CMake cache) |
| New PowerShell session | `setup_and_configure.ps1` once (re-imports MSVC env), then `build_and_install.ps1` for builds |

## End-to-end toy-model walkthrough

This is a compact smoke test that exercises the full stack: generate a
small Conv ONNX, then run it through the MorphiZen plugin EP via
`onnxruntime_perf_test.exe` for 60 seconds and observe latency stats.

### Step 1: Generate the toy Conv model

```powershell
cd C:\Users\<you>\ROCmEP\OnnxHipDNN\onnx-hipdnn-ep
.\.venv\Scripts\Activate.ps1
cd .\examples\quickstart
python .\gen_conv_only.py --output conv_hybrid.onnx `
    --batch 1 --in-channels 16 --out-channels 16 `
    --height 16 --width 16 --kernel 3 --pad 1 --stride 1
cd ..\..
```

Expected output:

```
Saved model to conv_hybrid.onnx
  Input shape: [1, 16, 16, 16]
  Weight shape: [16, 16, 3, 3]
  Output shape: [1, 16, 16, 16]
Saved weights to conv_hybrid_weights.npy
```

The default arguments to `gen_conv_only.py` produce a YOLOv8x-sized conv
(18.9 MB of weights, 7.5 GFLOPs); the overrides above shrink it to ~9 KB
of weights (`16 * 16 * 3 * 3 * 4 bytes = 9216 bytes`) so the model loads,
JIT-compiles, and runs in seconds. Feel free to scale up if you want
longer measurement windows.

### Step 2: Run the model through the MorphiZen plugin EP

```powershell
$bin = "C:\Users\<you>\ROCmEP\OnnxHipDNN\prebuilt-local\bin"
& "$bin\onnxruntime_perf_test.exe" `
    .\examples\quickstart\conv_hybrid.onnx `
    --plugin_ep_libs    "MorphiZenExecutionProvider|$bin\onnxruntime_morphizen_ep.dll" `
    --plugin_eps        "MorphiZenExecutionProvider" `
    --plugin_ep_options "config_file|$bin\morphizen_config.json" `
    -t 60 -c 1 -s -I
```

Flag-by-flag:

| Flag | Meaning |
|------|---------|
| First positional | Path to the `.onnx` model. |
| `--plugin_ep_libs "MorphiZenExecutionProvider\|<dll>"` | Tells ORT to load `<dll>` and register an EP with the given name. The name on the left of `\|` must match `--plugin_eps` exactly. |
| `--plugin_eps "MorphiZenExecutionProvider"` | Tells ORT which registered EP(s) to use for this run. |
| `--plugin_ep_options "config_file\|<json>"` | Passes per-EP configuration. The MorphiZen EP requires a `config_file` option pointing at a JSON file describing the compile passes (default: [etc/morphizen_config.json](../etc/morphizen_config.json), auto-installed to `prebuilt-local\bin\`). |
| `-t 60` | Run for 60 seconds wall-clock. |
| `-c 1` | One concurrent inference at a time (no parallelism). |
| `-s` | Show per-iteration latency statistics (P50, P90, P95, P99, average). |
| `-I` | Use sequential inputs without binding (no `-i` input-list file required). |

Output similar to (numbers will vary by hardware and run length):

```
Session creation time cost: 8.2 s
First inference time cost: 5.4 ms
Total inference time cost: 60.0 s
Total inference requests: <N>
Average inference time cost: 0.481 ms
Min Latency is 0.45 ms
Max Latency is 1.21 ms
P50 Latency is 0.48 ms
P90 Latency is 0.51 ms
P95 Latency is 0.54 ms
P99 Latency is 0.65 ms
```

The format strings (`Session creation time cost:`, etc.) are emitted by
`onnxruntime_perf_test.exe` itself; the same regexes are parsed by the
GitHub Actions CI in
[.github/workflows/gpu-perf-accuracy-test.yml:458](../.github/workflows/gpu-perf-accuracy-test.yml).

The first run in a fresh process spends 5-15 seconds on JIT compilation
inside `Session::Init` (you will see `[hipdnn-ep] compiling ...` log lines
on stderr). Subsequent invocations of the same process are instant; new
processes pay the JIT cost again unless you have caching configured.

### Why `MorphiZenExecutionProvider` (not `MorphiZenEP`)

The plugin EP name is registered in the EP DLL as
`MorphiZenExecutionProvider`
([hip-onnx-runner.cpp:872](../tools/hip-onnx-runner/hip-onnx-runner.cpp)).
Earlier versions of this guide used the short form `MorphiZenEP`; with
current EP DLLs that name is not registered, so ORT falls back to the CPU
EP without an explicit error. Use the full name `MorphiZenExecutionProvider`
to load the GPU EP.

## Smoke-test the build

To verify your build at a finer granularity than one model, run the E2E
ctest harness:

```powershell
ctest --test-dir ..\build\onnx-hipdnn-ep -R "^E2E_" --verbose
```

This auto-discovers every `.mlir` file under `test/lit/e2e/` (28 tests at
the time of writing) and verifies that each one compiles and executes
end-to-end. Each `.mlir` produces two ctest tests:

- `E2E_Compile_<name>` -- compiles the `.mlir` to a `.dll` via `hip-compiler`.
- `E2E_Execute_<name>` -- loads and runs the `.dll` via `hip-test-dll`.

See [test/e2e/README.md](../test/e2e/README.md) for full details and the LIT
test discovery convention.

To run only the LIT pass-transformation tests:

```powershell
ctest --test-dir ..\build\onnx-hipdnn-ep -R MorphizenMLIRLitTests --verbose
```

If LIT tests pass instantly without printing any test names, the venv is
likely missing `lit`. Run `pip install lit` inside the venv and reconfigure
(`. .\setup_and_configure.ps1`).

## Going further

The headline workflow above proves the build works. Once that is green, the
following recipes give you progressively richer measurements.

### Bit-for-bit correctness with hip-onnx-runner

`hip-onnx-runner.exe` runs a single ONNX model through MorphiZen EP (or CPU
only with `-n`), dumps tensors to disk in a fixed format, and compares two
dump directories element-wise via `-L`. It is the right tool for
verifying that an EP change does not regress numerical correctness against
ORT's own CPU EP.

```powershell
$bin = "C:\Users\<you>\ROCmEP\OnnxHipDNN\prebuilt-local\bin"
cd $bin

# Run on MorphiZen EP, dump outputs
.\hip-onnx-runner.exe -m C:\Users\<you>\...\conv_hybrid.onnx -d 2

# Run on CPU only, dump outputs
.\hip-onnx-runner.exe -m C:\Users\<you>\...\conv_hybrid.onnx -n -d 2

# L2-norm compare the two dump dirs
.\hip-onnx-runner.exe -L conv_hybrid_o_dump,conv_hybrid_cpu_o_dump
```

For a small fp32 conv the L2 norm should be `< 1e-5`. Larger tolerances are
expected for fp16 / bf16 graphs.

Note: `hip-onnx-runner.exe` loads the EP DLL from its current working
directory ([hip-onnx-runner.cpp:876-881](../tools/hip-onnx-runner/hip-onnx-runner.cpp)).
Run it from `<prebuilt-local>\bin` (or any directory containing
`onnxruntime_morphizen_ep.dll`); otherwise the EP DLL will not be found
and the runner exits during startup.

Runner flags (verified against [hip-onnx-runner.cpp:800-828](../tools/hip-onnx-runner/hip-onnx-runner.cpp)):

| Flag | Description |
|------|-------------|
| `-m, --model` | Path to `.onnx` model (required for inference). |
| `-n, --no-ep` | Boolean flag. CPU only; skip EP registration. Used with `-d 2` to produce a CPU reference dump for `-L` comparison. |
| `-d, --dump-level` | `0=off (default)`, `1=dump inputs to <stem>_i_dump/`, `2=dump outputs to <stem>_o_dump/`, `3=both`. With `-n`, dump dirs are suffixed `_cpu` (e.g. `<stem>_cpu_o_dump/`). |
| `-L, --l2norm` | Compare two dump dirs: `dir1,dir2` (comma-separated, no spaces). Each `.bin` filename must end with `..._<typetag>.bin` where `<typetag>` is one of `fp32`, `fp16`, `fp64`, `bf16`, `i64`, `i32`, `i16`, `i8`, `u8`, `u16`. With `-L`, `-m` is not required. |
| `-i, --input-dir` | Directory containing `input_<idx>_<name>_<type>.bin` files; if empty, the runner generates random inputs. Use this together with `-d 1` once to fix an input set, then re-use it across runs for deterministic comparison. |
| `-s, --seed` | RNG seed for random inputs (default: 42). Use this to reproduce a specific input set without writing files to disk. |
| `-o, --graph-opt-level` | Calls `session_options.SetGraphOptimizationLevel(level)` with the given level: `0=ORT_DISABLE_ALL`, `1=ORT_ENABLE_BASIC`, `2=ORT_ENABLE_EXTENDED`, `99=ORT_ENABLE_ALL`, `-1=do not call (default)`. Note: `-o` is **not** an output-directory flag; output dump dirs are auto-named based on the model stem. |
| `-p, --positive-only` | Boolean flag. Generate positive-only random inputs in `[0.1, 256.0]` (for `Sqrt` / `Reciprocal` testing where negative inputs would NaN). |

### DML EP comparison

The same `onnxruntime_perf_test.exe` can run any model through the
DirectML EP (built into ORT in [Section 4.3](#43-build-onnx-runtime-once-directml-build))
for a head-to-head latency comparison:

```powershell
& "$bin\onnxruntime_perf_test.exe" `
    .\examples\quickstart\conv_hybrid.onnx `
    -e dml `
    -C "ep.dml.disable_graph_fusion|1" `
    -t 60 -c 1 -s -I
```

`disable_graph_fusion|1` keeps the DML EP from re-fusing the graph, which
is closer to what the MorphiZen EP sees and makes the comparison fairer.

### Custom-kernel docs

If you are working on the HIP custom kernels, see
[3rd-party/custom_kernels/](../3rd-party/custom_kernels/) for
the source and CMake plumbing. The kernel ABI is documented in
[3rd-party/custom_kernels/include/hip_custom_kernels.h](../3rd-party/custom_kernels/include/hip_custom_kernels.h).

## ABI note

The prebuilt LLVM/MLIR/Protobuf/FlatBuffers archives are compiled with the
**Release `/MT`** runtime (`MultiThreaded` static CRT). Two consequences:

- Using `-DCMAKE_BUILD_TYPE=Debug` causes the project to link the Debug CRT
  (`/MTd`), which is incompatible -- you will see linker errors like
  `error LNK2038: mismatch detected for 'RuntimeLibrary'`.
- Using `-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDLL` (the `/MD` dynamic
  CRT) produces a different mismatch with the same outcome.

Build Release with `MultiThreaded` to match the prebuilt CRT. For
debuggable code, add `-DCMAKE_CXX_FLAGS="/Z7 /Od"` to the Release
configuration; this stays compatible with the prebuilt CRT.

## Updating prebuilt binaries

When new versions of LLVM/MLIR/Protobuf/FlatBuffers are published to
[wcy123/llvm-mlir-prebuilt/releases](https://github.com/wcy123/llvm-mlir-prebuilt/releases),
update the tag variables at the top of [scripts/setup-prebuilt.sh](../scripts/setup-prebuilt.sh)
and re-run it from Git Bash:

```bash
bash scripts/setup-prebuilt.sh
```

If you want a clean re-download, delete the old zip files and sentinel
files in `..\prebuilt-local\` first:

```powershell
Remove-Item ..\prebuilt-local\*.zip
Remove-Item ..\prebuilt-local\.extracted-*
```

After updating prebuilt binaries, re-run `setup_and_configure.ps1` once
to refresh CMake's cache, then `build_and_install.ps1` to rebuild.

## Troubleshooting

### CMake cannot find LLVM, MLIR, or onnxruntime

**Symptom:** `Could not find package LLVM` (or `MLIR`, `onnxruntime`) at
configure time.

**Cause:** Either step 4.2 (`scripts/setup-prebuilt.sh`) or step 4.3
(ONNX Runtime DML build) was not completed.

**Fix:** Verify both succeeded:

```powershell
ls ..\prebuilt-local\lib\cmake\llvm
ls ..\prebuilt-local\lib\cmake\onnxruntime
```

Both directories should be present. If `lib\cmake\llvm` is missing, re-run
`bash scripts/setup-prebuilt.sh`. If `lib\cmake\onnxruntime` is missing,
re-run the `cmake --install ..\..\build\onnxruntime\Release --prefix ...`
step from [Section 4.3](#43-build-onnx-runtime-once-directml-build).

### gh authentication error during setup-prebuilt.sh

**Symptom:** `gh release download` fails with `HTTP 401`.

**Fix:** Run `gh auth login` and authenticate with a GitHub account that
has access to the `wcy123/llvm-mlir-prebuilt` repository. Confirm with
`gh auth status`.

### sccache not found

**Symptom:** CMake configure error: `Could not find compiler launcher sccache`.

**Fix:** Install sccache (`winget install Mozilla.sccache`) or remove the
`-DCMAKE_C_COMPILER_LAUNCHER=sccache` / `-DCMAKE_CXX_COMPILER_LAUNCHER=sccache`
options from `setup_and_configure.ps1`.

### Runtime library mismatch (/MT vs /MTd or /MD)

**Symptom:** Linker errors like `error LNK2038: mismatch detected for
'RuntimeLibrary': value 'MT_StaticRelease' doesn't match value
'MTd_StaticDebug'`.

**Fix:** Set `-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded` and
`-DCMAKE_BUILD_TYPE=Release`, which is what `setup_and_configure.ps1`
already does. Mixed Debug+Release builds are not supported with the
prebuilts. See [ABI note](#abi-note) above.

### LIT tests pass instantly without running anything

**Symptom:** `MorphizenMLIRLitTests` completes in under one second and
reports `Passed` without printing any test names.

**Cause:** `lit` is not installed in the active Python environment, so
CMake's `find_program(lit)` either failed silently or found a different
`lit.exe` outside the venv.

**Fix:** With the venv active, `pip install lit`, then re-run
`. .\setup_and_configure.ps1` so CMake re-detects the new `lit.exe`.

### Missing DIA SDK library (diaguids.lib)

**Symptom:**
`ninja: error: 'C:/msvsn2022/DIA SDK/lib/amd64/diaguids.lib' missing`

**Cause:** The prebuilt LLVM has `C:\msvsn2022` hardcoded in
`LLVMExports.cmake` for the DIA SDK path. This is a known LLVM issue
(<https://github.com/llvm/llvm-project/issues/111829>).

**Fix:** Create a directory junction from `C:\msvsn2022` to your Visual
Studio installation:

```cmd
for /f "usebackq delims=" %i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -property installationPath`) do mklink /J C:\msvsn2022 "%i"
```

Run that from cmd (not PowerShell -- `mklink` is a cmd.exe builtin) with
admin privileges.

### vcvarsall.bat not found

**Symptom:** [setup_and_configure.ps1](../setup_and_configure.ps1) prints
`ERROR: vcvarsall.bat not found at <path>` and exits during MSVC env
setup.

**Cause:** The script first looks for `vcvarsall.bat` via `vswhere`
(`%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe`).
If `vswhere` is missing or returns no install path, the script falls back
to `C:\msvsn2022\VC\Auxiliary\Build\vcvarsall.bat`. The error means
neither path resolved.

**Fix:** Either install Visual Studio 2022 with the "Desktop development
with C++" workload (which installs `vswhere`), or create the
`C:\msvsn2022` junction described in
[Missing DIA SDK library](#missing-dia-sdk-library-diaguidslib) above.

### Activate.ps1 execution policy

**Symptom:** `.\.venv\Scripts\Activate.ps1 : File ... cannot be loaded
because running scripts is disabled on this system.`

**Fix:** Loosen the execution policy for your user once:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

Then re-open the PowerShell window and try again.

### HIP DLL load failure (amdhip64_7.dll)

**Symptom:** `The code execution cannot proceed because amdhip64_7.dll
was not found` when launching `hip-onnx-runner.exe` or
`onnxruntime_perf_test.exe`.

**Cause:** Either the TheRock SDK is not at the expected path, or
`prebuilt-local\bin` has a stale copy from an older SDK.

**Fix:**

1. Verify TheRock 7.11.0 is at the path the scripts expect:

   ```powershell
   ls C:\Users\<you>\ROCmEP\OnnxHipDNN\therock\bin\amdhip64_7.dll
   ```

2. Re-run `build_and_install.ps1`. The DLL-sync logic at lines 66-78
   force-copies the SDK DLLs into `prebuilt-local\bin` if sizes differ.

3. Confirm `<therock>\bin` is on `PATH` (the script prepends it; check
   with `$env:PATH -split ';' | Select-String therock`).

### Plugin EP not loading (perf_test silently runs on CPU)

**Symptom:** `onnxruntime_perf_test.exe` reports plausible-looking latency
numbers but the GPU is idle (check with Task Manager Performance tab) and
no `[hipdnn-ep]` log lines appear on stderr.

**Cause:** A typo in the `--plugin_eps` argument. The string passed to
`--plugin_eps` should match the left-hand side of `--plugin_ep_libs`
exactly, and both should read `MorphiZenExecutionProvider`. ORT silently
falls back to the CPU EP when the requested plugin EP is not registered.

**Fix:** Use the literal name `MorphiZenExecutionProvider` (full, not
`MorphiZenEP`):

```powershell
& "$bin\onnxruntime_perf_test.exe" `
    .\model.onnx `
    --plugin_ep_libs    "MorphiZenExecutionProvider|$bin\onnxruntime_morphizen_ep.dll" `
    --plugin_eps        "MorphiZenExecutionProvider" `
    --plugin_ep_options "config_file|$bin\morphizen_config.json" `
    -t 60 -c 1 -s -I
```

### onnxruntime_morphizen_ep.dll not found by hip-onnx-runner

**Symptom:** `EP library not found: onnxruntime_morphizen_ep.dll. Set
MORPHIZEN_VITISAI_EP or use -n.` when running `hip-onnx-runner.exe`.

**Cause:** `hip-onnx-runner.exe` loads the EP DLL from its **current
working directory**, not from `PATH`
([hip-onnx-runner.cpp:876-881](../tools/hip-onnx-runner/hip-onnx-runner.cpp)).

**Fix:** `cd` into `prebuilt-local\bin` before running the executable, or
use an absolute path to `-m` and run from any directory containing the
DLL. The simplest reliable invocation:

```powershell
cd C:\Users\<you>\ROCmEP\OnnxHipDNN\prebuilt-local\bin
.\hip-onnx-runner.exe -m C:\absolute\path\to\model.onnx -d 2
```
