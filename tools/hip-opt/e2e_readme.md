# End-to-End POC: Compile and Run a Transformer in HIP Dialect

This guide walks through compiling `test_e2e.mlir` (a self-contained 2-layer
transformer) into a native executable that runs every HIP dialect op on the GPU.

---

## Prerequisites

| Dependency | Path | Notes |
|---|---|---|
| Visual Studio 2022 | `vcvarsall.bat x64` | C++ compiler + linker |
| LLVM/MLIR (built) | `gpu/llvm-project/build/Debug/bin/` | `mlir-translate`, `llc` |
| TheRock ROCm dist | `gpu/TheRock/build/dist/rocm/` | HIP runtime + library DLLs + headers |
| hip-opt (built) | `tools/hip-opt/build/Debug/hip-opt.exe` | Custom MLIR compiler driver |
| Conda env `llvm` | `conda activate llvm` | For LLVM tool DLL dependencies |

---

## 0. Building TheRock (hipBLASLt + MIOpen)

TheRock provides the ROCm SDK: HIP runtime, hipBLASLt, and MIOpen. Source is at
`gpu/TheRock`.

### 0.1 Install build tools

```cmd
winget install Git.Git -e --source winget --custom "/o:PathOption=CmdTools"
winget install cmake ninja-build.ninja ccache python strawberryperl bloodrock.pkg-config-lite
winget install --id Iterative.DVC --silent --accept-source-agreements
```

```cmd
git config --global core.symlinks true
git config --global core.longpaths true
```

### 0.2 Clone and fetch sources

```cmd
conda create -n llvm python=3.12 pip cmake ninja pkg-config -y
conda activate llvm

git clone https://github.com/ROCm/TheRock.git
cd TheRock
pip install -r requirements.txt
python ./build_tools/fetch_sources.py
```

### 0.3 Configure (minimal build for this project)

Open an **x64 Native Tools Command Prompt for VS 2022**, then:

```cmd
cmake -B build -GNinja --preset windows-release ^
  -DTHEROCK_AMDGPU_TARGETS=gfx1151 ^
  -DTHEROCK_ENABLE_ALL=OFF ^
  -DTHEROCK_ENABLE_BLAS=ON ^
  -DTHEROCK_ENABLE_MIOPEN=ON ^
  -DTHEROCK_ENABLE_HIPDNN=ON ^
  -DTHEROCK_ENABLE_HIPBLASLT_PLUGIN=ON ^
  -DTHEROCK_DIST_AMDGPU_FAMILIES=gfx1151
```

Replace `gfx1151` with your GPU target (run `hipinfo` or `offload-arch` to find it).

| Flag | What it builds |
|---|---|
| `THEROCK_ENABLE_BLAS=ON` | hipBLASLt (`libhipblaslt.dll`) |
| `THEROCK_ENABLE_MIOPEN=ON` | MIOpen (`MIOpen.dll`) -- normalization, activation, tensor ops |
| `THEROCK_ENABLE_HIPDNN=ON` | hipDNN backend (`hipdnn_backend.dll`) |
| `THEROCK_ENABLE_HIPBLASLT_PLUGIN=ON` | hipBLASLt engine plugin for hipDNN matmul |

HIP runtime and compiler are implicitly enabled as dependencies.

### 0.4 Build

```cmd
cmake --build build --target therock-dist
```

Output at:

```
TheRock/build/dist/rocm/
  bin/    amdhip64_7.dll, libhipblaslt.dll, MIOpen.dll, hipdnn_backend.dll
  include/    hip/, hipblaslt/, miopen/, hipdnn/
  lib/    (import libraries)
```

### 0.5 Set environment variable

```cmd
set THEROCK_DIST=C:\Users\chiz\work\gpu\TheRock\build\dist\rocm
```

### Build troubleshooting

- **`No engine configurations available`**: Rebuild with `-DTHEROCK_ENABLE_HIPBLASLT_PLUGIN=ON`
- **`Could NOT find PkgConfig`**: `conda install -c conda-forge pkg-config`
- **Fortran compiler missing**: `conda install -n llvm -c conda-forge gfortran`
- See main [README.md](../README.md) for more troubleshooting

---

## 1. Build hip-opt

Only needed once (or after modifying `HipOps.td`, `HipToLLVM.cpp`, or `hip-opt.cpp`).

```bat
cd tools/hip-opt/build
cmake ..
cmake --build . --config Debug --target hip-opt
```

---

## 2. Quick Start

```bat
cd tools/hip-opt
run_e2e_pipeline.bat
```

This runs all 7 steps below. If any step fails it stops with the step number.

---

## 3. Step-by-Step Pipeline

### Step 1: Lower HIP dialect to LLVM dialect

```bat
hip-opt.exe test_e2e.mlir ^
  --convert-hip-to-llvm ^
  --convert-scf-to-cf ^
  --convert-func-to-llvm ^
  --convert-cf-to-llvm ^
  --reconcile-unrealized-casts ^
  -o e2e_lowered.mlir
```

Pass order:
1. `--convert-hip-to-llvm` -- HIP ops -> `llvm.call @hip_<op>(...)`, graph regions inlined
2. `--convert-scf-to-cf` -- `scf.for` -> `cf.br` / `cf.cond_br`
3. `--convert-func-to-llvm` -- `func.func` -> `llvm.func`
4. `--convert-cf-to-llvm` -- `cf.br` -> `llvm.br`
5. `--reconcile-unrealized-casts` -- clean up type casts

### Step 2: Translate to LLVM IR

```bat
mlir-translate.exe e2e_lowered.mlir --mlir-to-llvmir -o e2e.ll
```

### Step 3: Compile to object file

```bat
llc.exe e2e.ll -filetype=obj -o e2e.obj
```

### Step 4: Generate import libraries

Only needed once per DLL. The build script handles this automatically.

```bat
REM amdhip64
dumpbin /EXPORTS "%THEROCK_DIST%\bin\amdhip64_7.dll" | findstr /R "^  *[0-9]" > _exports.txt
echo LIBRARY amdhip64_7.dll > amdhip64.def
echo EXPORTS >> amdhip64.def
for /f "tokens=4" %%a in (_exports.txt) do echo   %%a >> amdhip64.def
lib /def:amdhip64.def /out:amdhip64.lib /machine:x64

REM hipblaslt (same procedure)
dumpbin /EXPORTS "%THEROCK_DIST%\bin\libhipblaslt.dll" | findstr /R "^  *[0-9]" > _exports.txt
echo LIBRARY libhipblaslt.dll > hipblaslt.def
echo EXPORTS >> hipblaslt.def
for /f "tokens=4" %%a in (_exports.txt) do echo   %%a >> hipblaslt.def
lib /def:hipblaslt.def /out:hipblaslt.lib /machine:x64

REM MIOpen (same procedure, when available)
dumpbin /EXPORTS "%THEROCK_DIST%\bin\MIOpen.dll" | findstr /R "^  *[0-9]" > _exports.txt
echo LIBRARY MIOpen.dll > MIOpen.def
echo EXPORTS >> MIOpen.def
for /f "tokens=4" %%a in (_exports.txt) do echo   %%a >> MIOpen.def
lib /def:MIOpen.def /out:MIOpen.lib /machine:x64
```

### Step 5: Compile the runtime

**Stubs only** (no library calls, all ops print + hipMemset):
```bat
cl.exe /c /EHsc /std:c++17 /D__HIP_PLATFORM_AMD__ ^
  /I"%THEROCK_DIST%\include" ^
  ops_runtime\all_runtime.cpp /Fo:runtime.obj
```

**With hipBLASLt** (real matmul):
```bat
cl.exe /c /EHsc /std:c++17 /D__HIP_PLATFORM_AMD__ /DHAS_HIPBLASLT ^
  /I"%THEROCK_DIST%\include" ^
  ops_runtime\all_runtime.cpp /Fo:runtime.obj
```

**With hipBLASLt + MIOpen** (real matmul + real norm/activation/tensor ops):
```bat
cl.exe /c /EHsc /std:c++17 /D__HIP_PLATFORM_AMD__ /DHAS_HIPBLASLT /DHAS_MIOPEN ^
  /I"%THEROCK_DIST%\include" ^
  ops_runtime\all_runtime.cpp /Fo:runtime.obj
```

Runtime function implementations:

| Function | Stub mode | `/DHAS_HIPBLASLT` | `/DHAS_MIOPEN` |
|---|---|---|---|
| `hip_hipblaslt_matmul` | print + sync | `hipblasLtMatmul` | -- |
| `hip_miopen_rms_norm` | print + sync | -- | `miopenT5LayerNormForward` |
| `hip_miopen_skip_rms_norm` | print + sync | -- | `miopenAddLayerNormForward` (T5) |
| `hip_miopen_add` | print + sync | -- | `miopenOpTensor(Add)` |
| `hip_miopen_mul` | print + sync | -- | `miopenOpTensor(Mul)` |
| `hip_miopen_rope` | print + sync | -- | experimental / stub |
| `hip_gather` | `hipMemset(0)` | `hipMemset(0)` | `hipMemset(0)` |
| `hip_silu` | `hipMemset(0)` | `hipMemset(0)` | `hipMemset(0)` |
| `hip_gqa` | `hipMemset(0)` | `hipMemset(0)` | `hipMemset(0)` |

### Step 6: Compile main driver

```bat
cl.exe /c /EHsc /std:c++17 main_e2e.cpp /Fo:main_e2e.obj
```

### Step 7: Link and run

**Stubs only:**
```bat
link.exe e2e.obj runtime.obj main_e2e.obj ^
  /LIBPATH:. amdhip64.lib /out:test_e2e.exe
```

**With hipBLASLt:**
```bat
link.exe e2e.obj runtime.obj main_e2e.obj ^
  /LIBPATH:. amdhip64.lib hipblaslt.lib /out:test_e2e.exe
```

**With hipBLASLt + MIOpen:**
```bat
link.exe e2e.obj runtime.obj main_e2e.obj ^
  /LIBPATH:. amdhip64.lib hipblaslt.lib MIOpen.lib /out:test_e2e.exe
```

**Run:**
```bat
set PATH=%THEROCK_DIST%\bin;%PATH%
test_e2e.exe
```

Expected stderr output:
```
[hip] create_handle
[hip] alloc 2048 bytes -> 0x...
...
[gather] executed (out=0x...)
[miopen.rms_norm] executed (in=0x..., w=0x..., out=0x...)
[hipblaslt.matmul] executed (A=0x..., B=0x..., C=0x...)
[miopen.rope] executed (q=0x..., k=0x..., pos=0)
[gqa] executed (out=0x..., layer=0, pos=0, seq=0)
...
[hip] free 0x...
[hip] destroy_handle
```

---

## Files

| File | Purpose |
|---|---|
| `test_e2e.mlir` | Self-contained transformer (2 layers, static shapes, all HIP ops) |
| `main_e2e.cpp` | C driver: calls `@run()`, prints status |
| `ops_runtime/all_runtime.cpp` | Unified runtime: stubs + `#ifdef` real library calls |
| `run_e2e_pipeline.bat` | Automated build script (steps 1-7) |

---

## Compilation Flow

```
test_e2e.mlir
    |  hip-opt (--convert-hip-to-llvm --convert-scf-to-cf
    |           --convert-func-to-llvm --convert-cf-to-llvm
    |           --reconcile-unrealized-casts)
    v
e2e_lowered.mlir    (LLVM dialect, all hip.* ops -> llvm.call @hip_*)
    |  mlir-translate (--mlir-to-llvmir)
    v
e2e.ll              (LLVM IR)
    |  llc (-filetype=obj)
    v
e2e.obj             (native object)
    +--- runtime.obj  (all_runtime.cpp, compiled with MSVC)
    +--- main_e2e.obj (main_e2e.cpp, compiled with MSVC)
    |  link.exe (+ amdhip64.lib [+ hipblaslt.lib] [+ MIOpen.lib])
    v
test_e2e.exe        (runnable on AMD GPU)
```
