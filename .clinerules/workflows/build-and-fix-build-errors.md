<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
Please build the MorphiZen project and fix any build errors that occur.

## CRITICAL: MSVC Environment Setup for Bash

**BEFORE ANY BUILD COMMAND in Git Bash**, you MUST set up MSVC environment:

```bash
source tools/setup_msvc_env_bash.sh
```

This is required for:
- CMake configuration (detects compiler)
- Building (cl.exe, link.exe)
- Running tests

**Pattern for first command**:
```bash
source tools/setup_msvc_env_bash.sh && cmake --preset "Morphizen Ninja"
```

Environment persists for all subsequent Bash commands in the session.

**Alternative: Use wrapper scripts** (auto-setup):
```bash
tools/cmake-msvc --preset "Morphizen Ninja"
tools/build-msvc "C:/Develop/m/build/morphizen.ninja" --config Debug --parallel
```

## Build Settings (Remember these):
- **Build Type**: Debug
- **Environment Variables**:
  - `VAI_RT_WORKSPACE=c:/Develop/m/source`
  - `VAI_RT_BUILD_DIR=C:/Develop/m/build`
  - `VAI_RT_PREFIX=C:/Develop/m/local`
- **CMake Preset**: "Morphizen Ninja"

## Build Steps:

### For Git Bash (Recommended):

1. Setup MSVC environment (FIRST COMMAND):
   ```bash
   source tools/setup_msvc_env_bash.sh
   ```

2. Configure with CMake preset:
   ```bash
   cmake --preset "Morphizen Ninja"
   ```

3. Build the project:
   ```bash
   cmake --build "C:/Develop/m/build/morphizen.ninja" --config Debug --parallel
   ```

### For PowerShell:

1. Set environment variables:
   ```powershell
   $env:VAI_RT_WORKSPACE = "c:/Develop/m/source"
   $env:VAI_RT_BUILD_DIR = "C:/Develop/m/build"
   $env:VAI_RT_PREFIX = "C:/Develop/m/local"
   ```

2. Configure with CMake preset:
   ```powershell
   cd C:\Develop\m\source\MorphiZen
   cmake --preset "Morphizen Ninja"
   ```

3. Build the project:
   ```powershell
   cmake --build "C:/Develop/m/build/morphizen.ninja" --config Debug --parallel
   ```

### Fixing Build Errors:

If there are any build errors:
- Read the error messages
- Fix the errors in the source files
- Re-run the build command

## Expected Outputs:
- `onnxruntime_vitisai_ep.dll` - Main MorphiZen library
- `morphizen-graph-opt.exe` - Graph optimization tool
- `morphizen-unit-tests.exe` - Unit test executable
- `ort-bridge-test.exe` - ORT bridge tests
- Various other tools and libraries

## Notes:
- The project uses Ninja generator for faster builds
- Dependencies are automatically fetched via CMake FetchContent
- Build output goes to `C:/Develop/m/build/morphizen.ninja/bin/`
- The preset includes `morphizen_DEMO_DIR` and other configurations from CMakePresets.json

## MSVC Environment Setup (IMPORTANT):
If the build fails with errors like "Cannot open include file: 'cstddef'" or other missing standard headers, you need to initialize the Visual Studio environment first:

```cmd
cmd /c "call ""C:\msvsn2022\VC\Auxiliary\Build\vcvars64.bat"" && cd /d C:\Develop\m\source\MorphiZen && cmake --build ""C:/Develop/m/build/morphizen.ninja"" --config Debug --parallel"
```

This sets up the MSVC compiler environment including:
- Standard library include paths
- Windows SDK paths
- Required compiler environment variables

## Verifying Build Outputs:
Use PowerShell cmdlets (NOT CMD-style commands) to list build outputs:
```powershell
# CORRECT - use Get-ChildItem
Get-ChildItem "C:\Develop\m\build\morphizen.ninja\bin" -Name

# WRONG - do NOT use CMD-style dir /b (causes errors in PowerShell)
# dir "C:\Develop\m\build\morphizen.ninja\bin\" /b
```
