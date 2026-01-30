<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
Please build the MorphiZen project and fix any build errors that occur.

## Build Settings (Remember these):
- **Build Type**: Debug
- **Environment Variables**:
  - `VAI_RT_WORKSPACE=c:/Develop/m/source`
  - `VAI_RT_BUILD_DIR=C:/Develop/m/build`
  - `VAI_RT_PREFIX=C:/Develop/m/local`

## Build Steps:

1. Set environment variables:
   ```bash
   export VAI_RT_WORKSPACE="c:/Develop/m/source"
   export VAI_RT_BUILD_DIR="C:/Develop/m/build"
   export VAI_RT_PREFIX="C:/Develop/m/local"
   ```

2. Configure with CMake:
   ```bash
   cmake -G Ninja -B ../../build/morphizen.ninja -S . \
     -DCMAKE_BUILD_TYPE=Debug \
     -DCMAKE_MSVC_RUNTIME_LIBRARY="MultiThreaded$<$<CONFIG:Debug>:Debug>" \
     -Dmorphizen_ENABLE_UNIT_TEST=ON \
     -Dmorphizen_ENABLE_ORT_BRIDGE=ON \
     -Dmorphizen_ENABLE_MLIR_BACKEND=ON \
     -DCMAKE_PREFIX_PATH=../../local
   ```

3. Build the project:
   ```bash
   cmake --build "C:/Develop/m/build/morphizen.ninja" --config Debug --parallel
   ```

**Note**: For Git Bash users, ensure bash is launched from an MSVC Developer Command Prompt to inherit the MSVC environment.

### Fixing Build Errors:

If there are any build errors:
- Read the error messages
- Fix the errors in the source files
- Re-run the build command

## Expected Outputs:
- `onnxruntime_morphizen_ep.dll` - Main MorphiZen library
- `morphizen-graph-opt.exe` - Graph optimization tool
- `morphizen-unit-tests.exe` - Unit test executable
- `ort-bridge-test.exe` - ORT bridge tests
- Various other tools and libraries

## Notes:
- The project uses Ninja generator for faster builds
- Dependencies are automatically fetched via CMake FetchContent
- Build output goes to `C:/Develop/m/build/morphizen.ninja/bin/`

## Verifying Build Outputs:
Use PowerShell cmdlets (NOT CMD-style commands) to list build outputs:
```powershell
# CORRECT - use Get-ChildItem
Get-ChildItem "C:\Develop\m\build\morphizen.ninja\bin" -Name

# WRONG - do NOT use CMD-style dir /b (causes errors in PowerShell)
# dir "C:\Develop\m\build\morphizen.ninja\bin\" /b
```
