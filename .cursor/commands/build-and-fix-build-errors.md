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
1. Configure with CMake (includes MSVC environment setup):
   ```powershell
   cmd /c "C:\msvsn2022\VC\Auxiliary\Build\vcvarsall.bat x64 && set VAI_RT_WORKSPACE=c:/Develop/m/source && set VAI_RT_BUILD_DIR=C:/Develop/m/build && set VAI_RT_PREFIX=C:/Develop/m/local && cd C:\Develop\m\source\MorphiZen && cmake -G Ninja -B ../../build/morphizen.ninja -S . -DCMAKE_BUILD_TYPE=Debug -DCMAKE_MSVC_RUNTIME_LIBRARY=""MultiThreaded$<$<CONFIG:Debug>:Debug>"" -Dmorphizen_ENABLE_UNIT_TEST=ON -Dmorphizen_ENABLE_ORT_BRIDGE=ON -Dmorphizen_ENABLE_MLIR_BACKEND=ON -DCMAKE_PREFIX_PATH=../../local"
   ```

2. Build the project (includes MSVC environment setup):
   ```powershell
   cmd /c "C:\msvsn2022\VC\Auxiliary\Build\vcvarsall.bat x64 && set VAI_RT_WORKSPACE=c:/Develop/m/source && set VAI_RT_BUILD_DIR=C:/Develop/m/build && set VAI_RT_PREFIX=C:/Develop/m/local && cmake --build ""C:/Develop/m/build/morphizen.ninja"" --parallel"
   ```

3. If there are any build errors:
   - Read the error messages
   - Fix the errors in the source files
   - Re-run the build command

## Expected Outputs:
- `onnxruntime_morphizen_ep.dll` - Main MorphiZen library
- `morphizen-graph-opt.exe` - Graph optimization tool
- `morphizen-unit-tests.exe` - Unit test executable
- `ort-bridge-test.exe` - ORT bridge tests
- Various other tools and libraries

4. Commit changes after successful build:
   ```powershell
   # Check git status to see what changed
   git status

   # Stage the modified files
   git add <modified-files>

   # Commit with a descriptive message.
   git commit -m "refactor: <brief summary>

   - <detailed change 1>
   - <detailed change 2>
   - <detailed change 3>

   <optional additional context>"
   ```
   **Note**: Only commit after verifying the build succeeds. Pre-commit hooks will automatically run linters and formatters, modify files if needed, please accept the changes made by pre-commit.

5. Run unit tests in parallel:
   ```powershell
   ctest --test-dir "C:/Develop/m/build/morphizen.ninja" -C Debug --output-on-failure --timeout 600 -j $(Get-Command -Name ctest | ForEach-Object { [Environment]::ProcessorCount })
   ```
   Or specify the number of parallel jobs explicitly (e.g., 8):
   ```powershell
   ctest --test-dir "C:/Develop/m/build/morphizen.ninja" -C Debug --output-on-failure --timeout 600 -j 8
   ```
   Alternatively, you may run test executables using parallel shell jobs for fine-grained control.




## Notes:
- The project uses Ninja generator for faster builds
- Dependencies are automatically fetched via CMake FetchContent
- Build output goes to `C:/Develop/m/build/morphizen.ninja/bin/`
