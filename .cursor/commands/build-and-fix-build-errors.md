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
- **CMake Preset**: "Morphizen Ninja"

## Build Steps:
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

4. If there are any build errors:
   - Read the error messages
   - Fix the errors in the source files
   - Re-run the build command

## Expected Outputs:
- `onnxruntime_vitisai_ep.dll` - Main MorphiZen library
- `morphizen-graph-opt.exe` - Graph optimization tool
- `morphizen-unit-tests.exe` - Unit test executable
- `ort-bridge-test.exe` - ORT bridge tests
- Various other tools and libraries

5. Commit changes after successful build:
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

6. Run unit tests in parallel:
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
- The preset includes `morphizen_DEMO_DIR` and other configurations from CMakePresets.json
