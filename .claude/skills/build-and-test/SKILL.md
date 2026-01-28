<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->

---
name: build-and-test
description: Configure, build, and test the MorphiZen project using CMake and CTest
allowed-tools: [Bash, Read, Grep]
---

# Build and Test MorphiZen Project

This skill configures, builds, and tests the MorphiZen project using CMake and CTest commands directly in git-bash. It works on both Windows and Linux platforms.

The skill automatically determines the appropriate build type (clean vs incremental) based on the current build state.

## When to Use This Skill

- **Clean build from scratch**: Configure and build the entire project with fresh CMake cache
- **Incremental build**: Rebuild only changed files without reconfiguring
- **Test-only**: Run CTest without rebuilding
- **Build after dependency updates**: Rebuild when dependencies in `../../local` have been updated

## Prerequisites & Dependencies

### For End Users
**ONNX Runtime MUST be built** and installed to `../../local` - it cannot be auto-fetched by CMake.

Other dependencies (protobuf, gtest, glog, LLVM) can be auto-fetched by CMake automatically.

### For Developers (Optional - Faster Builds)
Pre-building ALL dependencies to `../../local` significantly improves build times.

See **[docs/developer-guide.md](../../../docs/developer-guide.md)** for:
- Prerequisites installation (Git, CMake, MSVC, Ninja, Python)
- Step-by-step dependency build instructions (protobuf, gtest, glog, ONNX Runtime, LLVM, etc.)
- Troubleshooting common issues

**Quick summary:**
- **ONNX Runtime** - REQUIRED, must be built manually (cannot be auto-fetched)
- **Other dependencies** - Optional to pre-build for faster builds, or let CMake auto-fetch
- All pre-built dependencies must be installed to `../../local`
- All must use: `-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded$<$<CONFIG:Debug>:Debug>`

## Build Configuration

### Build Settings

- **Build Directory**: `../../build/$(basename $PWD)`
- **Install Prefix**: `../../local` (shared install directory for dependencies)
- **Build Type**: Debug
- **Shared Libraries**: OFF (static linking)
- **Runtime Library**: Static `/MTd` (Debug) via `CMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded$<$<CONFIG:Debug>:Debug>`

### Critical CMake Flags

```bash
-DBUILD_SHARED_LIBS=OFF                                                    # Static linking
-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded$<$<CONFIG:Debug>:Debug>        # Static runtime /MTd (Debug) or /MT (Release)
-DCMAKE_BUILD_TYPE=Debug                                                   # Debug configuration
-DCMAKE_PREFIX_PATH=../../local                                            # Find dependencies
-DCMAKE_EXPORT_COMPILE_COMMANDS=ON                                         # Generate compile_commands.json
-Dmorphizen_ENABLE_UNIT_TEST=ON                                            # Enable unit tests
```

## Command Templates

### Clean Build (from scratch)

```bash
# NOTE: CMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded$<$<CONFIG:Debug>:Debug>
# This sets /MTd (static runtime) for Debug, /MT for Release

cmake -S . -B ../../build/$(basename $PWD) \
  -DBUILD_SHARED_LIBS=OFF \
  "-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded$<$<CONFIG:Debug>:Debug>" \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_PREFIX_PATH=../../local \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -Dmorphizen_ENABLE_UNIT_TEST=ON \
  --fresh

cmake --build ../../build/$(basename $PWD) --config Debug --parallel

ctest --test-dir ../../build/$(basename $PWD) -C Debug --output-on-failure --timeout 600
```

### Incremental Build

```bash
cmake --build ../../build/$(basename $PWD) --config Debug --parallel

ctest --test-dir ../../build/$(basename $PWD) -C Debug --output-on-failure --timeout 600
```

### Test Only

```bash
ctest --test-dir ../../build/$(basename $PWD) -C Debug --output-on-failure --timeout 600
```

## Instructions for Claude

When this skill is invoked, follow these steps:

### Step 0: Verify Git Workflow Compliance (CRITICAL)

**BEFORE making ANY changes to code or files, enforce `.clinerules/git-rules.md`:**

1. **Check current branch:**
   ```bash
   git branch --show-current
   ```

2. **If on `main` branch:**
   - **STOP immediately** - Do NOT proceed with any file changes
   - Inform user: "You are on the `main` branch. Cannot make changes directly to main."
   - Instruct user to create a feature branch: `git checkout -b feature/<descriptive-name>`
   - Wait for user to create branch before proceeding

3. **If on a feature branch:**
   - Proceed to next steps
   - Remember: Commit and push frequently with clear, professional commit messages
   - No mentions of AI/tools in commits or PRs

4. **When making changes during build/fix cycles:**
   - After fixing build errors or making code changes, commit immediately
   - Use descriptive commit messages: `fix:`, `feat:`, `refactor:`, etc.
   - Push to fork: `git push fork <branch-name>`
   - Follow `.clinerules/git-rules.md` for all git operations

**This step is MANDATORY before any file modifications.**

### Step 1: Check MSVC Environment (Windows only - lightweight check)

On Windows, perform a quick check for the MSVC compiler environment:

1. Run a lightweight check:
   ```bash
   which cl.exe 2>/dev/null
   ```

2. If `cl.exe` is not found:
   - Stop the build process
   - Inform user: "MSVC compiler not found. Please launch git-bash from an MSVC Developer Command Prompt, then restart Claude Code."
   - Provide instructions:
     - Open "Developer Command Prompt for VS XXXX" (where XXXX is your Visual Studio version: 2019, 2022, 2026, etc.)
     - Run: `bash`
     - Navigate to project directory
     - Launch Claude Code
     - Run `/build-and-test` again

3. If `cl.exe` is found:
   - Proceed to dependency checking

**Note**: Other tools (git, cmake, ninja) are NOT checked upfront - they will fail naturally with clear error messages if missing when their commands are executed.

### Step 2: Check Dependencies

1. **Check ONNX Runtime (REQUIRED - cannot be auto-fetched):**
   - Check for: `../../local/lib/cmake/onnxruntime/*.cmake`
   - If ONNX Runtime is missing:
     - **STOP the build process**
     - Inform user: "ONNX Runtime not found in ../../local. It MUST be built manually (cannot be auto-fetched by CMake)."
     - Guide user: "See docs/developer-guide.md for ONNX Runtime build instructions."
     - Remind: "Use: -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded$<$<CONFIG:Debug>:Debug>"
     - Do NOT proceed until ONNX Runtime is installed

2. **Check other dependencies (optional - can be auto-fetched):**
   - Check for CMake config files:
     - LLVM: `../../local/lib/cmake/llvm/*.cmake`
     - gtest: `../../local/lib/cmake/GTest/*.cmake`
     - glog: `../../local/lib/cmake/glog/*.cmake`
     - protobuf: `../../local/lib/cmake/protobuf/*.cmake`

3. **If some optional dependencies are missing:**
   - Inform user: "Some dependencies not found in ../../local. CMake will auto-fetch them (slower)."
   - Suggest: "For faster builds, you can pre-build dependencies. See docs/developer-guide.md"
   - Ask user: "Continue with auto-fetch or exit to pre-build dependencies?"
     - If continue: Proceed to build (CMake will auto-fetch)
     - If exit: Stop and guide them to docs/developer-guide.md

4. **If all dependencies are present (including ONNX Runtime):**
   - Inform user: "All dependencies found in ../../local. Using pre-built dependencies for faster build."
   - Proceed to build

### Step 3: Determine Build Type

Automatically determine the build type based on the current state:

1. Check if build directory exists: `../../build/$(basename $PWD)`
2. Check if `CMakeCache.txt` exists in the build directory

**Decision logic:**
- **If build directory doesn't exist** → Clean build (configure + build + test)
- **If build directory exists but no `CMakeCache.txt`** → Clean build (configure + build + test)
- **If `CMakeCache.txt` exists** → Incremental build (build + test)

**Only ask the user if:**
- The previous build failed (CMake configuration error, build error, etc.)
- There's a specific reason to reconfigure (e.g., user says "reconfigure" or "clean build")

### Step 4: Execute Build

Based on the automatically determined build type from Step 3:

1. **Clean build**:
   - Run the "Clean Build" command template
   - Parse CMake output for configuration errors
   - Parse build output for compilation errors
   - Parse CTest output for test results

2. **Incremental build**:
   - Run the "Incremental Build" command template
   - Parse build output for compilation errors
   - Parse CTest output for test results

3. **Test only** (if user specifically requests):
   - Run the "Test Only" command template
   - Parse CTest output for test results

**Output parsing:**
- Show configuration progress and any warnings
- Show build progress (files compiled, linking status)
- Show clear error messages if build fails
- Show test results summary (passed/failed tests)
- Highlight failed tests with their output

### Step 5: Provide Remediation Guidance

If errors occur, provide specific guidance:

#### Missing ONNX Runtime

**Symptoms:**
- CMake error: "Could not find package onnxruntime" or similar
- Build fails looking for ONNX Runtime

**Solution:**
- ONNX Runtime MUST be built manually (cannot be auto-fetched)
- See `docs/developer-guide.md` section 7 for ONNX Runtime build instructions
- Ensure you use: `-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded$<$<CONFIG:Debug>:Debug>`

#### Missing Other Dependencies

**Symptoms:**
- CMake error: "Could not find package" for protobuf, gtest, glog, or LLVM

**Solution:**
- These dependencies can be auto-fetched by CMake (may be slower)
- OR you can pre-build them to `../../local` for faster builds
- See `docs/developer-guide.md` for pre-build instructions
- Ensure all use: `-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded$<$<CONFIG:Debug>:Debug>`

#### Runtime Library Mismatch

**Symptoms:**
- Link error: `LNK2038: mismatch detected for 'RuntimeLibrary'`
- Link error: `LNK2005: symbol already defined in libcmt.lib`

**Solution:**
1. Identify the problematic dependency from the linker error message
2. Rebuild only that dependency with correct CMAKE_MSVC_RUNTIME_LIBRARY
3. See `docs/developer-guide.md` for detailed rebuild instructions

#### Build Errors (Compiler/Linker)

**Symptoms:**
- Compiler errors: syntax errors, missing headers, undefined symbols
- Linker errors: undefined references, multiple definitions

**Solution:**
1. Show the compiler error message with file and line number
2. Read the affected file to show context (use Read tool)
3. Suggest potential fixes based on error type

**IMPORTANT: If fixing code errors:**
- **MUST** follow `.clinerules/git-rules.md` - verify you're on a feature branch first
- After making fixes, commit immediately with descriptive message (e.g., `fix: resolve undefined symbol in ...`)
- Push to fork: `git push fork <branch-name>`
- Never commit directly to `main` branch

#### Test Failures

**Symptoms:**
- CTest reports: "X% tests passed, Y tests failed out of Z"

**Solution:**
1. Show which tests failed with their names
2. Show test output (CTest provides this with `--output-on-failure`)
3. Suggest running individual tests for debugging:
   ```bash
   ctest --test-dir ../../build/$(basename $PWD) -C Debug -R <test_name> --verbose
   ```

#### MSVC Environment Errors

**Symptoms:**
- "Cannot open include file: 'cstddef'" or other standard library headers
- "cl.exe not found" or compiler not available

**Solution:**
```
Launch git-bash from an MSVC Developer Command Prompt:
1. Open "Developer Command Prompt for VS XXXX" (where XXXX is your Visual Studio version: 2019, 2022, 2026, etc.)
2. Run: bash
3. Navigate to project directory
4. Run Claude Code and invoke this skill again

For more details, see docs/developer-guide.md
```

## Expected Outputs

After a successful build, the following files should be present in `../../build/$(basename $PWD)/bin/`:

- `onnxruntime_vitisai_ep.dll` - Main MorphiZen library (ONNX Runtime Execution Provider)
- `morphizen-unit-tests.exe` - Unit test executable
- `ort-bridge-test.exe` - ORT bridge tests
- `morphizen-graph-opt.exe` - Graph optimization tool
- Various other tools and executables

Verify outputs:
```bash
ls ../../build/$(basename $PWD)/bin/
```

## Additional Notes

- **Parallel builds**: `--parallel` flag uses all available CPU cores
- **Test timeout**: CTest uses 600-second timeout to prevent hanging tests
- **Compile commands**: `CMAKE_EXPORT_COMPILE_COMMANDS=ON` generates `compile_commands.json` for IDE integration
- **Fresh configure**: `--fresh` flag removes existing CMake cache

**For detailed documentation**: See `docs/developer-guide.md`
