<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->

# MorphiZen Developer Guide

This guide provides detailed instructions for building MorphiZen dependencies and setting up your development environment.

## Dependencies Overview

### ONNX Runtime (REQUIRED)
**ONNX Runtime MUST be built manually** - it cannot be auto-fetched by CMake. You must build and install it to `../../local` before building MorphiZen.

### Other Dependencies (Optional Optimization)
Other dependencies (protobuf, gtest, glog, LLVM, etc.) can be:
- **Auto-fetched by CMake** - Slower but works automatically
- **Pre-built to `../../local`** - Faster builds, recommended for developers

### Why Pre-build Dependencies?

Pre-building ALL dependencies to `../../local` provides:
- **Faster incremental builds** - Dependencies are built once, reused across rebuilds
- **Consistent environment** - All dependencies use the same runtime library settings (/MTd for Debug)
- **Offline builds** - No internet required once dependencies are built
- **Developer productivity** - Significant time savings during development

## Prerequisites

Before building MorphiZen and its dependencies, ensure the following tools are installed.

**Note**: The `/build-and-test` skill does not check all prerequisites automatically. Only the MSVC compiler (on Windows) is checked proactively. Missing tools will be detected when their commands fail with clear error messages.

### Quick Install (Windows Command Line)

You can install most prerequisites using command-line package managers on Windows:

#### Option 1: Using winget (Built into Windows 10/11)

```powershell
# Open PowerShell as Administrator and run:
winget install Git.Git
winget install Kitware.CMake
winget install Ninja-build.Ninja
winget install Python.Python.3.11

# After installing Python, install pre-commit (from regular PowerShell):
pip install pre-commit

# Note: Visual Studio must be installed manually (see below)
# Download from: https://visualstudio.microsoft.com/downloads/
```

#### Option 2: Using Chocolatey

First, install Chocolatey if not already installed:

```powershell
# Open PowerShell as Administrator and run:
Set-ExecutionPolicy Bypass -Scope Process -Force
[System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
```

Then install all tools:

```powershell
# Open PowerShell as Administrator and run:
choco install git cmake ninja python -y

# After installing Python, install pre-commit (from regular PowerShell):
pip install pre-commit

# Note: Visual Studio must be installed manually (see below)
```

**Important**: After installation, restart your terminal or log out and log back in for PATH changes to take effect.

**Visual Studio Installation**: Must be installed manually from [https://visualstudio.microsoft.com/downloads/](https://visualstudio.microsoft.com/downloads/) with "Desktop development with C++" workload. Supported versions: VS 2019, 2022, 2026, or later.

---

### Detailed Installation Instructions

### 1. Git

**Windows:**
- Download and install from [https://git-scm.com/download/win](https://git-scm.com/download/win)
- During installation, select "Git from the command line and also from 3rd-party software"
- Select "Use Windows' default console window" or "Use MinTTY (the default terminal of MSYS2)"

**Linux:**
```bash
# Debian/Ubuntu
sudo apt-get update
sudo apt-get install git

# RHEL/CentOS/Fedora
sudo yum install git
```

**Verify installation:**
```bash
git --version
```

### 2. CMake (3.20 or later)

**Windows:**
- Download installer from [https://cmake.org/download/](https://cmake.org/download/)
- During installation, select "Add CMake to the system PATH for all users"
- Or use winget: `winget install Kitware.CMake`

**Linux:**
```bash
# Debian/Ubuntu
sudo apt-get install cmake

# RHEL/CentOS/Fedora
sudo yum install cmake

# Or download latest from cmake.org
```

**Verify installation:**
```bash
cmake --version
```

### 3. Microsoft Visual Studio (Windows only)

**Required for Windows builds:**
- Download and install Visual Studio (2019, 2022, 2026, or later - Community, Professional, or Enterprise)
- Download from [https://visualstudio.microsoft.com/downloads/](https://visualstudio.microsoft.com/downloads/)

**Required components during installation:**
- Workload: "Desktop development with C++"
- Individual components:
  - MSVC C++ x64/x86 build tools (Latest version for your VS)
  - Windows 11 SDK (or Windows 10 SDK)
  - C++ CMake tools for Windows
  - C++ Clang tools for Windows (optional)

**After installation:**
- Launch git-bash from "Developer Command Prompt for VS XXXX" (where XXXX is your VS version: 2019, 2022, 2026, etc.) to ensure MSVC environment is available
- Or launch Claude Code from Developer Command Prompt

**Verify installation:**
```bash
# From Developer Command Prompt or git-bash launched from it
cl.exe
```

### 4. Ninja Build System (recommended)

**Windows:**
- Download from [https://github.com/ninja-build/ninja/releases](https://github.com/ninja-build/ninja/releases)
- Extract `ninja.exe` to a directory in your PATH (e.g., `C:\Program Files\ninja\`)
- Or use winget: `winget install Ninja-build.Ninja`
- Or install via Visual Studio Installer (included in "C++ CMake tools for Windows")

**Linux:**
```bash
# Debian/Ubuntu
sudo apt-get install ninja-build

# RHEL/CentOS/Fedora
sudo yum install ninja-build
```

**Verify installation:**
```bash
ninja --version
```

**Note**: Ninja is optional but recommended for faster builds. If not installed, CMake will use the default generator (Visual Studio on Windows, Make on Linux).

### 5. Python (3.8 or later, optional)

**Windows:**
- Download from [https://www.python.org/downloads/](https://www.python.org/downloads/)
- During installation, check "Add Python to PATH"
- Or use winget: `winget install Python.Python.3.11`

**Linux:**
```bash
# Debian/Ubuntu
sudo apt-get install python3 python3-pip

# RHEL/CentOS/Fedora
sudo yum install python3 python3-pip
```

**Verify installation:**
```bash
python --version
```

**Note**: Python is required for pre-commit hooks (see Code Formatting section below).

### 6. Pre-commit Hooks (REQUIRED for Contributors)

**Purpose**: Pre-commit hooks automatically format code and run linters before each commit to ensure CI checks pass.

**Installation**:
```bash
# Install pre-commit
pip install pre-commit

# Install the git hooks (run from project root)
pre-commit install
```

**IMPORTANT - Do NOT Install clang-format Manually**:
Pre-commit manages all formatting tools (clang-format, lintrunner, ruff) in an isolated environment with versions pinned in `.pre-commit-config.yaml`. **Never install or run these tools manually** - pre-commit handles everything automatically.

**Usage**:
```bash
# Pre-commit runs automatically on 'git commit'
# To manually format all files:
pre-commit run --all-files

# To manually format only staged files:
pre-commit run
```

**How It Works**:
- Pre-commit reads `.pre-commit-config.yaml` which pins exact tool versions (e.g., clang-format 16.0.1)
- It creates an isolated Python environment with these exact versions
- Your local formatting will always match CI because both use the same pinned versions
- No version conflicts or "works on my machine" issues

**Troubleshooting**:
- If pre-commit fails: `pre-commit clean` then `pre-commit install` to reset
- If hooks don't run on commit: Check that `.git/hooks/pre-commit` exists
- **Never run `clang-format` directly** - always use `pre-commit run`

### 7. Additional Tools (Windows)

**wget (for downloading archives):**
- Download from [https://eternallybored.org/misc/wget/](https://eternallybored.org/misc/wget/)
- Or use chocolatey: `choco install wget`
- Or use built-in PowerShell: `Invoke-WebRequest -Uri <url> -OutFile <file>`

### Summary Checklist

**For first-time setup**, verify all tools are installed by running these commands in your terminal:

```bash
# Manual prerequisite check
git --version          # Should show Git version
cmake --version        # Should show CMake 3.20 or later
ninja --version        # Should show Ninja version (optional but recommended)
python --version       # Should show Python 3.8 or later (optional)
cl.exe                 # Should show MSVC compiler version (Windows only)
```

**Windows-specific**: Launch git-bash from "Developer Command Prompt for VS XXXX" (where XXXX is your Visual Studio version) before running any build commands to ensure MSVC environment is available.

## Building Dependencies

Pre-building dependencies to `../../local` significantly improves build times for development workflows.

### CRITICAL: Runtime Library Consistency

**ALL dependencies and MorphiZen MUST use the same runtime library:**
- **Debug builds**: Static runtime `/MTd` via `-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded$<$<CONFIG:Debug>:Debug>`
- **Release builds**: Static runtime `/MT` via same flag
- **Mismatched runtime libraries cause link errors** (LNK2038, LNK2005)

If you encounter runtime library mismatch errors, rebuild the problematic dependency with the correct `CMAKE_MSVC_RUNTIME_LIBRARY` setting.

### Dependencies to Build

**REQUIRED (must build):**
1. **ONNX Runtime** (main branch) - MUST be built manually, cannot be auto-fetched by CMake

**OPTIONAL (for faster builds - or let CMake auto-fetch):**
2. **protobuf** (v21.12) - Protocol Buffers library (required for ONNX)
3. **gtest** (v1.15.0) - Google Test framework (required for unit tests)
4. **glog** (v0.7.1) - Google Logging library (required for logging)
5. **gsl** (v4.0.0) - Microsoft GSL (header-only)
6. **LLVM** (commit f8cb7987) - LLVM toolchain (from source or pre-built)
7. **boost** (v1.84.0) - Optional, if morphizen_ENABLE_BOOST=ON

All dependencies install to `../../local`.

**CRITICAL**: All dependencies must be built with:
```cmake
-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded$<$<CONFIG:Debug>:Debug>
```

## Building MorphiZen

### Unit Tests (Disabled by Default)

**Default behavior**: Unit tests are disabled by default (`morphizen_ENABLE_UNIT_TEST=OFF`) to reduce build time for end-users.

**For developers**:
- Use the `/build-and-test` skill which automatically enables unit tests
- Or manually enable when configuring: `-Dmorphizen_ENABLE_UNIT_TEST=ON`

**For CI**: Unit tests are always enabled in CI workflows to ensure code quality.

**Example manual build with unit tests**:
```bash
LOCAL_DIR=$(cd ../../local && pwd)
cmake -S . -B ../../build/morphizen \
  -DCMAKE_BUILD_TYPE=Debug \
  "-DCMAKE_PREFIX_PATH=$LOCAL_DIR" \
  -Dmorphizen_ENABLE_UNIT_TEST=ON
cmake --build ../../build/morphizen --config Debug --parallel
ctest --test-dir ../../build/morphizen -C Debug --output-on-failure
```

### Build Instructions for Each Dependency

Dependencies are cloned in the parent directory of the project (outside the project root), built in `../../build/<dependency-name>`, and installed to `../../local`.

**IMPORTANT - Use Absolute Paths**:

For reliable builds, compute absolute paths before running CMake commands:

```bash
# Run from Morphizen project directory
# Compute absolute paths once
LOCAL_DIR=$(cd ../../local && pwd)
BUILD_DIR=$(cd ../.. && pwd)/build
```

Then use these variables in CMake commands:
- `-DCMAKE_PREFIX_PATH="$LOCAL_DIR"` (absolute path, not `../../local`)
- `-DCMAKE_INSTALL_PREFIX="$LOCAL_DIR"` (absolute path)

Relative paths like `../../local` can fail depending on CMake's working directory.

**Notes**:
- These commands use Ninja generator if available (faster builds). If Ninja is not installed, remove the `-G Ninja` flag to use the default generator.
- Shallow clones (`--depth 1`) are used to save time and disk space by only fetching recent commit history.
- All commands assume you are in the Morphizen project directory

#### 1. protobuf (v21.12)

```bash
# Clone protobuf in parent directory (if not already cloned)
git clone --branch v21.12 --depth 1 https://github.com/protocolbuffers/protobuf.git ../protobuf

# Compute absolute paths (recommended for reliability)
LOCAL_DIR=$(cd ../../local && pwd)
BUILD_DIR=$(cd ../.. && pwd)/build

# Configure and build
cmake -S ../protobuf -B "$BUILD_DIR/protobuf" \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  "-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded$<$<CONFIG:Debug>:Debug>" \
  "-DCMAKE_INSTALL_PREFIX=$LOCAL_DIR" \
  -Dprotobuf_BUILD_TESTS=OFF \
  -Dprotobuf_MSVC_STATIC_RUNTIME=ON

cmake --build "$BUILD_DIR/protobuf" --config Debug --parallel
cmake --install "$BUILD_DIR/protobuf" --config Debug
```

**Note**: Other dependencies (gtest, glog, etc.) follow the same pattern. For brevity, they show relative paths below, but you can compute and use absolute paths the same way.

#### 2. gtest (Google Test v1.15.0)

```bash
# Clone googletest in parent directory (if not already cloned)
git clone --branch v1.15.0 --depth 1 https://github.com/google/googletest.git ../googletest

# Configure and build
cmake -S ../googletest -B ../../build/googletest \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  "-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded$<$<CONFIG:Debug>:Debug>" \
  -DCMAKE_INSTALL_PREFIX=../../local \
  -DBUILD_GMOCK=ON

cmake --build ../../build/googletest --config Debug --parallel
cmake --install ../../build/googletest --config Debug
```

#### 3. glog (Google Logging v0.7.1)

**CRITICAL**: glog MUST be built as a **static library** (not DLL) with `-DBUILD_SHARED_LIBS=OFF`.

```bash
# Clone glog in parent directory (if not already cloned)
git clone --branch v0.7.1 --depth 1 https://github.com/google/glog.git ../glog

# Configure and build
cmake -S ../glog -B ../../build/glog \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  "-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded$<$<CONFIG:Debug>:Debug>" \
  -DCMAKE_INSTALL_PREFIX=../../local \
  -DBUILD_SHARED_LIBS=OFF \
  -DBUILD_TESTING=OFF

cmake --build ../../build/glog --config Debug --parallel
cmake --install ../../build/glog --config Debug
```

**Verification**: After install, you should see `glogd.lib` (static library) installed, NOT `glogd.dll`. If you see a DLL, the build is incorrect - rebuild with `-DBUILD_SHARED_LIBS=OFF`.

#### 4. gsl (Microsoft GSL v4.0.0)

```bash
# Clone gsl in parent directory (if not already cloned)
git clone --branch v4.0.0 --depth 1 https://github.com/microsoft/GSL.git ../GSL

# Configure and install (header-only library)
cmake -S ../GSL -B ../../build/GSL \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_INSTALL_PREFIX=../../local \
  -DGSL_TEST=OFF

cmake --install ../../build/GSL --config Debug
```

#### 5. boost (Optional, v1.84.0 - if morphizen_ENABLE_BOOST=ON)

```bash
# Clone boost in parent directory (if not already cloned)
git clone --branch boost-1.84.0 --depth 1 https://github.com/boostorg/boost.git ../boost
cd ../boost

# Initialize submodules
git submodule update --init --recursive

# Bootstrap and build (Windows)
./bootstrap.bat
./b2 --prefix=../../local --build-type=complete runtime-link=static link=static threading=multi variant=debug install

# Bootstrap and build (Linux)
./bootstrap.sh
./b2 --prefix=../../local --build-type=complete runtime-link=static link=static threading=multi variant=debug install

cd -  # Return to project directory
```

#### 6. LLVM (commit f8cb7987)

```bash
# Clone LLVM in parent directory with shallow clone (if not already cloned)
# Fetch specific commit with shallow clone
git clone --depth 1 https://github.com/llvm/llvm-project.git ../llvm-project
cd ../llvm-project
git fetch --depth 1 origin f8cb7987c64dcffb72414a40560055cb717dbf74
git checkout f8cb7987c64dcffb72414a40560055cb717dbf74
cd -  # Return to project directory

# Configure and build (this takes a long time - 30+ minutes)
cmake -S ../llvm-project/llvm -B ../../build/llvm \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  "-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded$<$<CONFIG:Debug>:Debug>" \
  -DCMAKE_INSTALL_PREFIX=../../local \
  -DLLVM_ENABLE_PROJECTS="mlir" \
  -DLLVM_TARGETS_TO_BUILD="X86" \
  -DLLVM_INCLUDE_TESTS=OFF \
  -DLLVM_INCLUDE_EXAMPLES=OFF

cmake --build ../../build/llvm --config Debug --parallel
cmake --install ../../build/llvm --config Debug
```

**Note**: LLVM build can take 30-60 minutes and requires significant disk space (20GB+).

#### 7. ONNX Runtime (main branch)

```bash
# Clone ONNX Runtime in parent directory with shallow clone (if not already cloned)
# Use --recursive for submodules, --depth 1 and --shallow-submodules for faster clone
git clone --recursive --depth 1 --shallow-submodules https://github.com/microsoft/onnxruntime.git ../onnxruntime

# Compute absolute paths (IMPORTANT - relative paths can fail)
LOCAL_DIR=$(cd ../../local && pwd)
BUILD_DIR=$(cd ../.. && pwd)/build

# Configure and build
# Windows: Add /bigobj flag to handle large object files
cmake -S ../onnxruntime -B "$BUILD_DIR/onnxruntime" \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  "-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded$<$<CONFIG:Debug>:Debug>" \
  "-DCMAKE_INSTALL_PREFIX=$LOCAL_DIR" \
  "-DCMAKE_PREFIX_PATH=$LOCAL_DIR" \
  "-DCMAKE_CXX_FLAGS=/bigobj" \
  -Donnxruntime_BUILD_SHARED_LIB=ON \
  -Donnxruntime_BUILD_UNIT_TESTS=OFF \
  -Donnxruntime_USE_FULL_PROTOBUF=ON

cmake --build "$BUILD_DIR/onnxruntime" --config Debug --parallel
cmake --install "$BUILD_DIR/onnxruntime" --config Debug
```

**Notes**:
- ONNX Runtime depends on protobuf, so build protobuf first
- The `/bigobj` flag is required on Windows to handle large object files
- Unit tests are disabled for faster builds
- **IMPORTANT**: Use absolute paths for CMAKE_PREFIX_PATH - relative paths can fail

### Dependency Checking

To verify dependencies are installed correctly, check for CMake config files:

- **ONNX Runtime**: `../../local/lib/cmake/onnxruntime/onnxruntime-config.cmake` or similar
- **LLVM**: `../../local/lib/cmake/llvm/LLVM*.cmake` or `llvm-config.cmake`
- **gtest**: `../../local/lib/cmake/GTest/GTestConfig.cmake`
- **glog**: `../../local/lib/cmake/glog/glog-config.cmake`
- **protobuf**: `../../local/lib/cmake/protobuf/protobuf-config.cmake`

## Troubleshooting

### Runtime Library Mismatch

**Symptoms:**
- Link error: `LNK2038: mismatch detected for 'RuntimeLibrary': value 'MDd_DynamicDebug' doesn't match value 'MTd_StaticDebug'`
- Link error: `LNK2005: symbol already defined in libcmt.lib`
- Link error: Multiple definitions of standard library functions

**Solution:**
1. Identify the problematic dependency from the linker error message
2. Rebuild only that dependency with: `-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded$<$<CONFIG:Debug>:Debug>`
3. If multiple dependencies have mismatches, consider cleaning `../../local` and rebuilding all

**Explanation**: The issue is that some dependencies were built with `/MDd` (dynamic runtime) while MorphiZen uses `/MTd` (static runtime). ALL components must use the same runtime library. Rebuild the problematic dependency with the correct runtime library setting.

### MSVC Environment Not Available

**Symptoms:**
- "Cannot open include file: 'cstddef'" or other standard library headers
- "cl.exe not found" or compiler not available
- Missing Windows SDK headers

**Solution:**
Launch git-bash from an MSVC Developer Command Prompt:
1. Open "Developer Command Prompt for VS XXXX" (where XXXX is your Visual Studio version: 2019, 2022, 2026, etc.)
2. Run: `bash`
3. Navigate to project directory
4. Run the build commands

Alternatively, wrap commands with vcvars64.bat (adjust path for your VS version):
```bash
# For VS 2022:
cmd /c "call \"\"C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Auxiliary\\Build\\vcvars64.bat\"\" && cd /d %CD% && cmake --build \"../../build/$(basename $PWD)\" --config Debug --parallel"

# For VS 2026 (adjust path as needed):
cmd /c "call \"\"C:\\Program Files\\Microsoft Visual Studio\\2026\\Community\\VC\\Auxiliary\\Build\\vcvars64.bat\"\" && cd /d %CD% && cmake --build \"../../build/$(basename $PWD)\" --config Debug --parallel"
```
