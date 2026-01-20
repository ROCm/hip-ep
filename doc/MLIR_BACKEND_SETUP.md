# MLIR Backend Setup Guide

## Overview

This document describes how to enable MLIR backend support in morphizen-rocm. The MLIR backend requires LLVM with MLIR enabled, which is not included in TheRock ROCm SDK and must be built separately.

## Quick Start

### Option 1: Automated Build (Recommended)

The easiest way to build with MLIR backend:

```batch
cd d:\Develop\m\morphizen-rocm
build_with_mlir_backend.bat
```

This script will:
1. Check if LLVM/MLIR is already installed
2. Build LLVM/MLIR if needed (takes several hours on first run)
3. Build morphizen-rocm with MLIR backend enabled

### Option 2: Manual Build

If you prefer manual control:

```batch
# Step 1: Build LLVM/MLIR (one-time setup, takes several hours)
cd d:\Develop\m\morphizen-rocm
build_llvm.bat

# Step 2: Build morphizen-rocm with MLIR backend
set WITH_MLIR_BACKEND=true
build.bat
```

### Option 3: Standard Build (Without MLIR)

To build without MLIR backend (default behavior):

```batch
cd d:\Develop\m\morphizen-rocm
build.bat
```

## Build Scripts

### build_llvm.bat

Builds LLVM with MLIR support and installs it to `%WORKSPACE_ROOT%/local`.

**Features:**
- Clones LLVM project from GitHub to `%WORKSPACE_ROOT%/llvm`
- Checks out specific commit: `f8cb7987c64dcffb72414a40560055cb717dbf74` (matches MorphiZen dependency)
- Configures CMake with MLIR enabled
- Builds and installs to `%WORKSPACE_ROOT%/local`
- Supports incremental builds (won't reconfigure if build.ninja exists)

**CMake Configuration:**
```cmake
-DLLVM_ENABLE_PROJECTS=mlir
-DLLVM_TARGETS_TO_BUILD=host
-DLLVM_ENABLE_ASSERTIONS=ON
-DLLVM_ENABLE_RTTI=ON
-DLLVM_ENABLE_LIBEDIT=OFF
-DLLVM_BUILD_TOOLS=ON
-DLLVM_INSTALL_UTILS=ON
-DLLVM_INCLUDE_TESTS=ON
-DZLIB_USE_STATIC_LIBS=ON
-DLLVM_ENABLE_ZSTD=OFF
-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDLL
```

**Installation Locations:**
- LLVM: `D:/Develop/m/local/lib/cmake/llvm`
- MLIR: `D:/Develop/m/local/lib/cmake/mlir`

**Build Time:** Several hours (typically 3-6 hours depending on hardware)

### build.bat

Main build script for morphizen-rocm with MLIR backend support.

**Environment Variables:**
- `WITH_MLIR_BACKEND` - Set to `true` to enable MLIR backend (default: `false`)

**How It Works:**
- When `WITH_MLIR_BACKEND=true`, adds `-Dmorphizen_ENABLE_MLIR_BACKEND=ON` to CMake
- Automatically finds LLVM/MLIR via `-DCMAKE_PREFIX_PATH=%WORKSPACE_ROOT%/local`
- No manual LLVM_DIR or MLIR_DIR configuration needed

**Example:**
```batch
set WITH_MLIR_BACKEND=true
build.bat
```

### build_with_mlir_backend.bat

All-in-one convenience script that:
1. Checks if LLVM/MLIR is installed at `%WORKSPACE_ROOT%/local/lib/cmake/mlir`
2. Runs `build_llvm.bat` if LLVM/MLIR is not found
3. Runs `build.bat` with `WITH_MLIR_BACKEND=true`

**Usage:**
```batch
build_with_mlir_backend.bat
```

### clean_build_with_mlir.bat

Performs a clean build with MLIR backend enabled:
1. Removes the morphizen-rocm build directory
2. Builds with `WITH_MLIR_BACKEND=true`

**Usage:**
```batch
clean_build_with_mlir.bat
```

## Architecture

### Dependency Flow

```
morphizen-rocm (WITH_MLIR_BACKEND=true)
    ↓
CMAKE_PREFIX_PATH=%WORKSPACE_ROOT%/local
    ↓
Finds LLVM at: %WORKSPACE_ROOT%/local/lib/cmake/llvm
Finds MLIR at: %WORKSPACE_ROOT%/local/lib/cmake/mlir
    ↓
MorphiZen CMake enables MLIR backend
```

### Directory Structure

```
D:/Develop/m/
├── llvm/                          # LLVM source (cloned by build_llvm.bat)
├── morphizen-rocm/                # morphizen-rocm source
│   ├── build.bat                  # Main build script
│   ├── build_llvm.bat             # LLVM/MLIR build script
│   ├── build_with_mlir_backend.bat # All-in-one script
│   └── clean_build_with_mlir.bat  # Clean build helper
├── build/
│   ├── llvm/                      # LLVM build directory
│   └── morphizen-rocm/            # morphizen-rocm build directory
└── local/                         # Installation directory
    ├── bin/                       # Executables
    ├── lib/
    │   └── cmake/
    │       ├── llvm/              # LLVM CMake config
    │       └── mlir/              # MLIR CMake config
    └── include/                   # Headers
```

## Troubleshooting

### LLVM/MLIR Not Found

**Error:**
```
CMake Error: Could not find a package configuration file provided by "MLIR"
```

**Solution:**
Run `build_llvm.bat` to build and install LLVM/MLIR first.

### Build Takes Too Long

**Issue:** LLVM build takes several hours

**Solutions:**
- Use incremental builds (don't delete build directory)
- Use a machine with more CPU cores
- Consider using pre-built LLVM binaries if available

### CMake Configuration Fails

**Issue:** CMake can't find LLVM/MLIR even after building

**Solution:**
Verify installation:
```batch
dir D:\Develop\m\local\lib\cmake\llvm
dir D:\Develop\m\local\lib\cmake\mlir
```

Both directories should exist and contain CMake config files.

### Force Reconfiguration

To force CMake to reconfigure (useful after changing options):

```batch
# For LLVM
del D:\Develop\m\build\llvm\build.ninja

# For morphizen-rocm
del D:\Develop\m\build\morphizen-rocm\build.ninja
```

## Technical Details

### LLVM Commit Version

The build uses LLVM commit `f8cb7987c64dcffb72414a40560055cb717dbf74`, which matches the version specified in MorphiZen's dependency list (`deps.cmake`).

### CMake Integration

The build.bat script uses delayed variable expansion (`setlocal enabledelayedexpansion`) to properly handle the MLIR backend option:

```batch
set MLIR_ENABLE_OPTION=
if /I "%WITH_MLIR_BACKEND%"=="true" (
    set "MLIR_ENABLE_OPTION=-Dmorphizen_ENABLE_MLIR_BACKEND=ON"
)
```

The option is then passed to CMake:
```batch
cmake ... -DCMAKE_PREFIX_PATH=%WORKSPACE_ROOT%/local !MLIR_ENABLE_OPTION!
```

### Why TheRock Doesn't Include MLIR

TheRock ROCm SDK includes LLVM but not MLIR. The available CMake packages in TheRock are:
- `llvm` ✅
- `clang` ✅
- `lld` ✅
- `mlir` ❌ (not included)

Therefore, MLIR must be built separately.

## Best Practices

1. **Build LLVM/MLIR once**: The build takes several hours, so build it once and reuse it for multiple morphizen-rocm builds.

2. **Use incremental builds**: Don't delete the build directory unless necessary. Both scripts support incremental compilation.

3. **Monitor disk space**: LLVM build requires significant disk space (~20-30 GB for source + build).

4. **Use build_with_mlir_backend.bat**: This is the simplest approach for most users.

## Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `WITH_MLIR_BACKEND` | `false` | Enable MLIR backend in morphizen-rocm build |
| `LLVM_DIR` | (auto-detected) | Custom LLVM installation path (optional) |
| `MLIR_DIR` | (auto-detected) | Custom MLIR installation path (optional) |

## Build Output

### Successful LLVM/MLIR Build

```
============================================================
LLVM/MLIR build and install completed successfully!
Installation directory: D:/Develop/m/local

LLVM installed to: D:/Develop/m/local/lib/cmake/llvm
MLIR installed to: D:/Develop/m/local/lib/cmake/mlir

You can now build morphizen-rocm with MLIR backend enabled:
  set WITH_MLIR_BACKEND=true
  build.bat
============================================================
```

### Successful morphizen-rocm Build with MLIR

```
-- MorphiZen OPTIONS:
--   morphizen_ENABLE_MLIR_BACKEND : ON
...
============================================================
Build and install completed successfully!
Installation directory: D:/Develop/m/local
============================================================
```

## FAQ

**Q: Do I need to rebuild LLVM/MLIR every time?**  
A: No, build it once and reuse it. The build_with_mlir_backend.bat script automatically skips the LLVM build if it's already installed.

**Q: Can I use a different LLVM version?**  
A: It's recommended to use the version specified in MorphiZen's deps.cmake to ensure compatibility.

**Q: How much disk space do I need?**  
A: Approximately 30-40 GB total (source code + build artifacts + installation).

**Q: Can I delete the LLVM source after building?**  
A: Yes, once LLVM/MLIR is installed to `%WORKSPACE_ROOT%/local`, you can delete the `%WORKSPACE_ROOT%/llvm` directory to save space.

**Q: How do I disable MLIR backend after enabling it?**  
A: Simply run `build.bat` without setting `WITH_MLIR_BACKEND=true`, or set it to `false`.

## Version History

- **2026-01-20**: Initial implementation of MLIR backend support
  - Added `WITH_MLIR_BACKEND` environment variable
  - Created `build_llvm.bat` for LLVM/MLIR compilation
  - Created `build_with_mlir_backend.bat` for automated workflow
  - Simplified build.bat to use CMAKE_PREFIX_PATH for automatic detection
