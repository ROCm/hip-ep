# DynamicDispatch Integration Guide

This document covers known issues, workarounds, and requirements for integrating the DynamicDispatch library (XRT/NPU backend) into hip-ep.

## Overview

DynamicDispatch provides NPU acceleration through the XRT runtime. The integration is **optional** and controlled by the availability of DynamicDispatch headers and libraries at build time. When unavailable, hip-ep falls back to mock implementations.

## Quick Start: Setting DYNAMICDISPATCH_ROOT

The `DYNAMICDISPATCH_ROOT` environment variable tells hip-ep's build system where to find DynamicDispatch headers and libraries. Set it **before** running `build.py` or `cmake`.

**IMPORTANT**: `DYNAMICDISPATCH_ROOT` should point to a directory containing `include/` and `lib/` subdirectories, **not** the actual repository root. Think of it as "the root of the DD installation tree" rather than "the root of the DD source tree."

### Example 1: vai-rt Install Tree (Recommended)

When pointing to a vai-rt installation, use the install prefix:

```bash
# Linux
export DYNAMICDISPATCH_ROOT=/opt/vai-rt/install
python3 build.py

# Windows
set DYNAMICDISPATCH_ROOT=C:\vai-rt\install
python build.py
```

Expected structure:
```
/opt/vai-rt/install/
  ├── include/
  │   └── ryzenai/dynamic_dispatch/ops/...
  ├── lib/
  │   └── dyn_dispatch_core.{a,lib}
  └── ...
```

### Example 2: DynamicDispatch Repo Build Tree

When pointing to a DD repository build, point to the **build output directory** (which contains `include/` and `lib/`), **not** the repo root:

```bash
# Linux (example)
export VAI_RT_ROOT=/home/user/vai-rt
export DYNAMICDISPATCH_ROOT=${VAI_RT_ROOT}/vai-rt-build/workspace/dod/build/Release
python3 build.py

# Windows (example)
set VAI_RT_ROOT=C:\vai-rt-0
set DYNAMICDISPATCH_ROOT=%VAI_RT_ROOT%\vai-rt-build\workspace\dod\build\Release
python build.py
```

Expected structure:
```
C:\vai-rt-0\vai-rt-build\workspace\dod\
  ├── src/              [DD source code - NOT included in DYNAMICDISPATCH_ROOT path]
  ├── CMakeLists.txt    [DD repo root - NOT included in DYNAMICDISPATCH_ROOT path]
  └── build/
      └── Release/      [THIS is DYNAMICDISPATCH_ROOT]
          ├── include/ryzenai/dynamic_dispatch/ops/...
          └── lib/dyn_dispatch_core.lib
```

**Why point to `build/Release` instead of the repo root?**

hip-ep's CMake scripts expect to find:
- Headers at: `${DYNAMICDISPATCH_ROOT}/include/...`
- Libraries at: `${DYNAMICDISPATCH_ROOT}/lib/...`

The DD build output directory (`build/Release`) has this structure, but the DD repo root does not. The repo root has `src/` instead of `include/`.

### Example 3: Debug Build of DD

You can link against Debug builds by pointing to the Debug output directory:

```bash
# Windows Debug build
set DYNAMICDISPATCH_ROOT=C:\vai-rt-0\vai-rt-build\workspace\dod\build\Debug
python build.py

# Linux Debug build
export DYNAMICDISPATCH_ROOT=/home/user/vai-rt/vai-rt-build/workspace/dod/build/Debug
python3 build.py
```

**NOTE**: Ensure the CRT matches (see "MSVC Runtime Library Mismatch" section). Debug builds use `/MTd` or `/MDd`.

### Verification

After setting `DYNAMICDISPATCH_ROOT`, check that the expected files exist:

```bash
# Check headers
ls $DYNAMICDISPATCH_ROOT/include/ryzenai/dynamic_dispatch/ops/op_interface.hpp

# Check libraries
ls $DYNAMICDISPATCH_ROOT/lib/dyn_dispatch_core.{lib,a}

# Windows
dir %DYNAMICDISPATCH_ROOT%\include\ryzenai\dynamic_dispatch\ops\op_interface.hpp
dir %DYNAMICDISPATCH_ROOT%\lib\dyn_dispatch_core.lib
```

If these files don't exist, `DYNAMICDISPATCH_ROOT` is pointing to the wrong directory.

## Architecture

```
hip-ep Runtime
  ├─ Bitcode (runtime.bc)
  │   └─ mock/dynamic_dispatch_mock.cpp  [fallback stubs]
  └─ Native Library (hipdnn-ep-dd.lib)
      └─ real/dynamic_dispatch.cpp       [XRT C++ wrapper, when DD available]
           └─ real/dd_static_init.cpp    [static member workarounds]
```

The native library is required because:
- XRT APIs are C++ with complex types (cannot be in bitcode)
- JIT cannot link external C++ libraries at runtime
- Native symbols take precedence over bitcode stubs via JIT's `ProcessSymbolSearchGenerator`

## Known Issues and Workarounds

### 1. Missing Static Member Definitions

**Problem**: DynamicDispatch headers (e.g., `xrt_context.hpp`) declare static members but the library may not provide their definitions:

```cpp
// In xrt_context.hpp (example):
class xrt_context {
  static std::unordered_map<std::string, std::shared_ptr<xrt_context>> *ctx_map_;
};
```

This causes linker errors:
```
unresolved external symbol "private: static ... *ryzenai::dynamic_dispatch::xrt_context::ctx_map_"
```

**Why hip-ep can't fix this automatically**:
- No version API from DD to detect which statics are declared
- Static member names/types may change between DD versions
- The DD library should provide these definitions, not consumers

**Current workaround**: `lib/Runtime/real/dd_static_init.cpp` contains conditional definitions that can be enabled if needed:

```cpp
#if 0  // Enable if you get unresolved external errors
std::unordered_map<std::string, std::shared_ptr<xrt_context>> *xrt_context::ctx_map_ = nullptr;
#endif
```

**To fix unresolved externals**:
1. Inspect your version of `xrt_context.hpp` to find the exact static member declaration
2. Uncomment and update the definition in `dd_static_init.cpp` to match
3. File a bug with the DD maintainers to provide these definitions in their library

**Long-term solution**: DD should either:
- Provide all static definitions in `libdyn_dispatch_core.lib`
- Provide a version macro (e.g., `DD_VERSION_MAJOR`) for conditional compilation
- Move to header-only statics with inline initialization

### 2. Missing Template Instantiations

**Problem**: `lib/Runtime/real/dynamic_dispatch.cpp` uses DD templates that may not be instantiated in `libdyn_dispatch_core.lib`:

```cpp
// Example usage in hip-ep:
combined_gemm<uint16_t, uint8_t, uint16_t>
iconv<uint16_t, uint8_t, uint16_t>
```

**Error**:
```
unresolved external symbol "public: ... combined_gemm<uint16_t, uint8_t, uint16_t>::..."
```

**Current status**: As of this writing, these instantiations are **required** but not guaranteed to be present in all DD builds.

**Workaround**: None available from hip-ep side. Contact DD maintainers to request these template instantiations be added to their library.

**Alternative**: If DD provides source access, add explicit instantiations to `dd_static_init.cpp`:
```cpp
template class ryzenai::dynamic_dispatch::combined_gemm<uint16_t, uint8_t, uint16_t>;
template class ryzenai::dynamic_dispatch::iconv<uint16_t, uint8_t, uint16_t>;
```

### 3. MSVC Runtime Library Mismatch (Windows Only)

**Problem**: DynamicDispatch libraries may be built with `/MD` (dynamic CRT), but hip-ep uses `/MT` (static CRT), causing link errors.

**Symptoms**:
```
dyn_dispatch_core.lib(combined_gemm.obj) : error LNK2038: mismatch detected for 'RuntimeLibrary': 
value 'MD_DynamicRelease' doesn't match value 'MT_StaticRelease' in main.obj
```

**Explanation**:
- `/MD` = Multi-threaded DLL runtime (dynamic CRT, links against `msvcrt.dll` or `msvcrtd.dll`)
- `/MT` = Multi-threaded static runtime (static CRT, embeds CRT in binary)
- `/MDd` and `/MTd` are debug variants

All C++ code in a binary and its dependencies **must** use the same CRT variant. Mixing them causes:
- Link errors (LNK2038, LNK2005)
- Runtime crashes (different heap allocators)
- Multiple CRT DLL loading (Debug + Release in same process)

**Why DLLs should use /MT (static CRT)**:

The **correct** architectural choice for distributable DLLs is `/MT` (static CRT), not `/MD`. Here's why:

1. The **EXE** chooses whether to use Debug or Release CRT DLL (`msvcrtd.dll` vs `msvcrt.dll`)
2. A DLL built with `/MD` is compiled for **Release CRT DLL only**
3. When a Debug EXE (using `msvcrtd.dll`) loads a DLL built with `/MD` (using `msvcrt.dll`), **both CRT DLLs get loaded**
4. Both CRT DLLs are singletons that think they own the process heap → **heap corruption, crashes, undefined behavior**

Using `/MT` in DLLs avoids this problem by giving each DLL its own isolated CRT copy. This is the recommended practice for ISV-distributed libraries.

**Solution 1: Request /MT build from DD maintainers** (Recommended)

Contact the DynamicDispatch maintainers and request that `dyn_dispatch_core.lib` and related libraries be built with `/MT` (static CRT). This is the architecturally correct solution.

**Why**: Distributable libraries should use static CRT to avoid Debug/Release CRT DLL conflicts in customer applications.

**Solution 2: Use a /MD build of DD** (If available)

If DD provides separate `/MT` and `/MD` builds, request the `/MT` variant. Some projects ship both.

**Solution 3: Rebuild DD from source with /MT** (If you have source access)

If you have access to DD source code:

```bash
# Configure DD with static CRT
cmake -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --config Release
```

**Workaround: Build hip-ep with /MD** (Not recommended for distribution)

As a **temporary workaround only**, you can build hip-ep with `/MD` to match DD:

```bash
cmake -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDLL ..
```

**IMPORTANT**: This creates the same problem downstream - any Debug EXE using hip-ep will load both Debug and Release CRT DLLs. Only use this for testing, not for customer-facing builds.

**Verification**:

Check which CRT a library uses:
```bash
dumpbin /DIRECTIVES dyn_dispatch_core.lib | findstr /C:"DEFAULTLIB"
```

Look for:
- `MSVCRT` or `MSVCRTD` → `/MD` or `/MDd` (dynamic CRT) ❌ Problem for DLLs
- `LIBCMT` or `LIBCMTD` → `/MT` or `/MTd` (static CRT) ✓ Correct for DLLs

Check hip-ep's CRT configuration:
```bash
cmake -L ../build/hip-ep | grep MSVC_RUNTIME
# Should show: CMAKE_MSVC_RUNTIME_LIBRARY:STRING=MultiThreaded
```

**References**:
- CMake MSVC Runtime Library: https://cmake.org/cmake/help/latest/variable/CMAKE_MSVC_RUNTIME_LIBRARY.html
- Why static CRT for DLLs: https://docs.microsoft.com/en-us/cpp/c-runtime-library/crt-library-features
- Mixing CRT libraries: https://docs.microsoft.com/en-us/cpp/c-runtime-library/potential-errors-passing-crt-objects-across-dll-boundaries

### 4. Boost Header Discovery

**Problem**: XRT (required by DD) depends on Boost headers, but the DD installation may not expose them in standard search paths.

**Symptoms**:
```
fatal error: boost/any.hpp: No such file or directory
```

**Workaround**: Set `BOOST_ROOT` environment variable before building:

```bash
# Windows
set BOOST_ROOT=C:\path\to\boost_1_84_0
python build.py

# Linux
export BOOST_ROOT=/opt/boost_1_84_0
python3 build.py
```

The CMake configuration (as of recent changes) checks:
1. `DYNAMICDISPATCH_ROOT/boost/include`
2. `BOOST_ROOT` environment variable
3. `BOOST_ROOT/include`

**See**: `lib/Runtime/CMakeLists.txt` lines 668-683

### 5. Install Layout vs. Source Layout

**Problem**: DD may be built from source or installed as a package, with different header paths. Additionally, some DD headers reference files outside the standard include structure.

**Specific issue**: `combined_gemm.hpp` (in `ops/`) includes `coeffs.hpp` (in `src/ops/`). This creates three different layout scenarios:

```
1. vai-rt install layout (COMPLETE):
   DD_ROOT/include/ryzenai/dynamic_dispatch/ops/combined_gemm.hpp
   DD_ROOT/include/ryzenai/dynamic_dispatch/src/ops/coeffs.hpp ✓

2. DD repo build layout (INCOMPLETE):
   DD_ROOT/build/relwithdebinfo/include/ryzenai/dynamic_dispatch/ops/combined_gemm.hpp
   DD_ROOT/build/relwithdebinfo/include/ryzenai/dynamic_dispatch/src/ops/coeffs.hpp ✗ (missing!)
   DD_ROOT/src/ops/coeffs.hpp ✓ (actual location)

3. Legacy source layout:
   DD_ROOT/include/ops/...
   DD_ROOT/include/src/...
```

**Symptoms**:
```
fatal error: coeffs.hpp: No such file or directory
  #include "coeffs.hpp"
```

**Current handling**: CMake auto-detects the layout:
- Checks for `ryzenai/dynamic_dispatch/ops` to distinguish modern vs. legacy
- Checks for `ryzenai/dynamic_dispatch/src/ops` to distinguish vai-rt install vs. DD repo build
- For DD repo build, attempts to find `src/` in the repository root (relative to build dir)
- Issues a warning if neither layout is detected

**Recommended solutions**:

For **end users**:
- Use the vai-rt install directory for `DYNAMICDISPATCH_ROOT` (preferred)
- If pointing to DD repo, ensure you're pointing to the root, not `build/*/include`

For **DD maintainers**:
- Copy `src/ops/*.hpp` to `build/*/include/ryzenai/dynamic_dispatch/src/ops/` during build
- This matches the vai-rt install layout and eliminates the inconsistency

**Workaround**: If auto-detection fails, manually set both include paths:
```cmake
target_include_directories(myTarget PRIVATE
  ${DYNAMICDISPATCH_ROOT}/include/ryzenai/dynamic_dispatch
  ${DYNAMICDISPATCH_ROOT}/src  # For coeffs.hpp
)
```

**See**: `lib/Runtime/CMakeLists.txt` lines 726-770

## Build Configuration

### Environment Variables

| Variable | Purpose | Example |
|----------|---------|---------|
| `DYNAMICDISPATCH_ROOT` | DD installation or build output directory (must contain `include/` and `lib/`) | `C:\vai-rt-0\vai-rt-build\workspace\dod\build\Release` (DD repo)<br>`/opt/vai-rt/install` (vai-rt install) |
| `BOOST_ROOT` | Boost headers (if not in DD tree) | `C:\boost_1_84_0` |
| `XILINX_XRT` | XRT installation (fallback search) | `/opt/xilinx/xrt` |

**See the "Quick Start: Setting DYNAMICDISPATCH_ROOT" section above for detailed examples and explanation.**

### CMake Variables

Set via `cmake -D<VAR>=<value>` or `build.py` extensions:

| Variable | Purpose | Default |
|----------|---------|---------|
| `DYNAMICDISPATCH_ROOT` | Override DD search path | `$ENV{DYNAMICDISPATCH_ROOT}` |

### Runtime Variables

| Variable | Purpose | Values |
|----------|---------|--------|
| `HIPEP_USE_DYNAMIC_DISPATCH` | Enable DD at runtime | `0` (disabled) / `1` (enabled) |

**Note**: The runtime variable controls whether DD operations are dispatched to NPU. It does **not** affect whether the DD library is linked.

## Verifying DynamicDispatch Integration

### At Build Time

Check CMake output for:

```
-- Found DynamicDispatch headers: <path>
-- Found DynamicDispatch library: <path>
-- Found XRT headers: <path>
-- Found Boost headers: <path>
-- Building DynamicDispatch native library (XRT backend)
```

If you see:
```
-- DynamicDispatch headers not found. DynamicDispatch will use mock implementation.
```

Then the build will succeed but DD operations will use mock stubs (CPU fallback).

### At Link Time

The hipdnn-ep-dd library should link against:
- `dyn_dispatch_core.lib` (required)
- `transaction.lib` (optional)
- `aie_codegen.lib` (optional)
- `xrt_coreutil.lib` (optional)
- `zlib.lib` (optional)

Missing optional libraries are non-fatal; missing `dyn_dispatch_core` will cause unresolved externals.

### At Runtime

With `HIPEP_USE_DYNAMIC_DISPATCH=1`, DD operations should dispatch to NPU. Check logs for DD initialization messages.

## Debugging Integration Issues

### "DynamicDispatch headers not found"

1. Verify `DYNAMICDISPATCH_ROOT` points to the DD installation
2. Check that `$DYNAMICDISPATCH_ROOT/include/ryzenai/dynamic_dispatch/ops/op_interface.hpp` exists (install layout) **OR** `$DYNAMICDISPATCH_ROOT/include/ops/op_interface.hpp` exists (source layout)
3. Try setting the path explicitly:
   ```bash
   cmake -DDYNAMICDISPATCH_ROOT=C:\vai-rt-0\vai-rt-build\workspace\dod ..
   ```

### "Boost headers not found"

1. Set `BOOST_ROOT` environment variable
2. Verify `$BOOST_ROOT/boost/any.hpp` exists (without `include/` in the path) **OR** `$BOOST_ROOT/include/boost/any.hpp` exists
3. Check that DD installation includes Boost: `$DYNAMICDISPATCH_ROOT/boost/include/boost/any.hpp`

### "XRT headers not found"

1. Check `$DYNAMICDISPATCH_ROOT/xrt/include/xrt/xrt_bo.h`
2. Or set `XILINX_XRT` environment variable
3. Or install XRT to `/opt/xilinx/xrt` (Linux default)

### "LNK2038: mismatch detected for 'RuntimeLibrary'" (Windows)

This is an MSVC CRT mismatch (see "MSVC Runtime Library Mismatch" above).

1. **Diagnose**: Check what CRT DD was built with:
   ```bash
   dumpbin /DIRECTIVES dyn_dispatch_core.lib | findstr /C:"DEFAULTLIB"
   ```
   Look for:
   - `MSVCRT` or `MSVCRTD` → `/MD` or `/MDd` (dynamic CRT) ❌ **Problem!**
   - `LIBCMT` or `LIBCMTD` → `/MT` or `/MTd` (static CRT) ✓ Correct

2. **Best solution**: Request `/MT` build from DD maintainers (see "MSVC Runtime Library Mismatch" section)

3. **Temporary workaround** (testing only, not for distribution):
   ```bash
   # Make hip-ep match DD's /MD (creates downstream issues!)
   python build.py --clean
   cmake -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDLL ..
   python build.py
   ```

4. **Verify** hip-ep's CRT setting:
   ```bash
   cmake -L ../build/hip-ep | grep MSVC_RUNTIME
   # For proper builds: CMAKE_MSVC_RUNTIME_LIBRARY:STRING=MultiThreaded (/MT)
   # For workaround:    CMAKE_MSVC_RUNTIME_LIBRARY:STRING=MultiThreadedDLL (/MD)
   ```

### "coeffs.hpp: No such file or directory"

This indicates a layout detection issue (see "Install Layout vs. Source Layout" above).

1. Check CMake output for the detected layout:
   ```
   -- Using DynamicDispatch vai-rt install layout (complete)
   -- Using DynamicDispatch repo build layout
   -- Using DynamicDispatch legacy source layout
   ```

2. If you see a warning about missing `src/ops/`:
   ```
   -- Found src tree at: <path>  # Good - auto-detection worked
   ```
   OR
   ```
   WARNING: DynamicDispatch layout issue: Found ops/ but not src/ops/
   ```

3. Solutions in order of preference:
   - **Point to vai-rt install**: `set DYNAMICDISPATCH_ROOT=C:\vai-rt-build\install`
   - **Point to DD repo root**: `set DYNAMICDISPATCH_ROOT=C:\DD_REPO` (not `C:\DD_REPO\build\...`)
   - **Manually copy headers**: Copy `DD_REPO/src/ops/*.hpp` to `DD_REPO/build/relwithdebinfo/include/ryzenai/dynamic_dispatch/src/ops/`

### Unresolved Externals at Link Time

**Static members**:
1. See "Missing Static Member Definitions" section above
2. Enable workarounds in `dd_static_init.cpp`

**Template instantiations**:
1. See "Missing Template Instantiations" section above
2. Contact DD maintainers for library updates

**Vendor symbols** (MIOpen/hipBLASLt):
1. These should **not** be needed by DD integration
2. If you see MIOpen/hipBLASLt errors, check that `hipdnn-ep-dd` doesn't accidentally pull in GPU runtime headers

## Testing DD Integration

### Compiler-Only Test (No Hardware)

```bash
# Build with DD but without ROCm GPU libraries
python build.py --mock
```

This verifies that DD headers/libraries are found and link correctly.

### Full GPU + NPU Test

```bash
HIPEP_USE_DYNAMIC_DISPATCH=1 python test/numeric/test_conv.py
```

Use numeric tests or E2E models that exercise DD-accelerated operations.

## Relationship to Other Runtime Features

### Mock Runtime

When `BUILD_MOCK_RUNTIME=ON` (or `--mock` flag), **DynamicDispatch is also mocked**. The GPU and NPU backends are mutually independent but both disabled in mock mode.

### Vendor BLAS Opt-Out

`HIPDNN_EP_DISABLE_VENDOR_BLAS=ON` disables MIOpen/hipBLASLt but does **not** affect DynamicDispatch. DD is an independent NPU backend, not a GPU vendor library.

## Contact and Support

- **hip-ep issues**: File issues in the hip-ep repository
- **DynamicDispatch issues** (missing definitions, template instantiations): Contact DD maintainers or file bugs in the DD repository
- **XRT issues**: Contact Xilinx XRT team

## References

- Runtime integration: `lib/Runtime/real/dynamic_dispatch.cpp`
- Static workarounds: `lib/Runtime/real/dd_static_init.cpp`
- Build configuration: `lib/Runtime/CMakeLists.txt` (lines 568-813)
- Mock fallback: `lib/Runtime/mock/dynamic_dispatch_mock.cpp`
- Runtime control: `docs/design/compiler-runtime-contract.md`
