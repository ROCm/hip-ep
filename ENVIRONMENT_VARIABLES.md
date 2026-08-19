# Environment Variables for Building hip-ep

This document lists the environment variables and CMake options that affect the hip-ep build process.

## Quick Reference

| Variable | Type | When | Purpose |
|----------|------|------|---------|
| DYNAMICDISPATCH_ROOT | Env/CMake | Build | Path to DynamicDispatch installation |
| HIPEP_USE_DYNAMIC_DISPATCH | Env | Runtime | Enable DD lowering during compilation |
| XILINX_XRT | Env | Build | Fallback for XRT location |
| BOOST_ROOT | Env | Build | Boost library location |
| BUILD_MOCK_RUNTIME | CMake | Build | Build CPU-only mock runtime |
| HIP_ARCHITECTURES | CMake | Build | GPU target (auto-detected) |

## Required Variables

None. The build system works with default settings for standard GPU builds.

## Optional Variables

### DynamicDispatch NPU/IPU Backend (Optional)

These variables enable NPU/IPU acceleration via AMD's DynamicDispatch library. **If not set, the build proceeds normally without DynamicDispatch support.**

**Important distinction:**
- **Build-time variables** (below) control whether the DynamicDispatch native library is linked into the EP
- **Runtime variable** (HIPEP_USE_DYNAMIC_DISPATCH) controls whether DynamicDispatch lowering is used during model compilation

#### Build-time Configuration

- **DYNAMICDISPATCH_ROOT** (environment or CMake variable)
  - Path to DynamicDispatch installation or build directory
  - Examples:
    - Install layout: `/path/to/vai-rt-build/install`
    - Workspace build: `/path/to/vai-rt-build/workspace/dod/build/Release`
  - CMake: `-DDYNAMICDISPATCH_ROOT=<path>`
  - Environment: `export DYNAMICDISPATCH_ROOT=<path>` (Linux) or `set DYNAMICDISPATCH_ROOT=<path>` (Windows)
  - Default: Not set (DynamicDispatch disabled, uses mock implementation)

- **XILINX_XRT** (environment variable)
  - Fallback location for XRT (Xilinx Runtime) headers and libraries
  - Only used if DYNAMICDISPATCH_ROOT is not set
  - Default: Not set

- **BOOST_ROOT** (environment variable)
  - Boost library installation directory
  - Required by XRT when using DynamicDispatch
  - Default: Not set (searched in DYNAMICDISPATCH_ROOT first)

#### Runtime Configuration

- **HIPEP_USE_DYNAMIC_DISPATCH** (environment variable, runtime only)
  - Controls whether the compiler uses DynamicDispatch lowering for operations
  - Values: `1`, `true` (enable) or `0`, `false` (disable)
  - Environment: `export HIPEP_USE_DYNAMIC_DISPATCH=1` (Linux) or `set HIPEP_USE_DYNAMIC_DISPATCH=1` (Windows)
  - Can also be set via ONNX Runtime provider option: `use_dynamic_dispatch=true`
  - Default: Not set (DynamicDispatch lowering disabled)
  - **Note**: This only affects model compilation, not the EP build itself
  - If set to `1` but DD library was not linked at build time, the mock implementation will be used (returns "not available" errors)

### Build Configuration

- **BUILD_MOCK_RUNTIME** (CMake variable only)
  - Build without GPU support (CPU-only mock runtime for testing)
  - CMake: `-DBUILD_MOCK_RUNTIME=ON`
  - Default: `OFF` (build real GPU runtime)
  - When `ON`, DynamicDispatch is automatically disabled

- **HIP_ARCHITECTURES** (CMake variable, auto-detected by build.py)
  - Target GPU architecture (e.g., `gfx1150`, `gfx90a`, `gfx942`)
  - Auto-detected from available GPU by build.py
  - Override: `python build.py --hip_arch gfx942`
  - CMake: `-DHIP_ARCHITECTURES=gfx942`

## Build Examples

### Standard GPU build (no DynamicDispatch)
```bash
python build.py
```

### GPU build with DynamicDispatch (NPU backend)
```bash
# Build with DD support (build-time)
export DYNAMICDISPATCH_ROOT=/opt/vai-rt/install
python build.py

# Enable DD lowering at runtime
export HIPEP_USE_DYNAMIC_DISPATCH=1
# Now when you run models through the EP, DD lowering will be used

# Or set via provider option in Python:
# session_options.add_session_config_entry("ep.amdgpuexecutionprovider.use_dynamic_dispatch", "1")
```

### Mock runtime (no GPU required)
```bash
python build.py --mock
```

### Clean build
```bash
python build.py --clean
```

## DynamicDispatch Integration Status

The DynamicDispatch integration is currently waiting for template instantiations to be added to `dyn_dispatch_core.lib`:

**Required template instantiations:**
- `ryzenai::combined_gemm<uint16_t, uint8_t, uint16_t>`
- `ryzenai::iconv<uint16_t, uint8_t, uint16_t>`

Without these, builds with DYNAMICDISPATCH_ROOT set will fail at link time with unresolved external symbols. This is expected and documented in `lib/Runtime/real/dynamic_dispatch.cpp`.

## Checking Your Build Configuration

During CMake configuration, look for these status messages:

### DynamicDispatch Enabled
```
-- Found DynamicDispatch headers: <path>
-- Found DynamicDispatch library: <path>
-- Found DD core library: <path>
-- Using DynamicDispatch install layout
-- Building DynamicDispatch native library (XRT backend)
-- Linked DynamicDispatch via hipdnn-ep-dd target
```

### DynamicDispatch Disabled (Normal)
```
-- DynamicDispatch headers not found. DynamicDispatch will use mock implementation.
```

### Mock Runtime
```
-- Mock runtime mode: DynamicDispatch will use mock implementation.
```

## Using DynamicDispatch at Runtime

After building with DYNAMICDISPATCH_ROOT set, you can enable DynamicDispatch lowering in two ways:

### Via Environment Variable
```bash
# Linux/Mac
export HIPEP_USE_DYNAMIC_DISPATCH=1
python your_model.py

# Windows
set HIPEP_USE_DYNAMIC_DISPATCH=1
python your_model.py
```

### Via Provider Option (Python)
```python
import onnxruntime as ort

session_options = ort.SessionOptions()
# Enable DynamicDispatch lowering
session_options.add_session_config_entry(
    "ep.amdgpuexecutionprovider.use_dynamic_dispatch", "1"
)

session = ort.InferenceSession(
    "model.onnx",
    session_options,
    providers=["AMDGPUExecutionProvider"]
)
```

The provider option takes precedence over the environment variable.

## Troubleshooting

### "DynamicDispatch headers not found"
- **Status**: Normal - build continues without DynamicDispatch
- **Action**: None required unless you intended to enable DynamicDispatch
- **To enable**: Set `DYNAMICDISPATCH_ROOT` to your DynamicDispatch installation

### "XRT headers not found" or "Boost headers not found"
- **Status**: DynamicDispatch found but dependencies missing
- **Action**: Set `BOOST_ROOT` or ensure XRT is installed in DYNAMICDISPATCH_ROOT
- **Result**: Build continues without DynamicDispatch

### Linker errors about `combined_gemm` or `iconv` symbols
- **Status**: Expected - DynamicDispatch library needs template instantiations
- **Action**: Contact DynamicDispatch maintainers to add instantiations
- **Workaround**: Build without DYNAMICDISPATCH_ROOT set
