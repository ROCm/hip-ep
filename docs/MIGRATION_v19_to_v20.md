<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Migration Guide: API v19 (vaip) to v20 (morphizen)

## Breaking Changes

This is a **breaking change** release. All "vaip" names have been systematically renamed to "morphizen" throughout the codebase.

### API Version

- **API version**: 19.x → **20.0**
- Major version bump indicates breaking changes
- No backward compatibility layer provided

### Critical Constraint Preserved

**Binary ABI compatibility maintained** - the function pointer order [0]-[110] in the OrtApiForMorphizen struct has been preserved. Internal ABI remains stable for the same major version.

## Namespace Changes

### C++ Namespace

```cpp
// OLD (v19)
namespace vaip_core {
  const OrtApiForVaip* api();
  void set_the_global_api(OrtApiForVaip* api);
}

// NEW (v20)
namespace morphizen {
  const OrtApiForMorphizen* api();
  void set_the_global_api(OrtApiForMorphizen* api);
}
```

**Note**: The `vaip_cxx::` namespace remains unchanged - it's the C++ wrapper API and was not part of the rename.

## Header File Changes

### Include Directives

```cpp
// OLD (v19)
#include <vaip/vaip_ort_api.h>
#include <vaip/vaip_gsl.h>
#include <morphizen/vaip_core.hpp>
#include <morphizen/vaip_ort.hpp>
#include <morphizen/vaip_plugin.hpp>

// NEW (v20)
#include <morphizen/morphizen_ort_api.h>
#include <morphizen/morphizen_gsl.h>
#include <morphizen/morphizen_core.hpp>
#include <morphizen/ort_api_wrapper.hpp>
#include <morphizen/plugin.hpp>
```

### Vendored Headers Location

```
OLD: 3rd-party/onnxruntime-morphizen-headers/vaip/
NEW: 3rd-party/onnxruntime-morphizen-headers/morphizen/
```

## API Struct Changes

### Main API Struct

```cpp
// OLD (v19)
struct OrtApiForVaip {
  uint32_t magic; // 'VAIP'
  uint32_t major; // 19
  // ... function pointers [0]-[110]
};

// NEW (v20)
struct OrtApiForMorphizen {
  uint32_t magic; // 'VAIP' (unchanged for compat check)
  uint32_t major; // 20
  // ... function pointers [0]-[110] (same order)
};
```

## Macro Changes

### Preprocessor Macros

```cpp
// OLD (v19)
VAIP_ORT_API_MAJOR
VAIP_ORT_API_MINOR
VAIP_ORT_API_PATCH
VAIP_DLL_SPEC
VAIP_PASS_ENTRY
VAIP_ORT_API(name)
VAIP_USER__*

// NEW (v20)
MORPHIZEN_ORT_API_MAJOR
MORPHIZEN_ORT_API_MINOR
MORPHIZEN_ORT_API_PATCH
MORPHIZEN_DLL_SPEC
MORPHIZEN_PASS_ENTRY
MORPHIZEN_ORT_API(name)
MORPHIZEN_USER__*
```

## C Function Exports

### Renamed Exports

```cpp
// OLD (v19)
extern "C" uint32_t vaip_get_version();
extern "C" void* vaip_get_execution_provider_deletor();

// NEW (v20)
extern "C" uint32_t morphizen_get_version();
extern "C" void* morphizen_get_execution_provider_deletor();
```

### Unchanged (Existing EPs)

```cpp
// These remain unchanged
extern "C" void compile_onnx_model_morphizen_ep_with_options(...);
extern "C" void compile_onnx_model_morphizen_ep_v4(...);
extern "C" void initialize_onnxruntime_morphizen_ep(...);
extern "C" void deinitialize_onnxruntime_morphizen_ep(...);
extern "C" void morphizen_ep_on_run_start(...);
extern "C" void morphizen_ep_set_ep_dynamic_options(...);
extern "C" void morphizen_get_build_info(...);
```

## CMake Target Changes

Most CMake targets already used "morphizen" names. Updated targets:

```cmake
# OLD (v19)
morphizen-io        # Was: vaip_io
morphizen-pass-init # Was: vaip_pass_init

# NEW (v20)
morphizen-io
morphizen-pass-init
```

Main library target remains:
```cmake
onnxruntime_morphizen_ep  # DLL name unchanged
```

## Directory Structure Changes

### Component Directories

```
OLD (v19)                    NEW (v20)
├── vaip-core/              ├── morphizen-core/
├── vaip-ort-api-ext/       ├── morphizen-ort-api-ext/
├── vaip_io/                ├── morphizen-io/
├── vaip_pass_init/         ├── morphizen-pass-init/
└── unit-test/vaip/         └── unit-test/morphizen/
```

## Configuration Files

### Config File Naming

```
OLD: vaip_config.json, vaip_config_*.json
NEW: morphizen_config.json, morphizen_config_*.json
```

Location: `morphizen-core/etc/`

## Internal Type Changes

```cpp
// OLD (v19)
cleanup_vaip()
vaip_error_report_func

// NEW (v20)
cleanup_morphizen()
morphizen_error_report_func
```

## Plugin Discovery (Unchanged)

**These function names remain unchanged** for binary compatibility with existing plugins:

```cpp
// Still used (not renamed)
extern "C" MORPHIZEN_PASS_ENTRY morphizen::PassInfo* vaip_pass_info();
extern "C" MORPHIZEN_PASS_ENTRY morphizen::OpDefInfo* vaip_op_def_info();
```

Plugins using these names will continue to work without recompilation.

## Migration Steps

### For Application Developers

1. **Update include paths**:
   ```cpp
   // Search and replace in your code
   #include <vaip/ → #include <morphizen/
   vaip_core::     → morphizen::
   OrtApiForVaip   → OrtApiForMorphizen
   ```

2. **Update macro names**:
   ```cpp
   VAIP_ORT_API(   → MORPHIZEN_ORT_API(
   VAIP_DLL_SPEC   → MORPHIZEN_DLL_SPEC
   ```

3. **Rebuild against new headers**:
   - Link against `morphizen-core`, `morphizen-ort-api-ext`, etc.
   - Update `CMAKE_PREFIX_PATH` if needed

4. **Test thoroughly**:
   - Verify function calls work
   - Check custom passes load correctly
   - Validate configuration parsing

### For Pass/Plugin Developers

1. **Update pass entry points** (optional):
   - `vaip_pass_info()` still works (backward compat)
   - Consider migrating to new registration API in future

2. **Update configuration**:
   - Rename `vaip_config.json` → `morphizen_config.json`
   - Update any hardcoded "vaip" strings in configs

3. **Recompile plugins**:
   - Link against v20 headers
   - Update namespace references
   - Test pass registration

### For Library Users (ONNX Runtime)

If you use MorphiZen as an ONNX Runtime execution provider:

```cpp
// OLD (v19)
Ort::SessionOptions().AppendExecutionProvider_VitisAI(vaip_options);

// NEW (v20) - API unchanged, internal impl updated
Ort::SessionOptions().AppendExecutionProvider_VitisAI(morphizen_options);
```

The external ONNX Runtime API remains unchanged - only internal symbols were renamed.

## Binary Compatibility

### What Changed

- ✅ Struct name: `OrtApiForVaip` → `OrtApiForMorphizen`
- ✅ Namespace: `vaip_core` → `morphizen`
- ✅ Symbols: All internal symbols renamed
- ✅ Version: 19 → 20 (major bump)

### What Stayed Same

- ✅ Function pointer order [0]-[110] (ABI preserved)
- ✅ Struct field order (memory layout unchanged)
- ✅ Plugin discovery functions (`vaip_pass_info`, `vaip_op_def_info`)
- ✅ Magic number compatibility check logic

### Compatibility Matrix

| Component | v19 (old) | v20 (new) | Compatible? |
|-----------|-----------|-----------|-------------|
| Applications | Uses OrtApiForVaip | Uses OrtApiForMorphizen | ❌ Must recompile |
| Custom Passes | Links v19 headers | Links v20 headers | ❌ Must recompile |
| Plugins (discovery) | vaip_pass_info() | vaip_pass_info() | ✅ Binary compat |
| Config Files | vaip_config.json | morphizen_config.json | ⚠️ Rename required |

## Testing Checklist

After migration, verify:

- [ ] Application compiles without errors
- [ ] Custom passes register successfully
- [ ] Configuration files load correctly
- [ ] Model compilation works
- [ ] Inference executes successfully
- [ ] No undefined symbol errors
- [ ] Version check passes (v20 detected)

## Known Issues

### Pre-existing Test Failures

Some unit tests fail due to pre-existing issues from component extraction (morphizen-graph, morphizen-pattern refactoring), not from the v19→v20 rename:

- `test_graph.cpp`: Methods moved to `graph_extensions.hpp`
- `test_node_builder.cpp`: NodeBuilder moved to separate component
- `test_immutable_map.cpp`: File path changed after extraction

These are unrelated to the rename and will be addressed separately.

## Getting Help

- **Documentation**: See `docs/architecture.md` for updated API reference
- **Issues**: Report problems at https://github.com/ROCm/MorphiZen/issues
- **Examples**: Check `unit-test/` for usage examples with v20 API

## Summary

The v19→v20 migration is a **systematic rename** with no functional changes:

- **What changed**: All "vaip" symbols → "morphizen"
- **What stayed**: Binary ABI (function pointer order), plugin discovery
- **Migration effort**: Search/replace + rebuild
- **Timeline**: 2-3 weeks implementation (completed)

This is a clean break with no backward compatibility layer, enabling clearer naming going forward.
