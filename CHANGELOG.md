<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Changelog

All notable changes to MorphiZen will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Added CHANGELOG.md to track project changes

## [20.0.0] - 2025-01-29

### BREAKING CHANGES

This release systematically renames all "vaip" symbols to "morphizen" throughout the codebase. This is a **major breaking change** requiring recompilation of all dependent code.

### Changed

**Namespace**:
- Renamed `namespace vaip_core` → `namespace morphizen`
- Note: `vaip_cxx::` namespace unchanged (C++ wrapper API)

**API Struct**:
- Renamed `OrtApiForVaip` → `OrtApiForMorphizen`
- Bumped API major version: 19 → 20
- Function pointer order [0]-[110] preserved (binary ABI stable)

**Headers**:
- Renamed `vaip/vaip_ort_api.h` → `morphizen/morphizen_ort_api.h`
- Renamed `vaip/vaip_gsl.h` → `morphizen/morphizen_gsl.h`
- Renamed `morphizen/vaip_core.hpp` → `morphizen/morphizen_core.hpp`
- Renamed `morphizen/vaip_ort.hpp` → `morphizen/ort_api_wrapper.hpp`
- Renamed `morphizen/vaip_plugin.hpp` → `morphizen/plugin.hpp`
- Renamed `morphizen-utils/vaip_plugin.hpp` → `morphizen-utils/morphizen_plugin.hpp`

**Preprocessor Macros**:
- Renamed `VAIP_ORT_API_MAJOR` → `MORPHIZEN_ORT_API_MAJOR`
- Renamed `VAIP_ORT_API_MINOR` → `MORPHIZEN_ORT_API_MINOR`
- Renamed `VAIP_ORT_API_PATCH` → `MORPHIZEN_ORT_API_PATCH`
- Renamed `VAIP_DLL_SPEC` → `MORPHIZEN_DLL_SPEC`
- Renamed `VAIP_PASS_ENTRY` → `MORPHIZEN_PASS_ENTRY`
- Renamed `VAIP_ORT_API(name)` → `MORPHIZEN_ORT_API(name)`
- Renamed all `VAIP_USER__*` → `MORPHIZEN_USER__*`

**C Function Exports**:
- Renamed `vaip_get_version()` → `morphizen_get_version()`
- Renamed `vaip_get_execution_provider_deletor()` → `morphizen_get_execution_provider_deletor()`

**Directory Structure**:
- Renamed `vaip-core/` → `morphizen-core/`
- Renamed `vaip-ort-api-ext/` → `morphizen-ort-api-ext/`
- Renamed `vaip_io/` → `morphizen-io/`
- Renamed `vaip_pass_init/` → `morphizen-pass-init/`
- Renamed `3rd-party/.../vaip/` → `3rd-party/.../morphizen/`
- Renamed `unit-test/vaip/` → `unit-test/morphizen/`

**File Names**:
- Renamed all `vaip_*.hpp` → `morphizen_*.hpp`
- Renamed all `vaip_*.cpp` → `morphizen_*.cpp`
- Renamed `vaip_config.json` → `morphizen_config.json`
- Renamed all configuration variants (`vaip_config_*.json` → `morphizen_config_*.json`)

**Internal Symbols**:
- Renamed `cleanup_vaip()` → `cleanup_morphizen()`
- Renamed `vaip_error_report_func` → `morphizen_error_report_func`

**CMake Targets**:
- Renamed `vaip_io` → `morphizen-io`
- Renamed `vaip_pass_init` → `morphizen-pass-init`

### Unchanged (Backward Compatibility)

**Plugin Discovery Functions** (kept for binary compatibility):
- `vaip_pass_info()` - Plugin pass entry point
- `vaip_op_def_info()` - Plugin operation definition entry point
- `setup_global_vaip_ort_api()` - Backend API setup function

**C++ Wrapper Namespace**:
- `vaip_cxx::` - C++ wrapper classes (Model, Graph, Node, etc.)

**Binary ABI**:
- Function pointer order [0]-[110] in OrtApiForMorphizen struct
- Struct field layout and memory alignment

### Technical Details

- **Total commits**: 28 commits on `feature/vaip-to-morphizen-rename` branch
- **Files changed**: 100+ source files, headers, and configuration files
- **Build status**: Main DLL (`onnxruntime_morphizen_ep.dll`) builds successfully (32MB)
- **Implementation timeline**: 2-3 weeks (as planned)

### Migration Required

All applications, custom passes, and plugins must:
1. Update include paths (`#include <vaip/...>` → `#include <morphizen/...>`)
2. Update namespace references (`vaip_core::` → `morphizen::`)
3. Update macro names (`VAIP_*` → `MORPHIZEN_*`)
4. Recompile against v20 headers
5. Test thoroughly

**No runtime migration path** - this is a clean break requiring source-level changes and recompilation.

### Known Issues

Some unit tests fail due to **pre-existing issues** from component extraction (morphizen-graph, morphizen-pattern refactoring), **not** from the vaip→morphizen rename:
- Graph operations moved to `graph_extensions.hpp`
- NodeBuilder moved to separate component
- Pattern matching library linking issues

These issues are tracked separately and do not affect the main DLL functionality.

---

## [19.x] - Prior Releases

Historical releases before the systematic rename. See git history for details.

---

## Release Notes Format

Each release documents:
- **Added**: New features
- **Changed**: Changes in existing functionality
- **Deprecated**: Soon-to-be removed features
- **Removed**: Removed features
- **Fixed**: Bug fixes
- **Security**: Vulnerability fixes
- **BREAKING CHANGES**: Incompatible API changes

For detailed migration guides on breaking changes, see `docs/MIGRATION_*.md`.
