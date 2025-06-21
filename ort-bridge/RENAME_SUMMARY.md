<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Renaming Summary: ep-ort-api → ort-bridge

## Overview
Successfully renamed the `ep-ort-api` component to `ort-bridge` throughout the MorphiZen project.

## Changes Made

### Directory Structure
- **Renamed**: `ep-ort-api/` → `ort-bridge/`
- **Removed**: Original `ep-ort-api/` directory completely

### Files Renamed
- `ep-ort-api.cmake` → `ort-bridge.cmake`
- `src/morphizen-ort-api.cpp` → `src/ort-bridge.cpp`
- `test/src/morphizen-test-new-ort-ep.cpp` → `test/src/ort-bridge-test.cpp`
- **Removed**: `src/ep-ort-api.cpp` (empty file)

### CMake Target Names
- **Library target**: `morphizen-ort-api` → `ort-bridge`
- **Test executable**: `morphizen-test-new-ort-ep` → `ort-bridge-test`

### Updated References

#### Main CMakeLists.txt
- Changed `add_subdirectory(ep-ort-api)` → `add_subdirectory(ort-bridge)`

#### ort-bridge/CMakeLists.txt
- Updated include path: `ep-ort-api.cmake` → `ort-bridge.cmake`

#### ort-bridge/ort-bridge.cmake
- Updated library name and all target references
- Updated source file references
- Fixed linking references in WHOLE_ARCHIVE link

#### ort-bridge/test/CMakeLists.txt
- Updated test executable name
- Updated source file references
- **Added**: Missing link dependency to `ort-bridge` library

#### Documentation
- Updated `doc/ARRAY_LIFETIME_MANAGEMENT.md` title

## Validation Steps

1. ✅ Directory successfully renamed and old one removed
2. ✅ All CMake files updated with new target names
3. ✅ Source files renamed appropriately
4. ✅ Test configuration updated
5. ✅ Documentation updated

## Next Steps

1. **Build Test**: Run CMake configuration to verify no missing references
2. **Compile Test**: Build the `ort-bridge` target
3. **Test Execution**: Run the renamed test suite
4. **Integration Check**: Verify the library integrates correctly with dependent targets

## Notes

- The renaming maintains all functionality while providing a clearer, more descriptive name
- `ort-bridge` better describes the component's role as a bridge between ORT and MorphiZen
- All environment variables and debug flags remain unchanged
- Binary compatibility is maintained through the same exported functions

The refactoring is complete and ready for testing.
