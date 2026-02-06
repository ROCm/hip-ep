<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Implementation Plan: Issue #060 - Create morphizen-foundation Library

## Overview

Create `morphizen-foundation/` as the lowest architectural layer, containing generic, reusable utilities with ZERO MorphiZen dependencies. Move env_config, parse_value, mem_binary, and encryption into this new foundation layer.

## Implementation Steps

### Step 1: Create morphizen-foundation Directory Structure

**Action:**
```bash
mkdir -p morphizen-foundation/include/morphizen-foundation
mkdir -p morphizen-foundation/src
mkdir -p morphizen-foundation/test
```

**Expected structure:**
```
morphizen-foundation/
├── CMakeLists.txt
├── README.md
├── include/morphizen-foundation/
│   ├── env_config.hpp
│   ├── parse_value.hpp
│   ├── mem_binary.hpp
│   └── encryption.hpp
├── src/
│   ├── env_config.cpp
│   ├── mem_binary.cpp
│   ├── encryption.cpp
│   └── compress_binary.py
└── test/
    └── (tests to be added later)
```

### Step 2: Create morphizen-foundation/CMakeLists.txt

**File:** `morphizen-foundation/CMakeLists.txt`

**Content structure:**
```cmake
cmake_minimum_required(VERSION 3.22)
project(
  morphizen-foundation
  VERSION 1.0.0
  LANGUAGES C CXX)

# CRITICAL DEPENDENCY CHECK
# morphizen-foundation MUST have ZERO MorphiZen dependencies
# Only external libraries allowed: glog, GSL, zlib, OpenSSL

find_package(ZLIB QUIET)
set(ENABLE_COMPRESSION "0")
if(TARGET ZLIB::ZLIB)
  set(ENABLE_COMPRESSION "1")
endif()

find_package(OpenSSL QUIET)
set(WITH_OPENSSL "0")
if(TARGET OpenSSL::Crypto)
  set(WITH_OPENSSL "1")
endif()

# Generate mem_binary embedded resources
set(MEM_BINARY_CONTENT_FILE mem_binary_file.hpp.inc)
add_custom_command(
  OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${MEM_BINARY_CONTENT_FILE}"
  COMMAND
    ${CMAKE_COMMAND} -E env "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/../tools"
    $<TARGET_FILE:Python3::Interpreter>
    ${CMAKE_CURRENT_SOURCE_DIR}/src/compress_binary.py
    "${ENABLE_COMPRESSION}"
    "${CMAKE_CURRENT_BINARY_DIR}/${MEM_BINARY_CONTENT_FILE}"
    "${MORPHIZEN_EMBEDDED_RESOURCE_PATH}")

set_source_files_properties(
  "${CMAKE_CURRENT_BINARY_DIR}/${MEM_BINARY_CONTENT_FILE}" PROPERTIES GENERATED TRUE)

configure_file(src/mem_binary.cpp "${CMAKE_CURRENT_BINARY_DIR}/mem_binary.cpp" @ONLY)

# Source files
set(MORPHIZEN_FOUNDATION_SOURCES
    src/env_config.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/mem_binary.cpp
    src/encryption.cpp
)

set(MORPHIZEN_FOUNDATION_HEADERS
    include/morphizen-foundation/env_config.hpp
    include/morphizen-foundation/parse_value.hpp
    include/morphizen-foundation/mem_binary.hpp
    include/morphizen-foundation/encryption.hpp
)

# Create library
add_library(morphizen-foundation STATIC
    ${MORPHIZEN_FOUNDATION_SOURCES}
    ${MORPHIZEN_FOUNDATION_HEADERS}
    "${CMAKE_CURRENT_BINARY_DIR}/${MEM_BINARY_CONTENT_FILE}"
)

add_library(morphizen::foundation ALIAS morphizen-foundation)

# Include directories
target_include_directories(morphizen-foundation
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# Compiler options
target_compile_options(morphizen-foundation PRIVATE ${MORPHIZEN_COMPILER_OPTIONS})

# Link external libraries ONLY (NO MorphiZen dependencies)
target_link_libraries(morphizen-foundation
    PUBLIC
        glog::glog
        Microsoft.GSL::GSL
)

if(TARGET ZLIB::ZLIB)
  target_link_libraries(morphizen-foundation PRIVATE ZLIB::ZLIB)
  target_compile_definitions(morphizen-foundation PRIVATE ENABLE_COMPRESSION=1)
endif()

if(TARGET OpenSSL::Crypto)
  target_link_libraries(morphizen-foundation PRIVATE OpenSSL::Crypto)
  target_compile_definitions(morphizen-foundation PRIVATE WITH_OPENSSL=1)
endif()

# Verification: Ensure no MorphiZen dependencies
# This comment serves as documentation - actual enforcement via code review
# If needed in future, add CMake check here to fail build on morphizen-* dependencies
```

**Key points:**
- ONLY external dependencies (glog, GSL, zlib, OpenSSL)
- Comment documenting dependency rule
- Handles optional dependencies (zlib, OpenSSL)
- Includes mem_binary code generation

### Step 3: Move Files to morphizen-foundation

**Files to move from morphizen-utils:**

```bash
# Move env_config
git mv morphizen-utils/include/morphizen-utils/env_config.hpp \
        morphizen-foundation/include/morphizen-foundation/env_config.hpp
git mv morphizen-utils/src/env_config.cpp \
        morphizen-foundation/src/env_config.cpp

# Move parse_value
git mv morphizen-utils/include/morphizen-utils/parse_value.hpp \
        morphizen-foundation/include/morphizen-foundation/parse_value.hpp
```

**Files to move from mem_binary:**

```bash
# Move mem_binary headers
git mv mem_binary/include/morphizen/mem_binary.hpp \
        morphizen-foundation/include/morphizen-foundation/mem_binary.hpp

# Move mem_binary source
git mv mem_binary/src/mem_binary.cpp \
        morphizen-foundation/src/mem_binary.cpp
git mv mem_binary/src/compress_binary.py \
        morphizen-foundation/src/compress_binary.py

# Move mem_binary tests (optional, for later)
# git mv mem_binary/test morphizen-foundation/test/mem_binary
```

**Files to move from encryption:**

```bash
# Move encryption headers
git mv encryption/include/morphizen/encryption.hpp \
        morphizen-foundation/include/morphizen-foundation/encryption.hpp

# Move encryption source
git mv encryption/src/encryption.cpp \
        morphizen-foundation/src/encryption.cpp
```

**After moving:**
```bash
# Remove old directories (if empty)
git rm -r mem_binary/
git rm -r encryption/
```

### Step 4: Update Header Files - Namespace and Includes

**Update all moved headers:**

**env_config.hpp:**
- Change namespace from `morphizen::utils` to `morphizen::foundation`
- Update internal includes: `"./parse_value.hpp"` (same directory)

**parse_value.hpp:**
- Change namespace from `morphizen::utils` to `morphizen::foundation`

**mem_binary.hpp:**
- Keep namespace `morphizen` (no change for backwards compatibility)
- Update includes to use foundation paths

**encryption.hpp:**
- Keep namespace `morphizen_encryption` (no change for backwards compatibility)

**mem_binary.cpp:**
- Update `#include "morphizen/mem_binary.hpp"` → `#include "morphizen-foundation/mem_binary.hpp"`
- Update `#include <morphizen-utils/morphizen-utils.hpp>` → `#include "morphizen-foundation/env_config.hpp"`
- Remove `#include <morphizen-utils/morphizen_plugin.hpp>` (not needed)
- Update namespace usage: `morphizen::foundation` for env_config

**encryption.cpp:**
- Update `#include "morphizen/encryption.hpp"` → `#include "morphizen-foundation/encryption.hpp"`

**env_config.cpp:**
- Update `#include` paths to foundation

### Step 5: Create morphizen-foundation/README.md

**File:** `morphizen-foundation/README.md`

**Content:**

```markdown
# MorphiZen Foundation

**Generic, reusable C++ utilities with ZERO MorphiZen dependencies.**

## ⚠️ CRITICAL DEPENDENCY RULE ⚠️

**This library MUST have ZERO dependencies on other MorphiZen components.**

### Forbidden Dependencies
- ❌ morphizen-utils
- ❌ morphizen-core
- ❌ morphizen-graph
- ❌ morphizen-pattern
- ❌ Any other morphizen-* component

### Allowed Dependencies
- ✅ External libraries only (glog, GSL, zlib, OpenSSL)
- ✅ C++ standard library

### Rationale
morphizen-foundation is designed to be reusable in ANY C++ project without
requiring the MorphiZen framework. All components here are generic infrastructure
utilities with no domain-specific logic.

### Code Review Checklist
When reviewing changes to morphizen-foundation:
- [ ] No `#include` statements reference morphizen-utils, morphizen-core, etc.
- [ ] No `target_link_libraries` includes MorphiZen components
- [ ] All dependencies are external libraries or standard library
- [ ] No domain-specific or framework-specific logic

---

## Purpose

morphizen-foundation provides the lowest architectural layer of the MorphiZen
project, containing generic, reusable utilities that can be used in any C++
project.

**Architectural layers:**
```
morphizen-foundation (generic utilities, zero MorphiZen coupling)
    ↓
morphizen-utils (framework-specific utilities)
    ↓
morphizen-core (compilation engine)
    ↓
applications (execution providers, demos)
```

---

## Components

### 1. env_config - Type-Safe Environment Variable Access

Provides compile-time type-safe access to environment variables with caching
and default values.

**Features:**
- Type-safe conversion (string, int, bool, etc.)
- Cached after first access for performance
- Supports default values
- Cross-platform (uses `vitis_ai_getenv_s` when available)

**Example usage:**
```cpp
#include <morphizen-foundation/env_config.hpp>

// Define environment parameter
DEF_ENV_PARAM(DEBUG_LEVEL, "0")

// Use it
int level = ENV_PARAM(DEBUG_LEVEL);
if (level > 0) {
    LOG(INFO) << "Debug level: " << level;
}
```

**Namespace:** `morphizen::foundation`

### 2. parse_value - Generic String Parsing

Robust string-to-type conversion utilities with error checking.

**Features:**
- Parse strings to common types (int, double, bool, etc.)
- Error handling for invalid conversions
- Used internally by env_config

**Example usage:**
```cpp
#include <morphizen-foundation/parse_value.hpp>

std::string str = "42";
int value = morphizen::foundation::parse_value<int>(str);
```

**Namespace:** `morphizen::foundation`

### 3. mem_binary - Build-Time Binary Resource Embedding

Embeds binary files directly into the library at build time with optional
compression support.

**Features:**
- Embed any binary file at build time
- Optional zlib compression (if available)
- Simple API: `get_mem_binary()`, `has_mem_binary()`, `get_mem_binary_span()`
- Python-based code generation

**Build-time configuration:**
- Set `MORPHIZEN_EMBEDDED_RESOURCE_PATH` to path of `embedded_resource.txt`
- Resource file format: Python literal eval (JSON-like with comments)

**Example usage:**
```cpp
#include <morphizen-foundation/mem_binary.hpp>

// Check if resource exists
if (morphizen::has_mem_binary("config.json")) {
    // Get resource as vector
    auto data = morphizen::get_mem_binary("config.json");

    // Or get as span (zero-copy)
    auto span = morphizen::get_mem_binary_span("config.json");
    std::string content(span->data(), span->size());
}
```

**Namespace:** `morphizen`

### 4. encryption - AES-256 Encryption/Decryption

Provides AES-256 ECB encryption and decryption for streams.

**Features:**
- AES-256 ECB mode encryption/decryption
- Stream-based API (works with any std::istream/ostream)
- Optional dependency (requires OpenSSL)
- Exception-based error handling

**Example usage:**
```cpp
#include <morphizen-foundation/encryption.hpp>

// Check if encryption support is available
if (morphizen_encryption::has_encryption_support()) {
    std::ifstream src("plaintext.txt", std::ios::binary);
    std::ofstream dst("encrypted.bin", std::ios::binary);

    morphizen_encryption::aes_encryption(src, dst, "my-secret-key");
}
```

**Namespace:** `morphizen_encryption`

---

## Building

morphizen-foundation is built as part of the MorphiZen project.

**Dependencies:**
- **Required:** glog, Microsoft.GSL
- **Optional:** zlib (for mem_binary compression), OpenSSL (for encryption)

**Build:**
```bash
cmake -S . -B ../../build/$(basename $PWD) -DCMAKE_BUILD_TYPE=Debug
cmake --build ../../build/$(basename $PWD) --config Debug
```

---

## Architecture Context

morphizen-foundation sits at the bottom of the MorphiZen architectural stack:

**Layer 0: Foundation** (this library)
- Generic utilities with zero MorphiZen coupling
- Reusable in any C++ project
- Components: env_config, parse_value, mem_binary, encryption

**Layer 1: morphizen-utils**
- Framework-specific utilities
- Depends on foundation
- Components: morphizen_plugin, cleanup, weak_refs

**Layer 2+: morphizen-core, morphizen-graph, etc.**
- Core framework components
- Depend on utils and foundation

See `docs/architecture.md` for complete architectural overview.

---

## Adding New Components

Before adding a component to morphizen-foundation, verify it meets ALL criteria:

**✅ Should go in morphizen-foundation if:**
- Generic, domain-agnostic utility
- No MorphiZen-specific logic or dependencies
- Reusable in other C++ projects
- Only depends on external libraries (glog, GSL, etc.)
- Size: typically <300 LOC per component

**❌ Should NOT go in morphizen-foundation if:**
- MorphiZen-specific functionality
- Depends on morphizen-utils, morphizen-core, etc.
- Domain-specific logic (graph manipulation, ONNX, etc.)
- Framework coupling required

See `docs/technical/component-organization-guidelines.md` for detailed decision criteria.

---

## License

Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
```

### Step 6: Update morphizen-utils

**Update morphizen-utils/CMakeLists.txt:**

Remove env_config and parse_value from sources/headers:
```cmake
set(MORPHIZEN_UTILS_SOURCES
    src/morphizen_plugin.cpp
    src/cleanup.cpp
)

set(MORPHIZEN_UTILS_HEADERS
    include/morphizen-utils/morphizen-utils.hpp
    include/morphizen-utils/morphizen_plugin.hpp
    include/morphizen-utils/cleanup.hpp
    include/morphizen-utils/weak_refs.hpp
)
```

Add foundation dependency:
```cmake
target_link_libraries(morphizen-utils
    PUBLIC morphizen::foundation
    PRIVATE glog::glog
)
```

**Update morphizen-utils/include/morphizen-utils/morphizen-utils.hpp:**

Remove old includes, add foundation includes:
```cpp
#pragma once

// Import foundation utilities
#include <morphizen-foundation/env_config.hpp>
#include <morphizen-foundation/parse_value.hpp>

// Local utilities
#include "./morphizen_plugin.hpp"
#include "./cleanup.hpp"
#include "./weak_refs.hpp"

// Re-export foundation namespace for backwards compatibility
namespace morphizen::utils {
    using namespace morphizen::foundation;
}
```

### Step 7: Update Root CMakeLists.txt

**Update build order:**

Change from:
```cmake
add_subdirectory(morphizen-utils)
add_subdirectory(mem_binary)
add_subdirectory(encryption)
```

To:
```cmake
add_subdirectory(morphizen-foundation)  # NEW - first
add_subdirectory(morphizen-utils)       # depends on foundation
# mem_binary and encryption removed (moved to foundation)
```

**Complete ordering:**
```cmake
# Foundation layer (no MorphiZen dependencies)
add_subdirectory(morphizen-foundation)

# Utility layer (depends on foundation)
add_subdirectory(morphizen-utils)

# Rest of the project...
add_subdirectory(morphizen-ort-api-ext)
add_subdirectory(onnx-ir-imp)
# ...
```

### Step 8: Update morphizen-core Dependencies

**Update morphizen-core/cmake/morphizen-core-static.cmake:**

Change from:
```cmake
target_link_libraries(
  morphizen-core-static
  PRIVATE
  # ...
  morphizen::encryption
  morphizen::mem_binary
  # ...
)
```

To:
```cmake
target_link_libraries(
  morphizen-core-static
  PRIVATE
  # ...
  morphizen::foundation  # Replaces encryption + mem_binary
  # ...
)
```

### Step 9: Update Include Paths Across Codebase

**Find all files that include moved headers:**

```bash
# Find files including old paths
grep -r "#include.*morphizen/mem_binary.hpp" --include="*.cpp" --include="*.hpp"
grep -r "#include.*morphizen/encryption.hpp" --include="*.cpp" --include="*.hpp"
grep -r "#include.*morphizen-utils/env_config.hpp" --include="*.cpp" --include="*.hpp"
grep -r "#include.*morphizen-utils/parse_value.hpp" --include="*.cpp" --include="*.hpp"
```

**Update all includes:**

- `#include "morphizen/mem_binary.hpp"` → `#include "morphizen-foundation/mem_binary.hpp"`
- `#include "morphizen/encryption.hpp"` → `#include "morphizen-foundation/encryption.hpp"`
- `#include <morphizen-utils/env_config.hpp>` → `#include <morphizen-foundation/env_config.hpp>`
- `#include <morphizen-utils/parse_value.hpp>` → `#include <morphizen-foundation/parse_value.hpp>`

**Files likely affected:**
- morphizen-core/src/pass_context_imp.cpp
- morphizen-core/src/morphizen_compile_model.cpp
- Any other files using these utilities

**For env_config users:** If they include `<morphizen-utils/morphizen-utils.hpp>`, no change needed (re-exported).

### Step 10: Update Namespace Usage

**For env_config/parse_value users:**

Option 1: Update to use foundation namespace directly:
```cpp
// Old
using morphizen::utils::ENV_PARAM;

// New
using morphizen::foundation::ENV_PARAM;
```

Option 2: Keep using utils namespace (re-exported for backwards compatibility):
```cpp
// Still works due to re-export in morphizen-utils.hpp
using morphizen::utils::ENV_PARAM;
```

**Recommendation:** Use Option 2 for minimal disruption, update to Option 1 incrementally.

### Step 11: Build and Test

**Build commands:**
```bash
# Configure
LOCAL_DIR=$(cd ../../local && pwd)
cmake -S . -B ../../build/$(basename $PWD) -DBUILD_SHARED_LIBS=OFF \
  "-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded\$<\$<CONFIG:Debug>:Debug>" \
  -DCMAKE_BUILD_TYPE=Debug "-DCMAKE_PREFIX_PATH=$LOCAL_DIR" \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -Dmorphizen_ENABLE_UNIT_TEST=ON --fresh

# Build
cmake --build ../../build/$(basename $PWD) --config Debug --parallel

# Test
../../build/$(basename $PWD)/bin/morphizen-unit-tests.exe
```

**Verify:**
- [ ] morphizen-foundation builds without errors
- [ ] morphizen-utils builds (depends on foundation)
- [ ] morphizen-core builds (depends on foundation)
- [ ] All unit tests pass
- [ ] No linker errors

### Step 12: Commit and Push

**Commit message:**
```bash
git add morphizen-foundation/ \
        morphizen-utils/ \
        CMakeLists.txt \
        morphizen-core/cmake/morphizen-core-static.cmake \
        $(git status --porcelain | grep "^ M" | awk '{print $2}')  # All modified files

git commit -m "$(cat <<'EOF'
refactor: create morphizen-foundation library

Creates morphizen-foundation as the lowest architectural layer, containing
generic, reusable utilities with ZERO MorphiZen dependencies.

Changes:
- Create morphizen-foundation/ library (new)
- Move env_config, parse_value from morphizen-utils
- Move mem_binary/ (entire component)
- Move encryption/ (entire component)
- Update morphizen-utils to depend on foundation
- Update morphizen-core to link against foundation
- Update root CMakeLists.txt build order

Components in morphizen-foundation (~600 LOC):
- env_config: Type-safe environment variable access
- parse_value: Generic string parsing utilities
- mem_binary: Build-time binary resource embedding
- encryption: AES-256 encryption/decryption

Architectural layers:
  morphizen-foundation (generic, zero MorphiZen coupling)
      ↓
  morphizen-utils (framework-specific utilities)
      ↓
  morphizen-core (compilation engine)

CRITICAL DEPENDENCY RULE:
morphizen-foundation MUST have ZERO dependencies on morphizen-utils,
morphizen-core, or any MorphiZen component. Only external dependencies
allowed (glog, GSL, zlib, OpenSSL). See morphizen-foundation/README.md.

Rationale:
- Generic utilities should be reusable in ANY C++ project
- Clear architectural separation (foundation vs framework)
- Eliminates conceptual circular dependency (mem_binary → utils)
- Provides home for future generic utilities

Implements Issue #060 and follows guidelines from Issue #059.
EOF
)"

git push -u fork feature/create-morphizen-foundation
```

### Step 13: Create Draft PR

**Command:**
```bash
gh pr create --draft --title "Issue #060: refactor: create morphizen-foundation library" --body "$(cat <<'EOF'
## Summary

Creates `morphizen-foundation` as the lowest architectural layer, containing generic, reusable utilities with ZERO MorphiZen dependencies.

This refactoring consolidates scattered generic utilities (mem_binary, encryption) and extracts generic parts from morphizen-utils (env_config, parse_value) into a clean foundation layer that can be reused in any C++ project.

## Problem

**Before this PR:**
- Generic utilities scattered: mem_binary/ (standalone), encryption/ (standalone), env_config (in morphizen-utils)
- Conceptual circular dependency: mem_binary depends on morphizen-utils for env_config
- No clear separation between "generic infrastructure" and "MorphiZen framework utilities"
- Cannot reuse generic utilities in other projects (MorphiZen coupling)

## Solution

**After this PR:**
```
morphizen-foundation/     (~600 LOC, ZERO MorphiZen deps)
├── env_config           (type-safe env vars)
├── parse_value          (string parsing)
├── mem_binary           (resource embedding)
└── encryption           (AES-256)
    ↓
morphizen-utils/         (~400 LOC, framework-specific)
├── morphizen_plugin     (plugin system)
├── cleanup              (cleanup utilities)
└── weak_refs            (weak singleton patterns)
    ↓
morphizen-core/          (compilation engine)
```

**Clear architectural layers:**
1. **Foundation** (generic, reusable anywhere)
2. **Utils** (framework-specific)
3. **Core** (compilation engine)

## Critical Dependency Rule

⚠️ **morphizen-foundation MUST have ZERO MorphiZen dependencies** ⚠️

**Forbidden:**
- ❌ morphizen-utils
- ❌ morphizen-core
- ❌ morphizen-graph
- ❌ Any morphizen-* component

**Allowed:**
- ✅ External libs only (glog, GSL, zlib, OpenSSL)
- ✅ Standard library

**Rationale:** Foundation must be reusable in ANY C++ project without requiring MorphiZen framework.

**Enforcement:** Documented prominently in README.md, enforced via code review.

## Changes

### Created
- `morphizen-foundation/` directory with CMakeLists.txt
- `morphizen-foundation/README.md` - Comprehensive documentation with:
  - Critical dependency rules (prominently displayed)
  - Purpose and rationale
  - Component descriptions
  - Usage examples
  - Architecture context
  - Guidelines for adding new components

### Moved
- `morphizen-utils/include/morphizen-utils/env_config.hpp` → `morphizen-foundation/include/morphizen-foundation/env_config.hpp`
- `morphizen-utils/src/env_config.cpp` → `morphizen-foundation/src/env_config.cpp`
- `morphizen-utils/include/morphizen-utils/parse_value.hpp` → `morphizen-foundation/include/morphizen-foundation/parse_value.hpp`
- `mem_binary/*` → `morphizen-foundation/` (entire component)
- `encryption/*` → `morphizen-foundation/` (entire component)

### Updated
- `CMakeLists.txt` - Build order: foundation before utils
- `morphizen-utils/CMakeLists.txt` - Depend on foundation, remove moved files
- `morphizen-utils/include/morphizen-utils/morphizen-utils.hpp` - Re-export foundation for backwards compatibility
- `morphizen-core/cmake/morphizen-core-static.cmake` - Link foundation instead of individual components
- Include paths across codebase (mem_binary.hpp, encryption.hpp, env_config.hpp)

### Removed
- `mem_binary/` directory (moved to foundation)
- `encryption/` directory (moved to foundation)

## Benefits

✅ **Reusability**: Generic utilities can be used in ANY C++ project
✅ **Architectural clarity**: Clear separation foundation vs framework
✅ **Clean dependencies**: Foundation → Utils → Core (no circular concepts)
✅ **Developer guidance**: Clear home for future generic utilities
✅ **Consistency**: Implements two-tier utilities pattern from #059

## Testing

- [ ] morphizen-foundation builds successfully
- [ ] morphizen-utils builds (depends on foundation)
- [ ] morphizen-core builds (depends on foundation)
- [ ] All unit tests pass
- [ ] No linker errors
- [ ] Include paths correctly updated

## Backwards Compatibility

**Namespace re-export:**
```cpp
// morphizen-utils/morphizen-utils.hpp re-exports foundation
namespace morphizen::utils {
    using namespace morphizen::foundation;
}
```

**Result:** Existing code using `morphizen::utils::ENV_PARAM` continues to work.

**Include paths:** Updated throughout codebase.

## Related Issues

- Implements: Issue #060 (Create morphizen-foundation)
- Follows: Issue #059 (Component organization guidelines) - two-tier utilities pattern
- Originated from: mem_binary organization discussion

## Files Changed

- **Created**: ~15 files in morphizen-foundation/
- **Moved**: ~10 files from mem_binary/, encryption/, morphizen-utils/
- **Updated**: ~10 files (CMakeLists.txt, includes, dependencies)
- **Removed**: 2 directories (mem_binary/, encryption/)

**Total lines changed:** ~600 LOC moved + ~100 LOC CMake updates + comprehensive README.md

## Next Steps

After merge:
1. Future generic utilities go in morphizen-foundation
2. Framework-specific utilities go in morphizen-utils
3. Reference #059 guidelines for decision criteria

---

**Review focus areas:**
1. ✅ Verify morphizen-foundation has NO morphizen-* dependencies in CMakeLists.txt
2. ✅ Check README.md clearly documents dependency rules
3. ✅ Ensure all include paths updated correctly
4. ✅ Verify build order (foundation before utils)
5. ✅ Confirm tests pass
EOF
)"
```

## Verification Steps

After implementation:

1. **Check directory structure:**
   ```bash
   ls -la morphizen-foundation/
   # Should show: CMakeLists.txt, README.md, include/, src/
   ```

2. **Verify dependency rule in CMakeLists.txt:**
   ```bash
   grep "target_link_libraries.*morphizen-foundation" morphizen-foundation/CMakeLists.txt
   # Should ONLY show external libs (glog, GSL, zlib, OpenSSL)
   # Should NOT show morphizen::utils or morphizen::core
   ```

3. **Check README.md has dependency warning:**
   ```bash
   grep "CRITICAL DEPENDENCY RULE" morphizen-foundation/README.md
   # Should find prominent warning section
   ```

4. **Verify build order:**
   ```bash
   grep -A5 "add_subdirectory(morphizen-foundation)" CMakeLists.txt
   # morphizen-foundation should come BEFORE morphizen-utils
   ```

5. **Build test:**
   ```bash
   cmake --build ../../build/$(basename $PWD) --config Debug --parallel
   # Should build successfully
   ```

6. **Run tests:**
   ```bash
   ../../build/$(basename $PWD)/bin/morphizen-unit-tests.exe
   # All tests should pass
   ```

7. **Check for broken includes:**
   ```bash
   # Should find NO results:
   grep -r "#include.*morphizen/mem_binary.hpp" --include="*.cpp" --include="*.hpp"
   grep -r "#include.*morphizen/encryption.hpp" --include="*.cpp" --include="*.hpp"
   ```

## Success Criteria

- [ ] morphizen-foundation/ directory created with proper structure
- [ ] CMakeLists.txt has NO morphizen-* dependencies
- [ ] README.md prominently displays dependency rules
- [ ] All 4 components moved (env_config, parse_value, mem_binary, encryption)
- [ ] morphizen-utils depends on foundation
- [ ] morphizen-core links against foundation
- [ ] Root CMakeLists.txt updated (foundation before utils)
- [ ] All include paths updated
- [ ] Project builds successfully
- [ ] All unit tests pass
- [ ] No linker errors
- [ ] Branch pushed to fork
- [ ] Draft PR created with comprehensive description

## Edge Cases and Considerations

1. **Namespace migration:**
   - env_config/parse_value change from `morphizen::utils` to `morphizen::foundation`
   - Provide re-export in morphizen-utils.hpp for backwards compatibility
   - Can migrate callers incrementally

2. **Build-time code generation:**
   - mem_binary uses Python script to generate .hpp.inc file
   - Ensure compress_binary.py moved and CMake paths updated
   - PYTHONPATH may need adjustment

3. **Optional dependencies:**
   - zlib (for mem_binary compression)
   - OpenSSL (for encryption)
   - Handle gracefully when not available (feature flags)

4. **Test migration:**
   - mem_binary has tests in mem_binary/test/
   - Consider moving to morphizen-foundation/test/ (or defer to future PR)

5. **Documentation references:**
   - Update any docs that reference mem_binary/ or encryption/ paths
   - Update architecture.md to mention foundation layer

## Notes

**Why all-at-once migration?**
- Single coherent architectural change
- Cleaner git history
- Foundation only useful with multiple components
- Total change manageable (~600 LOC moved, not modified)

**Why these 4 components?**
- **env_config/parse_value**: Generic utilities, no domain logic
- **mem_binary**: Generic resource embedding, no MorphiZen logic
- **encryption**: Generic AES, already has zero MorphiZen deps

**Why NOT include in foundation?**
- **morphizen_plugin**: MorphiZen-specific plugin system
- **cleanup**: Framework cleanup utilities
- **weak_refs**: Framework-specific singleton patterns

**Relationship to Issue #059:**
- #059 documents guidelines
- #060 implements two-tier utilities pattern
- Foundation = lowest tier, utils = framework tier
