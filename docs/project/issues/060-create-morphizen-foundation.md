<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Issue #060: Create morphizen-foundation Library

## Problem

Small, generic utility components (mem_binary ~111 LOC, encryption ~184 LOC) are scattered as standalone top-level directories alongside MorphiZen-specific utilities (morphizen-utils ~500+ LOC). This creates several architectural issues:

1. **Missed reusability opportunity**: Generic utilities (env_config, mem_binary, encryption) are coupled with MorphiZen-specific code, preventing reuse in other projects

2. **Unclear architectural boundaries**: No clear separation between "generic infrastructure" and "framework-specific utilities"

3. **Dependency confusion**: mem_binary depends on morphizen-utils for env_config, creating circular conceptual dependency (utility depends on utilities)

4. **Inconsistent organization**: Some generic utilities are standalone (mem_binary, encryption), others are in morphizen-utils (env_config)

## Current State

**Directory structure:**
```
morphizen-utils/        (framework-specific utilities)
├── env_config.*        (GENERIC - env var access)
├── parse_value.hpp     (GENERIC - string parsing)
├── morphizen_plugin.*  (MorphiZen-specific)
├── cleanup.*           (MorphiZen-specific)
└── weak_refs.hpp       (MorphiZen-specific)

mem_binary/             (standalone, GENERIC)
├── depends on morphizen-utils (for env_config)
└── ~111 LOC

encryption/             (standalone, GENERIC)
└── ~184 LOC (no MorphiZen dependencies)
```

**Dependency graph:**
```
mem_binary → morphizen-utils
encryption (standalone)
morphizen-utils → morphizen-core
```

## Impact

**Without clear foundation layer:**
- Generic utilities cannot be reused in non-MorphiZen projects
- Conceptual coupling (mem_binary → morphizen-utils for env_config)
- No clear principle for where new generic utilities belong
- Missed opportunity for architectural clarity

**Benefits of foundation layer:**
- **Zero MorphiZen coupling**: Reusable in ANY C++ project
- **Clear separation**: Generic infrastructure vs framework utilities
- **Better dependencies**: Foundation → Utils → Core (clean hierarchy)
- **Developer guidance**: Clear home for generic utilities

## Solution

Create `morphizen-foundation/` library as the lowest architectural layer, containing ONLY generic, reusable utilities with ZERO MorphiZen dependencies.

### Components to Move

**Into morphizen-foundation:**
1. **env_config.hpp/cpp** - Type-safe environment variable access (~200 LOC)
2. **parse_value.hpp** - Generic string parsing utilities (~100 LOC)
3. **mem_binary/*** - Build-time binary resource embedding (~200 LOC)
4. **encryption/*** - AES-256 encryption/decryption (~200 LOC)

**Total:** ~600 LOC of generic, reusable utilities

**Remain in morphizen-utils:**
1. **morphizen_plugin.*** - MorphiZen plugin loading system (MorphiZen-specific)
2. **cleanup.*** - Framework cleanup utilities (MorphiZen-specific)
3. **weak_refs.hpp** - Weak singleton patterns (framework-specific)

**Total:** ~400 LOC of MorphiZen-specific utilities

### New Dependency Graph

```
morphizen-foundation (ZERO MorphiZen deps, only external libs)
    ↓
morphizen-utils (depends on foundation)
    ↓
morphizen-core (depends on utils + foundation)
```

### Critical Dependency Rule

**⚠️ CRITICAL ⚠️**

`morphizen-foundation` MUST have ZERO dependencies on:
- morphizen-utils
- morphizen-core
- morphizen-graph
- Any other MorphiZen-specific component

**Only allowed dependencies:**
- External libraries: glog, GSL, zlib, OpenSSL
- Standard library

**Enforcement:** Prominent README.md documentation + code review

**Rationale:** Foundation must be reusable in ANY C++ project without requiring MorphiZen framework.

## Deliverables

1. **Create morphizen-foundation/ directory** with proper CMake structure
2. **Move 4 components** (env_config, parse_value, mem_binary, encryption)
3. **Comprehensive README.md** documenting:
   - Purpose (reusable foundation with zero MorphiZen coupling)
   - Critical dependency rules (prominently displayed)
   - Component descriptions
   - Usage examples
   - Architecture context
4. **Update morphizen-utils** to depend on foundation
5. **Update morphizen-core** to depend on foundation (not individual components)
6. **Update root CMakeLists.txt** build order (foundation before utils)
7. **Update all include paths** across codebase

## Success Criteria

- [ ] morphizen-foundation builds successfully
- [ ] morphizen-foundation has ZERO MorphiZen dependencies (verified in CMakeLists.txt)
- [ ] All unit tests pass
- [ ] morphizen-utils depends on foundation
- [ ] morphizen-core links against foundation
- [ ] README.md clearly documents dependency rules
- [ ] No broken includes across codebase

## Metadata

- **Type:** Architecture / Refactoring
- **Priority:** MEDIUM
- **Created:** 2026-02-06
- **Component:** Project organization
- **Dependencies:** Related to #059 (Component organization guidelines) - implements two-tier utilities pattern
