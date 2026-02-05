<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Issue #050: Standardize TarFile Factory Method Naming

## Metadata
- **Type:** Refactoring
- **Priority:** MEDIUM
- **Dependencies:** None

## Description

Standardize TarFile factory method naming to use consistent "create_from_X" pattern. Currently only `create_from_path()` uses explicit naming, while all other factories use overloaded `create()`, making the API confusing and hard to discover.

## Problem

**Current naming pattern:**

**Explicit naming (1 method):**
- `create_from_path(path, enable_mmap)` - Reads existing tar file from disk

**Overloaded naming (6 methods):**
- `create()` - Creates empty tar from tmpfile
- `create(stream)` - Takes pre-constructed stream (base factory)
- `create(vector<char>)` - From vector buffer
- `create(string)` - From string buffer (default mmap)
- `create(string, bool)` - From string buffer with mmap control
- `create(char*, size_t)` - From raw data pointer

**Why this is problematic:**

1. **Inconsistent pattern** - Only path-based method has explicit name
2. **Hard to discover** - Users must rely on parameter types, not method names
3. **Unclear purpose** - What does `create()` do? (Creates empty tar? From what?)
4. **Poor IDE experience** - Autocomplete shows 6 overloads of `create()` instead of descriptive names
5. **Maintenance burden** - Must read documentation to know which overload to use

**Example of confusion:**
```cpp
// Which create() should I use?
auto tar1 = TarFile::create();              // ❓ What does this create?
auto tar2 = TarFile::create(my_buffer);     // ❓ Vector or string? Both work
auto tar3 = TarFile::create_from_path(p);   // ✓ Clear!
```

## Solution

### Rename Factory Methods to Follow Consistent Pattern

**Pattern:**
- Low-level: `create(stream)` - Keep as-is (base factory)
- High-level: `create_from_X()` - Use explicit naming

**Renamings:**

```cpp
// Keep as-is (already good):
create(unique_ptr<iostream>&&)          // Base factory - low level
create_from_path(path, enable_mmap)     // Already clear

// Rename (high-level factories):
create()                    → create_from_tmpfile()
create(vector<char>&&)      → create_from_buffer(vector<char>&&)
create(string&&)            → (removed - consolidated into next)
create(string&&, bool)      → create_from_buffer(string&&, bool enable_mmap = true)
create(char*, size_t)       → create_from_data(char*, size_t)
```

**Result after refactoring:**
```cpp
// Low-level (advanced usage)
create(unique_ptr<iostream>&&)

// High-level (self-documenting names)
create_from_path(path, enable_mmap)
create_from_tmpfile()
create_from_buffer(vector<char>&&)
create_from_buffer(string&&, bool enable_mmap = true)
create_from_data(char*, size_t)
```

### Design Decisions

**1. Consolidate string variants:**
- Current: `create(string)` + `create(string, bool)` (wrapper + implementation)
- After: `create_from_buffer(string, bool = true)` (single method with default)
- Reason: The first method just calls the second with `true`, unnecessary duplication

**2. Direct breaking change (no deprecation):**
- Delete old method names
- Update all callers in same PR
- Reason: Internal codebase, no external users, cleaner result

**3. Name choices:**
- `create_from_tmpfile()` - Clear that it creates new empty tar
- `create_from_buffer()` - Owns the buffer (vector or string)
- `create_from_data()` - Non-owning view of raw data

## Plans

- [050-standardize-tarfile-factory-method-naming-plan.md](../plans/050-standardize-tarfile-factory-method-naming-plan.md) - Created 2026-02-03

## Notes

**Discovery:** While analyzing TarFile God Class Pattern, identified factory method proliferation as a sub-problem. Further breakdown revealed naming inconsistency as the first issue to address.

**Benefits of consistent naming:**
1. Self-documenting code
2. Better IDE autocomplete experience
3. Easier to find the right factory method
4. Makes code duplication more visible (helps with Issue #051 - extracting complex logic)

**Related work:**
- This addresses the naming aspect of factory method proliferation
- Follow-up work: Issue #051 will extract complex logic from the long factory methods
