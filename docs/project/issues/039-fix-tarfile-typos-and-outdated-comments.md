<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Issue #039: Fix Typos and Outdated Comments in TarFile

## Metadata
- **Type:** Documentation / Code Cleanup
- **Priority:** LOW
- **Created:** 2026-02-02
- **Dependencies:** None

## Description

Fix multiple documentation issues in TarFile implementation: outdated comment describing removed renaming behavior, typos in comments, incorrect FIXME comment, and unused parameter removal.

## Problem

**Current documentation issues:**

**1. Outdated comment (tar_file.hpp:103-105):**
```cpp
// if the filename already exists, the orinal entry will be renamed to a wired
// invisiable name,
// TODO: check if tar file header support delete flag.
```

❌ **Problem**: Describes behavior that was removed in April 2025 (commit 9081edc). The `rename_existing_entry()` function was abandoned 5 days after creation when design shifted to append-only semantics.

**2. Typos (tar_file.hpp:107, 110):**
```cpp
// after the stream is closed, a new entries is append to the tar stream_.
// ... indidcate the end of the tar file.
```

❌ **Problems**: "entries is append" → should be "entry is appended", "indidcate" → should be "indicate"

**3. Incorrect FIXME (tar_entry.cpp:342):**
```cpp
// FIXME , the param "name" is not used
void TarEntryOutputStream::maybe_add_4k_align(TarFile& tar_file,
                                              const std::string& name) {
```

❌ **Problem**: FIXME is **wrong** - `name` IS used in `add_padding_block_for_4k()` at line 468 for logging.

**4. Unused parameter (tar_entry.cpp:433):**
```cpp
// FIXME , the param "name" is not used
void TarEntryOutputStream::add_1024_padding(const std::string& /*name*/) {
```

❌ **Problem**: FIXME is correct - parameter truly unused and should be removed.

**Why this is problematic:**

1. **Misleading documentation** - Developers expect renaming behavior that doesn't exist
2. **Maintenance confusion** - Outdated comments create false understanding of the code
3. **Code quality** - Typos and incorrect comments suggest lack of attention to detail
4. **Unused code** - Dead parameter clutters API

**Code locations:**
- `tar_file.hpp:103-105` - Outdated renaming comment
- `tar_file.hpp:107, 110` - Typos
- `tar_entry.cpp:342` - Incorrect FIXME
- `tar_entry.cpp:433` - Unused parameter with correct FIXME
- `tar_entry.hpp:150` - Parameter declaration to remove
- `tar_entry.cpp:513, 517, 526, 533` - Caller sites to update

## Solution

**Fix outdated comment with accurate description:**
```cpp
// If the filename already exists, a new entry is appended to the tar file.
// When reading, the last entry with a given name takes precedence (standard
// TAR semantics). The tar file is append-only; existing entries are never
// modified.
```

**Approach:**
1. Replace outdated renaming description with accurate append-only semantics
2. Fix typos: "entries is append" → "entry is appended", "indidcate" → "indicate"
3. Remove incorrect FIXME at line 342 (parameter IS used)
4. Remove unused `name` parameter from `add_1024_padding()`:
   - Update declaration in tar_entry.hpp
   - Update implementation in tar_entry.cpp
   - Update 4 caller sites to remove argument

**Benefits:**
- ✅ Accurate documentation matching actual behavior
- ✅ Remove misleading information about non-existent features
- ✅ Professional code quality (no typos)
- ✅ Cleaner API (remove unused parameter)
- ✅ Zero behavioral changes (documentation/cleanup only)

## Plans

- [039-fix-tarfile-typos-and-outdated-comments-plan.md](../plans/039-fix-tarfile-typos-and-outdated-comments-plan.md) - Created 2026-02-02

## Notes

**Context**: This issue was identified as Topic #4 during comprehensive TarFile code review. Previous topics:
- Topic #1 (Issue #037): Remove dead code
- Topic #2 (Issue #038): Document lazy symlink const_cast
- Topic #3: Circular symlink vulnerability → TODO document

**Impact**: Documentation-only changes with one minor API cleanup (unused parameter removal). No runtime behavior changes.
