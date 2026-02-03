<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Issue #037: Remove Dead Code from TarFile (rename_symlink and rename_existing_entry)

## Metadata
- **Type:** Refactoring
- **Priority:** MEDIUM
- **Created:** 2026-02-02
- **Dependencies:** None

## Description

Remove `rename_symlink()` and `rename_existing_entry()` functions that have been dead code since April 2025. These functions were abandoned when TarFile design shifted from "modify-in-place" to "append-only" semantics but were never deleted.

## Problem

**Current code contains unused functions with const_cast violations:**

```cpp
// tar_entry.cpp:65-81
bool TarEntryInputStreamBuffer::rename_symlink(const std::string& new_name,
                                               pos_type data_begin_pos,
                                               pos_type data_end_pos) {
  if (real_path_) {
    const_cast<std::optional<std::string>&>(real_path_) = new_name;  // ❌
    const_cast<pos_type&>(data_begin_pos_) = data_begin_pos;         // ❌
    const_cast<pos_type&>(data_end_pos_) = data_end_pos;             // ❌
    const_cast<pos_type&>(buffer_pos_) = data_begin_pos;             // ❌
    // ...
  }
}

// tar_entry.cpp:222-226 - Wrapper function
bool TarEntryInputStream::rename_symlink(...) {
  return buf_->rename_symlink(...);
}

// tar_entry.cpp:475-496 - Only caller (never invoked)
void TarEntryOutputStream::rename_existing_entry(...) {
  prev_entry.rename_symlink(...);
  // ... modify tar header in-place
}
```

**Why this is problematic:**

1. **Security/Correctness**: Four `const_cast` violations that modify const members
2. **Misleading Code**: Functions suggest features that don't exist and violate append-only design
3. **Maintenance Burden**: ~60 lines of dead code that needs to be maintained
4. **Documentation Mismatch**: Comment in `tar_file.hpp:103` incorrectly describes renaming behavior

**Git History - How It Became Dead:**

**April 9, 2025** (Commit `87d108d`) - "refactor tar entry (#32)"
- Initial implementation where `rename_existing_entry()` **was called** in Case 4 (file overwrite)
- Original design: Modify existing tar header in-place, convert to symlink
- Related JIRA: VAI-10873, VAI-10864

**April 14, 2025** (Commit `9081edc`) - "Br verify write same content twice (#74)"
- **Removed the call** to `rename_existing_entry()` - only 5 days after adding it!
- Reason: "TAR is append only" - modifying headers violates this principle
- New approach: Just append new entry, last entry wins (standard TAR semantics)
- Functions left in codebase but never deleted (oversight)

**Result**: Dead code for **9 months** (April 2025 - Feb 2026)

**Code locations:**
- `tar_entry.cpp:65-81` - `TarEntryInputStreamBuffer::rename_symlink()` implementation
- `tar_entry.cpp:222-226` - `TarEntryInputStream::rename_symlink()` wrapper
- `tar_entry.cpp:475-496` - `TarEntryOutputStream::rename_existing_entry()` (only caller, never invoked)
- `tar_entry.hpp:28-29, 105-106, 155-156` - Function declarations
- `tar_file.hpp:103-105` - Misleading comment about renaming entries

## Solution

**Delete all dead code:**

1. Remove 3 function implementations from `tar_entry.cpp`
2. Remove 3 function declarations from `tar_entry.hpp`
3. Update misleading comment in `tar_file.hpp`

**Updated documentation (tar_file.hpp:103-105):**
```cpp
// OLD (incorrect):
// if the filename already exists, the orinal entry will be renamed to a wired
// invisiable name,
// TODO: check if tar file header support delete flag.

// NEW (correct):
// If the filename already exists, a new entry is appended to the tar file.
// When reading, the last entry with a given name takes precedence (standard
// TAR semantics). The tar file is append-only; existing entries are never
// modified.
```

**Approach:**
1. Delete dead function implementations and declarations
2. Fix documentation to reflect actual append-only behavior
3. Verify no references remain via `git grep`
4. Run all tar-related tests to confirm no behavioral changes

**Benefits:**
- ✅ Eliminate const_cast violations (security/correctness improvement)
- ✅ Remove misleading code (~60 lines)
- ✅ Clarify append-only design with accurate documentation
- ✅ Reduce maintenance burden
- ✅ No behavioral changes (code already unused)

## Plans

- [037-remove-tarfile-dead-code-plan.md](../plans/037-remove-tarfile-dead-code-plan.md) - Created 2026-02-02

## Notes

Verification confirmed no code calls these functions:
```bash
git grep -n "rename_existing_entry" -- "*.cpp" "*.hpp"
# Only finds: declaration, definition

git grep -n "\.rename_symlink" -- "*.cpp" "*.hpp"
# Only finds: internal call within rename_existing_entry()
```

This is pure code removal - the functions have been abandoned for 9 months with no impact on functionality.
