<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Issue #047: Extract Duplicate Entry Deduplication Logic in TarFile

## Metadata
- **Type:** Refactoring
- **Priority:** MEDIUM
- **Created:** 2026-02-03
- **Dependencies:** Related to #044 (both improve TarFile deduplication code)

## Description

Extract duplicated entry deduplication logic from `add_regular_entry()` and `add_symlink_entry()` into a private helper function. Both methods contain identical ~14 lines of code that remove existing entries by path, violating the DRY principle.

## Problem

**Current behavior:**

Both `add_regular_entry()` and `add_symlink_entry()` in tar_file.cpp contain nearly identical deduplication logic:

**Location 1:** `add_regular_entry()` (lines 303-316):
```cpp
auto it = std::remove_if(
    entries_.begin(), entries_.end(), [&path](const auto& entry) {
      auto ret = entry->path() == path;
      if (ret) {
        MY_LOG(1)
            << " add_symlink_entry: duplicated entry found, remove old entry "
            << entry->to_string() //
            ;
      }
      return ret;
    });
if (it != entries_.end()) {
  entries_.erase(it, entries_.end());
}
```

**Location 2:** `add_symlink_entry()` (lines 362-376):
```cpp
auto it = std::remove_if(
    entries_.begin(), entries_.end(), [&symlink_name](const auto& entry) {
      auto ret = entry->path() == symlink_name;
      if (ret) {
        MY_LOG(1)
            << " add_symlink_entry: duplicated entry found, remove old entry "
            << entry->to_string() //
            ;
      }
      return ret;
    });
if (it != entries_.end()) {

  entries_.erase(it, entries_.end());
}
```

**The duplication has caused a bug:** The log message in `add_regular_entry()` incorrectly says "add_symlink_entry" (copy-paste error).

**Why this is problematic:**

1. **Violates DRY principle** - Same logic duplicated in two places
2. **Already caused a bug** - Wrong function name in log message
3. **Maintenance burden** - Changes must be made in both places
4. **Risk of divergence** - One location might get updated without the other

## Solution

### 1. Extract Helper Function

Create private helper `remove_duplicate_entry(const std::string& path)`:

```cpp
private:
  void remove_duplicate_entry(const std::string& path);
```

**Implementation:**
```cpp
void TarFile::remove_duplicate_entry(const std::string& path) {
  entries_.erase(
    std::remove_if(entries_.begin(), entries_.end(),
                   [&path](const auto& entry) { return entry->path() == path; }),
    entries_.end());
}
```

**Design decisions:**
- **No logging inside helper** - Simpler implementation, callers can log if needed
- **Uses erase-remove idiom** - Addresses Issue #044 at the same time (no unnecessary if-check)
- **Name:** `remove_duplicate_entry` - more semantic than `remove_entry_by_path`

### 2. Update Callers

Replace duplicated code in both `add_regular_entry()` and `add_symlink_entry()`:

```cpp
// Before (14 lines of duplicated code)
auto it = std::remove_if(...);
if (it != entries_.end()) {
  entries_.erase(it, entries_.end());
}

// After (1 line)
remove_duplicate_entry(path); // or symlink_name
```

### 3. Add Documentation

Add concise documentation in tar_file.hpp explaining the difference between the two methods:

**For `add_regular_entry()`:**
```cpp
/// Adds a regular file entry to the tar archive.
/// Removes any existing entry with the same path (TAR last-wins semantics).
/// @param path - Entry path in the archive
/// @param data_begin_pos - Start of file data in tar stream
/// ...
```

**For `add_symlink_entry()`:**
```cpp
/// Adds a symlink entry to the tar archive.
/// Removes any existing entry with the same path (TAR last-wins semantics).
/// Resolves symlink target via find_real_entry(); uses lazy resolution if not found.
/// @param symlink_name - Symlink path in the archive
/// @param real_path_name - Target path that the symlink points to
/// ...
```

## Rationale

**Why these methods are different (but share deduplication):**

- **`add_regular_entry()`**: Adds a regular file entry with actual data positions from tar stream. After deduplication, calls `add_entry(path, std::nullopt, ...)` where `std::nullopt` means "no symlink target".

- **`add_symlink_entry()`**: Adds a symlink entry pointing to another file. After deduplication, resolves the symlink by calling `find_real_entry(real_path_name)`. If found, creates symlink pointing to real entry's data. If not found, creates symlink with invalid positions (-1) for lazy resolution.

**The key difference:**
- Regular entry: Stores actual file data at specific positions
- Symlink entry: Points to another entry and needs to resolve the target

**Why deduplication is the SAME:**
Both need to remove old entries with the same path (TAR "last entry wins" semantics). Deduplication doesn't care about entry type - just removes by path.

## Plans

- [047-extract-duplicate-entry-deduplication-logic-plan.md](../plans/047-extract-duplicate-entry-deduplication-logic-plan.md) - Created 2026-02-03

## Notes

**Discovery:** While exploring TarFile organizational improvements, found identical deduplication pattern in two methods with a copy-paste bug in the log message.

**Related issues:**
- Issue #044: Use erase-remove idiom (this issue addresses the same code locations)
- Both can be fixed together in the helper function implementation
