<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Issue #044: Use Standard Erase-Remove Idiom in TarFile

## Metadata
- **Type:** Code Quality / Refactoring
- **Priority:** LOW
- **Started:** 2026-02-05
- **Dependencies:** None

## Description

Replace verbose erase-remove pattern with standard C++ erase-remove idiom in TarFile entry deduplication logic. Current code has unnecessary `if` check that adds complexity without benefit.

## Problem

**Current code** in `morphizen-core/src/tar_file.cpp`:

**Lines 303-316** (`add_regular_entry()`):
```cpp
auto it = std::remove_if(
    entries_.begin(), entries_.end(), [&path](const auto& entry) {
      auto ret = entry->path() == path;
      if (ret) {
        MY_LOG(1)
            << " add_symlink_entry: duplicated entry found, remove old entry "
            << entry->to_string();
      }
      return ret;
    });
if (it != entries_.end()) {  // ❌ Unnecessary check
  entries_.erase(it, entries_.end());
}
```

**Lines 362-376** (`add_symlink_entry()`):
```cpp
auto it = std::remove_if(
    entries_.begin(), entries_.end(), [&symlink_name](const auto& entry) {
      auto ret = entry->path() == symlink_name;
      if (ret) {
        MY_LOG(1)
            << " add_symlink_entry: duplicated entry found, remove old entry "
            << entry->to_string();
      }
      return ret;
    });
if (it != entries_.end()) {  // ❌ Unnecessary check
  entries_.erase(it, entries_.end());
}
```

**Why the check is unnecessary:**
- If `std::remove_if` finds no matches, it returns `end()`
- Calling `erase(end(), end())` is valid and does nothing (empty range)
- The `if` check adds 2 lines of code without providing any value
- This pattern appears in both functions (duplicate verbose code)

## Solution

Use the standard **erase-remove idiom** - single line, no intermediate variable:

**Before:**
```cpp
auto it = std::remove_if(entries_.begin(), entries_.end(), predicate);
if (it != entries_.end()) {
  entries_.erase(it, entries_.end());
}
```

**After:**
```cpp
entries_.erase(
    std::remove_if(entries_.begin(), entries_.end(), predicate),
    entries_.end()
);
```

**Files to modify:**
- `morphizen-core/src/tar_file.cpp:303-316` - `add_regular_entry()`
- `morphizen-core/src/tar_file.cpp:362-376` - `add_symlink_entry()`

**Benefits:**
- ✅ Follows standard C++ idiom (more recognizable)
- ✅ Reduces code from 4 lines to 1 line (per location)
- ✅ Removes unnecessary conditional check
- ✅ Total reduction: ~6 lines (3 lines × 2 locations)

## Implementation

1. Replace verbose pattern in `add_regular_entry()` (lines 303-316)
2. Replace verbose pattern in `add_symlink_entry()` (lines 362-376)
3. Keep the logging lambda unchanged (still needed)
4. Build and verify tests pass

**Verification:**
```bash
# Build
cmake --build ../../build/morphizen-core --config Debug --parallel

# Run tar tests
../../build/morphizen-core/bin/morphizen-unit-tests.exe --gtest_filter=*Tar*
```

## Notes

**This is a pure refactoring** - no behavioral changes, just cleaner code following standard C++ idioms.

**Related:** This issue was discovered while analyzing code duplication in TarFile entry management. A separate issue will address the duplication itself.
