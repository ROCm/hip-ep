<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Issue #048: Extract Platform-Specific tmpfile Helper Function

## Metadata
- **Type:** Refactoring
- **Priority:** LOW
- **Dependencies:** None

## Description

Extract duplicated platform-specific tmpfile creation code into a reusable helper function. The same `#ifdef _WIN32` pattern for creating temporary files is duplicated in 4 different locations across the codebase.

## Problem

**Current behavior:**

Platform-specific tmpfile creation code is duplicated in 4 locations:

**Location 1:** `tar_file.cpp:63-67` (TarFile::create)
```cpp
#ifdef _WIN32
  auto file = tmpfile_with_posix_delete();
#else
  auto file = std::tmpfile();
#endif
```

**Location 2:** `tar_file.cpp:111-115` (TarFile::create with mmap)
```cpp
#ifdef _WIN32
  auto file = tmpfile_with_posix_delete();
#else
  auto file = std::tmpfile();
#endif
```

**Location 3:** `pass_context_imp.cpp:66-71` (write_to_tmp_file)
```cpp
#if _WIN32
  FILE* tmp_file = tmpfile_with_posix_delete();
  CHECK(tmp_file != nullptr) << "tmpfile_with_posix_delete error";
#else
  FILE* tmp_file = tmpfile();
  CHECK(tmp_file != nullptr) << "cannot create tmp file";
#endif
```

**Location 4:** `temp_file_stream.cpp:12-16` (TempFileStream constructor)
```cpp
#ifdef _WIN32
  FILE* file = tmpfile_with_posix_delete();
#else
  FILE* file = tmpfile();
#endif
```

**Why this is problematic:**

1. **Violates DRY principle** - Same platform-specific logic in 4 places
2. **Maintenance burden** - Platform changes need updates in 4 locations
3. **Inconsistent formatting** - Some use `#if _WIN32`, others use `#ifdef _WIN32`
4. **Harder to test** - Platform-specific code scattered across multiple files

## Solution

### 1. Create Helper Function

Add inline helper function `create_tmpfile()` in `util.hpp`:

```cpp
/// Creates a temporary file using platform-specific tmpfile implementation.
/// @return FILE* pointer to temporary file, or nullptr on failure.
///         Callers MUST check for nullptr and handle errors appropriately.
inline FILE* create_tmpfile() {
#ifdef _WIN32
  return tmpfile_with_posix_delete();
#else
  return std::tmpfile();
#endif
}
```

**Design decisions:**
- **Location:** `util.hpp` - alongside existing `tmpfile_with_posix_delete()` declaration
- **Inline function:** Simple wrapper, no .cpp file needed
- **No error checking:** Preserves existing behavior - some callers want fallback, others want to crash
- **Documentation:** Clear comment that callers must check for nullptr

### 2. Replace All Usages

Replace the 4 duplicated blocks with calls to `create_tmpfile()`.

**Locations to update:**
- tar_file.cpp (2 locations)
- pass_context_imp.cpp (1 location)
- temp_file_stream.cpp (1 location)

**Example replacement:**

Before:
```cpp
#ifdef _WIN32
  auto file = tmpfile_with_posix_delete();
#else
  auto file = std::tmpfile();
#endif
```

After:
```cpp
auto file = create_tmpfile();
```

**Note:** Error handling remains unchanged - each caller continues to handle nullptr as appropriate for their context.

## Plans

- [048-extract-platform-specific-tmpfile-helper-plan.md](../plans/048-extract-platform-specific-tmpfile-helper-plan.md) - Created 2026-02-03

## Notes

**Discovery:** While exploring TarFile organizational improvements, found platform-specific tmpfile creation duplicated in 4 locations.

**Error handling patterns:**
- tar_file.cpp: Checks for nullptr, falls back to stringstream if tmpfile fails
- pass_context_imp.cpp: CHECKs and crashes if tmpfile fails
- temp_file_stream.cpp: CHECKs and crashes if tmpfile fails

The helper function preserves these different error handling strategies by not including error checking itself.

**Future improvement noted:** User observed that `tmpfile_with_posix_delete()` location in util.hpp might not be ideal - can address in separate issue.
