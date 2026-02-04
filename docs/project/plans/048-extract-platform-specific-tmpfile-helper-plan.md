<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Extract Platform-Specific tmpfile Helper Function

**Issue:** #048
**Created:** 2026-02-03
**Status:** READY

## Objective

Extract duplicated platform-specific tmpfile creation code into a reusable inline helper function `create_tmpfile()` in util.hpp. This eliminates code duplication in 4 locations.

## Background

**Problem discovered:** While reviewing TarFile organization, found identical `#ifdef _WIN32` pattern for tmpfile creation in 4 different files.

**Why extraction makes sense:** Platform-specific code should be centralized for maintainability and consistency.

## Implementation Steps

### Step 1: Add Helper Function to util.hpp

**File:** `morphizen-core/src/util.hpp`

**Location:** After the `tmpfile_with_posix_delete()` declaration (around line 50-60)

**Find:**
```cpp
MORPHIZEN_DLL_SPEC FILE* tmpfile_with_posix_delete();
```

**Add after it:**
```cpp
/// Creates a temporary file using platform-specific tmpfile implementation.
/// On Windows, uses tmpfile_with_posix_delete() for better cleanup behavior.
/// On other platforms, uses standard std::tmpfile().
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

### Step 2: Update tar_file.cpp - Location 1

**File:** `morphizen-core/src/tar_file.cpp`

**Location:** Lines 63-67 in `TarFile::create()`

**Find:**
```cpp
std::unique_ptr<TarFile> TarFile::create() {
#ifdef _WIN32
  auto file = tmpfile_with_posix_delete();
#else
  auto file = std::tmpfile();
#endif

  std::unique_ptr<std::iostream> stream;
```

**Replace with:**
```cpp
std::unique_ptr<TarFile> TarFile::create() {
  auto file = create_tmpfile();

  std::unique_ptr<std::iostream> stream;
```

**Changes:** 5 lines → 1 line

### Step 3: Update tar_file.cpp - Location 2

**File:** `morphizen-core/src/tar_file.cpp`

**Location:** Lines 111-115 in `TarFile::create(std::string&&, bool)`

**Find:**
```cpp
std::unique_ptr<TarFile> TarFile::create(std::string&& buffer0,
                                         bool enable_mmap) {
  std::unique_ptr<std::iostream> stream;
#ifdef _WIN32
  auto file = tmpfile_with_posix_delete();
#else
  auto file = std::tmpfile();
#endif
  // by default, the stream will be from a tmp file to decrease memory,
```

**Replace with:**
```cpp
std::unique_ptr<TarFile> TarFile::create(std::string&& buffer0,
                                         bool enable_mmap) {
  std::unique_ptr<std::iostream> stream;
  auto file = create_tmpfile();
  // by default, the stream will be from a tmp file to decrease memory,
```

**Changes:** 5 lines → 1 line

### Step 4: Update pass_context_imp.cpp

**File:** `morphizen-core/src/pass_context_imp.cpp`

**Location:** Lines 66-71 in `write_to_tmp_file()`

**Find:**
```cpp
static FILE* write_to_tmp_file(gsl::span<const char> data) {
#if _WIN32
  FILE* tmp_file = tmpfile_with_posix_delete();
  CHECK(tmp_file != nullptr) << "tmpfile_with_posix_delete error";
#else
  FILE* tmp_file = tmpfile();
  CHECK(tmp_file != nullptr) << "cannot create tmp file";
#endif
  auto write_size = std::fwrite(data.data(), 1, data.size(), tmp_file);
```

**Replace with:**
```cpp
static FILE* write_to_tmp_file(gsl::span<const char> data) {
  FILE* tmp_file = create_tmpfile();
  CHECK(tmp_file != nullptr) << "tmpfile creation error";
  auto write_size = std::fwrite(data.data(), 1, data.size(), tmp_file);
```

**Changes:**
- 6 lines → 2 lines
- Unified error message (was inconsistent between platforms)

### Step 5: Update temp_file_stream.cpp

**File:** `morphizen-core/src/temp_file_stream.cpp`

**Location:** Lines 12-16 in `TempFileStream::TempFileStream()`

**Find:**
```cpp
TempFileStream::TempFileStream() {
#ifdef _WIN32
  FILE* file = tmpfile_with_posix_delete();
#else
  FILE* file = tmpfile();
#endif
  CHECK(file != nullptr) << "Failed to create temporary file";

  // FileStream takes ownership and will close file in destructor
```

**Replace with:**
```cpp
TempFileStream::TempFileStream() {
  FILE* file = create_tmpfile();
  CHECK(file != nullptr) << "Failed to create temporary file";

  // FileStream takes ownership and will close file in destructor
```

**Changes:** 5 lines → 1 line

### Step 6: Ensure util.hpp is included

Verify all modified files include util.hpp:

**tar_file.cpp:** Already includes platform-specific headers
- Add `#include "./util.hpp"` if not present

**pass_context_imp.cpp:** Already uses `tmpfile_with_posix_delete()`
- Should already have util.hpp included

**temp_file_stream.cpp:** Already uses `tmpfile_with_posix_delete()`
- Should already have util.hpp included

**Action:** Check each file's includes and add `#include "./util.hpp"` if missing.

## Verification

### Build
```bash
cmake --build ../../build/morphizen-core --config Debug --parallel
```

### Test
```bash
../../build/morphizen-core/bin/morphizen-unit-tests.exe
```

**Expected:** All tests pass (no behavior change, only refactoring)

### Code Review Checklist

- [ ] `create_tmpfile()` added to util.hpp with documentation
- [ ] tar_file.cpp updated (2 locations)
- [ ] pass_context_imp.cpp updated (1 location)
- [ ] temp_file_stream.cpp updated (1 location)
- [ ] All files include util.hpp
- [ ] Error handling preserved (each caller handles nullptr as before)
- [ ] Build succeeds
- [ ] All tests pass
- [ ] Code reduction: ~20 lines eliminated (5+5+6+5 - 1 helper = 20 lines)

## Success Criteria

- [ ] `create_tmpfile()` helper function implemented in util.hpp
- [ ] All 4 duplicated blocks replaced with helper call
- [ ] Platform-specific `#ifdef` code centralized in one location
- [ ] Error handling behavior unchanged for each caller
- [ ] All tests pass
- [ ] Build succeeds without warnings

## Files Modified

- `morphizen-core/src/util.hpp` - Add inline helper function
- `morphizen-core/src/tar_file.cpp` - Replace 2 locations
- `morphizen-core/src/pass_context_imp.cpp` - Replace 1 location
- `morphizen-core/src/temp_file_stream.cpp` - Replace 1 location

## Notes

**Why no error checking in helper:**
Different callers have different error handling needs:
- tar_file.cpp allows tmpfile to fail and falls back to stringstream
- pass_context_imp.cpp and temp_file_stream.cpp crash with CHECK if tmpfile fails

By keeping the helper minimal (just platform selection), each caller can handle errors appropriately.

**Why inline function:**
Simple wrapper with no implementation complexity - perfect candidate for inline function in header. No .cpp file needed, zero runtime overhead.

**Why util.hpp:**
Logical grouping with existing `tmpfile_with_posix_delete()` function. Platform-specific utilities already live in util.hpp/util_mswin.cpp.

**Code reduction:**
- Before: 5 + 5 + 6 + 5 = 21 lines of duplicated platform-specific code
- After: 1 helper function (7 lines including doc) + 4 call sites = 11 lines
- Net reduction: 10 lines eliminated
