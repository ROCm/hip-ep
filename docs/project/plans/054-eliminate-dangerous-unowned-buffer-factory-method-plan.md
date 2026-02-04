<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Eliminate Dangerous Un-Owned Buffer Factory Method

**Issue:** #054
**Created:** 2026-02-04
**Status:** READY

## Objective

Remove the dangerous `TarFile::create(const char* data, size_t size)` factory method and migrate the single test to use safer owned-buffer API.

## Background

**Discovery:** Found during Task #14 (Public API Surface Too Large) analysis. This method requires caller-managed lifetime, is only used in one test, and contributes to factory method proliferation.

## Implementation Steps

### Step 1: Remove Factory Method Declaration

**File:** `morphizen-core/src/tar_file.hpp`

**Location:** Lines 82-95

**Action:** Delete the entire documentation block and declaration

```cpp
// DELETE THESE LINES (82-95):
/**
 * @brief Creates a TarFile instance from raw data.
 *
 * This function initializes and returns a unique pointer to a TarFile
 * object, which represents the tar file created from the provided raw data
 * and size.
 *
 * the caller need to ensure `data` and `size` lifetime is long enough
 *
 * @param data Pointer to the raw data buffer containing the tar file data.
 * @param size The size of the raw data buffer.
 * @return A unique pointer to the created TarFile instance.
 */
static std::unique_ptr<TarFile> create(const char* data, size_t size);
```

### Step 2: Remove Factory Method Implementation

**File:** `morphizen-core/src/tar_file.cpp`

**Location:** Lines 101-107

**Action:** Delete the implementation

```cpp
// DELETE THESE LINES (101-107):
std::unique_ptr<TarFile> TarFile::create(const char* base, size_t size) {
  auto stream = std::make_unique<MemStream<int>>(
      MemBuffer<int>::create(base, size, std::unique_ptr<int>()));
  LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_TAR_CACHE))
      << " create a tar file from memory " << (void*)base << " " << size;
  return create(std::move(stream));
}
```

### Step 3: Migrate Test to Safer API

**File:** `unit-test/morphizen/test_tar_file.cpp`

**Location:** Line 392

**Change:**

```cpp
// OLD (line 392):
auto tar_file_obj = morphizen::TarFile::create(buf.data(), buf.size());

// NEW (line 392):
auto tar_file_obj = morphizen::TarFile::create(std::move(buf));
```

**Rationale:** Zero-cost migration - `buf` is not used after line 392 (scope ends at line 398), so we can safely move ownership.

## Verification

### Build

```bash
cmake --build ../../build/morphizen-core --config Debug --parallel
```

**Expected:** Build succeeds with no errors

### Test

```bash
../../build/morphizen-core/bin/morphizen-unit-tests.exe --gtest_filter=*Tar*
```

**Expected:** All TAR-related tests pass (including the migrated test)

### Code Review Checklist

**tar_file.hpp:**
- [ ] Declaration removed (lines 82-95)
- [ ] No other references to `create(const char*, size_t)`

**tar_file.cpp:**
- [ ] Implementation removed (lines 101-107)

**test_tar_file.cpp:**
- [ ] Line 392 uses `create(std::move(buf))`
- [ ] Test still validates tar file reading from buffer

**General:**
- [ ] No compilation errors
- [ ] All tests pass
- [ ] No new warnings introduced

## Success Criteria

- [ ] `create(const char*, size_t)` declaration removed from tar_file.hpp
- [ ] `create(const char*, size_t)` implementation removed from tar_file.cpp
- [ ] Test migrated to use `create(std::move(buf))`
- [ ] Build succeeds
- [ ] All TAR tests pass
- [ ] Public API reduced from 15 to 14 methods

## Files Modified

- `morphizen-core/src/tar_file.hpp` - Remove declaration (lines 82-95)
- `morphizen-core/src/tar_file.cpp` - Remove implementation (lines 101-107)
- `unit-test/morphizen/test_tar_file.cpp` - Migrate test (line 392)

## Notes

**API count after this change:**
- Factory methods: 6 (was 7)
- Instance methods: 7 (unchanged)
- Total public API: 14 (was 15)

**Related work:**
- Part of Task #14: Public API Surface Too Large in TarFile
- Contributes to reducing factory method proliferation
