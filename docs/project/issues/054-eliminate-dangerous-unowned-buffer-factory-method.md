<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Issue #054: Eliminate Dangerous Un-Owned Buffer Factory Method

## Metadata
- **Type:** Refactoring
- **Priority:** LOW
- **Created:** 2026-02-04
- **Dependencies:** None

## Description

Remove the dangerous `TarFile::create(const char* data, size_t size)` factory method that requires caller-managed lifetime. This method is only used in one test and contributes to factory method proliferation (7 methods).

## Problem

**Current API:**

```cpp
// tar_file.hpp:82-95
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

**Why this is problematic:**

1. **Dangerous lifetime pattern** - Documentation warns "caller need to ensure `data` and `size` lifetime is long enough" - easy to misuse
2. **Test-only usage** - Only used in `test_tar_file.cpp:392`, NOT in production code
3. **API bloat** - Contributes to 7 factory methods (factory method proliferation)
4. **Encapsulation violation** - Uses `MemStream<int>` with null unique_ptr (no ownership)

**Usage analysis:**

Only 1 call site exists:

```cpp
// test_tar_file.cpp:392
auto tar_file_obj = morphizen::TarFile::create(buf.data(), buf.size());
```

The test doesn't need un-owned buffer semantics - `buf` is not used after this line (scope ends at line 398).

## Solution

Remove the dangerous factory method and migrate the test to use safer owned-buffer API.

### Step 1: Remove Declaration

**File:** `morphizen-core/src/tar_file.hpp:82-95`

Delete the entire documentation block and declaration:
```cpp
// DELETE LINES 82-95
/**
 * @brief Creates a TarFile instance from raw data.
 * ...
 */
static std::unique_ptr<TarFile> create(const char* data, size_t size);
```

### Step 2: Remove Implementation

**File:** `morphizen-core/src/tar_file.cpp:101-107`

Delete the implementation:
```cpp
// DELETE LINES 101-107
std::unique_ptr<TarFile> TarFile::create(const char* base, size_t size) {
  auto stream = std::make_unique<MemStream<int>>(
      MemBuffer<int>::create(base, size, std::unique_ptr<int>()));
  LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_TAR_CACHE))
      << " create a tar file from memory " << (void*)base << " " << size;
  return create(std::move(stream));
}
```

### Step 3: Migrate Test to Safer API

**File:** `unit-test/morphizen/test_tar_file.cpp:392`

Replace un-owned buffer call with owned buffer:

```cpp
// OLD (line 392):
auto tar_file_obj = morphizen::TarFile::create(buf.data(), buf.size());

// NEW (line 392):
auto tar_file_obj = morphizen::TarFile::create(std::move(buf));
```

**Note:** Zero-cost migration - `buf` is not used after line 392 (scope ends at line 398), so we can safely move it.

## Benefits

- ✅ Eliminates dangerous un-owned buffer pattern
- ✅ Reduces public API surface (7 → 6 factory methods)
- ✅ Zero-cost test migration (move instead of copy)
- ✅ Prevents potential lifetime bugs in future code

## Plans

- [054-eliminate-dangerous-unowned-buffer-factory-method-plan.md](../plans/054-eliminate-dangerous-unowned-buffer-factory-method-plan.md) - Created 2026-02-04

## Notes

**Discovery:** Found during Task #14 (Public API Surface Too Large) while analyzing TarFile's 15 public methods.

**Out of scope:** Non-const `entries()` overload removal - will be addressed separately as part of Task #3 (Encapsulation Violation - entries() API).

**API count after this change:**
- Factory methods: 6 (was 7)
- Instance methods: 7 (unchanged)
- Total public API: 14 (was 15)
