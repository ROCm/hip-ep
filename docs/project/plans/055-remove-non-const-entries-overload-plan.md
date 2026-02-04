<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Remove Non-Const entries() Overload from TarFile

**Issue:** #055
**Created:** 2026-02-04
**Status:** READY

## Objective

Remove the unused non-const `entries()` overload from TarFile, enforcing read-only access to the entries vector.

## Implementation Steps

### Step 1: Remove Non-Const Declaration

**File:** `morphizen-core/src/tar_file.hpp`

**Action:** Delete lines 105-106 (non-const overload)

### Step 2: Remove Non-Const Implementation

**File:** `morphizen-core/src/tar_file.cpp`

**Action:** Find and delete the non-const `entries()` implementation

## Verification

**Build:**
```bash
cmake --build ../../build/morphizen-core --config Debug --parallel
```

**Test:**
```bash
../../build/morphizen-core/bin/morphizen-unit-tests.exe --gtest_filter=*Tar*
```

## Success Criteria

- [ ] Non-const `entries()` declaration removed
- [ ] Non-const `entries()` implementation removed
- [ ] Build succeeds
- [ ] All tests pass
- [ ] Public API reduced from 14 to 13 methods

## Files Modified

- `morphizen-core/src/tar_file.hpp` - Remove declaration (lines 105-106)
- `morphizen-core/src/tar_file.cpp` - Remove implementation
