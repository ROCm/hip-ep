<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Issue #041: Remove Duplicate friend Declaration in TarEntryInputStream

## Metadata
- **Type:** Bug Fix / Code Quality
- **Priority:** LOW
- **Created:** 2026-02-03
- **Dependencies:** None

## Description

Remove duplicate `friend class TarFile;` declaration in `TarEntryInputStream` class (tar_entry.hpp:127). This is a copy-paste error with no functional impact but affects code quality.

## Problem

**File:** `morphizen-core/src/tar_entry.hpp`

**Lines 126-127:**
```cpp
friend class TarFile;
friend class TarFile;  // ❌ Duplicate
```

**Why this needs fixing:**
- Copy-paste error
- Violates clean code principles
- While C++ ignores duplicate friend declarations, it's sloppy and may confuse developers

## Solution

**Delete line 127** - keep only one friend declaration.

**Before:**
```cpp
private:
  std::unique_ptr<TarEntryInputStreamBuffer> buf_;
  MemStream<MemFile>* mem_buf_;
  friend class TarFile;
  friend class TarFile;  // Remove this line
};
```

**After:**
```cpp
private:
  std::unique_ptr<TarEntryInputStreamBuffer> buf_;
  MemStream<MemFile>* mem_buf_;
  friend class TarFile;
};
```

## Implementation

1. Delete line 127 in `morphizen-core/src/tar_entry.hpp`
2. Build and verify compilation succeeds
3. Run tests to confirm no behavioral changes

**Verification:**
```bash
# Build
cmake --build ../../build/morphizen-core --config Debug --parallel

# Run tar tests
../../build/morphizen-core/bin/morphizen-unit-tests.exe --gtest_filter=*Tar*
```

## Notes

Trivial 1-line fix. No functional changes expected - C++ allows duplicate friend declarations without error.
