<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Fix Typos and Outdated Comments in TarFile

**Issue:** #039
**Created:** 2026-02-02
**Status:** READY

## Objective

Fix typos and update outdated documentation in TarFile implementation to accurately reflect current append-only semantics and remove misleading comments.

## Background

### Context from Code Review

During comprehensive TarFile code review (Topics #1-4), identified multiple documentation issues:
1. **Outdated comment** describing renaming behavior that was removed in April 2025 (commit 9081edc)
2. **Multiple typos** in tar_file.hpp comments
3. **Incorrect FIXME** claiming parameter is unused when it actually is used
4. **Unused parameter** with correct FIXME that should be removed

These issues accumulated over time and make the code harder to understand.

### Why This Matters

**Outdated comments are worse than no comments** - they actively mislead developers. The comment at tar_file.hpp:103-105 describes entry renaming that never happens in current code, creating false expectations about behavior.

## Implementation Steps

### Step 1: Fix Outdated Comment About Renaming Behavior

**File**: `morphizen-core/src/tar_file.hpp:103-105`

**OLD (WRONG - describes dead code removed in commit 9081edc):**
```cpp
// if the filename already exists, the orinal entry will be renamed to a wired
// invisiable name,
// TODO: check if tar file header support delete flag.
```

**NEW (CORRECT - describes actual append-only behavior):**
```cpp
// If the filename already exists, a new entry is appended to the tar file.
// When reading, the last entry with a given name takes precedence (standard
// TAR semantics). The tar file is append-only; existing entries are never
// modified.
```

**Why**: The old comment describes the `rename_existing_entry()` approach that was abandoned 5 days after creation. Current implementation appends duplicates and relies on "last wins" semantics during reading.

### Step 2: Fix Grammar Typo

**File**: `morphizen-core/src/tar_file.hpp:107`

**OLD:**
```cpp
// after the stream is closed, a new entries is append to the tar stream_.
```

**NEW:**
```cpp
// after the stream is closed, a new entry is appended to the tar stream_.
```

**Fix**: "entries is append" → "entry is appended"

### Step 3: Fix Spelling Typo

**File**: `morphizen-core/src/tar_file.hpp:110`

**OLD:**
```cpp
// there are 1024 zero bytes at the end of the tar file, which is used to
// indidcate the end of the tar file. it should be also available after the
```

**NEW:**
```cpp
// there are 1024 zero bytes at the end of the tar file, which is used to
// indicate the end of the tar file. it should be also available after the
```

**Fix**: "indidcate" → "indicate"

### Step 4: Remove Incorrect FIXME Comment

**File**: `morphizen-core/src/tar_entry.cpp:342`

**OLD:**
```cpp
// FIXME , the param "name" is not used
void TarEntryOutputStream::maybe_add_4k_align(TarFile& tar_file,
                                              const std::string& name) {
```

**NEW:**
```cpp
void TarEntryOutputStream::maybe_add_4k_align(TarFile& tar_file,
                                              const std::string& name) {
```

**Why**: The FIXME is **incorrect** - `name` IS used in `add_padding_block_for_4k()` at line 468 for logging. Just remove the misleading comment.

### Step 5: Remove Unused Parameter

**File 1**: `morphizen-core/src/tar_entry.hpp:150` (declaration)

**OLD:**
```cpp
void add_1024_padding(const std::string& name);
```

**NEW:**
```cpp
void add_1024_padding();
```

**File 2**: `morphizen-core/src/tar_entry.cpp:433-434` (implementation)

**OLD:**
```cpp
// FIXME , the param "name" is not used
void TarEntryOutputStream::add_1024_padding(const std::string& /*name*/) {
```

**NEW:**
```cpp
void TarEntryOutputStream::add_1024_padding() {
```

**File 3**: `morphizen-core/src/tar_entry.cpp` (4 caller sites)

Update all 4 calls to remove the `name_` argument:

**Line 513:**
```cpp
// OLD: add_1024_padding(name_);
// NEW: add_1024_padding();
```

**Line 517:**
```cpp
// OLD: add_1024_padding(name_);
// NEW: add_1024_padding();
```

**Line 526:**
```cpp
// OLD: add_1024_padding(name_);
// NEW: add_1024_padding();
```

**Line 533:**
```cpp
// OLD: add_1024_padding(name_);
// NEW: add_1024_padding();
```

**Why**: Parameter was already commented out as `/*name*/` and FIXME correctly identified it as unused. Clean removal improves code clarity.

## Critical Files

- `morphizen-core/src/tar_file.hpp` - Fix outdated comment (lines 103-105) and typos (lines 107, 110)
- `morphizen-core/src/tar_entry.hpp` - Remove unused parameter from declaration (line 150)
- `morphizen-core/src/tar_entry.cpp` - Remove FIXME (line 342), remove unused parameter (line 433), update 4 caller sites (lines 513, 517, 526, 533)

## Benefits

1. ✅ **Accurate documentation** - Comments now match actual behavior
2. ✅ **Remove misleading info** - No more references to non-existent renaming behavior
3. ✅ **Fix typos** - Professional code quality
4. ✅ **Cleaner API** - Remove unused parameter reduces cognitive load
5. ✅ **No behavioral changes** - Pure documentation/cleanup, zero runtime impact

## Verification

### Build
```bash
cmake --build ../../build/morphizen-core --config Debug --parallel
```

### Test
```bash
../../build/morphizen-core/bin/morphizen-unit-tests.exe --gtest_filter=*Tar*
```

**Expected**: All tests pass (no code behavior changes)

## Success Criteria

- [ ] Outdated comment about renaming replaced with accurate append-only description
- [ ] Typos fixed: "entries is append" → "entry is appended", "indidcate" → "indicate"
- [ ] Incorrect FIXME removed from line 342
- [ ] Unused `name` parameter removed from `add_1024_padding()` declaration and implementation
- [ ] All 4 caller sites updated to remove the argument
- [ ] All tests pass

## Notes

This is **documentation and cleanup only** - no behavioral changes. These issues were identified during comprehensive TarFile code review as Topic #4 (following dead code removal in #037, const_cast documentation in #038, and circular symlink TODO).
