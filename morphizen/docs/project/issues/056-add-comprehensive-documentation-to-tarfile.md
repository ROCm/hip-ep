<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Issue #056: Add Comprehensive Documentation to TarFile Class

## Metadata
- **Type:** Documentation
- **Priority:** MEDIUM
- **Dependencies:** None

## Description

Add comprehensive inline documentation to TarFile class, including class-level overview and missing method documentation. Fix outdated/misleading comments in open_for_write() and review all existing documentation for inconsistencies.

## Problem

**Current documentation gaps:**

### 1. Missing Class-Level Documentation

```cpp
// tar_file.hpp:16
class TarFile {  // NO class-level documentation
```

**Needs:**
- Purpose and responsibilities
- Usage patterns
- Thread-safety guarantees
- Ownership model

### 2. Undocumented Public Methods (4 methods)

**a) `create(std::unique_ptr<std::iostream>&&)` (line 18-19)**
```cpp
MORPHIZEN_DLL_SPEC static std::unique_ptr<TarFile>
create(std::unique_ptr<std::iostream>&& stream);  // NO documentation
```

**b) `has_file(const std::string& filename)` (line 104)**
```cpp
MORPHIZEN_DLL_SPEC
bool has_file(const std::string& filename) const;  // NO documentation
```

**c) `entries()` (lines 106-108)**
```cpp
MORPHIZEN_DLL_SPEC
std::vector<std::unique_ptr<TarEntryInputStream>>& entries();  // NO documentation
MORPHIZEN_DLL_SPEC
const std::vector<std::unique_ptr<TarEntryInputStream>>& entries() const;  // NO documentation
```

**d) `open_for_read(const std::string& filename)` (line 112)**
```cpp
// user must not close this stream.
// stream->close() is a noop.
MORPHIZEN_DLL_SPEC
TarEntryInputStream* open_for_read(const std::string& filename);  // Has comment but not Doxygen format
```

### 3. Outdated/Misleading Documentation

**`open_for_write()` comments (lines 113-130):**

```cpp
// user must close this stream
// NOTE: there should be only one writer, otherwise return nullptr;
//
// there could be many streams already opened for read.
//
// if the filename already exists, the orinal entry will be renamed to a wired
// invisiable name,
// TODO: check if tar file header support delete flag.
//
// after the stream is closed, a new entries is append to the tar stream_.
//
// there are 1024 zero bytes at the end of the tar file, which is used to
// indidcate the end of the tar file. it should be also available after the
// ostream is closed. so that when stream_ is closed, the tar file is still
// valid.
//
// NOTE: the stream is not thread safe.
```

**Problems:**
- **Lines 118-119: WRONG** - "if the filename already exists, the orinal entry will be renamed to a wired invisiable name" - This describes dead code behavior removed in April 2025 (commit 9081edc)
- **Actual behavior:** Append-only, last entry with given name takes precedence (standard TAR semantics)
- **Lines 118-120:** Contains typos fixed by Issue #030 but content is still wrong
- **Not Doxygen format:** Should use @brief, @param, @return tags

### 4. Documentation Review Findings

**Existing documentation to review:**
- `create_from_path()` (lines 21-34) - Already flagged by Issue #051 for improvement
- `create()` tmpfile (lines 36-45) - Generic, could be more specific
- `create(vector)` (lines 46-55) - OK
- `create(string)` variants (lines 57-81) - OK
- `current_size()` (lines 132-145) - Good, has example
- `dump_to()` (lines 146-159) - OK

## Solution

Add comprehensive Doxygen documentation following the style of `current_size()` and `dump_to()`.

### 1. Add Class-Level Documentation

**Location:** Before line 16

**Content to cover:**
- Brief: TAR archive abstraction with read/write capabilities
- Details:
  - Content-addressable storage (MD5 deduplication)
  - Append-only semantics (last entry wins)
  - Two access patterns: buffered I/O and mmap
  - Thread-safety: NOT thread-safe, single-threaded access only
  - Ownership: Manages stream lifetime and entries

### 2. Document `create(std::unique_ptr<std::iostream>&&)`

**Location:** Before line 18

**Content:**
- Brief: Creates TarFile from pre-constructed iostream (base factory)
- Param: stream - Pre-constructed iostream containing TAR data
- Return: Unique pointer to TarFile instance
- Note: This is the base factory used by other create() variants

### 3. Document `has_file(const std::string& filename)`

**Location:** Before line 104

**Content:**
- Brief: Checks if entry exists in TAR archive
- Param: filename - Path of entry to check
- Return: true if entry exists, false otherwise
- Note: Uses last-entry-wins semantics for duplicates

### 4. Document `entries()`

**Location:** Before line 106

**Content:**
- Brief: Returns all entries in the TAR archive
- Return: Vector of TarEntryInputStream unique pointers
- Note: Entries ordered by position in TAR file, includes duplicates

### 5. Document `open_for_read(const std::string& filename)`

**Location:** Replace inline comments (lines 109-111) with Doxygen

**Content:**
- Brief: Opens TAR entry for reading
- Param: filename - Path of entry to read
- Return: Pointer to TarEntryInputStream, or nullptr if not found
- Note:
  - Returned stream managed by TarFile, do not close
  - Uses last-entry-wins semantics for duplicates
  - Resolves symlinks automatically

### 6. Fix `open_for_write()` Documentation

**Location:** Replace comment block (lines 113-130) with Doxygen

**Content:**
- Brief: Opens TAR entry for writing (append-only)
- Param: filename - Path of entry to write
- Return: Unique pointer to ostream, or nullptr if read-only (mmap)
- Note:
  - Only one writer allowed at a time
  - Multiple readers allowed concurrently
  - Caller must close returned stream
  - If entry exists, new entry appended (last-entry-wins)
  - TAR remains valid after close (1024-byte trailer)
  - NOT thread-safe

**Remove misleading content:**
- ❌ "the orinal entry will be renamed" - WRONG, no renaming happens
- ❌ "TODO: check if tar file header support delete flag" - Not relevant

### 7. Review Existing Documentation

**Items to check:**
- Verify all @param names match actual parameter names
- Check for typos in existing docs
- Ensure consistent terminology (e.g., "TAR archive" vs "tar file")
- Update any references to removed behavior

## Benefits

- ✅ Complete public API documentation (no undocumented methods)
- ✅ Accurate documentation (removes misleading renaming behavior)
- ✅ Consistent Doxygen format throughout
- ✅ Users understand class purpose, usage patterns, thread-safety
- ✅ Better IDE tooltips and generated documentation

## Plans

- [056-add-comprehensive-documentation-to-tarfile-plan.md](../plans/056-add-comprehensive-documentation-to-tarfile-plan.md) - Created 2026-02-04

## Notes

**Discovery:** Found during comprehensive TarFile review after completing all organizational tasks.

**Related issues:**
- Issue #030: Fixes typos in open_for_write() comments (but content still wrong)
- Issue #051: Documents complex factory methods (create_from_path behavior)

**Documentation count after this change:**
- Undocumented methods: 0 (was 4)
- Class-level docs: Added
- Outdated/misleading docs: Fixed (open_for_write)
