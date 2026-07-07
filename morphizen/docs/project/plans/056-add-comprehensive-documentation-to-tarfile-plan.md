<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Add Comprehensive Documentation to TarFile Class

**Issue:** #056
**Created:** 2026-02-04
**Status:** READY

## Objective

Add comprehensive Doxygen documentation to TarFile class, including class-level overview, 4 undocumented methods, and fix outdated open_for_write() comments.

## Implementation Steps

### Step 1: Add Class-Level Documentation

**File:** `morphizen-core/src/tar_file.hpp`

**Location:** Before line 16 (before `class TarFile {`)

**Add Doxygen comment:**
```cpp
/**
 * @brief TAR archive abstraction with read/write capabilities.
 *
 * TarFile provides an interface for reading and writing TAR archives with
 * content-addressable storage using MD5 deduplication. The archive uses
 * append-only semantics where the last entry with a given name takes
 * precedence.
 *
 * Features:
 * - Two access patterns: buffered I/O and memory-mapped (mmap)
 * - Automatic MD5-based deduplication for identical content
 * - Symlink support with automatic resolution
 * - Multiple factory methods for different data sources
 *
 * Thread-Safety:
 * - NOT thread-safe - designed for single-threaded access
 * - Multiple readers allowed, but only one writer at a time
 *
 * Ownership:
 * - TarFile owns the underlying stream and all entries
 * - Entries managed via unique_ptr, streams via shared_ptr
 */
class TarFile {
```

### Step 2: Document `create(std::unique_ptr<std::iostream>&&)`

**File:** `morphizen-core/src/tar_file.hpp`

**Location:** Before line 18

**Add Doxygen comment:**
```cpp
/**
 * @brief Creates a TarFile instance from a pre-constructed iostream.
 *
 * This is the base factory method used by other create() variants. Reads
 * the TAR archive header and populates the entries vector.
 *
 * @param stream Pre-constructed iostream containing TAR data
 * @return Unique pointer to the created TarFile instance
 */
MORPHIZEN_DLL_SPEC static std::unique_ptr<TarFile>
create(std::unique_ptr<std::iostream>&& stream);
```

### Step 3: Document `has_file(const std::string& filename)`

**File:** `morphizen-core/src/tar_file.hpp`

**Location:** Before line 104

**Add Doxygen comment:**
```cpp
/**
 * @brief Checks if an entry exists in the TAR archive.
 *
 * Uses last-entry-wins semantics: if multiple entries have the same name,
 * only the last one is considered.
 *
 * @param filename Path of the entry to check
 * @return true if entry exists, false otherwise
 */
MORPHIZEN_DLL_SPEC
bool has_file(const std::string& filename) const;
```

### Step 4: Document `entries()`

**File:** `morphizen-core/src/tar_file.hpp`

**Location:** Before line 106

**Add Doxygen comment:**
```cpp
/**
 * @brief Returns all entries in the TAR archive.
 *
 * Entries are ordered by their position in the TAR file. May include
 * duplicate entries (same filename), where the last entry takes precedence.
 *
 * @return Vector of TarEntryInputStream unique pointers
 */
MORPHIZEN_DLL_SPEC
std::vector<std::unique_ptr<TarEntryInputStream>>& entries();
MORPHIZEN_DLL_SPEC
const std::vector<std::unique_ptr<TarEntryInputStream>>& entries() const;
```

### Step 5: Document `open_for_read(const std::string& filename)`

**File:** `morphizen-core/src/tar_file.hpp`

**Location:** Replace lines 109-111 with Doxygen comment

**Replace:**
```cpp
// OLD (lines 109-111):
// user must not close this stream.
// stream->close() is a noop.
MORPHIZEN_DLL_SPEC
TarEntryInputStream* open_for_read(const std::string& filename);

// NEW:
/**
 * @brief Opens a TAR entry for reading.
 *
 * Returns a stream for reading the entry's content. The stream is managed
 * by TarFile and must not be closed by the caller. Automatically resolves
 * symlinks to their target entries.
 *
 * Uses last-entry-wins semantics: if multiple entries have the same name,
 * returns the last one.
 *
 * @param filename Path of the entry to read
 * @return Pointer to TarEntryInputStream, or nullptr if entry not found
 * @note Stream managed by TarFile - do not close
 */
MORPHIZEN_DLL_SPEC
TarEntryInputStream* open_for_read(const std::string& filename);
```

### Step 6: Fix `open_for_write()` Documentation

**File:** `morphizen-core/src/tar_file.hpp`

**Location:** Replace lines 113-130 with Doxygen comment

**Replace:**
```cpp
// OLD (lines 113-130):
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
MORPHIZEN_DLL_SPEC
std::unique_ptr<std::ostream> open_for_write(const std::string& filename);

// NEW:
/**
 * @brief Opens a TAR entry for writing (append-only).
 *
 * Creates a new entry in the TAR archive. If an entry with the same name
 * already exists, a new entry is appended (last-entry-wins semantics).
 * Existing entries are never modified or deleted.
 *
 * Returns nullptr if the TAR is memory-mapped (read-only mode).
 *
 * Concurrency:
 * - Only one writer allowed at a time (returns nullptr if writing)
 * - Multiple readers allowed concurrently with writer
 *
 * The TAR file remains valid after closing the stream due to the 1024-byte
 * end-of-archive trailer.
 *
 * @param filename Path of the entry to write
 * @return Unique pointer to ostream, or nullptr if read-only or already writing
 * @note Caller must close the returned stream
 * @note NOT thread-safe
 */
MORPHIZEN_DLL_SPEC
std::unique_ptr<std::ostream> open_for_write(const std::string& filename);
```

### Step 7: Review Existing Documentation

**File:** `morphizen-core/src/tar_file.hpp`

**Review these existing docs for consistency:**

1. **`create_from_path()` (lines 21-34)** - Already addressed by Issue #051
2. **`create()` tmpfile (lines 36-45)** - Check for clarity
3. **`create(vector)` (lines 46-55)** - Verify @param names
4. **`create(string)` variants (lines 57-81)** - Verify @param names
5. **`current_size()` (lines 132-145)** - Good reference example
6. **`dump_to()` (lines 146-159)** - Good reference example

**Check for:**
- Parameter name mismatches
- Typos
- Inconsistent terminology
- References to removed behavior

## Verification

### Documentation Checklist

**Class-level:**
- [ ] Class documentation added before line 16
- [ ] Covers purpose, features, thread-safety, ownership

**Undocumented methods:**
- [ ] `create(stream)` documented (line 18)
- [ ] `has_file()` documented (line 104)
- [ ] `entries()` documented (line 106)
- [ ] `open_for_read()` documented (line 112)

**Fixed documentation:**
- [ ] `open_for_write()` uses Doxygen format
- [ ] Removes misleading "renamed" behavior
- [ ] Describes correct append-only semantics
- [ ] Includes all relevant notes (concurrency, thread-safety, closure)

**Existing docs review:**
- [ ] All @param names match actual parameters
- [ ] No typos
- [ ] Consistent terminology
- [ ] No references to removed behavior

### Build

```bash
cmake --build ../../build/morphizen-core --config Debug --parallel
```

**Expected:** Build succeeds (documentation changes only)

## Success Criteria

- [ ] Class-level documentation added
- [ ] All 4 undocumented methods have Doxygen docs
- [ ] `open_for_write()` documentation fixed (correct behavior, Doxygen format)
- [ ] All existing documentation reviewed and consistent
- [ ] No undocumented public methods remain
- [ ] Build succeeds

## Files Modified

- `morphizen-core/src/tar_file.hpp` - Add class and method documentation

## Notes

**Documentation style:** Follow the pattern of `current_size()` and `dump_to()` - comprehensive Doxygen with @brief, @param, @return, @note tags.

**Key improvements:**
- Removes misleading "renaming" behavior from open_for_write()
- Adds class-level context for users
- Completes documentation coverage for entire public API
