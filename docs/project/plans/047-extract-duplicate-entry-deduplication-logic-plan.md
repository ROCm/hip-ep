<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Extract Duplicate Entry Deduplication Logic in TarFile

**Issue:** #047
**Created:** 2026-02-03
**Status:** READY

## Objective

Extract duplicated entry deduplication logic from `add_regular_entry()` and `add_symlink_entry()` into a private helper function `remove_duplicate_entry()`. This eliminates ~14 lines of duplicate code and fixes a copy-paste bug in the log message.

## Background

**Problem discovered:** While reviewing TarFile organization, found identical deduplication logic in two methods:
- `add_regular_entry()` (tar_file.cpp:303-316)
- `add_symlink_entry()` (tar_file.cpp:362-376)

**Copy-paste bug:** The log message in `add_regular_entry()` says "add_symlink_entry" instead of the correct function name.

**Why these methods need deduplication:** Both implement TAR "last entry wins" semantics - when adding an entry, remove any existing entry with the same path first.

## Implementation Steps

### Step 1: Add Helper Function Declaration

**File:** `morphizen-core/src/tar_file.hpp`

**Location:** In the `private:` section of the `TarFile` class

**Find the private section** (around line 200+):
```cpp
private:
  // existing private members...
```

**Add declaration:**
```cpp
private:
  /// Removes any existing entry with the given path (TAR last-wins semantics).
  /// Uses standard erase-remove idiom.
  /// @param path - Entry path to remove duplicates for
  void remove_duplicate_entry(const std::string& path);
```

### Step 2: Add Helper Function Implementation

**File:** `morphizen-core/src/tar_file.cpp`

**Location:** Add after other helper functions (before or after `find_real_entry()`)

**Implementation:**
```cpp
void TarFile::remove_duplicate_entry(const std::string& path) {
  entries_.erase(
      std::remove_if(entries_.begin(), entries_.end(),
                     [&path](const auto& entry) {
                       return entry->path() == path;
                     }),
      entries_.end());
}
```

**Key design decisions:**
1. **No logging** - Keep helper simple, let callers handle logging if needed
2. **Erase-remove idiom** - Standard C++ pattern, no unnecessary if-check (addresses Issue #044)
3. **One-liner lambda** - No need for intermediate `ret` variable

### Step 3: Update `add_regular_entry()` to Use Helper

**File:** `morphizen-core/src/tar_file.cpp`

**Location:** Lines 303-316

**Find:**
```cpp
TarEntryInputStream&
TarFile::add_regular_entry(const std::string& path,
                           std::streambuf::pos_type data_begin_pos,
                           std::streambuf::pos_type data_end_pos,
                           std::streambuf::pos_type block_begin_pos,
                           std::streambuf::pos_type block_end_pos) {
  // erase the old entry if found
  auto it = std::remove_if(
      entries_.begin(), entries_.end(), [&path](const auto& entry) {
        auto ret = entry->path() == path;
        if (ret) {
          MY_LOG(1)
              << " add_symlink_entry: duplicated entry found, remove old entry "
              << entry->to_string() //
              ;
        }
        return ret;
      });
  if (it != entries_.end()) {
    entries_.erase(it, entries_.end());
  }

  auto ret = add_entry(path, std::nullopt, data_begin_pos, data_end_pos,
                       block_begin_pos, block_end_pos);
```

**Replace with:**
```cpp
TarEntryInputStream&
TarFile::add_regular_entry(const std::string& path,
                           std::streambuf::pos_type data_begin_pos,
                           std::streambuf::pos_type data_end_pos,
                           std::streambuf::pos_type block_begin_pos,
                           std::streambuf::pos_type block_end_pos) {
  // Remove any existing entry with the same path (TAR last-wins semantics)
  remove_duplicate_entry(path);

  auto ret = add_entry(path, std::nullopt, data_begin_pos, data_end_pos,
                       block_begin_pos, block_end_pos);
```

**Changes:**
- Lines 303-316 (14 lines) → 1 line
- Updated comment to be more concise
- Removed logging (can be added back if needed, but simpler without)
- Fixes the copy-paste bug (no more incorrect "add_symlink_entry" log message)

### Step 4: Update `add_symlink_entry()` to Use Helper

**File:** `morphizen-core/src/tar_file.cpp`

**Location:** Lines 362-376

**Find:**
```cpp
TarEntryInputStream*
TarFile::add_symlink_entry(const std::string& symlink_name,
                           const std::string& real_path_name,
                           std::streambuf::pos_type block_begin_pos,
                           std::streambuf::pos_type block_end_pos) {
  // erase the old entry if found
  auto it = std::remove_if(
      entries_.begin(), entries_.end(), [&symlink_name](const auto& entry) {
        auto ret = entry->path() == symlink_name;
        if (ret) {
          MY_LOG(1)
              << " add_symlink_entry: duplicated entry found, remove old entry "
              << entry->to_string() //
              ;
        }
        return ret;
      });
  if (it != entries_.end()) {

    entries_.erase(it, entries_.end());
  }
  TarEntryInputStream* ret = nullptr;
  auto real_entry = find_real_entry(real_path_name);
```

**Replace with:**
```cpp
TarEntryInputStream*
TarFile::add_symlink_entry(const std::string& symlink_name,
                           const std::string& real_path_name,
                           std::streambuf::pos_type block_begin_pos,
                           std::streambuf::pos_type block_end_pos) {
  // Remove any existing entry with the same path (TAR last-wins semantics)
  remove_duplicate_entry(symlink_name);

  TarEntryInputStream* ret = nullptr;
  auto real_entry = find_real_entry(real_path_name);
```

**Changes:**
- Lines 362-376 (14 lines) → 1 line
- Updated comment to match `add_regular_entry()`
- Removed logging for consistency

### Step 5: Add Documentation to Header

**File:** `morphizen-core/src/tar_file.hpp`

**Location:** Above `add_regular_entry()` and `add_symlink_entry()` declarations

**Find `add_regular_entry()` declaration** (around line 80-90):
```cpp
  TarEntryInputStream& add_regular_entry(const std::string& path,
                                         std::streambuf::pos_type data_begin_pos,
                                         std::streambuf::pos_type data_end_pos,
                                         std::streambuf::pos_type block_begin_pos,
                                         std::streambuf::pos_type block_end_pos);
```

**Add documentation above it:**
```cpp
  /// Adds a regular file entry to the tar archive.
  /// Removes any existing entry with the same path (TAR last-wins semantics).
  /// Regular entries store actual file data at specified positions in the tar stream.
  /// @param path - Entry path in the archive
  /// @param data_begin_pos - Start position of file data in tar stream
  /// @param data_end_pos - End position of file data in tar stream
  /// @param block_begin_pos - Start position of tar block (including header)
  /// @param block_end_pos - End position of tar block (including padding)
  /// @return Reference to the created entry input stream
  TarEntryInputStream& add_regular_entry(const std::string& path,
                                         std::streambuf::pos_type data_begin_pos,
                                         std::streambuf::pos_type data_end_pos,
                                         std::streambuf::pos_type block_begin_pos,
                                         std::streambuf::pos_type block_end_pos);
```

**Find `add_symlink_entry()` declaration** (a few lines below):
```cpp
  TarEntryInputStream* add_symlink_entry(const std::string& symlink_name,
                                         const std::string& real_path_name,
                                         std::streambuf::pos_type block_begin_pos,
                                         std::streambuf::pos_type block_end_pos);
```

**Add documentation above it:**
```cpp
  /// Adds a symlink entry to the tar archive.
  /// Removes any existing entry with the same path (TAR last-wins semantics).
  /// Symlinks point to another entry; this method resolves the target via find_real_entry().
  /// If the target is not found, creates a symlink with invalid positions (-1) for lazy resolution.
  /// @param symlink_name - Symlink path in the archive
  /// @param real_path_name - Target path that the symlink points to
  /// @param block_begin_pos - Start position of tar block (including header)
  /// @param block_end_pos - End position of tar block (including padding)
  /// @return Pointer to the created entry input stream, or nullptr on failure
  TarEntryInputStream* add_symlink_entry(const std::string& symlink_name,
                                         const std::string& real_path_name,
                                         std::streambuf::pos_type block_begin_pos,
                                         std::streambuf::pos_type block_end_pos);
```

## Verification

### Build
```bash
cmake --build ../../build/morphizen-core --config Debug --parallel
```

### Test
```bash
../../build/morphizen-core/bin/morphizen-unit-tests.exe --gtest_filter=*Tar*
```

**Expected:** All TarFile-related tests pass (no behavior change, only refactoring)

### Code Review Checklist

- [ ] Helper function added to private section
- [ ] Helper uses erase-remove idiom (no if-check)
- [ ] `add_regular_entry()` uses helper (14 lines → 1 line)
- [ ] `add_symlink_entry()` uses helper (14 lines → 1 line)
- [ ] Copy-paste bug fixed (no more incorrect log message)
- [ ] Documentation added to header for both methods
- [ ] All tests pass
- [ ] Code reduction: ~26 lines eliminated (14 + 14 - 2 helper lines)

## Success Criteria

- [ ] `remove_duplicate_entry()` helper function implemented
- [ ] Both `add_regular_entry()` and `add_symlink_entry()` use the helper
- [ ] Duplicate code eliminated (~26 lines removed)
- [ ] Copy-paste bug fixed (log message was incorrect in add_regular_entry)
- [ ] Documentation added to header explaining the difference between the methods
- [ ] All TarFile tests pass
- [ ] Build succeeds without warnings

## Files Modified

- `morphizen-core/src/tar_file.hpp` - Add helper declaration + documentation
- `morphizen-core/src/tar_file.cpp` - Add helper implementation, update both callers

## Notes

**Why no logging in helper:**
The current logging happens inside the lambda and includes the function name. If we keep logging in the helper, we'd need to pass the caller's name as a parameter, adding complexity. Simpler to remove logging entirely - it's a debug log that isn't critical.

**Relationship to Issue #044:**
Issue #044 addresses the erase-remove idiom (removing unnecessary if-check). This issue extracts the duplicated logic. Both improve the same code locations, and the helper function implementation uses the corrected erase-remove idiom from #044.

**Why these methods do different things:**
- `add_regular_entry()`: Adds regular file with actual data positions, calls `add_entry(path, std::nullopt, ...)` where `std::nullopt` means "not a symlink"
- `add_symlink_entry()`: Adds symlink entry, resolves target via `find_real_entry()`, supports lazy resolution if target not found

The deduplication logic is identical because both need TAR last-wins semantics regardless of entry type.

**Code reduction:**
- Before: 14 lines × 2 = 28 lines of duplicate code
- After: 2 lines of helper + 1 line × 2 callers = 4 lines
- Net reduction: 24 lines eliminated
