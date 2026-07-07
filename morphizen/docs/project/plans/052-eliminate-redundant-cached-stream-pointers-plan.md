<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Eliminate Redundant Cached Stream Pointers

**Issue:** #052
**Created:** 2026-02-04
**Status:** READY

## Objective

Eliminate redundant cached pointer members (`TarFile::mem_stream_` and `TarEntryInputStream::mem_buf_`) by using inline helper functions with on-demand dynamic_cast. This removes N+1 redundant pointers and simplifies the ownership model.

## Background

**Problem:** TAR streaming architecture caches the same `MemStream<MemFile>*` pointer in multiple locations:
- 1× in `TarFile::mem_stream_`
- N× in `TarEntryInputStream::mem_buf_` (once per entry)

With 100 entries: 101 redundant pointers, all pointing to the same object.

**Discovery:** Found during stream management responsibility analysis (Task #11). Further exploration revealed full extent of duplication.

## Implementation Steps

### Step 1: Add underlying_tar_stream() Getter to TarEntryInputStreamBuffer

**File:** `morphizen-core/src/tar_entry.hpp`

**Location:** After existing public methods in `TarEntryInputStreamBuffer` class (around line 32)

**Add public method:**

```cpp
class TarEntryInputStreamBuffer : public std::streambuf {
public:
  // ... existing methods (path(), real_path(), size(), etc.)

  // Get the underlying TAR stream this buffer reads from
  MORPHIZEN_DLL_SPEC std::shared_ptr<std::istream> underlying_tar_stream() const {
    return stream_;
  }

  // ... rest of class
};
```

**Note:** Add after `size()` method, before the constructor section.

### Step 2: Remove mem_buf_ from TarEntryInputStream Constructor

**File:** `morphizen-core/src/tar_entry.hpp`

**Location:** Line 120-121 (TarEntryInputStream constructor)

**Change constructor signature:**

```cpp
// OLD:
explicit TarEntryInputStream(std::unique_ptr<TarEntryInputStreamBuffer> buf,
                             MemStream<MemFile>* mem_file);

// NEW:
explicit TarEntryInputStream(std::unique_ptr<TarEntryInputStreamBuffer> buf);
```

### Step 3: Remove mem_buf_ Member from TarEntryInputStream

**File:** `morphizen-core/src/tar_entry.hpp`

**Location:** Line 125 (TarEntryInputStream private members)

**Delete line:**

```cpp
// DELETE THIS LINE:
MemStream<MemFile>* mem_buf_;
```

### Step 4: Update TarEntryInputStream Constructor Implementation

**File:** `morphizen-core/src/tar_entry.cpp`

**Location:** Lines 168-172

**Update constructor:**

```cpp
// OLD:
TarEntryInputStream::TarEntryInputStream(
    std::unique_ptr<TarEntryInputStreamBuffer> buf, MemStream<MemFile>* mem_buf)
    : std::istream(buf.get()), buf_{nullptr}, mem_buf_{mem_buf} {
  buf_ = std::move(buf);
}

// NEW:
TarEntryInputStream::TarEntryInputStream(
    std::unique_ptr<TarEntryInputStreamBuffer> buf)
    : std::istream(buf.get()), buf_{nullptr} {
  buf_ = std::move(buf);
}
```

### Step 5: Update TarEntryInputStream::mmap() Implementation

**File:** `morphizen-core/src/tar_entry.cpp`

**Location:** Lines 228-233

**Update mmap() method:**

```cpp
// OLD:
void* TarEntryInputStream::mmap() {
  if (mem_buf_) {
    return (void*)mem_buf_->offset(buf_->data_begin_pos());
  }
  return nullptr;
}

// NEW:
void* TarEntryInputStream::mmap() {
  auto* mem_stream = dynamic_cast<MemStream<MemFile>*>(
      buf_->underlying_tar_stream().get());
  if (mem_stream) {
    return (void*)mem_stream->offset(buf_->data_begin_pos());
  }
  return nullptr;
}
```

### Step 6: Add get_mem_stream() Helper to TarFile

**File:** `morphizen-core/src/tar_file.hpp`

**Location:** Private section, after member declarations (around line 213)

**Add private helper:**

```cpp
class TarFile {
  // ... public methods ...

private:
  std::shared_ptr<std::iostream> stream_;
  MemStream<MemFile>* mem_stream_;  // Will be deleted in Step 8

  // Get memory-mapped stream if available (nullptr otherwise)
  MemStream<MemFile>* get_mem_stream() const {
    return dynamic_cast<MemStream<MemFile>*>(stream_.get());
  }

  // ... rest of private section
};
```

### Step 7: Update TarFile Call Sites

**File:** `morphizen-core/src/tar_file.cpp`

**Location 1:** Line 289 (TarFile::open_for_write)

```cpp
// OLD:
if (mem_stream_) {
  // mem_stream_ is not nullptr, it means tar file is created in memory, it is
  // readonly, we cannot expand the size of memroy dynamically
  return nullptr;
}

// NEW:
if (get_mem_stream()) {
  // get_mem_stream() returns non-null if tar file is memory-mapped, meaning
  // it is readonly - we cannot expand the size of memory dynamically
  return nullptr;
}
```

**Location 2:** Line 452 (TarFile::add_entry)

```cpp
// OLD:
auto entry = std::make_unique<TarEntryInputStream>(
    std::make_unique<TarEntryInputStreamBuffer>(path,
                                                real_path,       //
                                                data_begin_pos,  //
                                                data_end_pos,    //
                                                block_begin_pos, //
                                                block_end_pos,   //
                                                stream_),
    mem_stream_);

// NEW:
auto entry = std::make_unique<TarEntryInputStream>(
    std::make_unique<TarEntryInputStreamBuffer>(path,
                                                real_path,       //
                                                data_begin_pos,  //
                                                data_end_pos,    //
                                                block_begin_pos, //
                                                block_end_pos,   //
                                                stream_));
```

**Note:** Remove the second parameter (`mem_stream_`)

### Step 8: Remove mem_stream_ Member from TarFile

**File:** `morphizen-core/src/tar_file.hpp`

**Location:** Line 212 (TarFile private members)

**Delete lines:**

```cpp
// DELETE THESE LINES:
// it is nullptr if stream_ is not a
// memory map file. it is stream_.get() if
// stream_ is a memory map file.
MemStream<MemFile>* mem_stream_;
```

### Step 9: Update TarFile Constructor

**File:** `morphizen-core/src/tar_file.cpp`

**Location:** Lines 191-193

**Update constructor:**

```cpp
// OLD:
TarFile::TarFile(PrivateTag, std::unique_ptr<std::iostream>&& stream)
    : stream_(std::move(stream)),
      mem_stream_{dynamic_cast<decltype(mem_stream_)>(stream_.get())} {

// NEW:
TarFile::TarFile(PrivateTag, std::unique_ptr<std::iostream>&& stream)
    : stream_(std::move(stream)) {
```

### Step 10: Remove Old Issue File

**Action:** Delete the old issue file since we renamed it

```bash
git rm docs/project/issues/052-replace-tarfile-memstream-member-with-helper.md
```

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

**Expected:** All TAR-related tests pass

### Code Review Checklist

**TarEntryInputStreamBuffer:**
- [ ] `underlying_tar_stream()` getter added
- [ ] Returns `std::shared_ptr<std::istream>`
- [ ] Marked with MORPHIZEN_DLL_SPEC

**TarEntryInputStream:**
- [ ] Constructor signature updated (removed mem_buf parameter)
- [ ] `mem_buf_` member deleted
- [ ] Constructor implementation updated
- [ ] `mmap()` uses `buf_->underlying_tar_stream()` + dynamic_cast

**TarFile:**
- [ ] `get_mem_stream()` helper added
- [ ] `mem_stream_` member deleted
- [ ] Constructor no longer initializes `mem_stream_`
- [ ] `open_for_write()` uses `get_mem_stream()`
- [ ] `add_entry()` no longer passes second parameter

**General:**
- [ ] No compilation errors
- [ ] All tests pass
- [ ] No new warnings introduced

## Success Criteria

- [ ] Both cached pointer members eliminated (`mem_stream_` and `mem_buf_`)
- [ ] `underlying_tar_stream()` getter added to `TarEntryInputStreamBuffer`
- [ ] `get_mem_stream()` helper added to `TarFile`
- [ ] All call sites updated (3 locations)
- [ ] Build succeeds
- [ ] All TAR tests pass
- [ ] Reduces memory usage for TARs with many entries

## Files Modified

- `morphizen-core/src/tar_entry.hpp` - Add getter, update constructor, remove member
- `morphizen-core/src/tar_entry.cpp` - Update constructor and mmap() implementation
- `morphizen-core/src/tar_file.hpp` - Add helper, remove member
- `morphizen-core/src/tar_file.cpp` - Update constructor and 2 call sites

## Notes

**Order of changes:**

1. Add getters/helpers first (Steps 1, 6)
2. Update call sites to use new getters (Steps 5, 7)
3. Remove constructor parameters (Steps 2, 4)
4. Remove members last (Steps 3, 8, 9)

This order ensures code always compiles during refactoring.

**Performance impact:**

- Dynamic_cast called O(N) times during construction (non-hot path)
- `mmap()` calls dynamic_cast on-demand (at most once per entry, non-hot path)
- No performance degradation expected

**Memory savings:**

With N entries: Saves (N+1) × 8 bytes = 808 bytes for 100-entry TAR
