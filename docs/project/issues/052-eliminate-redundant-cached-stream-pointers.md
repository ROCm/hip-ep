<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Issue #052: Eliminate Redundant Cached Stream Pointers

## Metadata
- **Type:** Refactoring
- **Priority:** LOW
- **Dependencies:** None

## Description

Eliminate redundant cached pointer members (`TarFile::mem_stream_` and `TarEntryInputStream::mem_buf_`) by using inline helper functions with on-demand dynamic_cast. This removes N+1 redundant pointers (1 in TarFile + N entries) and simplifies the ownership model.

## Problem

**Current design:**

The TAR streaming architecture caches the same `MemStream<MemFile>*` pointer in multiple locations:

```cpp
// TarFile - caches mem_stream_
class TarFile {
  std::shared_ptr<std::iostream> stream_;
  MemStream<MemFile>* mem_stream_;  // Cached dynamic_cast result
};

// TarEntryInputStream - caches mem_buf_ (N times, once per entry)
class TarEntryInputStream {
  std::unique_ptr<TarEntryInputStreamBuffer> buf_;
  MemStream<MemFile>* mem_buf_;  // Same pointer as TarFile::mem_stream_
};
```

**Construction flow:**
```cpp
// TarFile constructor
TarFile::TarFile(PrivateTag, std::unique_ptr<std::iostream>&& stream)
    : stream_(std::move(stream)),
      mem_stream_{dynamic_cast<MemStream<MemFile>*>(stream_.get())} {
  // Creates N entries, each gets same mem_stream_ pointer
}

// TarFile::add_entry() - called N times
auto entry = std::make_unique<TarEntryInputStream>(
    std::make_unique<TarEntryInputStreamBuffer>(..., stream_),
    mem_stream_);  // Passes cached pointer to every entry
```

**Why this is problematic:**

1. **Massive redundancy** - With 100 TAR entries: 101 pointers (1 in TarFile + 100 in entries), all pointing to the same object
2. **Memory waste** - Each cached pointer is 8 bytes × 101 = 808 bytes for same value
3. **Not self-documenting** - Unclear why we cache the same pointer everywhere
4. **Maintenance burden** - Multiple places store derived information from `stream_`
5. **Violates DRY** - Same dynamic_cast result duplicated N+1 times

**Code locations:**

TarFile:
- `tar_file.hpp:208-212` - Member declarations (stream_ + mem_stream_)
- `tar_file.cpp:191-193` - Constructor initializes both
- `tar_file.cpp:289` - `open_for_write()` checks `mem_stream_` for read-only
- `tar_file.cpp:452` - `add_entry()` passes `mem_stream_` to entries

TarEntryInputStream:
- `tar_entry.hpp:125` - Member declaration (mem_buf_)
- `tar_entry.cpp:170` - Constructor stores mem_buf_
- `tar_entry.cpp:229-230` - `mmap()` uses mem_buf_ to get memory pointer

TarEntryInputStreamBuffer:
- `tar_entry.hpp:71` - Stores `stream_` (shared_ptr, the actual data source)
- No public accessor for stream_

## Solution

**Eliminate cached pointers, use on-demand dynamic_cast:**

### Part 1: TarFile Cleanup

Replace `mem_stream_` member with inline helper:

```cpp
// tar_file.hpp - Remove mem_stream_, add helper
class TarFile {
  std::shared_ptr<std::iostream> stream_;

  // Get memory-mapped stream if available (nullptr otherwise)
  MemStream<MemFile>* get_mem_stream() const {
    return dynamic_cast<MemStream<MemFile>*>(stream_.get());
  }
};
```

### Part 2: TarEntryInputStream Cleanup

Remove `mem_buf_` parameter and member, access via buffer's stream:

```cpp
// tar_entry.hpp - TarEntryInputStreamBuffer - add public getter
class TarEntryInputStreamBuffer : public std::streambuf {
public:
  // Existing methods...

  // Get the underlying TAR stream this buffer reads from
  std::shared_ptr<std::istream> underlying_tar_stream() const {
    return stream_;
  }

private:
  std::shared_ptr<std::istream> stream_;  // Already exists
};

// tar_entry.hpp - TarEntryInputStream - remove mem_buf_
class TarEntryInputStream : public std::istream {
public:
  // Remove mem_buf_ parameter from constructor
  explicit TarEntryInputStream(std::unique_ptr<TarEntryInputStreamBuffer> buf);

private:
  std::unique_ptr<TarEntryInputStreamBuffer> buf_;
  // MemStream<MemFile>* mem_buf_;  // DELETE THIS
};

// tar_entry.cpp - TarEntryInputStream::mmap() - use buf_->underlying_tar_stream()
void* TarEntryInputStream::mmap() {
  auto* mem_stream = dynamic_cast<MemStream<MemFile>*>(
      buf_->underlying_tar_stream().get());
  if (mem_stream) {
    return (void*)mem_stream->offset(buf_->data_begin_pos());
  }
  return nullptr;
}
```

### Part 3: Update Call Sites

```cpp
// tar_file.cpp - TarFile::open_for_write()
if (get_mem_stream()) {  // Was: if (mem_stream_)
  return nullptr;
}

// tar_file.cpp - TarFile::add_entry()
auto entry = std::make_unique<TarEntryInputStream>(
    std::make_unique<TarEntryInputStreamBuffer>(..., stream_));
    // Remove second parameter (mem_stream_)
```

**Benefits:**
- ✅ Eliminates N+1 redundant pointers (massive reduction with many entries)
- ✅ Single source of truth for stream type checking
- ✅ Self-documenting - clear we're checking stream type on-demand
- ✅ Simpler ownership model
- ✅ Low risk - well-defined changes with clear verification
- ✅ Negligible performance impact - dynamic_cast called only when needed (not in hot loops)

## Plans

- [052-eliminate-redundant-cached-stream-pointers-plan.md](../plans/052-eliminate-redundant-cached-stream-pointers-plan.md) - Created 2026-02-04

## Notes

**Performance consideration:**

The dynamic_cast overhead is negligible:

**TarFile usage (2 call sites):**
1. `open_for_write()` - Called once per write session (initialization)
2. `add_entry()` - Called N times during construction (one-time setup)

**TarEntryInputStream usage (1 call site per entry):**
1. `mmap()` - Called at most once per entry (when requesting mmap access)

Total: O(N) dynamic_cast operations during construction/setup, not in any hot loops.

**Discovery:** Found during TarFile God Class analysis (Task #11 / Sub-topic 5.2.2: Stream Management Responsibility). Further exploration revealed the full extent of pointer duplication across all entries.

**Related:**
- Issue #053: Documents the TAR streaming architecture that this issue refactors
