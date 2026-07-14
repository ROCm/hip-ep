<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Document TAR Streaming Architecture

**Issue:** #053
**Created:** 2026-02-04
**Status:** READY

## Objective

Create comprehensive technical documentation explaining the TAR file streaming architecture at `docs/technical/tar-streaming-architecture.md`. Document design rationale, maintenance guidance, and current state before refactoring.

## Background

**Discovery:** While analyzing Issue #052 (stream management responsibility), explored the full TAR streaming architecture and discovered several interconnected design patterns that are not documented anywhere.

**Why document now:** Knowledge is fresh from exploration, and this serves as prerequisite before refactoring work.

## Implementation Steps

### Step 1: Create Documentation File Structure

**File:** `docs/technical/tar-streaming-architecture.md`

**Add standard header:**
```markdown
<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# TAR Streaming Architecture

**Purpose:** Design rationale, maintenance guide, and decision record

**Last Updated:** 2026-02-04

## Overview

[Brief introduction to the TAR streaming system]
```

### Step 2: Document Stream Ownership Model

**Section 2: Stream Ownership Model**

Document:
- `TarFile::stream_` is `std::shared_ptr<iostream>`
- Passed to all `TarEntryInputStreamBuffer` objects
- Reference count = N+1 (TarFile + N entries)
- Why shared_ptr: Multiple entry buffers need simultaneous access

**Code example:**
```cpp
// TarFile constructor (tar_file.cpp:191-198)
TarFile::TarFile(PrivateTag, std::unique_ptr<std::iostream>&& stream)
    : stream_(std::move(stream)) {
  // ...
  do {
  } while (read_tar_entry(stream_));  // Passes stream_ to all entries
}

// TarEntryInputStreamBuffer (tar_entry.hpp:71)
std::shared_ptr<std::istream> stream_;  // Shares TarFile's stream
```

### Step 3: Document Cached Pointer Pattern

**Section 3: Cached Pointer Pattern**

Document:
- `TarFile::mem_stream_` - Cached `MemStream<MemFile>*` pointer
- `TarEntryInputStream::mem_buf_` - Same pointer, duplicated N times
- Initialized in constructors via `dynamic_cast`
- Used for mmap access optimization

**Code example:**
```cpp
// TarFile (tar_file.cpp:193)
mem_stream_{dynamic_cast<MemStream<MemFile>*>(stream_.get())}

// TarEntryInputStream (tar_entry.cpp:170)
mem_buf_{mem_buf}  // Receives mem_stream_ from TarFile::add_entry()
```

**Result:** With 100 entries, 101 pointers (1 in TarFile + 100 in entries)

### Step 4: Document Buffering Mechanism

**Section 4: Buffering Mechanism**

Document:
- `TarEntryInputStreamBuffer` extends `std::streambuf`
- Implements `underflow()` for lazy reading
- Each entry tracks position range: `data_begin_pos_` to `data_end_pos_`
- Seeking pattern: `stream_->seekg(buffer_pos_)` before reading
- Internal buffer: `std::vector<char> buffer_`

**Code example:**
```cpp
// TarEntryInputStreamBuffer::underflow() (tar_entry.cpp:106-135)
std::streambuf::int_type TarEntryInputStreamBuffer::underflow() {
  if (buffer_pos_ >= data_end_pos_) {
    return traits_type::eof();
  }
  // Seek to this entry's position in shared stream
  CHECK(stream_->seekg(buffer_pos_).good());

  // Read from stream into internal buffer
  CHECK(stream_->read(buffer_.data(), readSize).good());

  buffer_pos_ += bytesRead;
  setg(buffer_.data(), buffer_.data(), buffer_.data() + bytesRead);
  return traits_type::to_int_type(*gptr());
}
```

### Step 5: Document Dual Access Patterns

**Section 5: Dual Access Patterns**

Document two ways to access TAR entry data:

**Pattern 1: Buffered I/O**
- Via `std::istream` interface
- Uses `underflow()` mechanism
- Seeks and reads from shared stream

**Pattern 2: mmap Access**
- Via `TarEntryInputStream::mmap()` method
- Returns direct pointer: `mem_buf_->offset(data_begin_pos_)`
- Bypasses stream entirely

**Code example:**
```cpp
// Buffered I/O - standard stream reading
entry->read(buffer, size);  // Triggers underflow()

// mmap - direct memory access
void* ptr = entry->mmap();  // Returns raw pointer if available
if (ptr) {
  // Direct access to memory-mapped data
}
```

### Step 6: Document Thread-Safety Considerations

**Section 6: Thread-Safety Considerations**

Document:
- **Shared stream position:** All entries seek the same `stream_`
- **Non-thread-safe buffered I/O:** Reading different entries concurrently will interfere
- **Thread-safe mmap:** Direct memory access, no shared state

**Explanation:**
```cpp
// NOT thread-safe:
Thread 1: entry[0]->read()  // Seeks stream to entry[0] position
Thread 2: entry[1]->read()  // Seeks stream to entry[1] position
// Race condition on shared stream position

// Thread-safe:
Thread 1: entry[0]->mmap()  // Returns pointer into mmap region
Thread 2: entry[1]->mmap()  // Returns different pointer, no shared state
```

### Step 7: Document Object Lifetime and Relationships

**Section 7: Object Lifetime and Relationships**

**Ownership diagram:**
```
TarFile (owns stream)
├── stream_ : shared_ptr<iostream> (refcount = N+1)
├── mem_stream_ : MemStream<MemFile>* (cached pointer)
└── entries_ : vector<unique_ptr<TarEntryInputStream>> (N entries)
    │
    ├── TarEntryInputStream[0]
    │   ├── buf_ : unique_ptr<TarEntryInputStreamBuffer>
    │   │   └── stream_ : shared_ptr<istream> (shares TarFile::stream_)
    │   └── mem_buf_ : MemStream<MemFile>* (cached, same as TarFile::mem_stream_)
    │
    ├── TarEntryInputStream[1]
    │   ├── buf_->stream_ : shared_ptr (shares TarFile::stream_)
    │   └── mem_buf_ : MemStream<MemFile>* (cached, same as TarFile::mem_stream_)
    │
    └── ... (N total entries)
```

**Construction flow:**
```cpp
1. TarFile::TarFile()
   - Stores stream_ as shared_ptr
   - Caches mem_stream_ = dynamic_cast<MemStream<MemFile>*>(stream_.get())

2. For each TAR entry:
   - read_tar_entry(stream_) - passes stream_ to create entry
   - add_entry() creates TarEntryInputStreamBuffer with shared stream_
   - TarEntryInputStream stores mem_buf_ (same as TarFile::mem_stream_)
```

### Step 8: Document Key Files

**Section 8: Key Source Files**

List relevant source files and their responsibilities:

- `morphizen-core/src/tar_file.hpp` - TarFile class, owns stream
- `morphizen-core/src/tar_file.cpp` - TarFile implementation, constructs entries
- `morphizen-core/src/tar_entry.hpp` - TarEntryInputStream, TarEntryInputStreamBuffer classes
- `morphizen-core/src/tar_entry.cpp` - Buffering implementation (underflow), mmap access
- `morphizen-core/src/mem_stream_buffer.hpp` - MemStream template, offset() for mmap

## Verification

### Review Checklist

- [ ] All 6 main sections documented
- [ ] Code examples provided for key concepts
- [ ] Ownership diagram included
- [ ] Thread-safety implications explained
- [ ] No opinions or recommendations (purely factual)
- [ ] Cross-references to source files included
- [ ] Standard header and copyright added

## Success Criteria

- [ ] Technical documentation file created at `docs/technical/tar-streaming-architecture.md`
- [ ] All design patterns documented with code examples
- [ ] Future maintainers can understand the architecture without reading entire codebase
- [ ] Serves as decision record before refactoring work

## Files Modified

- `docs/technical/tar-streaming-architecture.md` - New technical documentation (create)

## Notes

**Style:** Purely factual, no opinions. This is "how it works" not "how it should work."

**Audience:** Future maintainers and developers working on TAR streaming code.

**Related issues:**
- Issue #052: Will refactor mem_stream_ based on this architecture understanding
- Future refactoring issues may reference this doc as baseline
