<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Issue #053: Document TAR Streaming Architecture

## Metadata
- **Type:** Documentation
- **Priority:** MEDIUM
- **Dependencies:** None

## Description

Create comprehensive technical documentation explaining the TAR file streaming architecture, including stream ownership model, cached pointer pattern, buffering mechanism, mmap access, and thread-safety considerations. This serves as design rationale, maintenance guide, and decision record before potential refactoring work.

## Problem

**Current situation:**

The TAR streaming architecture has several interconnected design patterns that are not documented:

1. **Shared ownership model** - One `std::shared_ptr<iostream>` shared by TarFile and all TarEntryInputStreamBuffer objects
2. **Cached pointer pattern** - `mem_stream_` and `mem_buf_` cache dynamic_cast results
3. **Lazy buffering** - TarEntryInputStreamBuffer::underflow() seeks and reads on-demand
4. **Dual access patterns** - Both buffered I/O and direct mmap access
5. **Thread-safety implications** - Shared stream position, non-concurrent access

**Why documentation is needed:**

1. **Complex architecture** - Multiple objects share ownership of single stream
2. **Non-obvious design decisions** - Why shared_ptr? Why cached pointers? Why lazy buffering?
3. **Maintenance burden** - Future developers need to understand the model before making changes
4. **Refactoring prerequisite** - Document current design before considering improvements
5. **Knowledge preservation** - Capture architectural decisions while context is fresh

**Example of undocumented design:**

```cpp
// TarFile - owns the stream
std::shared_ptr<std::iostream> stream_;
MemStream<MemFile>* mem_stream_;  // Why cached?

// TarEntryInputStreamBuffer - shares the stream
std::shared_ptr<std::istream> stream_;  // Why shared_ptr?

// TarEntryInputStream - caches pointer again
MemStream<MemFile>* mem_buf_;  // Why N duplicates?
```

Without documentation, it's unclear:
- Why use shared_ptr instead of unique_ptr?
- Why cache the MemStream pointer in two places?
- How does shared stream seeking work with multiple entries?
- When is mmap used vs buffered I/O?

## Solution

Create technical documentation at `docs/technical/tar-streaming-architecture.md` covering:

### 1. Stream Ownership Model
- Single `std::shared_ptr<iostream>` owned by TarFile
- Shared by all N TarEntryInputStreamBuffer objects
- Reference count = N+1 (TarFile + N entries)
- Why shared_ptr: Multiple entries need simultaneous access to same stream

### 2. Cached Pointer Pattern
- `TarFile::mem_stream_` - Cached MemStream<MemFile>* pointer
- `TarEntryInputStream::mem_buf_` - Same pointer cached N times (once per entry)
- Purpose: Enable mmap access without repeated dynamic_cast
- Result: With 100 entries, 101 pointers to same MemStream object

### 3. Buffering Mechanism
- TarEntryInputStreamBuffer inherits from std::streambuf
- Implements underflow() for lazy reading
- Each entry knows its position range (data_begin_pos to data_end_pos)
- Seeks to position before reading: `stream_->seekg(buffer_pos_)`
- Reads on-demand into internal buffer

### 4. Dual Access Patterns
- **Buffered I/O:** Via std::istream interface, uses underflow() mechanism
- **mmap:** Direct memory access via `mem_buf_->offset(data_begin_pos_)`
- mmap bypasses stream entirely, returns raw pointer into mapped memory

### 5. Thread-Safety Considerations
- Shared stream position: All entries seek the same stream
- Non-thread-safe: Reading different entries concurrently will interfere
- mmap access: Thread-safe for reads (direct memory access)

### 6. Object Lifetime and Relationships
```
TarFile
├── stream_ (shared_ptr) - reference count = N+1
├── mem_stream_ (raw pointer to stream_ if MemStream)
└── entries_ (vector of N unique_ptr<TarEntryInputStream>)
    └── Each TarEntryInputStream
        ├── buf_ (unique_ptr<TarEntryInputStreamBuffer>)
        │   └── stream_ (shared_ptr) - shares TarFile::stream_
        └── mem_buf_ (raw pointer, same as TarFile::mem_stream_)
```

## Plans

- [053-document-tar-streaming-architecture-plan.md](../plans/053-document-tar-streaming-architecture-plan.md) - Created 2026-02-04

## Notes

**Discovery:** While analyzing stream management responsibility (Task #11 / Issue #052), explored the full architecture and discovered undocumented design patterns that should be captured before refactoring.

**Purpose:** This is purely factual documentation - no opinions, trade-off analysis, or recommendations. Serves as:
- Design rationale (why it's designed this way)
- Maintenance guide (how it works)
- Decision record (current state before potential changes)

**Related work:**
- Issue #052: Replace TarFile mem_stream_ with helper function (refactoring)
- This issue documents the architecture that #052 will modify
