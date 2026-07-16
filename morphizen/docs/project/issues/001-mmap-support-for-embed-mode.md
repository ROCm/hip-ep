<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Issue #001: Add mmap Support for Embed Mode

## Metadata
- **Status:** BACKLOG
- **Priority:** MEDIUM
- **Type:** Feature
- **Dependencies:** None

## Description

Add memory-mapped file (mmap) support for EP context embed mode to improve memory efficiency and performance when reading cached tar files created via tmpfile().

Currently, mmap is only available in non-embed mode where tar files are persisted to disk. Embed mode uses `std::tmpfile()` which creates a FileStream that cannot be memory-mapped. This issue explores making mmap available in embed mode.

## Context

### Current Implementation

**Non-Embed Mode (with mmap):**
- Uses `TarFile::create_from_path()` for reading
- Can enable mmap via provider option `ep_context_enable_mmap`
- Creates `MemStream<MemFile>` wrapping memory-mapped file
- Direct pointer access via `mmap()` method reduces memory copies

**Embed Mode (no mmap):**
- Uses `TarFile::create()` with `tmpfile()`
- Creates `FileStream` wrapping FILE* pointer
- No mmap support - all reads go through buffered FILE* I/O
- Tar file deleted after session creation

### Why Embed Mode Currently Lacks Mmap

1. **tmpfile() returns FILE\***: Cannot easily extract underlying HANDLE for CreateFileMapping on Windows
2. **FileStream Architecture**: FileStream wraps FILE* with buffering, not designed for mmap
3. **Temporary Nature**: Embed mode tar files are temporary and meant to be ephemeral
4. **Design Intent**: Originally intended as write-only temporary buffer during processing

### Technical Constraints

**Windows (mmap_file_win.cpp:35-90):**
- Requires file HANDLE for `CreateFileMappingW()` and `MapViewOfFile()`
- FILE* from tmpfile() doesn't directly expose HANDLE
- Could use `_fileno()` and `_get_osfhandle()` to extract HANDLE (similar to util_mswin.cpp:140)

**Linux:**
- mmap not currently implemented (mmap_file.hpp returns nullptr on Linux)
- Would need similar FILE* to fd extraction

## Acceptance Criteria

- [ ] Embed mode tar files can be memory-mapped on Windows
- [ ] Provider option `ep_context_enable_mmap` controls mmap for both embed and non-embed modes
- [ ] Fallback to FileStream if mmap creation fails (maintain current robustness)
- [ ] No performance regression for existing non-embed mode mmap
- [ ] Memory consumption measured and compared (before/after)
- [ ] Unit tests verify mmap functionality in embed mode
- [ ] Documentation updated explaining mmap behavior in both modes

## Technical Design Options

### Option 1: Extract HANDLE from tmpfile() FILE*

**Approach:**
- After `tmpfile_with_posix_delete()` creates FILE* (util_mswin.cpp:115)
- Extract HANDLE using `_fileno()` and `_get_osfhandle()` (similar to line 133-145)
- Create MemFile wrapper around the HANDLE
- Wrap in MemStream and pass to TarFile

**Pros:**
- Reuses existing MemFile/MemStream infrastructure
- Minimal changes to TarFile architecture
- Leverages proven mmap implementation

**Cons:**
- Platform-specific code (Windows HANDLE extraction)
- FILE* and HANDLE both reference same file (requires careful lifetime management)
- Closing FILE* might invalidate HANDLE or vice versa

**Files to modify:**
- `morphizen-core/src/tar_file.cpp` - create() method
- `morphizen-core/src/mmap_file_win.cpp` - new MemFile variant accepting HANDLE
- New class: `MemFileTmpHandle` or similar

### Option 2: tmpfile() on Known Path, Then Mmap

**Approach:**
- Create temporary file with known path (using GetTempPath/tmpnam)
- Use FILE* for initial writing
- After writing complete, close FILE*
- Reopen with TarFile::create_from_path() enabling mmap

**Pros:**
- Clean separation: write with FILE*, read with mmap
- Reuses existing create_from_path() logic
- No mixing of FILE* and HANDLE lifetime

**Cons:**
- Temporary file not automatically deleted (requires explicit cleanup)
- Loses POSIX delete semantics from tmpfile_with_posix_delete()
- More file I/O overhead (close and reopen)

**Files to modify:**
- `morphizen-core/src/pass_context_imp.cpp` - maybe_create_tar_file_for_write()
- Cleanup logic to delete temp file after session

### Option 3: Dual-Stream TarFile for Embed Mode

**Approach:**
- Create TarFile with both FileStream (for writing) and MemStream (for reading)
- Initially write to tmpfile via FileStream
- After write phase, create MemFile from same HANDLE
- Switch to MemStream for read operations

**Pros:**
- No file reopening
- Supports both write and mmap read from same file
- Maintains tmpfile cleanup semantics

**Cons:**
- More complex TarFile state management
- Need careful synchronization between streams
- Larger architectural change to TarFile

**Files to modify:**
- `morphizen-core/src/tar_file.hpp` - add dual-stream mode
- `morphizen-core/src/tar_file.cpp` - stream switching logic

## Feasibility Assessment

### Windows: FEASIBLE
- HANDLE extraction from FILE* is proven (util_mswin.cpp:140)
- CreateFileMapping works with HANDLEs from tmpfile
- Need careful lifetime management to avoid double-close issues

### Linux: REQUIRES INVESTIGATION
- Current mmap implementation returns nullptr
- Need to implement mmap support first
- fd extraction from FILE* using `fileno()` is standard

## Performance Considerations

**Expected Benefits:**
- Reduced memory copies when reading cached tar entries
- Lower memory pressure (OS can page out unused mmap regions)
- Faster random access to tar entries

**Potential Concerns:**
- tmpfile() files may be on slower temp partition
- mmap overhead for small tar files might exceed buffered I/O benefits
- Need benchmark comparison

**Benchmark Plan:**
1. Measure memory usage: embed mode with/without mmap
2. Measure read latency: mmap vs FileStream for various tar sizes
3. Measure session creation time with mmap enabled

## Sub-tasks

- [ ] Research HANDLE lifetime management when extracted from FILE*
- [ ] Prototype Option 1 (extract HANDLE from tmpfile)
- [ ] Create benchmark comparing mmap vs FileStream performance
- [ ] Implement MemFileTmpHandle wrapper class
- [ ] Modify TarFile::create() to support mmap
- [ ] Add provider option handling for embed mode mmap
- [ ] Write unit tests for embed mode mmap
- [ ] Measure memory consumption before/after
- [ ] Update documentation

## Notes

### Key Code References

**Embed mode tar creation:**
- `morphizen-core/src/pass_context_imp.cpp:1080-1086` - Uses TarFile::create()

**Non-embed mode mmap:**
- `morphizen-core/src/pass_context_imp.cpp:1114-1116` - Uses create_from_path() with mmap
- `morphizen-core/src/tar_file.cpp:25-58` - create_from_path() implementation

**Windows tmpfile with POSIX delete:**
- `morphizen-core/src/util_mswin.cpp:115-232` - tmpfile_with_posix_delete()
- `morphizen-core/src/util_mswin.cpp:140` - HANDLE extraction example

**MemFile implementation:**
- `morphizen-core/src/mmap_file_win.cpp:35-90` - Windows mmap via CreateFileMapping

**Mmap access in tar entries:**
- `morphizen-core/src/tar_entry.cpp:228-233` - TarEntryInputStream::mmap()

### Questions to Resolve

1. **Lifetime Management**: If we extract HANDLE from FILE*, which should close first?
   - Hypothesis: Close FILE* first, then close HANDLE manually
   - Need to verify Windows behavior

2. **POSIX Delete Compatibility**: Does mmap interfere with POSIX delete semantics?
   - tmpfile_with_posix_delete() sets FILE_DISPOSITION_POSIX_SEMANTICS
   - Need to verify mmap doesn't prevent deletion

3. **Performance Threshold**: At what tar file size does mmap become beneficial?
   - Small files (<1MB): buffered I/O might be faster
   - Large files (>10MB): mmap should win
   - Need empirical data

4. **Linux Support**: Should we implement Linux mmap as part of this issue?
   - Currently Linux returns nullptr for MemFile::create()
   - Consider separate issue for Linux mmap support

### References

- tmpfile() specification: https://en.cppreference.com/w/cpp/io/c/tmpfile
- POSIX delete on Windows: See technical/tmpfile-posix-delete.md
