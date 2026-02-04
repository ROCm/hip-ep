<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Document Complex TarFile Factory Methods

**Issue:** #051
**Created:** 2026-02-03
**Status:** READY

## Objective

Add comprehensive comments to two complex TarFile factory methods to document mmap strategy, platform differences, fallback logic, and troubleshooting context while knowledge is fresh (code added Feb 1-3, 2026).

## Background

**Problem discovered:** While analyzing factory method proliferation, found the 78-line `create(string, bool)` method is too complex to safely refactor (just added 1-3 days ago). Better to document comprehensively first.

**Why document now:** Knowledge is fresh, code needs to stabilize before refactoring, comprehensive comments enable future maintenance.

## Implementation Steps

### Step 1: Add High-level Strategy Comment to create_from_buffer()

**File:** `morphizen-core/src/tar_file.cpp`

**Location:** Above `TarFile::create(std::string&& buffer0, bool enable_mmap)` at line 108

**Note:** After Issue #050, this will be renamed to `create_from_buffer()`. Use the current name for now.

**Add comment:**

```cpp
/* Creates TarFile from string buffer.
 *
 * Strategy: Optimize for different deployment environments:
 * - Windows: tmpfile + optional mmap (best performance, Issue #001)
 * - Linux: tmpfile + FileStream (mmap not implemented yet, TODO)
 * - WebNN/Browser: In-memory buffer (sandboxed, no tmpfile access)
 *
 * Mmap can be disabled for troubleshooting hard-to-debug system failures:
 * - Production: enable_mmap=false (provider option, user-configurable)
 * - Development: MORPHIZEN_ENABLE_TAR_MMAP=0 (env var, developer-only)
 *
 * Falls back gracefully: tmpfile+mmap → tmpfile+FileStream → memory buffer
 */
std::unique_ptr<TarFile> TarFile::create(std::string&& buffer0,
                                         bool enable_mmap) {
```

### Step 2: Document tmpfile Creation Section

**File:** `morphizen-core/src/tar_file.cpp`

**Location:** Lines 111-115 (tmpfile creation)

**Update comment:**

```cpp
  std::unique_ptr<std::iostream> stream;
  // Create tmpfile (platform-specific, see Issue #048 for consolidation)
#ifdef _WIN32
  auto file = tmpfile_with_posix_delete();
#else
  auto file = std::tmpfile();
#endif
```

### Step 3: Document tmpfile Success Path

**File:** `morphizen-core/src/tar_file.cpp`

**Location:** Line 119 (if file succeeded)

**Update comment:**

```cpp
  if (file) {
    // tmpfile created successfully - use disk-backed stream for lower memory usage
    LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_TAR_CACHE))
        << " create a tar file from temp file";

    // Write buffer data to tmpfile
    auto r = fwrite(buffer0.data(), 1, buffer0.size(), file);
    CHECK_EQ(r, buffer0.size()) << "write error";
    fflush(file); // Ensure data is written before mmap
    r = fseek(file, 0, SEEK_SET);
    CHECK_EQ(r, 0);
```

### Step 4: Document Two-level Mmap Control

**File:** `morphizen-core/src/tar_file.cpp`

**Location:** Lines 130-136 (mmap control logic)

**Keep existing comment, enhance it:**

```cpp
    // Try to create memory-mapped stream for better performance
    // Two-level control (intentional, do not simplify):
    // 1. enable_mmap: User preference via provider option (ep_context_enable_mmap)
    //    - Users can disable in production for troubleshooting
    // 2. ENV_PARAM: Global override for debugging/compatibility
    //    - Developers only, cannot change in production
    // This allows disabling mmap when it causes hard-to-debug system failures
    bool use_mmap = enable_mmap && (ENV_PARAM(MORPHIZEN_ENABLE_TAR_MMAP) != 0);
```

### Step 5: Document Windows Mmap Logic

**File:** `morphizen-core/src/tar_file.cpp`

**Location:** Lines 137-168 (Windows mmap section)

**Update comments:**

```cpp
#ifdef _WIN32
    // Windows: tmpfile mmap supported via MemFileTmpHandle (Issue #001)
    // Linux: TODO - not implemented yet, no customer request
    if (use_mmap) {
      try {
        auto mem_file = MemFileTmpHandle::create(file);
        if (mem_file != nullptr) {
          LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_TAR_CACHE))
              << "created memory-mapped stream from tmpfile (embed mode)";
          auto base = mem_file->base();
          auto size = mem_file->size();
          stream = std::make_unique<MemStream<MemFile>>(
              MemBuffer<MemFile>::create(base, size, std::move(mem_file)));
          // Safe to close FILE* - mmap keeps data accessible
          fclose(file);
        } else {
          // Fallback 1: mmap creation failed - use regular FileStream
          MY_LOG(1) << "mmap creation failed, falling back to FileStream";
          stream = std::make_unique<FileStream>(file);
        }
      } catch (const std::exception& e) {
        // Fallback 2: exception during mmap - use regular FileStream
        // Common when troubleshooting mmap-related issues
        MY_LOG(1) << "mmap creation exception: " << e.what()
                  << ", falling back to FileStream";
        stream = std::make_unique<FileStream>(file);
      }
    } else {
      // mmap disabled via configuration - use regular FileStream
      stream = std::make_unique<FileStream>(file);
    }
#else
    // Non-Windows platforms: tmpfile mmap not implemented yet (TODO)
    // Falls back to regular FileStream
    (void)use_mmap; // Suppress unused variable warning
    stream = std::make_unique<FileStream>(file);
#endif
```

### Step 6: Document Memory Buffer Fallback

**File:** `morphizen-core/src/tar_file.cpp`

**Location:** Lines 169-178 (memory buffer fallback)

**Update comment:**

```cpp
  } else {
    // Fallback 3: tmpfile creation failed - use in-memory buffer
    // Common in WebNN/Browser environments (sandboxed, no filesystem access)
    LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_TAR_CACHE))
        << " create a tar file from memory buffer";
    auto buff_owner = std::make_unique<std::string>(std::move(buffer0));
    auto buff_base = buff_owner->data();
    auto buff_size = buff_owner->size();
    stream =
        std::make_unique<MemStream<std::string>>(MemBuffer<std::string>::create(
            buff_base, buff_size, std::move(buff_owner)));
  }
```

### Step 7: Add High-level Strategy Comment to create_from_path()

**File:** `morphizen-core/src/tar_file.cpp`

**Location:** Above `TarFile::create_from_path()` at line 28

**Add comment:**

```cpp
/* Creates TarFile from existing file on disk.
 *
 * Strategy: Use mmap for better performance when reading tar files.
 * - Windows: MemFile::create() for mmap support
 * - Linux: MemFile::create() for mmap support
 * - Fallback: Regular fstream if mmap fails or disabled
 *
 * Mmap can be disabled via enable_mmap parameter or ENV_PARAM
 * for troubleshooting (same controls as create_from_buffer).
 */
std::unique_ptr<TarFile>
TarFile::create_from_path(const std::filesystem::path& path, bool enable_mmap) {
```

### Step 8: Document create_from_path Fallback Lambda

**File:** `morphizen-core/src/tar_file.cpp`

**Location:** Lines 29-37 (fallback lambda)

**Update comment:**

```cpp
  // Fallback helper: creates regular fstream (no mmap)
  // Used when mmap disabled or mmap creation fails
  auto create_with_regular_stream = [&]() -> std::unique_ptr<TarFile> {
    auto stream =
        std::make_unique<std::fstream>(path, std::ios::binary | std::ios::in);
    if (!stream->is_open()) {
      MY_LOG(1) << "Failed to open file: " << path.string();
      return nullptr;
    }
    return TarFile::create(std::move(stream));
  };
```

### Step 9: Document create_from_path Mmap Logic

**File:** `morphizen-core/src/tar_file.cpp`

**Location:** Lines 38-60 (mmap attempt with fallbacks)

**Update comments:**

```cpp
  // Check if mmap is enabled (two-level control, same as create_from_buffer)
  if (!enable_mmap) {
    return create_with_regular_stream();
  }
  if (ENV_PARAM(MORPHIZEN_ENABLE_TAR_MMAP) == 0) {
    return create_with_regular_stream();
  }

  // Try to create mmap-backed stream
  try {
    auto mem_file = MemFile::create(path);
    if (mem_file == nullptr) {
      // Fallback 1: mmap creation failed
      MY_LOG(1) << "do not support to create MMapFile object: ";
      return create_with_regular_stream();
    }
    auto base = mem_file->base();
    auto size = mem_file->size();
    auto stream = std::make_unique<MemStream<MemFile>>(
        MemBuffer<MemFile>::create(base, size, std::move(mem_file)));
    return TarFile::create(std::move(stream));
  } catch (const std::exception& e) {
    // Fallback 2: exception during mmap
    MY_LOG(1) << "Failed to create MMapFile object: " << e.what();
  }
  return create_with_regular_stream();
```

## Verification

### Build
```bash
cmake --build ../../build/morphizen-core --config Debug --parallel
```

**Expected:** Build succeeds (no code changes, only comments)

### Code Review

- [ ] High-level strategy documented for both methods
- [ ] Platform differences explained (Windows mmap, Linux TODO)
- [ ] Troubleshooting context documented (why disable mmap)
- [ ] User vs developer controls explained
- [ ] All three fallback paths documented with reasons
- [ ] Cross-references to Issue #001 and #048 added
- [ ] WebNN/Browser environment mentioned

## Success Criteria

- [ ] `create(string, bool)` has comprehensive comments
- [ ] `create_from_path()` has comprehensive comments
- [ ] Strategy, platform differences, and fallbacks documented
- [ ] Troubleshooting context captured (disable mmap for hard-to-debug failures)
- [ ] WebNN environment requirements documented
- [ ] Cross-references to related issues added
- [ ] Build succeeds
- [ ] Future developers can understand the design without asking

## Files Modified

- `morphizen-core/src/tar_file.cpp` - Add comprehensive comments to 2 methods

## Notes

**Why comprehensive comments:**
This code was added 1-3 days ago (Feb 1-3, 2026) with complex mmap logic. Knowledge is fresh now but will fade. Comments preserve the design decisions and context.

**Timing is critical:**
- Code is fresh - you remember all the context
- Code is complex - 78 lines with multiple fallback paths
- Code is high-stakes - affects performance and compatibility
- Code should stabilize - too early to refactor

**Future work:**
After this code stabilizes in production (few months), consider extracting helper functions if patterns emerge. But for now, comprehensive documentation is the safest improvement.
