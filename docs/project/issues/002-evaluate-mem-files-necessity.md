<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Issue #002: Remove mem_files_ - Always Create tar_file_ for Cache

## Metadata
- **Status:** BACKLOG
- **Priority:** MEDIUM
- **Type:** Refactoring / Bug Fix
- **Owner:** TBD
- **Created:** 2026-01-30
- **Updated:** 2026-01-30 (Final Correct Understanding)

## Description

Remove `mem_files_` (~200 LOC) by fixing the root cause: tar_file_ should always be created for cache files, regardless of EP context state.

**Root Cause:** Cache files serve TWO purposes:
1. **Inter-pass communication** (Level 1 passes, custom ops) - ALWAYS needed
2. **EP context persistence** (external save) - Only when enabled

Current code conflates these: when EP context disabled → no tar_file_ → fallback to mem_files_

**Correct design:** Always create tar_file_, EP context flag only controls external persistence.

## Problem

Current flawed logic:

```cpp
// pass_context_imp.cpp:1036-1040
if (!is_ep_context_enabled) {
    return;  // Don't create tar_file_ - WRONG!
}
```

**Why this is wrong:**
- Passes and custom ops need cache files to communicate
- Cache needed even when EP context disabled
- mem_files_ exists only as workaround for this design flaw

**Additional bug:**
- TarFile::create() crashes on tmpfile() failure (tar_file.cpp:64)
- Should fallback to MemStream like create(string&&) does

## Solution

### Part 1: Fix TarFile::create() Crash

**Current (tar_file.cpp:59-72):**
```cpp
std::unique_ptr<TarFile> TarFile::create() {
  auto file = tmpfile_with_posix_delete();
  CHECK(file != nullptr) << "cannot create a tmpfile";  // CRASHES
  auto stream = std::unique_ptr<std::iostream>(new FileStream(file));
  return create(std::move(stream));
}
```

**Fixed:**
```cpp
std::unique_ptr<TarFile> TarFile::create() {
  auto file = tmpfile_with_posix_delete();
  std::unique_ptr<std::iostream> stream;

  if (file) {
    // Success: use tmpfile
    stream = std::make_unique<FileStream>(file);
  } else {
    // Fallback: use in-memory MemStream (sandbox/restricted environments)
    LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_TAR_CACHE))
        << "tmpfile() failed, using in-memory stream";
    stream = std::make_unique<MemStream<std::string>>(
        MemBuffer<std::string>::create());
  }

  return create(std::move(stream));
}
```

**Matches existing pattern in TarFile::create(string&&) at lines 90-126.**

### Part 2: Always Create tar_file_

**Current (pass_context_imp.cpp:1023-1096):**
```cpp
void PassContextImp::maybe_create_tar_file_for_write() {
  auto is_ep_context_enabled =
      get_session_config(kOrtSessionOptionEpContextEnable, "0") == "1";

  if (!is_ep_context_enabled) {
    return;  // REMOVE THIS - cache needed regardless
  }

  // ... create tar_file_ ...
}
```

**Fixed:**
```cpp
void PassContextImp::maybe_create_tar_file_for_write() {
  auto is_ep_context_enabled =
      get_session_config(kOrtSessionOptionEpContextEnable, "0") == "1";
  auto is_ep_context_embed_mode =
      get_session_config(kOrtSessionOptionEpContextEmbedMode, "1") == "1";

  // ALWAYS create tar_file_ for cache (pass communication)
  // EP context flag only controls WHERE to persist, not WHETHER to create

  if (!is_ep_context_enabled) {
    // EP disabled: tmpfile (in-memory), discarded after session
    tar_file_ = TarFile::create();
    CHECK(tar_file_ != nullptr);
    return;
  }

  // EP enabled: save to external file or embed in model
  if (is_ep_context_embed_mode) {
    tar_file_ = TarFile::create();  // tmpfile, serialized to model later
  } else {
    // Non-embed: create persistent file
    auto ep_context_binary_file = get_dir_of_ep_context_model() /
                                  get_basename_of_ep_context_binary_file();
    // ... existing file creation code ...
  }
}
```

### Part 3: Remove mem_files_

**After Parts 1 & 2, tar_file_ always exists successfully:**
- Works in sandbox (MemStream fallback)
- Works when EP disabled (tmpfile/MemStream)
- Works when EP enabled (file or tmpfile)

**Remove:**
- `mem_files_` declaration (pass_context_imp.hpp:358)
- MemoryFileReaderImp/WriterImp (~80 LOC)
- CacheFile adapters (~70 LOC)
- Fallback logic in open_file_for_read/write (~50 LOC)

**Simplify:**
```cpp
std::unique_ptr<CacheFileWriter>
PassContextImp::open_file_for_write(const std::string& filename) {
  CHECK(tar_file_ != nullptr) << "tar_file_ should always exist";
  return open_file_for_write_with_tar_file(filename);
}
```

## Acceptance Criteria

- [ ] TarFile::create() handles tmpfile() failure gracefully (MemStream fallback)
- [ ] tar_file_ always created in maybe_create_tar_file_for_write()
- [ ] mem_files_ and related code removed (~200 LOC)
- [ ] Cache works in all scenarios (EP enabled/disabled, sandbox/normal)
- [ ] Unit tests pass for sandbox mode
- [ ] No regressions in existing functionality

## Implementation Plan

**Phase 1: Fix TarFile::create() (1 day)**
1. Add MemStream fallback when tmpfile() fails
2. Match pattern from create(string&&)
3. Add logging when fallback activates
4. Write unit test mocking tmpfile() failure

**Phase 2: Always Create tar_file_ (1 day)**
1. Remove EP context check from maybe_create_tar_file_for_write()
2. Create tar_file_ via TarFile::create() when EP disabled
3. Verify passes/custom ops work with tar_file_
4. Test EP enabled/disabled scenarios

**Phase 3: Remove mem_files_ (1 day)**
1. Remove mem_files_ map and all related classes
2. Simplify open_file_for_read/write (remove fallback logic)
3. Update tests
4. Verify no regressions

**Total: 3 days**

## Plans

_No plans yet._

## Sessions

### 2026-01-30: Root Cause Identified

**Key insight from user:**
> "we need cache files even when ep context is disabled. enabling ep context means we would like to save the tar file in an external ep context model. when it is disabled, all level 1 passes and my custom ops need cache files as a bridge to communicate with each other."

**This revealed:**
- Cache files have TWO purposes (communication + persistence)
- Current design conflates them
- mem_files_ is workaround for design flaw
- Proper fix: always create tar_file_, EP flag controls persistence

**Previous misunderstandings:**
- ~~mem_files_ for sandbox/tmpfile() failure~~ NO - TarFile handles that internally
- ~~mem_files_ for EP-disabled caching~~ PARTIAL - needed due to design flaw
- **CORRECT:** mem_files_ compensates for missing tar_file_ when EP disabled

## Related PRs

- #35 (748a5b6): Remove cache_files_ and unify storage

## Notes

**Cache purposes:**
1. **Inter-pass communication** - Always needed (passes, custom ops)
2. **EP context persistence** - Optional (external save when enabled)

**EP context flag semantics:**
- **Enabled:** Persist cache to external file/model
- **Disabled:** Keep cache in memory only (discard after session)

**Code locations:**
- `tar_file.cpp:64` - TarFile::create() crash point
- `tar_file.cpp:90-126` - TarFile::create(string&&) graceful fallback pattern
- `pass_context_imp.cpp:1036` - Incorrect EP context check
- `pass_context_imp.hpp:358` - mem_files_ declaration (to be removed)
