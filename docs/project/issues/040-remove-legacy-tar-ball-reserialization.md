<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Issue #040: Remove Legacy tar_ball.cpp/hpp Re-Serialization

## Metadata
- **Type:** Tech Debt / Refactoring
- **Priority:** MEDIUM
- **Dependencies:** None

## Description

Remove ~750 lines of legacy TarWriter/TarReader code that wastefully re-serializes tar data when creating EP context embeddings. The code reads entries from an existing TarFile and re-serializes them using TarWriter, when TarFile already has a `dump_to()` method that directly copies the raw tar bytes.

## Problem

**Current design/code:**
```cpp
// pass_context_imp.cpp:572-580
bool PassContextImp::cache_files_to_tar_file(std::ostream& writer) const {
  TarWriter tar_writer(writer);
  auto file_names = get_cache_file_names();
  for (const auto& file_name : file_names) {
    // 1. Read entry from tar_file_ (deserialize)
    CacheFileIstreamAdapter reader(open_file_for_read(file_name));
    // 2. Write using TarWriter (re-serialize to new tar)
    tar_writer.write(reader, file_name);
  }
  return true;
}
```

**Why this is problematic:**
1. **Unnecessary re-serialization**: Reads each entry from `tar_file_` and writes them to a new tar using TarWriter, when `tar_file_` already contains properly formatted tar data
2. **Performance waste**: Double I/O operations (read + write) for every cached file entry
3. **Code duplication**: Two separate tar implementations (TarWriter/TarReader vs TarFile/TarHeader/TarEntry) with duplicate tar_checksum() functions
4. **External dependency**: Requires tar.h from FreeBSD project (patches/tar.h.force_align_1.patch)
5. **API clutter**: Three unused/test-only functions in PassContext public API

**Code locations:**
- `pass_context_imp.cpp:572-580` - `cache_files_to_tar_file()` re-serializes entire tar
- `pass_context_imp.cpp:562-570` - `cache_files_to_tar_mem()` wrapper (test-only)
- `pass_context_imp.cpp:582-606` - `tar_file_to_cache_files()` (test-only, commented out)
- `util.cpp:216-220` - `context_cache_files_to_tar_stream()` uses cache_files_to_tar_file()
- `morphizen_compile_model.cpp:419` - EP context creation calls context_cache_files_to_tar_stream()
- `tar_ball.cpp:61-94, 291-324` - Duplicate tar_checksum() implementations

## Solution

**Proposed design:**
```cpp
// util.cpp - Replace context_cache_files_to_tar_stream()
std::unique_ptr<std::istream>
context_cache_files_to_tar_stream(PassContext& context) {
  auto& ctx_imp = dynamic_cast<PassContextImp&>(context);
  CHECK(ctx_imp.tar_file_ != nullptr) << "tar_file_ should exist";

  // Direct dump - no re-serialization!
  auto size = ctx_imp.tar_file_->current_size();
  auto buffer = std::make_shared<std::string>(size, '\0');
  CHECK(ctx_imp.tar_file_->dump_to(buffer->data(), size))
      << "failed to dump tar file";

  return std::make_unique<std::istringstream>(*buffer, std::ios::binary);
}
```

**Approach:**
1. Simplify `context_cache_files_to_tar_stream()` to use TarFile::dump_to() directly (3 lines of logic)
2. Remove 3 dead functions from PassContext API: `cache_files_to_tar_mem()`, `cache_files_to_tar_file()`, `tar_file_to_cache_files()`
3. Delete tar_ball.cpp (~524 lines), tar_ball.hpp (~78 lines), test_tarball.cpp (~100+ lines)
4. Remove tar.h external dependency from CMakeLists
5. Update/remove tests in test_pass_context.cpp
6. Verify EP context creation (embed/non-embed modes) and encryption still work

**Benefits:**
- ✅ Eliminates ~750 lines of legacy code
- ✅ Removes code duplication (duplicate tar_checksum() functions)
- ✅ Better performance - direct memory dump instead of re-serialization
- ✅ Removes external tar.h dependency
- ✅ Cleaner PassContext API (3 unused functions removed)
- ✅ Single tar implementation (TarFile) instead of two

## Plans

- [040-remove-legacy-tar-ball-reserialization-plan.md](../plans/040-remove-legacy-tar-ball-reserialization-plan.md) - Created 2026-02-02
