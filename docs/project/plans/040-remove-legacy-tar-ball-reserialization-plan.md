<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Plan: Remove Legacy tar_ball.cpp/hpp Re-Serialization

**Issue:** #040
**Created:** 2026-02-02
**Status:** READY

## Summary

Remove ~750 lines of legacy TarWriter/TarReader code that wastefully re-serializes tar data. Replace with direct TarFile::dump_to() usage which already exists.

## Root Cause: Unnecessary Re-Serialization

**Current implementation waste in `pass_context_imp.cpp:572-580`:**
```cpp
bool PassContextImp::cache_files_to_tar_file(std::ostream& writer) const {
  TarWriter tar_writer(writer);
  auto file_names = get_cache_file_names();
  for (const auto& file_name : file_names) {
    // 1. Open file from tar_file_ (deserialize)
    CacheFileIstreamAdapter reader(open_file_for_read(file_name));
    // 2. Write using TarWriter (re-serialize)
    tar_writer.write(reader, file_name);
  }
  return true;
}
```

This **RE-SERIALIZES the entire tar_file_**! It reads entries from the existing tar and writes them to a new tar.

**But tar_file_ ALREADY contains the properly formatted tar data!**

TarFile has:
- `current_size()` - returns total tar file size
- `dump_to(char*, size)` - copies raw tar bytes to buffer

## EP Context Caching Flow

**Write path (Creating EP Context):**
```
create_ep_context_nodes() [EXPORTED API]
  → create_ep_context_node() (morphizen_compile_model.cpp:560)
    → get_ep_cache_context() (line 635, if main_context)
      → get_ep_cache_context_embed_mode() (line 493)
        → get_ep_cache_context_common() (line 450)
          → context_cache_files_to_tar_stream() (line 419)
            → cache_files_to_tar_file() (util.cpp:218) ← WASTE HERE
              → TarWriter (tar_ball.cpp) ← LEGACY CODE
    → attrs.add("ep_cache_context", ep_cache_context) (line 637)
```

**Purpose**: The `ep_cache_context` attribute stores the ENTIRE tar file content as a string in the EPContext ONNX node for model embedding.

## Dead Code Analysis

**Functions to remove:**
1. ✅ `cache_files_to_tar_mem()` - test-only wrapper
2. ✅ `tar_file_to_cache_files()` - test-only (all uses commented out!)
3. ✅ `cache_files_to_tar_file()` - replaced by TarFile::dump_to()
4. ✅ `context_cache_files_to_tar_stream()` - simplified to use dump_to()
5. ✅ All of tar_ball.cpp/hpp (TarWriter, TarReader)

## Implementation Steps

### Step 1: Simplify context_cache_files_to_tar_stream()

Replace implementation at `util.cpp:216-220`:

```cpp
std::unique_ptr<std::istream>
context_cache_files_to_tar_stream(PassContext& context) {
  auto& ctx_imp = dynamic_cast<PassContextImp&>(context);
  CHECK(ctx_imp.tar_file_ != nullptr) << "tar_file_ should exist";

  auto size = ctx_imp.tar_file_->current_size();
  auto buffer = std::make_shared<std::string>(size, '\0');
  CHECK(ctx_imp.tar_file_->dump_to(buffer->data(), size))
      << "failed to dump tar file";

  return std::make_unique<std::istringstream>(*buffer, std::ios::binary);
}
```

Add friend declaration in `pass_context_imp.hpp`:
```cpp
friend std::unique_ptr<std::istream>
context_cache_files_to_tar_stream(PassContext& context);
```

### Step 2: Remove Dead Functions from PassContext API

Remove from `pass_context.hpp` (public API):
- Line 289: `virtual std::vector<char> cache_files_to_tar_mem() const = 0;`
- Line 281: `virtual bool cache_files_to_tar_file(std::ostream& writer) const = 0;`
- Line 301: `virtual bool tar_file_to_cache_files(std::istream& src) = 0;`
- Lines 30-32: Update comment to remove mentions of these functions

Remove from `pass_context_imp.hpp`:
- Line 310: `cache_files_to_tar_mem()` declaration
- Line 313: `cache_files_to_tar_file()` declaration
- Line 314: `tar_file_to_cache_files()` declaration

Remove from `pass_context_imp.cpp`:
- Lines 562-570: `cache_files_to_tar_mem()` implementation
- Lines 572-580: `cache_files_to_tar_file()` implementation
- Lines 582-606: `tar_file_to_cache_files()` implementation

### Step 3: Delete tar_ball Files

- `morphizen-core/src/tar_ball.cpp` (~524 lines)
- `morphizen-core/src/tar_ball.hpp` (~78 lines)
- `unit-test/morphizen/test_tarball.cpp` (~100+ lines)

### Step 4: Update Build System

In `morphizen-core/cmake/morphizen-core-static.cmake`:
- Remove tar_ball.cpp/hpp from source list (lines ~46-53)
- Remove tar.h external fetch (lines ~5-10) - only used by tar_ball.cpp

### Step 5: Update Tests

In `unit-test/morphizen/test_pass_context.cpp`:
- Remove tests using `cache_files_to_tar_mem()`
- Remove tests using `tar_file_to_cache_files()`
- Update tests using `cache_files_to_tar_file()` if needed

## Critical Files

### Files to DELETE (3 files, ~700 lines)
1. `morphizen-core/src/tar_ball.cpp` (~524 lines)
2. `morphizen-core/src/tar_ball.hpp` (~78 lines)
3. `unit-test/morphizen/test_tarball.cpp` (~100+ lines)

### Files to MODIFY (6 files)
1. `morphizen-core/src/util.cpp` - Simplify context_cache_files_to_tar_stream() (~10 lines)
2. `morphizen-core/src/pass_context_imp.cpp` - Remove 3 functions (~45 lines removed)
3. `morphizen-core/src/pass_context_imp.hpp` - Remove 3 declarations, add friend (~5 lines)
4. `morphizen-core/include/morphizen/pass_context.hpp` - Remove 3 virtual functions (~30 lines)
5. `morphizen-core/cmake/morphizen-core-static.cmake` - Remove tar_ball files + tar.h fetch (~10 lines)
6. `unit-test/morphizen/test_pass_context.cpp` - Update/remove tests

### Files UNCHANGED
- `morphizen-core/src/tar_file.cpp` - Already has dump_to()
- `morphizen-core/src/tar_file.hpp` - Already has dump_to()
- `morphizen-core/src/morphizen_compile_model.cpp` - Uses context_cache_files_to_tar_stream()

## Verification

1. **Build**: `cmake --build ../../build/morphizen-core --config Debug --parallel`
2. **Unit tests**: `../../build/morphizen-core/bin/morphizen-unit-tests.exe --gtest_filter="*PassContext*"`
3. **EP context test**: Verify EP context creation works:
   - Embed mode: Check get_ep_cache_context_embed_mode() produces valid tar
   - Non-embed mode: Check files are written correctly
   - Encryption: Verify encryption still works in get_ep_cache_context_common()
4. **Full regression**: Run complete test suite

## Benefits

1. **Eliminates code duplication**: Removes duplicate tar_checksum() functions
2. **Simplifies architecture**: Single tar implementation (TarFile) instead of two
3. **Removes external dependency**: No more tar.h from FreeBSD
4. **Cleaner API**: Removes 3 unused functions from PassContext public API
5. **Better performance**: Direct memory dump instead of stream re-serialization
6. **Easier maintenance**: ~750 lines removed

## Success Criteria

- All builds complete successfully
- All tests pass
- EP context creation works in both embed and non-embed modes
- Encryption functionality preserved
- No runtime performance degradation
