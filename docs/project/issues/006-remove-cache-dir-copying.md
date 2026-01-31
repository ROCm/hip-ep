<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Issue #006: Remove cache_dir Entirely (Legacy Disk-Based Cache System)

## Metadata
- **Status:** BACKLOG
- **Priority:** MEDIUM
- **Type:** Tech Debt / Refactoring / Dead Code Removal
- **Owner:** TBD
- **Created:** 2026-01-30
- **Updated:** 2026-01-30 (Expanded scope after investigation)
- **Dependencies:** None (can be done independently)

## Description

Remove the entire `cache_dir` system - it's a vestige of the old vendor-specific disk-based cache format that was replaced by EP context (unified tar-based cache).

**cache_dir is DEAD CODE:**
- Old system wrote cache files to disk directories (vendor-specific format)
- New system uses tar_file_ (tmpfile or EP context binary file) - always in memory or in EP context
- cache_dir still exists but doesn't actually do anything - all I/O goes through tar_file_
- get_log_dir() returns fake paths that don't exist on disk
- Code comment explicitly says: "Cache is always in memory, skip creating cache directory"

**This isn't just bad copying - the entire cache_dir concept is obsolete.**

## Problem

**cache_dir is a vestige of the OLD cache system:**

### Old System (Vendor-Specific, Disk-Based)
- Cache files written to disk directories: `cache_dir/cache_key/filename`
- Users could specify cache_dir via provider_options
- Files persisted on disk for reuse across sessions

### New System (EP Context, Tar-Based)
- Cache files in tar_file_ (tmpfile or EP context binary file)
- Always in memory (tmpfile) or in EP context model
- No disk directories involved
- Unified format across vendors

### Evidence cache_dir is Dead

**1. Explicit comment (cache_dir.cpp:69):**
```cpp
// Cache is always in memory, skip creating cache directory
```

**2. update_cache_dir() is commented out (pass_context_imp.cpp:1239):**
```cpp
// update_cache_dir(*this);  // COMMENTED OUT
```

**3. All cache I/O goes through tar_file_:**
- open_file_for_read() → tar_file_
- open_file_for_write() → tar_file_
- No actual disk I/O to cache_dir paths

**4. get_log_dir() returns FAKE PATH:**
```cpp
// cache_dir.cpp:65-66
context.pass_context_log_dir_ =
    cache_dir / fs::u8path(context.context_proto.config().cache_key());
```
This path doesn't exist on disk - it's just a logical path for legacy compatibility.

**5. Dead code using fake paths:**
- `xclbin_path_to_cache_files()`: Returns fake path, NEVER called
- `model_set_meta_data("morphizen_log_dir")`: SET but NEVER read
- `load_context_json_2("context_dod.json")`: Tries to load from fake path (dead code)
- `collect_stat_and_dump()`: Tries to save to fake path (probably fails)

**6. Only real use:** `get_nodes()` for `MORPHIZEN_ORT_API_MAJOR < 6` - legacy ORT API compatibility

## Context

**Historical evolution:**
1. **Original design:** Vendor-specific disk-based cache (cache_dir system)
2. **Upstream evolution:** ORT proposed unified EP context format (tar-based)
3. **Current state:** EP context implemented, but cache_dir code remains as dead legacy

**Related to update_config_proto_root_field() cleanup:**
- encryption_key (Issue #004) - has security reason to remove
- cache_key (Issue #005) - unclear if needed
- cache_dir (this issue) - **DEAD CODE, entire system obsolete**
- target (Issue #007) - unclear if needed

**This issue is different:** Not just bad copying - the entire cache_dir concept should be removed.

## Solution

### Phase 1: Remove cache_dir Field from Proto

**1. Remove from config.proto:**
```protobuf
message ConfigProto {
  repeated PassProto passes = 1;
  // REMOVE: string cache_dir = 2;
  reserved 2;
  reserved "cache_dir";
  string cache_key = 3;
  // ...
}
```

**2. Remove copying in update_config_proto_root_field():**
```cpp
// DELETE lines 1266-1268
if (auto cache_dir = get_provider_option_local({"cache_dir", "cacheDir"})) {
  context_proto.mutable_config()->set_cache_dir(*cache_dir);
}
```

### Phase 2: Remove cache_dir Infrastructure

**1. Remove cache_dir.cpp and cache_dir.hpp (~73 lines):**
- `update_cache_dir()` - already commented out
- `get_cache_file_name()` - returns fake paths
- `default_cache_directory()` - unused

**2. Remove get_log_dir() system:**
- Remove `pass_context_log_dir_` member (pass_context_imp.hpp:205)
- Remove `get_log_dir()` function (pass_context_imp.cpp:134-136)
- Remove `cache_dir_set` member (pass_context_imp.hpp, line 105 usage)

**3. Remove dead callers:**
- `xclbin_path_to_cache_files()` (pass_context_imp.cpp:659-683) - NEVER called
- `load_context_json_2()` (morphizen_compile_model.cpp:149-154) - loads from fake path
- `update_primary_context()` (morphizen_compile_model.cpp:172-177) - calls above
- `model_set_meta_data("morphizen_log_dir")` (line 428-429) - never read

### Phase 3: Update Legacy ORT API Support

**For MORPHIZEN_ORT_API_MAJOR < 6 (morphizen_compile_model.cpp:535-547):**
```cpp
// OLD: get_nodes() uses cache_dir and cache_key from get_log_dir()
static std::string get_nodes(PassContextImp& context) {
  auto log_dir = context.get_log_dir();
  auto cache_dir = log_dir.parent_path().u8string();
  auto cache_key = log_dir.filename().u8string();
  // ...
}

// NEW: Read directly from provider_options or return empty
static std::string get_nodes(PassContextImp& context) {
  auto cache_dir = context.get_provider_option("cache_dir", "");
  auto cache_key = context.get_config_proto().cache_key();
  // ...
}
```

### Phase 4: Cleanup References

**Update or remove:**
- `collect_stat_and_dump()` - tries to save to fake path (line 163)
- Unit tests referencing cache_dir or get_log_dir()
- Documentation mentioning cache_dir

## Acceptance Criteria

**Phase 1: Proto Changes**
- [ ] cache_dir field removed from config.proto (field 2 reserved)
- [ ] Copying logic removed from update_config_proto_root_field()
- [ ] Proto regenerated, code compiles

**Phase 2: Infrastructure Removal**
- [ ] cache_dir.cpp and cache_dir.hpp deleted
- [ ] get_log_dir() function removed
- [ ] pass_context_log_dir_ member removed
- [ ] cache_dir_set member removed
- [ ] Dead callers removed (xclbin_path_to_cache_files, load_context_json_2, etc.)

**Phase 3: Legacy API Support**
- [ ] get_nodes() updated for ORT API < 6 (if still needed)
- [ ] Or entire legacy path removed if ORT API < 6 no longer supported

**Phase 4: Verification**
- [ ] All tests pass
- [ ] No regressions in EP context functionality
- [ ] ~200-300 lines of dead code removed
- [ ] Documentation updated

## Sessions

### 2026-01-30: Initial Creation

**User guidance:**
> "we can come back to `cache_key`, `cache_dir` and `target` later. I don't have a clear plan. I need you help to write them down and clean my mind, then we have better picture and then we have a better plan."

**Initial scope:** Just remove copying in update_config_proto_root_field().

### 2026-01-30: Investigation and Scope Expansion

**User:** "let's focus on issue 006. again, let's update the issue until you fully understand the issue."

**Investigation findings:**
1. Comment says "Cache is always in memory, skip creating cache directory"
2. update_cache_dir() is commented out
3. All cache I/O goes through tar_file_, not disk
4. get_log_dir() returns fake paths that don't exist
5. Multiple dead callers trying to use fake paths

**User confirmation:**
> "yes, I think cache_dir is now completely legacy. please correct me if I am wrong. we should not have `get_log_dir()`."

**Key insight from user:**
> "the cache system was designed to write cache files into tmp directory, or a directory specified by end-users. this is a vendor specific cache format. later on, upstream ort propose a unified cache format, i.e. ep context model."

**Decision:** Expand scope from "remove copying" to "remove entire cache_dir system"

**Reasoning:**
- cache_dir isn't just bad design - it's obsolete
- Old vendor-specific disk cache replaced by EP context tar-based cache
- Removing just the copying leaves ~200 lines of dead code
- Complete removal is cleaner and more accurate solution

## Related Issues

- **Issue #001:** Add mmap Support for Embed Mode - discusses EP context unified cache format

## Notes

### Files to Delete Entirely

- `morphizen-core/src/cache_dir.cpp` (~73 lines)
- `morphizen-core/src/cache_dir.hpp` (~16 lines)

### Code to Remove

**Proto definition:**
- `morphizen-core/src/config.proto:45` - cache_dir field (reserve field 2)

**Copying logic:**
- `pass_context_imp.cpp:1266-1268` - cache_dir copying in update_config_proto_root_field()

**Infrastructure:**
- `pass_context_imp.hpp:205` - pass_context_log_dir_ member
- `pass_context_imp.cpp:134-136` - get_log_dir() function
- `pass_context_imp.cpp:105` - cache_dir_set flag

**Dead callers (entire functions):**
- `pass_context_imp.cpp:659-683` - xclbin_path_to_cache_files() (24 lines, NEVER called)
- `morphizen_compile_model.cpp:149-154` - load_context_json_2() (5 lines, loads from fake path)
- `morphizen_compile_model.cpp:172-177` - update_primary_context() (5 lines, calls above)
- `morphizen_compile_model.cpp:428-429` - model_set_meta_data("morphizen_log_dir") (never read)

**To update:**
- `morphizen_compile_model.cpp:535-547` - get_nodes() for ORT API < 6 (read directly from provider_options)
- `morphizen_compile_model.cpp:163` - collect_stat_and_dump() (tries to save to fake path)
- `cache_dir.cpp:48-52` - get_cache_file_name() (called from dead code only)

**Unit tests:**
- `test_pass_context.cpp:34` - Sets pass_context_log_dir_ manually
- `test_pass_context.cpp:375, 379, 408` - Tests get_log_dir()

### Code That's Already Dead/Commented

- `pass_context_imp.cpp:1239` - update_cache_dir() call is commented out
- `cache_dir.cpp:69` - Comment: "Cache is always in memory, skip creating cache directory"
- `pass_context_imp.cpp:579-582` - restore_cache_files() is no-op

### Historical Context

**Old vendor-specific cache system:**
```
cache_dir/
  cache_key/
    file1.bin
    file2.bin
    context.json
```

**New EP context system:**
```
tar_file_ (tmpfile or ep_context.bin)
  └─ TAR archive:
       cache_key/file1.bin
       cache_key/file2.bin
       cache_key/context.json
```

**Key difference:** Old = disk directories, New = tar archives (in memory or EP context file)

### Estimated Removal

- **Files deleted:** 2 files (~89 lines)
- **Code removed:** ~200-300 lines total
- **Dead functions removed:** 5+ functions
- **Simplified:** No more fake paths, clearer architecture
