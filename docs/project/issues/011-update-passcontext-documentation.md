<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Issue #011: Update PassContext Header Documentation

## Metadata
- **Status:** BACKLOG
- **Priority:** LOW
- **Type:** Documentation / Cleanup
- **Owner:** TBD
- **Created:** 2026-01-30
- **Dependencies:** Related to Issue #006 (cache_dir removal)

## Description

Update outdated ASCII art diagram in pass_context.hpp - references non-existent functions.

**What's wrong:**
- ASCII art diagram (lines 21-43) references `cache_files_to_dir` and `dir_to_cache_files`
- These functions don't exist in the codebase
- Legacy documentation from old disk-based cache system (cache_dir - Issue #006)

**Why fix:**
- **Misleading documentation** - Shows functions that don't exist
- **Confusing** - Developers might look for non-existent functions
- **Outdated** - References obsolete cache_dir system

**Current status:**
- ❌ Outdated ASCII art in header (lines 21-43)
- ✅ Actual functions work correctly (cache_files_to_tar_file, tar_file_to_cache_files, etc.)

## Context

**Part of cache system cleanup effort:**
- Issue #006: cache_dir removal (obsolete disk-based cache)
- Issue #010: cache_files removal (dead proto field)
- This issue: Documentation cleanup (outdated ASCII art)

**Current ASCII art (pass_context.hpp:21-43):**
```cpp
/**


                          write_file     read_file

                               │           ▲
                               │           │
   cache_files_to_tar_mem      │           │     tar_file_to_cache_files
                               ▼           │
   ┌─────────┐    ◄──────     ┌────────────────┴─────┐   ◄────── ┌─────────┐
   │ tar mem │                │   memory cache files │           │ tar file│
   └─────────┘    ──────►     └──────────────────────┘   ──────► └─────────┘

       tar_mem_to_cache_files     │            ▲   cache_files_to_tar_file
                                  │            │
                                  │            │
                                  │            │
                                  ▼            │
                    cache_files_to_dir        dir_to_cache_files


*/
```

**Problems:**
1. `cache_files_to_dir` - Function doesn't exist
2. `dir_to_cache_files` - Function doesn't exist
3. These reference the old disk-based cache_dir system (removed in Issue #006)

**Actual functions that exist:**
- ✅ `cache_files_to_tar_mem()` - Creates tar in memory
- ✅ `cache_files_to_tar_file()` - Writes tar to stream
- ✅ `tar_file_to_cache_files()` - Extracts tar into cache
- ✅ `write_file()` - Writes individual cache file
- ✅ `read_file_c8()`, `read_file_u8()` - Reads individual cache file
- ✅ `open_file_for_read()`, `open_file_for_write()` - Opens cache file streams

## Solution

### Update Documentation Strategy

**Option A: Remove the ASCII art entirely**
- Simplest solution
- Documentation can go stale
- Code is self-documenting with good function names

**Option B: Update ASCII art to reflect current implementation**
- Show actual functions (cache_files_to_tar_file, tar_file_to_cache_files)
- Remove non-existent functions (cache_files_to_dir, dir_to_cache_files)
- Keep it accurate for current tar_file_ system

**Option C: Replace with simple text description**
- Brief description of cache system architecture
- Link to actual function documentation
- Less maintenance burden than ASCII art

### Recommended: Option A (Remove ASCII art)

**Rationale:**
- ASCII art is outdated and hard to maintain
- Function names are self-explanatory
- Detailed documentation is in function comments
- Reduces maintenance burden

### Implementation Steps

**Step 1: Remove ASCII art**
- Edit pass_context.hpp
- Delete lines 21-43 (entire ASCII art comment block)

**Step 2: Optionally add brief description**
```cpp
// PassContext provides access to cached compilation artifacts via tar_file_ system.
// Use cache_files_to_tar_mem() to export cache, tar_file_to_cache_files() to import.
// Use write_file() and read_file_*() for individual files.
```

**Step 3: Verify**
- Check that header compiles
- Verify documentation still makes sense

### Benefits

- ✅ Remove misleading documentation
- ✅ Prevent confusion about non-existent functions
- ✅ Cleaner header file
- ✅ Consistent with cache_dir removal (Issue #006)

### No Code Changes

This is documentation-only cleanup:
- No code changes required
- No compilation impact
- No behavior changes
- Safe documentation update

## Plans

_No plans needed - simple documentation update._

## Sessions

### 2026-01-30: Issue Created After PassContext API Evaluation

**Context:** User asked to evaluate pass_context.hpp for dead code.

**Evaluation approach:**
1. Systematically searched for usage of all public API functions
2. Identified outdated documentation
3. Found ASCII art references non-existent functions

**Investigation:**

**Search for functions mentioned in ASCII art:**
```bash
grep -r "cache_files_to_dir\|dir_to_cache_files" morphizen-core/ --include="*.cpp" --include="*.hpp"
# Result: No matches (functions don't exist)
```

**Only found in:**
- pass_context.hpp:40 - ASCII art diagram (documentation only)

**Actual cache system functions:**
```cpp
// Export cache to tar
virtual std::vector<char> cache_files_to_tar_mem() const = 0;
virtual bool cache_files_to_tar_file(std::ostream& writer) const = 0;

// Import cache from tar
virtual bool tar_file_to_cache_files(std::istream& src) = 0;

// Individual file access
virtual bool write_file(const std::string& filename, gsl::span<const char> data) = 0;
virtual std::optional<std::vector<char>> read_file_c8(const std::string& filename) const = 0;
```

**Analysis:**

The ASCII art shows:
- `cache_files_to_dir` - Doesn't exist (legacy from cache_dir system)
- `dir_to_cache_files` - Doesn't exist (legacy from cache_dir system)

**Original design (obsolete):**
- Old cache_dir system could export/import cache to/from disk directory
- `cache_files_to_dir` would copy mem_files_ to cache_dir
- `dir_to_cache_files` would load cache_dir into mem_files_

**Current design:**
- Only tar_file_ system exists
- No directory-based import/export
- Functions shown in diagram don't exist

**Decision:** Remove or update ASCII art to reflect current implementation.

**Recommendation:** Remove ASCII art entirely (Option A) - simpler and less maintenance.

## Related PRs

_None yet._

## Related Branches

_None yet._

## Notes

### ASCII Art Location

**File:** morphizen-core/include/morphizen/pass_context.hpp
**Lines:** 21-43

```cpp
/**


                          write_file     read_file

                               │           ▲
                               │           │
   cache_files_to_tar_mem      │           │     tar_file_to_cache_files
                               ▼           │
   ┌─────────┐    ◄──────     ┌────────────────┴─────┐   ◄────── ┌─────────┐
   │ tar mem │                │   memory cache files │           │ tar file│
   └─────────┘    ──────►     └──────────────────────┘   ──────► └─────────┘

       tar_mem_to_cache_files     │            ▲   cache_files_to_tar_file
                                  │            │
                                  │            │
                                  │            │
                                  ▼            │
                    cache_files_to_dir        dir_to_cache_files


*/
```

### What Functions Actually Exist

**Cache export (to tar):**
```cpp
// line 293
virtual bool cache_files_to_tar_file(std::ostream& writer) const = 0;

// line 301
virtual std::vector<char> cache_files_to_tar_mem() const = 0;
```

**Cache import (from tar):**
```cpp
// line 313
virtual bool tar_file_to_cache_files(std::istream& src) = 0;
```

**Individual file access:**
```cpp
// line 243-244
virtual std::optional<std::vector<char>> read_file_c8(const std::string& filename) const = 0;
virtual std::optional<std::vector<uint8_t>> read_file_u8(const std::string& filename) const = 0;

// line 249-252
virtual std::unique_ptr<CacheFileReader> open_file_for_read(const std::string& filename) const = 0;
virtual std::unique_ptr<CacheFileWriter> open_file_for_write(const std::string& filename) = 0;

// line 262
virtual bool write_file(const std::string& filename, gsl::span<const char> data) = 0;
```

### Relationship to Other Issues

**Issue #006 (cache_dir removal):**
- cache_dir was disk-based cache system
- `cache_files_to_dir` and `dir_to_cache_files` were for cache_dir
- After cache_dir removal, these functions are obsolete
- ASCII art still shows them (outdated documentation)

**Pattern:**
- Issue #006: Remove cache_dir code
- Issue #010: Remove cache_files proto field
- This issue: Update documentation to reflect removals

### Why ASCII Art Became Outdated

**Timeline:**
1. **Original design:** cache_dir system (disk-based)
   - Functions: cache_files_to_dir, dir_to_cache_files
   - ASCII art documented this flow

2. **Transition:** Added tar_file_ system (tar-based)
   - Functions: cache_files_to_tar_file, tar_file_to_cache_files
   - ASCII art updated to show both

3. **Current:** Removed cache_dir (Issue #006)
   - Only tar_file_ remains
   - ASCII art NOT updated (still shows dir functions)

**Result:** Documentation is stale, shows functions that don't exist.
