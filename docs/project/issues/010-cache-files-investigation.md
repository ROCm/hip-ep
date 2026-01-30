<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Issue #010: cache_files Field Investigation

## Metadata
- **Status:** BACKLOG
- **Priority:** LOW
- **Type:** Investigation / Tech Debt
- **Owner:** TBD
- **Created:** 2026-01-30
- **Dependencies:** Related to Issue #005 (cache_key)

## Description

Remove `cache_files` field from ContextProto - it's dead code from the obsolete cache system.

**Decision after investigation:** cache_files proto field is NEVER READ - only written but not used. Remove completely.

**Current location:** ContextProto field 12 (pass_context.proto) - NOTE: Not field 6 as initially stated

**From Issue #005 discussion:**
> User: "cache_files is another issue, we should put it aside, but don't forget it."

**What cache_files was:**
- Proto field tracking list of cached compilation artifacts
- Intended for restoring cache files from saved state
- Part of obsolete disk-based cache system

**Why remove:**
- **Dead code** - Written to but NEVER read from
- **restore_cache_files() is NO-OP** - Explicitly says "No longer needed with tar_file_"
- **Redundant** - tar_file_.entries() provides actual file list
- **Legacy** - From old cache_dir system (Issue #006)

**Current status:**
- ✅ Investigated completely
- ✅ Confirmed dead code (no read usage)
- ❌ Proto field still exists (field 12)
- ❌ restore_cache_files() still exists (but is no-op)
- ❌ Outdated API comments still reference cache_files_

## Context

**Part of cache system cleanup:**
- Issue #005: cache_key - Move from ConfigProto to ContextProto (cache metadata, needs persistence)
- Issue #006: cache_dir - Remove entirely (obsolete legacy system)
- This issue: cache_files - Investigate usage and design

**ContextProto structure (after Issue #003 and #005):**
```protobuf
message ContextProto {
  repeated MetaDefProto meta_def = 1;     // Compilation results
  // ConfigProto config = 3;              // Removed in Issue #003
  map<string, AnchorPointProto> origin_nodes = 4;
  repeated PassEventProto events = 5;
  repeated string cache_files = 6;        // THIS ISSUE
  string cache_key = 7;                   // Added in Issue #005
}
```

**cache_files is already in ContextProto (correct location for cache metadata)**, but need to investigate:
1. What does it contain? (list of cached file names?)
2. How is it populated?
3. How is it used?
4. Is it related to tar_file_ system?
5. Any issues similar to cache_key or cache_dir?

## Solution

### Complete Removal Strategy

**Decision:** Remove cache_files field and restore_cache_files() function entirely.

**What to remove:**

1. **Remove cache_files from proto** (pass_context.proto:56)
   ```protobuf
   message ContextProto {
     repeated MetaDefProto meta_def = 1;
     ConfigProto config = 3;
     map<string, AnchorPointProto> origin_nodes = 4;
     map<string, int32> fix_info = 6;
     map<string, int32> device_subgraph_count = 7;
     repeated string stacks = 8;
     repeated EventProto events = 9;
     repeated CPUUsageProto cpu_usage = 10;
     map<string, SubgraphProto> subgraph_metadefs = 11;
     // DELETE THIS LINE:
     repeated string cache_files = 12;
   }
   ```

   Reserve the field number:
   ```protobuf
   reserved 12;
   reserved "cache_files";
   ```

2. **Remove restore_cache_files() from public API** (pass_context.hpp:266-268)
   ```cpp
   // DELETE:
   /**
    * restore cache_files_ from saved file
    */
   virtual void restore_cache_files() = 0;
   ```

3. **Remove restore_cache_files() implementation** (pass_context_imp.cpp:578-582)
   ```cpp
   // DELETE:
   void PassContextImp::restore_cache_files() {
     // No longer needed with tar_file_ - this is a no-op
     // Cache files are loaded directly via tar_file_ or mem_files_
     LOG_VERBOSE(2) << "restore_cache_files: no-op with tar_file_";
   }
   ```

4. **Remove restore_cache_files() declaration** (pass_context_imp.hpp:320)
   ```cpp
   // DELETE:
   virtual void restore_cache_files() override final;
   ```

5. **Remove restore_cache_files() call** (pass_context_imp.cpp:1249)
   ```cpp
   pass_context_update_context_json(context_context_json_text);
   // DELETE THIS LINE:
   restore_cache_files();
   ```

6. **Remove cache_files population** (pass_context_imp.cpp:715-718)
   ```cpp
   // In save_context_json(), DELETE:
   if (std::find(proto.mutable_cache_files()->begin(),
                 proto.mutable_cache_files()->end(),
                 "context.json") == proto.mutable_cache_files()->end()) {
     proto.add_cache_files("context.json");
   }
   ```

**What to keep:**

1. **tar_file_ system** - This is the actual cache implementation
2. **get_cache_file_names()** - Returns tar_file_.entries() (useful)
3. **cache_files_to_tar_file() / tar_file_to_cache_files()** - Named for clarity, not related to proto field

### Implementation Steps

**Step 1: Remove proto field**
- Edit pass_context.proto
- Delete `repeated string cache_files = 12;`
- Add `reserved 12;` and `reserved "cache_files";`

**Step 2: Remove API function**
- Edit pass_context.hpp
- Delete restore_cache_files() virtual function declaration and comment

**Step 3: Remove implementation**
- Edit pass_context_imp.cpp
- Delete restore_cache_files() implementation (lines 578-582)
- Delete restore_cache_files() call (line 1249)
- Delete cache_files population in save_context_json() (lines 715-718)

**Step 4: Remove declaration**
- Edit pass_context_imp.hpp
- Delete restore_cache_files() override declaration

**Step 5: Verify compilation**
- Build the project
- Confirm no linker errors
- All tests pass

### Benefits

- ✅ Remove ~10 lines of dead code
- ✅ Remove confusing no-op function
- ✅ Remove obsolete proto field
- ✅ Cleaner API (no misleading comments)
- ✅ Consistent with cache_dir removal (Issue #006)

### No Breaking Changes Risk

**Why safe to remove:**
- cache_files proto field is NEVER READ (verified by grep)
- restore_cache_files() is already a NO-OP (does nothing)
- Proto field backward compatible (old binaries can still read new proto with reserved field)
- No external code relies on this (internal implementation detail)

**Proto compatibility:**
- Reserving field 12 ensures no future field reuses this number
- Old binaries reading new proto: Field is missing, ignored safely
- New binaries reading old proto: Field is present but ignored (not read)

## Plans

_No plans yet - investigation needed first._

## Sessions

### 2026-01-30: Issue Created as Placeholder

**Context:** Identified during Issue #005 discussion about cache_key.

**User guidance:**
> "cache_files is another issue, we should put it aside, but don't forget it."

**Purpose:** Document that cache_files needs investigation. Defer until after cache_key and cache_dir are resolved.

### 2026-01-30: Investigation and Decision

**User:** "lets discuss about 010"

**Investigation approach:**
1. Search for cache_files usage in codebase
2. Trace how it's populated and consumed
3. Understand relationship to tar_file_ system
4. Determine if cleanup needed

**Key findings:**

**1. Proto field location:**
- Initially thought field 6, actually field 12 in pass_context.proto:56
- No comment explaining purpose

**2. Where cache_files is WRITTEN:**
- Only one location: pass_context_imp.cpp:715-718
- In `save_context_json()` - adds "context.json" if not already present
- That's it - no other code writes to it

**3. Where cache_files is READ:**
- NOWHERE! Searched for `.cache_files()` accessor - zero usages
- Searched for `mutable_cache_files()` - only the one write location above
- **cache_files proto field is NEVER READ**

**4. get_cache_file_names() function:**
```cpp
// pass_context_imp.cpp:595-612
std::vector<std::string> PassContextImp::get_cache_file_names() const {
  auto ret = std::vector<std::string>{};
  if (tar_file_) {
    const auto& entries = tar_file_->entries();  // ← Reads from tar_file_
    ret.reserve(entries.size());
    for (const auto& entry : entries) {
      if (entry && !entry->is_symlink()) {
        ret.push_back(entry->path());
      }
    }
  } else {
    ret.reserve(mem_files_.size());
    for (const auto& [name, _] : mem_files_) {
      ret.push_back(name);
    }
  }
  return ret;
}
```
- **Does NOT read from cache_files proto field**
- Reads directly from tar_file_.entries() or mem_files_

**5. restore_cache_files() function:**
```cpp
// pass_context_imp.cpp:578-582
void PassContextImp::restore_cache_files() {
  // No longer needed with tar_file_ - this is a no-op
  // Cache files are loaded directly via tar_file_ or mem_files_
  LOG_VERBOSE(2) << "restore_cache_files: no-op with tar_file_";
}
```
- **Explicitly a NO-OP!**
- Comment says: "No longer needed with tar_file_"
- Called from line 1249 but does nothing

**6. Public API comment (pass_context.hpp:266):**
```cpp
/**
 * restore cache_files_ from saved file
 */
virtual void restore_cache_files() = 0;
```
- Comment refers to "cache_files_" member (with underscore)
- **This member doesn't exist** - misleading comment
- Function is a no-op anyway

**Analysis:**

**Original design (obsolete):**
- cache_files proto field tracked which files were in cache
- restore_cache_files() would restore files from disk based on this list
- Part of old disk-based cache system (cache_dir - Issue #006)

**Current design:**
- tar_file_ directly contains all cached files
- get_cache_file_names() reads from tar_file_.entries() directly
- cache_files proto field is legacy dead code
- restore_cache_files() is a no-op placeholder

**Verification:**
```bash
# Search for any read usage of cache_files proto field
grep -r "\.cache_files()" morphizen-core/src/
# Result: No matches

# Search for write usage
grep -r "mutable_cache_files\|add_cache_files" morphizen-core/src/
# Result: Only one location - save_context_json()
```

**Conclusion:**

cache_files is **DEAD CODE**:
1. WRITTEN: Only in save_context_json() (adds "context.json")
2. READ: NEVER - no code reads from it
3. restore_cache_files(): NO-OP - does nothing, obsolete function
4. Redundant with tar_file_.entries() which provides actual file list

**Decision:** Remove cache_files proto field and restore_cache_files() function entirely.

**Rationale:**
- Same pattern as Issue #006 (cache_dir) and Issue #009 (xclbin) - removing obsolete legacy code
- tar_file_ system is the actual implementation, cache_files is vestigial
- No breaking changes (proto field never read, function is no-op)
- Cleaner codebase, less confusion

## Related PRs

_None yet._

## Related Branches

_None yet._

## Notes

### Investigation Results

**Proto field location:**
- pass_context.proto:56 - `repeated string cache_files = 12;`
- NOTE: Initially thought field 6, but ContextProto has many reserved/removed fields
- Actual field number is 12

**Where cache_files is written (only 1 location):**
- pass_context_imp.cpp:715-718 in save_context_json()
  ```cpp
  if (std::find(proto.mutable_cache_files()->begin(),
                proto.mutable_cache_files()->end(),
                "context.json") == proto.mutable_cache_files()->end()) {
    proto.add_cache_files("context.json");
  }
  ```
- Adds "context.json" if not already present
- That's the ONLY place cache_files is modified

**Where cache_files is read (0 locations):**
- Searched for `.cache_files()` accessor - NO MATCHES
- Searched for usage of field - NO MATCHES
- **cache_files proto field is NEVER READ**

**restore_cache_files() - NO-OP function:**
- Public API: pass_context.hpp:266-268
  ```cpp
  /**
   * restore cache_files_ from saved file
   */
  virtual void restore_cache_files() = 0;
  ```
  Comment is misleading - refers to cache_files_ member that doesn't exist

- Implementation: pass_context_imp.cpp:578-582
  ```cpp
  void PassContextImp::restore_cache_files() {
    // No longer needed with tar_file_ - this is a no-op
    // Cache files are loaded directly via tar_file_ or mem_files_
    LOG_VERBOSE(2) << "restore_cache_files: no-op with tar_file_";
  }
  ```
  Explicitly does nothing!

- Called from: pass_context_imp.cpp:1249
  ```cpp
  restore_cache_files();  // This call does nothing
  ```

**get_cache_file_names() - Reads from tar_file_, NOT proto:**
- pass_context_imp.cpp:595-612
- Returns file list from tar_file_.entries() or mem_files_
- Does NOT read from cache_files proto field
- Provides actual list of cached files

**cache_files_to_tar_file() - Unrelated to proto field:**
- pass_context_imp.cpp:623-631
- Creates tar file from cache contents
- Does NOT read from cache_files proto field
- Uses get_cache_file_names() which reads from tar_file_.entries()

**tar_file_to_cache_files() - Unrelated to proto field:**
- pass_context_imp.cpp:633-657
- Extracts tar file into cache
- Does NOT populate cache_files proto field
- Name is about extracting "files to cache", not about proto field

### Code Locations to Remove

**1. Proto field:**
- morphizen-core/src/pass_context.proto:56
  ```protobuf
  repeated string cache_files = 12;
  ```

**2. Public API:**
- morphizen-core/include/morphizen/pass_context.hpp:266-268
  ```cpp
  virtual void restore_cache_files() = 0;
  ```

**3. Implementation:**
- morphizen-core/src/pass_context_imp.cpp:578-582 (restore_cache_files implementation)
- morphizen-core/src/pass_context_imp.cpp:715-718 (cache_files population)
- morphizen-core/src/pass_context_imp.cpp:1249 (restore_cache_files call)

**4. Declaration:**
- morphizen-core/src/pass_context_imp.hpp:320
  ```cpp
  virtual void restore_cache_files() override final;
  ```

### Why cache_files Became Dead Code

**Original design (disk-based cache):**
1. Compilation creates files in cache_dir on disk
2. cache_files proto field tracks which files were created
3. save_context_json() saves ContextProto with cache_files list
4. restore_cache_files() reads cache_files list and loads files from disk

**New design (tar_file_-based cache):**
1. Compilation creates files in tar_file_ (in-memory tar archive)
2. tar_file_ itself knows what files it contains (tar_file_.entries())
3. get_cache_file_names() directly queries tar_file_.entries()
4. No need for cache_files proto field - redundant with tar directory

**Transition left dead code:**
- cache_files proto field still written to (save_context_json adds "context.json")
- But never read from (tar_file_.entries() is used instead)
- restore_cache_files() made a no-op but not removed
- API comments still refer to obsolete cache_files_ member

### Relationship to Other Issues

**Issue #006 (cache_dir removal):**
- cache_dir is the old disk-based cache system
- cache_files was part of that system
- Both are obsolete legacy code

**Issue #005 (cache_key):**
- cache_key is legitimate metadata (namespace prefix)
- cache_files was supposed to be metadata but became dead code
- Different outcomes: keep cache_key, remove cache_files

**Issue #009 (xclbin removal):**
- Similar pattern: functions exist but are dead code (never called)
- Similar solution: remove entirely
- Both are NPU→GPU transition artifacts

### From Issue #005:**
- cache_files is field 12 in ContextProto (not field 6)
- It's alongside cache_key (which IS used - cache metadata)
- User mentioned it during cache_key discussion but deferred

**From Issue #006:**
- cache_dir is obsolete legacy disk-based cache system
- New system uses tar_file_ (tmpfile or EP context binary file)
- cache_files was part of old system, now redundant

### Questions for Investigation

1. **What is cache_files?**
   - List of filenames in the cache?
   - Metadata about cached compilation artifacts?

2. **How is it populated?**
   - During compilation passes?
   - When files are written to tar_file_?

3. **How is it used?**
   - During cache loading?
   - For validation?
   - For shared context scenarios?

4. **Relationship to tar_file_:**
   - Does cache_files list the entries in tar_file_?
   - Is it redundant with tar_file_'s internal directory?

5. **Relationship to cache_key:**
   - cache_key is namespace prefix
   - cache_files might be: `["cache_key/file1.bin", "cache_key/file2.bin"]`?

6. **Is there any bad design similar to cache_key or cache_dir?**
   - Unnecessary copying?
   - Wrong location?
   - Obsolete functionality?

### Investigation Questions (ANSWERED)

**Q1: What is cache_files?**
- ✅ Proto field tracking list of cached compilation artifacts
- ✅ Originally used to restore cache files from disk
- ✅ Now obsolete with tar_file_ system

**Q2: How is it populated?**
- ✅ Only one place: save_context_json() adds "context.json"
- ✅ No other code writes to it

**Q3: How is it used?**
- ✅ NOT USED - never read from anywhere
- ✅ Dead code

**Q4: Relationship to tar_file_:**
- ✅ Redundant - tar_file_.entries() provides actual file list
- ✅ cache_files was supposed to track tar contents but isn't used

**Q5: Relationship to cache_key:**
- ✅ Unrelated - cache_key is namespace prefix (still used)
- ✅ cache_files is legacy dead code (to be removed)

**Q6: Is there bad design?**
- ✅ YES - dead code, never read, no-op functions, misleading comments
- ✅ Similar to cache_dir (Issue #006) - obsolete legacy system

### Proto Definition (Current State)

```protobuf
// pass_context.proto:46-57
message ContextProto {
  repeated MetaDefProto meta_def = 1;
  ConfigProto config = 3;
  map<string, AnchorPointProto> origin_nodes = 4;
  map<string, int32> fix_info = 6;
  map<string, int32> device_subgraph_count = 7; // only used for fused dpu
  repeated string stacks = 8;
  repeated EventProto events = 9;
  repeated CPUUsageProto cpu_usage = 10;
  map<string, SubgraphProto> subgraph_metadefs = 11;
  repeated string cache_files = 12;  // ← TO BE REMOVED (dead code)
}
```

**After Issue #003 and #005 (ConfigProto removed, cache_key added):**
```protobuf
message ContextProto {
  repeated MetaDefProto meta_def = 1;
  // ConfigProto config = 3;  // Removed in Issue #003
  reserved 3;
  reserved "config";
  map<string, AnchorPointProto> origin_nodes = 4;
  repeated PassEventProto events = 5;
  repeated string cache_files = 12;  // ← Still here, dead code
  string cache_key = 13;              // Added in Issue #005
}
```

**After this issue (cache_files removed):**
```protobuf
message ContextProto {
  repeated MetaDefProto meta_def = 1;
  reserved 3;
  reserved "config";
  map<string, AnchorPointProto> origin_nodes = 4;
  repeated PassEventProto events = 5;
  reserved 12;
  reserved "cache_files";  // Reserve to prevent reuse
  string cache_key = 13;
}
```

### What cache_files Field Contains (Irrelevant - Never Read)

**Answer:** Only "context.json" is added to it.

In save_context_json() (pass_context_imp.cpp:715-718):
```cpp
if (std::find(proto.mutable_cache_files()->begin(),
              proto.mutable_cache_files()->end(),
              "context.json") == proto.mutable_cache_files()->end()) {
  proto.add_cache_files("context.json");
}
```

**But this is meaningless** because:
- No code reads from cache_files proto field
- get_cache_file_names() reads from tar_file_.entries() instead
- The proto field is just written to and ignored

**What SHOULD be used** (and IS used):
```cpp
auto file_names = get_cache_file_names();  // Reads from tar_file_.entries()
// This returns ALL files in tar, not from proto field
```
