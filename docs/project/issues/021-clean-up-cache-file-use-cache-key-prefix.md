<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Issue #021: Clean Up cache_file_use_cache_key_prefix_ Implementation

## Metadata
- **Status:** BACKLOG
- **Priority:** MEDIUM
- **Type:** Tech Debt / Refactoring
- **Owner:** TBD
- **Created:** 2026-02-01
- **Updated:** 2026-02-01 (Simplified: Remove flag entirely, always use prefix)
- **Dependencies:** None
- **Related:** Issue #002 (mem_files_ removal), Issue #005 (cache_key), Issue #006 (cache_dir cleanup)

## Description

**Remove `cache_file_use_cache_key_prefix_` entirely and always use cache_key prefix.**

This flag adds unnecessary complexity (~20-25 LOC) for a feature that should always be enabled. Analysis shows:
- Prefix is TRUE by default (99% of cases)
- Required for shared EP context
- cache_key always exists (set in initialize_context)
- **No backwards compatibility needed** - new project ported from NPU EP, no existing deployed EP context models

**Solution:** Replace conditional logic with simple deterministic behavior: `cache_key + "/" + filename`.

**Benefits:**
- ✅ Remove ~20-25 LOC
- ✅ Simpler code, no conditionals
- ✅ Consistent tar structure
- ✅ Fewer validation points
- ✅ One code path instead of two

## Problem

**What is cache_file_use_cache_key_prefix_?**

A boolean flag that controls whether cache files are prefixed with `cache_key` in tar archives:
- **When true:** Files stored as `cache_key/file.bin` (namespaced)
- **When false:** Files stored as `file.bin` (flat)

**Declaration (pass_context_imp.hpp:384):**
```cpp
bool cache_file_use_cache_key_prefix_ = false;
```

**Why this flag is unnecessary:**

1. **Always TRUE in practice:**
   ```cpp
   // pass_context_imp.cpp:1087-1095
   if (is_shared_context_enabled) {
     cache_file_use_cache_key_prefix_ = true;  // REQUIRED for shared context
   } else {
     cache_file_use_cache_key_prefix_ =
         get_provider_option("use_cache_key_prefix", "1") == "1";  // DEFAULTS to "1"!
   }
   ```
   Only false if user explicitly sets undocumented `use_cache_key_prefix=0` option.

2. **cache_key always exists** - set by initialize_context from 4 code paths (never empty)

   **Analysis of initialize_context() (morphizen_compile_model.cpp:387-409):**

   ```cpp
   // Path 1: User provided via ConfigProto
   if (!context->context_proto.config().cache_key().empty()) {
     // Already set by user - use as-is (non-empty by definition)
   }
   // Path 2: Model metadata (embedded in ONNX model)
   else if (model.has_metadata("morphizen_model_md5sum")) {
     auto cache_key = model.get_metadata("morphizen_model_md5sum");
     *context->context_proto.mutable_config()->mutable_cache_key() = cache_key;
   }
   // Path 3: File-based MD5 (hash of model file on disk)
   else if (ENV_PARAM(XLNX_ENABLE_FILE_BASED_CACHE_KEY) && !model_path.empty()) {
     auto cache_key = get_md5_of_file(context->model_path.string());
     *context->context_proto.mutable_config()->mutable_cache_key() = cache_key;
   }
   // Path 4: Memory signature (DEFAULT FALLBACK - ALWAYS EXECUTES)
   else {
     auto new_cache_key = md5;  // From get_model_signature() - MD5 of graph structure
     *context->context_proto.mutable_config()->mutable_cache_key() = new_cache_key;
   }
   ```

   **Why cache_key is NEVER empty:**
   - **Path 4 is guaranteed**: The else clause ALWAYS executes if paths 1-3 don't apply
   - **MD5 hash is never empty**: `get_model_signature()` (lines 310-338) returns MD5 hash:
     ```cpp
     static std::string get_model_signature(const Graph& onnx_graph) {
       auto md5 = MD5Sig(".data");
       // ... loop through nodes, add to hash ...
       return md5.getHash();  // Always returns 32 hex chars, never ""
     }
     ```
   - **MD5.getHash() always returns 32 characters**: MD5 hash is always a 32-character hexadecimal string
   - **Result**: cache_key is ALWAYS set to a non-empty value by the time initialize_context() completes

3. **Adds complexity with no benefit:**
   - Conditional logic in 3 file operations (read/write/has)
   - Flag setting in 4 locations
   - Two tar file formats to maintain
   - More validation complexity
   - More code paths = more bugs

4. **No backwards compatibility needed:**
   - New project ported from NPU EP
   - No existing deployed EP context models to support
   - Can enforce consistent structure from day 1

**Current problems with the flag:**

1. **Unnecessary complexity** - Two code paths (prefix / flat) when only one is needed
2. **Conditional logic** - 3 file operations have `if (cache_file_use_cache_key_prefix_)` checks
3. **Flag setting overhead** - 4 locations set the flag
4. **More validation needed** - Must validate both flag state AND cache_key
5. **Undocumented provider option** - `use_cache_key_prefix` is internal only
6. **Code maintenance burden** - More code paths = more potential bugs

## Solution

**Remove the flag entirely and always use prefix.**

### Step 1: Add Helper Function

```cpp
// pass_context_imp.cpp or pass_context_imp.hpp
std::string PassContextImp::get_cache_filename(const std::string& filename) const {
  auto cache_key = get_config_proto().cache_key();
  CHECK(!cache_key.empty()) << "cache_key required for cache file operations";
  return cache_key + "/" + filename;
}
```

### Step 2: Replace All Conditional Logic

**Before (pass_context_imp.cpp:491):**
```cpp
auto prefix = get_config_proto().cache_key();
auto filename = cache_file_use_cache_key_prefix_ ? prefix + "/" + filename1 : filename1;
auto stream = tar_file_->open_for_read(filename);
```

**After:**
```cpp
auto stream = tar_file_->open_for_read(get_cache_filename(filename1));
```

**Apply to:**
- pass_context_imp.cpp:491 - open_file_for_read_with_tar_file
- pass_context_imp.cpp:509 - open_file_for_write_with_tar_file
- pass_context_imp.cpp:587 - has_cache_file
- pass_context_imp.cpp:1163 - create_tar_file_for_prebuild_cache

### Step 3: Remove Flag Setting Logic

**Delete entirely:**
```cpp
// pass_context_imp.cpp:1087-1095
if (is_shared_context_enabled) {
  cache_file_use_cache_key_prefix_ = true;
} else {
  cache_file_use_cache_key_prefix_ =
      get_provider_option("use_cache_key_prefix", "1") == "1";
}
```

**Delete:**
- pass_context_imp.cpp:1160 - `cache_file_use_cache_key_prefix_ = false;`
- pass_context_imp.cpp:1162 - `cache_file_use_cache_key_prefix_ = true;`

### Step 4: Remove EP Context Attribute

**Delete from create_ep_context_node (morphizen_compile_model.cpp:642-643):**
```cpp
// DELETE these lines:
attrs.add("cache_file_use_cache_key_prefix",
          (int64_t)context.cache_file_use_cache_key_prefix_);
```

**Simplify store_cache_directory_from_main_node (morphizen_compile_model.cpp:845-855):**
```cpp
// DELETE:
context.cache_file_use_cache_key_prefix_ =
    main_node.get_attr_int("cache_file_use_cache_key_prefix", 0) != 0;
if (context.cache_file_use_cache_key_prefix_) {
  CHECK_NE(context.get_context_proto().config().cache_key(), "") << ...;
}

// REPLACE WITH (always validate):
auto loaded_cache_key = main_node.get_attr_string("cache_file_prefix", "");
CHECK(!loaded_cache_key.empty())
    << "EP context node must have non-empty cache_file_prefix attribute";
*context.get_context_proto().mutable_config()->mutable_cache_key() = loaded_cache_key;
```

### Step 5: Remove Member Variable and Comments

**Delete from pass_context_imp.hpp:**
```cpp
// DELETE lines 379-383 (comment)
// DELETE line 384 (member variable)
bool cache_file_use_cache_key_prefix_ = false;
```

### Step 6: Update Validation

**Add to initialize_context() (morphizen_compile_model.cpp:409):**
```cpp
// After all cache_key initialization paths
CHECK(!context->context_proto.config().cache_key().empty())
    << "cache_key must be set during initialization";
```

**Fix error message (pass_context_imp.cpp:1076) if it still exists:**
```cpp
CHECK(!get_config_proto().cache_key().empty())
    << "cache_key required for tar file operations";
```

### Result: Simplified Code

**All tar files use consistent structure:**
```
A_ctx.onnx_MORPHIZEN.bin
└─ cache_key_A1/
   ├─ context.json
   ├─ weights.bin
   └─ ...
```

**Benefits:**
- ✅ Remove ~20-25 LOC
- ✅ No conditional logic in file operations
- ✅ Simpler validation (only cache_key, not flag + cache_key)
- ✅ One code path instead of two
- ✅ Consistent tar structure for all models
- ✅ Fewer potential bugs
- ✅ Easier to understand and maintain

## Evidence

**Flag is always TRUE by default:**
- pass_context_imp.cpp:1093-1094 - Defaults to `get_provider_option("use_cache_key_prefix", "1") == "1"`
- Only false if user sets undocumented option to "0"

**Flag declaration and comment:**
- pass_context_imp.hpp:379-383 - Comment (to be removed)
- pass_context_imp.hpp:384 - Member variable (to be removed)

**Conditional logic using the flag (to be simplified):**
- pass_context_imp.cpp:493 - `cache_file_use_cache_key_prefix_ ? prefix + "/" + filename1 : filename1`
- pass_context_imp.cpp:511 - `cache_file_use_cache_key_prefix_ ? prefix + "/" + filename1 : filename1`
- pass_context_imp.cpp:586 - `if (cache_file_use_cache_key_prefix_)`

**Flag setting locations (to be removed):**
- pass_context_imp.cpp:1087-1095 - Sets based on shared context + provider option
- pass_context_imp.cpp:1160 - Sets to false
- pass_context_imp.cpp:1162 - Sets to true
- morphizen_compile_model.cpp:845-846 - Reads from EP context node

**EP context attribute (to be removed):**
- morphizen_compile_model.cpp:642-643 - Saves flag as node attribute
- morphizen_compile_model.cpp:845 - Loads flag from node attribute

**Feature documentation:**
- docs/technical/ep_shard_context.md - Shared EP context examples

## Context

### Why This Feature Exists

**Shared EP Context Scenario:**
- Multiple models share a single tar archive as prebuild cache
- Each model needs isolated namespace for cache files
- cache_key prefix prevents file collisions (e.g., "model_A/file.bin" vs "model_B/file.bin")

**How It Works:**
1. Flag is enabled when creating tar_file_ for shared context
2. File operations (read/write/has) prefix paths with `cache_key + "/"`
3. Tar archive contains directory structure: `cache_key_A1/context.json`, `cache_key_A1/weights.bin`, etc.

### Impact of the Bug (RESOLVED BY ISSUE #002)

**Scenario where bug manifests:**
1. Shared EP context enabled (cache_file_use_cache_key_prefix_ = true)
2. tar_file_ is null (using mem_files_ for some reason)
3. Code calls has_cache_file("weights.bin")
4. Expected: Check for "cache_key_A1/weights.bin" in mem_files_
5. Actual: Check for "weights.bin" in mem_files_ (unprefixed)
6. Result: File lookup failure, incorrect behavior

**Resolution:**
- Issue #002 removes mem_files_ entirely (~200 LOC)
- After #002, tar_file_ will always exist (sandbox fallback via MemStream)
- The mem_files_ code path (and the bug) will be removed
- This issue becomes documentation cleanup only

### Relationship to Other Issues

**Related to Issue #002 (mem_files_ removal):**
- Issue #002: Remove mem_files_ by always creating tar_file_
- Issue #021: Logic bug exists in mem_files_ code path
- **Resolution:** After #002 completes, the buggy code path is removed
- Issue #021 becomes documentation cleanup (grammar, comments, validation)

**Related to Issue #005 (cache_key):**
- Issue #005: Move cache_key from ConfigProto to ContextProto
- Issue #021: cache_key is used as prefix when this flag is enabled
- Coordinate: If #005 moves cache_key, update prefix computation here

**Related to Issue #006 (cache_dir cleanup):**
- Both involve cache system cleanup
- Both improve cache-related code clarity
- Independent implementations

## Acceptance Criteria

**Implementation:**
- [ ] Add `get_cache_filename()` helper function
- [ ] Replace conditional logic in open_file_for_read_with_tar_file (line 491)
- [ ] Replace conditional logic in open_file_for_write_with_tar_file (line 509)
- [ ] Replace conditional logic in has_cache_file (line 587)
- [ ] Replace conditional logic in create_tar_file_for_prebuild_cache (line 1163)
- [ ] Remove flag setting logic (lines 1087-1095, 1160, 1162)
- [ ] Remove EP context attribute save (lines 642-643)
- [ ] Simplify EP context attribute load (lines 845-855, just validate cache_key)
- [ ] Remove member variable declaration (line 384)
- [ ] Remove comment (lines 379-383)
- [ ] Add final validation after initialize_context (line 409)
- [ ] All tests pass

**Code reduction:**
- [ ] Verify ~20-25 LOC removed
- [ ] No conditional `if (cache_file_use_cache_key_prefix_)` checks remain
- [ ] No references to `use_cache_key_prefix` provider option

**Verification:**
- [ ] All tar files use prefixed structure: `cache_key/filename`
- [ ] Shared EP context works correctly
- [ ] Non-shared EP context works correctly (still uses prefix)
- [ ] EP context loading validates cache_key is non-empty
- [ ] Clear error message when cache_key is empty
- [ ] No regressions in existing functionality

**Testing:**
- [ ] Test shared EP context (multiple models, same tar file)
- [ ] Test non-shared EP context (single model, prefixed structure)
- [ ] Test EP context loading with valid cache_file_prefix
- [ ] Test EP context loading with empty cache_file_prefix (should fail)
- [ ] Verify tar file structure: `cache_key/context.json`, `cache_key/weights.bin`, etc.
- [ ] All unit tests pass
- [ ] No regressions

## Notes

### Scope Simplified After Analysis

**Original scope:**
- Fix validation bugs with flag enabled
- Fix documentation
- Fix grammar errors

**Final simplified scope:**
- **Remove the flag entirely**
- Always use prefix (simpler, cleaner, no conditionals)
- Add helper function `get_cache_filename()`
- Validate cache_key is non-empty

**Why this is better:**
- Removes ~20-25 LOC instead of adding validation
- Eliminates all conditional logic
- Simpler to understand and maintain
- Consistent behavior for all models
- Fewer code paths = fewer bugs

### Implementation Effort

**Complexity:** LOW to MEDIUM
- Add 1 helper function
- Replace 4 conditional usages
- Remove 4 flag setting locations
- Remove 1 member variable
- Remove 1 comment block
- Remove 2 EP context attribute operations
- Add 1 validation CHECK

**Risk:** LOW
- Deterministic behavior (always prefix)
- cache_key always exists (set in initialize_context)
- No backwards compatibility needed (new project)
- Tests will catch any issues

**Effort:** 2-3 hours
- Write helper function: 15 minutes
- Replace conditional logic (4 locations): 30 minutes
- Remove flag setting (4 locations): 15 minutes
- Remove member variable + comment: 5 minutes
- Update EP context attribute handling: 30 minutes
- Test all scenarios: 1 hour
- Verify no regressions: 30 minutes

**Code reduction:** ~20-25 LOC removed, ~10 LOC added → **Net -15 LOC**

### Why Always Prefix is Safe

**cache_key always exists:**
```cpp
// initialize_context() sets cache_key from one of 4 paths:
if (!context->context_proto.config().cache_key().empty()) {
  // User provided
} else if (model.has_metadata("morphizen_model_md5sum")) {
  // Model metadata
} else if (ENV_PARAM(XLNX_ENABLE_FILE_BASED_CACHE_KEY) && !model_path.empty()) {
  // File-based MD5
} else {
  // Memory signature (MD5 of model graph)
  auto new_cache_key = md5;  // Never empty
}
```

**All scenarios work:**
- Shared EP context: Requires prefix ✓
- Non-shared EP context: Works with prefix ✓
- No downsides to always using prefix ✓

**No backwards compat needed:**
- New project ported from NPU EP
- No existing deployed EP context models
- Can enforce structure from day 1 ✓
