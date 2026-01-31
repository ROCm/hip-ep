<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Issue #005: Move cache_key from ConfigProto to ContextProto

## Metadata
- **Status:** BACKLOG
- **Priority:** MEDIUM
- **Type:** Refactoring
- **Owner:** TBD
- **Created:** 2026-01-30
- **Updated:** 2026-01-30 (Solution designed after discussion)
- **Dependencies:** Should be done after or with Issue #003
- **Strategic Goal:** Immutable ConfigProto

## Description

Move `cache_key` from ConfigProto to ContextProto as a top-level field.

**cache_key is cache metadata, not configuration:**
- ConfigProto = INPUT (how to compile)
- cache_key = OUTPUT metadata (how cache is organized)
- It needs to be persisted with the cache, especially for shared context scenarios

**Correct location:** ContextProto (alongside cache_files)

## Problem

**Current design (wrong location):**

cache_key is currently in ConfigProto (config.proto:3), but ConfigProto will become runtime-only after Issue #003.

**Why cache_key is in the wrong place:**

### cache_key is Cache Metadata, Not Configuration

**What cache_key does:**
- Namespace prefix for files in tar archive
- Files become: `cache_key/file1.bin`, `cache_key/context.json`

**Why it needs to be persisted:**

**Shared Context Scenario** (when `ep.share_ep_contexts=1`):
- Multiple models (A1, A2) write to SAME tar file (`A1_ctx.onnx_MORPHIZEN.bin`)
- Without namespace, their files would collide
- Each model uses unique cache_key to namespace its files:
  - Model A1: `cache_key_A1/file.bin`
  - Model A2: `cache_key_A2/file.bin`

**Loading from cache:**
1. Read context.json from cache
2. Get cache_key from context.json
3. Use cache_key to find files: `cache_key/filename`

**If cache_key is NOT persisted:**
- Can't load files from cache (don't know the prefix)
- Shared context breaks (namespace information lost)

### Current Wrong Location

```cpp
// config.proto (wrong - will NOT be persisted after #003)
message ConfigProto {
  string cache_key = 3;  // ← Should NOT be here
}

// pass_context.proto (current)
message ContextProto {
  ConfigProto config = 3;  // ← ConfigProto will be removed in #003
  repeated string cache_files = 6;  // ← cache_key should be alongside this
}
```

**After Issue #003:** ConfigProto removed from ContextProto → cache_key won't be persisted → broken!

## Context

**Semantic categories:**
- **ConfigProto** (after #003): Runtime INPUT, never persisted
  - passes: which optimizations to run
  - target: target device
  - provider_options: execution settings
- **ContextProto**: Compilation OUTPUT + cache metadata, persisted
  - MetaDefProto: execution subgraphs (OUTPUT)
  - events: pass execution history (OUTPUT)
  - cache_files: list of cached files (cache metadata)
  - **cache_key: namespace prefix for files (cache metadata) ← BELONGS HERE**

**cache_key is neither pure input nor pure output** - it's cache organization metadata that needs to persist with the cache.

**Related to update_config_proto_root_field() cleanup:**
- encryption_key (Issue #004) - runtime secret, move to RuntimeConfig
- cache_key (this issue) - cache metadata, move to ContextProto
- cache_dir (Issue #006) - obsolete legacy system, remove entirely
- target (Issue #007) - unclear, needs investigation

## Solution

### Move cache_key from ConfigProto to ContextProto

**Step 1: Add to ContextProto (pass_context.proto)**
```protobuf
message ContextProto {
  repeated MetaDefProto meta_def = 1;
  // ConfigProto config = 3;  // Removed in #003
  map<string, AnchorPointProto> origin_nodes = 4;
  repeated PassEventProto events = 5;
  repeated string cache_files = 6;
  string cache_key = 7;  // NEW - cache namespace prefix
}
```

**Step 2: Remove from ConfigProto (config.proto)**
```protobuf
message ConfigProto {
  repeated PassProto passes = 1;
  string cache_dir = 2;
  // REMOVED: string cache_key = 3;
  reserved 3;
  reserved "cache_key";
  AllVersionInfoProto version = 4;
  // ...
}
```

**Step 3: Update Initialization**
```cpp
// Pass context creation - read from provider_options
void PassContextImp::initialize() {
  auto cache_key = get_provider_option("cache_key", "default_key");
  context_proto.set_cache_key(cache_key);  // Set directly in ContextProto
}
```

**Step 4: Update All Usage**
```cpp
// OLD: Read from ConfigProto
auto prefix = get_config_proto().cache_key();

// NEW: Read from ContextProto
auto prefix = context_proto.cache_key();
```

**Step 5: Remove Copying Logic**
```cpp
// DELETE lines 1263-1265 in update_config_proto_root_field()
if (auto cache_key = get_provider_option_local({"cache_key", "cacheKey"})) {
  context_proto.mutable_config()->set_cache_key(*cache_key);
}
```

**Step 6: Update Loading from Cache**
```cpp
// When loading context.json, cache_key is already in ContextProto
// No special handling needed - just use context_proto.cache_key()
```

### No Member Variable Needed

**Do NOT add cache_key_ member to PassContextImp:**
- cache_key is already in ContextProto (source of truth)
- No need to duplicate
- Read directly: `context_proto.cache_key()`

## Acceptance Criteria

- [ ] cache_key field added to ContextProto (field 7)
- [ ] cache_key removed from ConfigProto (field 3 reserved)
- [ ] Protos regenerated, code compiles
- [ ] Initialization sets context_proto.cache_key() from provider_options
- [ ] All usage updated to read from context_proto.cache_key()
- [ ] Copying logic removed from update_config_proto_root_field()
- [ ] Shared context scenarios work correctly (multiple models, same tar file)
- [ ] Cache loading works (cache_key restored from context.json)
- [ ] No separate cache_key_ member variable added

## Sessions

### 2026-01-30: Initial Creation

**User guidance:**
> "we can come back to `cache_key`, `cache_dir` and `target` later. I don't have a clear plan. I need you help to write them down and clean my mind, then we have better picture and then we have a better plan."

**Initial scope:** Just remove copying in update_config_proto_root_field().

### 2026-01-30: Deep Discussion and Solution Design

**User:** "let's focus on issue 005, again, let's discuss"

**Key questions explored:**

**Q1: What problem does cache_key solve?**
- A: Namespacing inside tar archive for shared context
- Files become: `cache_key/file1.bin`, `cache_key/file2.bin`

**Q2: Why namespace?**
- A: When `ep.share_ep_contexts=1`, multiple models write to SAME tar file
- Without namespace, files collide
- Each model uses unique cache_key

**Example from docs/technical/ep_shard_context.md:**
```
A1_ctx.onnx_MORPHIZEN.bin (shared tar file)
├─ cache_key_A1/file1.bin
├─ cache_key_A1/context.json
├─ cache_key_A2/file1.bin
└─ cache_key_A2/context.json
```

**Critical realization:**
> "cache_key is NOT just input configuration - it's OUTPUT metadata describing how the cache is organized!"

**Q3: After Issue #003 (ConfigProto removed from ContextProto), where should cache_key go?**

**Options considered:**
- A) Keep in ConfigProto (but then not persisted - broken!)
- B) Move to ContextProto directly
- C) Create new CacheMetadata message
- D) Something else

**Decision: Option B - Move to ContextProto**

**Reasoning:**
- ContextProto already has `cache_files` (list of files)
- Natural to have `cache_key` (namespace prefix) alongside it
- cache_key is cache metadata, belongs with other cache metadata
- Self-contained: ContextProto describes what was compiled AND how cache is organized

**Q4: Do we need a member variable for PassContextImp?**

**User confirmed:** "yes, I like your solution"

**Decision: No member variable needed**
- cache_key IS in ContextProto (source of truth)
- Read directly: `context_proto.cache_key()`
- Avoid duplication, keep it simple

**Final understanding:**
- cache_key = cache organization metadata (neither pure input nor pure output)
- Needs to persist with the cache
- Belongs in ContextProto alongside cache_files
- No separate member variable

## Related Issues

- **Issue #003:** Remove ConfigProto from ContextProto - this issue should be coordinated with #003
- **Issue #006:** Remove cache_dir entirely - cache_files is another issue to address later

## Notes

### Shared Context Use Case

**Scenario:** Multiple models share weights via shared EP context binary

```python
# Model A1 and A2 share A1_ctx.onnx_MORPHIZEN.bin
options.add_session_config_entry("ep.share_ep_contexts", "1")
create_session("A1.onnx", options)  # Creates tar with cache_key_A1/*
create_session("A2.onnx", options)  # Appends to tar with cache_key_A2/*
options.add_session_config_entry("ep.stop_share_ep_contexts", "1")
```

**Resulting tar structure:**
```
A1_ctx.onnx_MORPHIZEN.bin
├─ cache_key_A1/
│  ├─ file1.bin
│  └─ context.json
└─ cache_key_A2/
   ├─ file1.bin
   └─ context.json
```

**Code enforces prefix for shared context (pass_context_imp.cpp:1087-1094):**
```cpp
if (is_shared_context_enabled) {
    // for shared ep context, we must enable file prefix.
    cache_file_use_cache_key_prefix_ = true;
}
```

### Current Code Locations

**Proto definitions:**
- `config.proto:3` - cache_key field (to be removed, reserve field 3)
- `pass_context.proto` - add cache_key as field 7

**Copying logic:**
- `pass_context_imp.cpp:1263-1265` - cache_key copying (to be removed)

**Usage locations (need to update):**
- `pass_context_imp.cpp:509` - Builds filename with prefix: `prefix + "/" + filename1`
- `cache_dir.cpp:66` - Uses cache_key in path construction (but cache_dir is being removed in #006)
- `pass_context_imp.cpp:1163` - Checks for `prefix + "/context.json"` in tar
- All code reading `get_config_proto().cache_key()` → change to `context_proto.cache_key()`

### Important: cache_files is Another Issue

**User note:** "cache_files is another issue, we should put it aside, but don't forget it."

`cache_files` field in ContextProto (field 6) likely has similar issues to cache_key and needs investigation. Defer to future issue.

### Coordination with Issue #003

**Timing consideration:**
- Issue #003 removes ConfigProto from ContextProto
- This issue moves cache_key from ConfigProto to ContextProto
- These can be done together in same refactoring, or sequentially

**If done sequentially:**
1. Do #003 first (remove ConfigProto from ContextProto, move to PassContextImp member)
2. Then do #005 (move cache_key from ConfigProto to ContextProto)

**If done together:**
- More efficient (one proto change)
- Single migration path
