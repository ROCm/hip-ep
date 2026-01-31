<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Issue #015: Configuration Initialization Mutations

## Metadata
- Status: BACKLOG
- Priority: MEDIUM
- Type: Tech Debt / Refactoring
- Created: 2026-01-31
- Updated: 2026-01-31 (Complete solution designed)
- Dependencies: Must coordinate with Issue #003 (can do before or after #003)
- Strategic Goal: Immutable ConfigProto

## Description

Move `AllVersionInfoProto` from ConfigProto to ContextProto. Version info is OUTPUT metadata (describes what compiler produced this), not INPUT configuration.

## Problem

**Current design (WRONG):**

```protobuf
message ConfigProto {
  AllVersionInfoProto version = 1;  // ← Version info in INPUT structure
  // ...
}

message ContextProto {
  repeated MetaDefProto meta_def = 1;
  ConfigProto config = 3;  // ← ConfigProto saved (includes version info)
  // ...
}
```

**Why this is wrong:**

1. **Version info is OUTPUT metadata, not INPUT configuration**
   - Version info describes "what compiler produced this result" (OUTPUT)
   - ConfigProto describes "what user wants to compile" (INPUT)
   - Version info belongs with compilation results, not user configuration

2. **Causes mutation during initialization:**
   ```cpp
   auto config_proto = ConfigProto(config_proto1);  // Copy input
   Config::add_version_info(config_proto);  // ← MUTATE ConfigProto
   ret->context_proto.mutable_config()->Swap(&config_proto);  // Move into ContextProto
   ```

3. **After Issue #003, version info will be LOST:**
   - Issue #003 removes ConfigProto from ContextProto (makes it runtime-only)
   - ConfigProto won't be saved to cache anymore
   - Version info will be lost (but needed for troubleshooting!)

**Why version info must be saved:**
- Troubleshooting requires knowing exact compiler version that produced a result
- Must be saved in cache (context.json) for reproducibility
- When user reports bug, you need to know: package name, commit ID, version numbers

## Solution

Move `AllVersionInfoProto` from ConfigProto to ContextProto.

**After this change (CORRECT):**

```protobuf
message ConfigProto {
  reserved 1;  // Was: AllVersionInfoProto version = 1;
  reserved "version";
  // ... other config fields (no version info)
}

message ContextProto {
  repeated MetaDefProto meta_def = 1;
  // ConfigProto removed in Issue #003 (field 3 reserved)
  map<string, AnchorPointProto> origin_nodes = 4;
  repeated PassEventProto events = 5;
  repeated string cache_files = 6;
  AllVersionInfoProto version = 7;  // ← NEW: Version info directly in OUTPUT
}
```

**Benefits:**
- ✅ Version info saved to cache (ContextProto is persisted)
- ✅ No mutation of ConfigProto needed
- ✅ Version info in correct location (OUTPUT metadata, not INPUT config)
- ✅ Works with or without Issue #003

### Implementation Steps

**Step 1: Add version field to ContextProto**

```protobuf
// pass_context.proto
message ContextProto {
  repeated MetaDefProto meta_def = 1;
  ConfigProto config = 3;  // Will be removed in Issue #003
  map<string, AnchorPointProto> origin_nodes = 4;
  repeated PassEventProto events = 5;
  repeated string cache_files = 6;
  AllVersionInfoProto version = 7;  // ← ADD THIS
}
```

**Step 2: Update add_version_info() to populate ContextProto**

```cpp
// config.cpp - UPDATE function signature
static void add_version_info(ContextProto& context_proto) {  // Changed parameter
  auto temp_version = context_proto.mutable_version()->add_version_infos();  // Changed
  temp_version->set_package_name(package_name);
  temp_version->set_commit(commit_id);
  // ... rest unchanged
}
```

**Step 3: Update call site in create_pass_context()**

```cpp
// pass_context_imp.cpp:100-109 - UPDATE
std::unique_ptr<PassContextImp>
PassContextImp::create_pass_context(const ConfigProto& config_proto1) {
  auto ret = std::make_unique<PassContextImp>();
  auto config_proto = ConfigProto(config_proto1);
  // REMOVED: Config::add_version_info(config_proto);  // Old - mutates ConfigProto
  ret->context_proto.mutable_config()->Swap(&config_proto);
  Config::add_version_info(ret->context_proto);  // NEW - populates ContextProto directly
  ret->update_config_proto_root_field();
  return ret;
}
```

**Step 4: Remove version field from ConfigProto**

```protobuf
// config.proto - RESERVE field 1
message ConfigProto {
  reserved 1;  // Was: AllVersionInfoProto version = 1;
  reserved "version";
  // ... other fields
}
```

**Step 5: Regenerate proto code**

```bash
# Regenerate C++ code from updated protos
cmake --build ../../build/$(basename $PWD) --target regenerate-protos
```

**Step 6: Update all version info access**

Search for and update all code that reads version info:
```cpp
// OLD (reading from ConfigProto):
context->get_config().version()

// NEW (reading from ContextProto):
context->get_context_proto().version()
```

**Step 7: Verify version info saved to cache**

After implementation:
- Compile a model
- Check context.json contains version field at top level
- Version info persisted correctly

## Evidence

- pass_context_imp.cpp:104 - add_version_info() call (mutation)
- config.cpp:63-67 - add_version_info() implementation
- pass_context_imp.cpp:106 - Config Swap (moves mutated ConfigProto into ContextProto)

## Acceptance Criteria

**Implementation:**
- [ ] `version` field added to ContextProto (field 7)
- [ ] `add_version_info()` updated to populate ContextProto.version
- [ ] Call site updated (pass context_proto, not config_proto)
- [ ] `version` field removed from ConfigProto (field 1 reserved)
- [ ] Proto code regenerated
- [ ] All version info access points updated
- [ ] All tests pass

**Verification:**
- [ ] Version info saved to cache (inspect context.json)
- [ ] Version info contains: package_name, commit, version numbers
- [ ] No mutation of ConfigProto during initialization
- [ ] Works correctly with Issue #003 (ConfigProto runtime-only)

## Notes

### Why Version Info Must Be Saved

**Purpose:** Troubleshooting and reproducibility

When user reports a bug, you need to know:
- Which package built the model? (package_name)
- Which commit was it? (commit_id)
- Which version? (version numbers)

Without this information, bug reproduction is difficult or impossible.

**Where version info is used:**
- Saved to cache (context.json) ← REQUIRED for troubleshooting
- May appear in logs
- May be included in error messages

### Version Info = OUTPUT Metadata

**What version info describes:**
- "This result was produced by compiler version X.Y.Z"
- "This compilation used commit abc123"
- "This was built from package morphizen-core"

**This is OUTPUT metadata** (describes what compiler produced this), not INPUT configuration (describes what user wants).

**Correct location:** ContextProto alongside other OUTPUT:
- MetaDefProto: execution subgraphs (OUTPUT)
- events: pass execution history (OUTPUT)
- cache_files: generated cache entries (OUTPUT)
- **version: compiler version that produced this** (OUTPUT)

### Relationship to Issue #003

**Issue #003:** Remove ConfigProto from ContextProto (make it runtime-only)

**This issue can be done before OR after Issue #003:**

**Option A: Do #015 first, then #003:**
1. Move version from ConfigProto to ContextProto
2. ConfigProto.version removed
3. Then #003 removes entire ConfigProto from ContextProto
4. Result: Version info preserved in ContextProto ✓

**Option B: Do #003 first, then #015:**
1. ConfigProto removed from ContextProto (becomes runtime-only)
2. Version info lost (currently in ConfigProto) ✗
3. Then #015 adds version back to ContextProto directly
4. Result: Version info preserved in ContextProto ✓

**Recommendation:** Do #015 first to avoid losing version info between issues.

### No ConfigProto Mutation After This Issue

**Current flow (BAD):**
```cpp
auto config_proto = ConfigProto(input);  // Copy
Config::add_version_info(config_proto);  // ← MUTATE ConfigProto
context_proto.mutable_config()->Swap(&config_proto);  // Move
```

**After this issue (GOOD):**
```cpp
auto config_proto = ConfigProto(input);  // Copy (unchanged)
context_proto.mutable_config()->Swap(&config_proto);  // Move (unchanged)
Config::add_version_info(context_proto);  // ← Populate ContextProto (no ConfigProto mutation)
```

ConfigProto never mutated - version info added directly to ContextProto.

### Proto Field Assignments

**ConfigProto.version:**
- Currently field 1
- After removal: `reserved 1; reserved "version";`

**ContextProto.version:**
- New field 7 (fields 1-6 already used)
- `AllVersionInfoProto version = 7;`
