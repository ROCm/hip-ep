<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Issue #003: Remove ConfigProto from ContextProto - Make Configuration Runtime-Only

## Metadata
- **Status:** BACKLOG
- **Priority:** HIGH
- **Type:** Architecture / Refactoring
- **Owner:** TBD
- **Created:** 2026-01-30
- **Updated:** 2026-01-31 (Removed encryption_key work - moved to #004)
- **Dependencies:** Issues #004, #012 (ConfigProto must be clean before removal)
- **Strategic Goal:** Immutable ConfigProto that is never persisted

## Description

Remove `ConfigProto` field from `ContextProto` entirely and make it a runtime-only member of `PassContextImp`.

**Architectural Issue:** ConfigProto is INPUT (configuration), ContextProto should only contain OUTPUT (compilation results). Mixing input with output violates semantic clarity.

**Design Goal:** ConfigProto should be runtime-only, never persisted to EP context.

**Part of strategic objective:** This issue is a major architectural step toward achieving immutable ConfigProto across the codebase.

## Problem

**Current Design (pass_context.proto:46-57):**
```protobuf
message ContextProto {
  repeated MetaDefProto meta_def = 1;  // Compilation results
  ConfigProto config = 3;               // ← INPUT, should NOT be in OUTPUT
  map<string, AnchorPointProto> origin_nodes = 4;
  repeated PassEventProto events = 5;
  repeated string cache_files = 6;
}
```

**Why This is Wrong:**

1. **Semantic Confusion: Input vs Output**
   - **ContextProto = Compilation OUTPUT** (results to be persisted to EP context)
     - MetaDefProto: execution subgraphs (RESULT)
     - events: pass execution history (RESULT)
     - cache_files: generated cache entries (RESULT)
   - **ConfigProto = Compilation INPUT** (user configuration)
     - passes: which optimizations to run
     - cache_dir, cache_key: where to cache
     - provider_options: execution settings
   - **Mixing input with output violates single responsibility principle**

2. **Violates Strategic Goal:**
   - Goal: "Immutable ConfigProto which should not be persisted"
   - Current: ConfigProto IS in ContextProto which IS persisted
   - ConfigProto should be runtime-only input, never saved to EP context

3. **Why ConfigProto Doesn't Belong in Persistent Context:**
   - EP context = compiled model state (for session reuse)
   - ConfigProto = how to compile (not part of the compiled result)
   - Persisting ConfigProto is like saving compiler flags in a .exe file
   - Session creation doesn't need original config, just the compiled MetaDefProto

4. **Causes Mutation Problems:**
   - ConfigProto persistence forces swapping (Issues #012, #013)
   - Runtime input (session_configs, provider_options) must be restored after cache load
   - These mutations exist only because ConfigProto is persisted

## Solution

Remove `ConfigProto` field from `ContextProto` and make it a runtime-only member of `PassContextImp`.

**Remove from ContextProto (pass_context.proto):**
```protobuf
message ContextProto {
  repeated MetaDefProto meta_def = 1;
  // REMOVED: ConfigProto config = 3;
  reserved 3;
  reserved "config";
  map<string, AnchorPointProto> origin_nodes = 4;
  repeated PassEventProto events = 5;
  repeated string cache_files = 6;
}
```

**Update PassContextImp (pass_context_imp.hpp):**
```cpp
class PassContextImp : public PassContext {
  ContextProto context_proto;  // Persistent OUTPUT only (MetaDefProto, events, etc.)
  ConfigProto config_;         // Runtime INPUT (never serialized)

  // Accessor
  const ConfigProto& get_config() const { return config_; }
};
```

### Implementation Steps

1. Add config_ member to PassContextImp
2. Find all access points to context_proto.config():
   ```bash
   grep -r "context_proto.*config()" morphizen-core/src/
   grep -r "context_proto.*mutable_config()" morphizen-core/src/
   ```
3. Update each access point:
   - Read access: `context_proto.config()` → `config_` or `get_config()`
   - Write access: `context_proto.mutable_config()` → `config_`
4. Update initialization logic to populate config_ directly
5. Remove config field from pass_context.proto (reserve field 3)
6. Regenerate proto C++ code
7. Update save_context_json() to serialize ContextProto (now without config)
8. Verify no compilation errors

**Result:**
- ContextProto only contains compilation results (immutable output)
- ConfigProto is runtime-only input (never persisted)
- Impossible to accidentally serialize ConfigProto

## Benefits

**Architecture:**
- ✅ **Clear Semantic Separation**: Input (ConfigProto) vs Output (ContextProto)
- ✅ **Advances Strategic Goal**: Major step toward immutable ConfigProto
- ✅ **Single Responsibility**: ContextProto only contains compilation results
- ✅ **Self-Documenting**: Structure clearly shows what gets persisted vs runtime-only

**Design Quality:**
- ✅ **Fail-Safe**: New serialization paths can't leak ConfigProto
- ✅ **Less Cognitive Load**: Don't need to track what to clear before save
- ✅ **Future-Proof**: Runtime-only data has clear home (config_)
- ✅ **Easier Testing**: Can verify ContextProto never contains config in unit tests

**Maintainability:**
- ✅ **Easier to Reason About**: ContextProto = output to persist, ConfigProto = input to use
- ✅ **Prevents Mistakes**: Can't accidentally include config in EP context
- ✅ **Cleaner Code**: No more clearing fields before serialization

**Enables Other Improvements:**
- ✅ **Eliminates Swapping**: Issues #012, #013 resolved (no cache persistence of config)
- ✅ **Clean Foundation**: ConfigProto mutations (#004, #012, #014, #015) easier to address

## Acceptance Criteria

**Prerequisites (must complete first):**
- [ ] Issue #004 completed (encryption_key removed from ConfigProto)
- [ ] Issue #012 completed (session_configs removed from ConfigProto)
- [ ] ConfigProto is clean (no fields that need swapping or special handling)

**Implementation:**
- [ ] PassContextImp has config_ member (ConfigProto)
- [ ] All config access updated to use config_ instead of context_proto.config()
- [ ] config field removed from pass_context.proto (field 3 reserved)
- [ ] save_context_json() serializes ContextProto without config
- [ ] Proto regenerated, code compiles

**Verification:**
- [ ] Unit tests verify ConfigProto never in serialized ContextProto
- [ ] All existing tests pass (no regressions)
- [ ] EP context save/load works correctly
- [ ] Cache load no longer requires swapping (Issues #012, #013 resolved)

## Implementation Plan

### Prerequisites
- Complete Issues #004, #012 (remove encryption_key, session_configs from ConfigProto)
- ConfigProto is clean and ready for architectural change

### Phase 1: Code Migration (1 day)

1. Add config_ member to PassContextImp
2. Find all access points to context_proto.config():
   ```bash
   grep -r "context_proto.*config()" morphizen-core/src/
   grep -r "context_proto.*mutable_config()" morphizen-core/src/
   ```
3. Update each access point:
   - Read access: `context_proto.config()` → `config_` or `get_config()`
   - Write access: `context_proto.mutable_config()` → `config_`
4. Update initialization logic to populate config_ directly

### Phase 2: Proto Changes (1 day)

1. Remove config field from pass_context.proto (reserve field 3)
2. Regenerate proto C++ code
3. Update save_context_json() to serialize ContextProto (now without config)
4. Verify no compilation errors

### Phase 3: Testing & Validation (1 day)

1. **Unit tests:**
   - Verify ConfigProto never appears in serialized ContextProto
   - Verify swapping no longer occurs (Issues #012, #013 resolved)

2. **Integration tests:**
   - Test EP context save (embed mode and non-embed mode)
   - Test EP context load and session creation
   - Verify loaded ContextProto has MetaDefProto but no ConfigProto

3. **Regression tests:**
   - All existing unit tests pass
   - Cache files work correctly

4. **Verification:**
   - Inspect saved context.json - should NOT contain config
   - Verify ContextProto only contains: meta_def, origin_nodes, events, cache_files

**Total Estimate: 3 days**

## Sessions

### 2026-01-30: Issue Identified and Scoped

**Initial observation:**
> "it should be in a separate runtime-only structure... if we put it in a runtime-only structure, we should not have this issue. I am right?"

**Confirmed:** YES - separating runtime config from persistent config eliminates accidental leakage risk.

**User's strategic goal:**
> "my final goal would create an immutable ConfigProto object, which should not be persisted."

**Deep dive discussion:**
- Initially focused only on encryption_key
- Through discussion, discovered broader architectural issue
- User clarified: "I think we should persist ContextProto which contains many MetaDefProto, but it should not contains ConfigProto"
- **Key insight:** ContextProto = OUTPUT (results), ConfigProto = INPUT (configuration)
- Mixing input with output violates semantic clarity

**Final scope:**
1. Remove ConfigProto from ContextProto entirely (architecture)
2. Make ConfigProto runtime-only member of PassContextImp

**Recommendation agreed:** Remove ConfigProto from ContextProto completely (fail-safe design) rather than just excluding during serialization (fragile pattern).

### 2026-01-31: Scope Clarified - Focus Only on ContextProto Removal

**Strategic clarification:**
- Issue #003 is ONE step toward final goal: Immutable ConfigProto
- #003 should focus ONLY on removing ConfigProto from ContextProto
- Other mutations (encryption_key, session_configs) addressed in separate issues

**Updated scope:**
- Remove ConfigProto field from pass_context.proto
- Make ConfigProto runtime-only member of PassContextImp
- Prerequisites: Issues #004, #012 (clean ConfigProto first)

**Dependency order:**
1. Clean provider_options (#008, #009)
2. Clean ConfigProto fields (#004, #012)
3. Architectural change (#003) ← THIS ISSUE

## Notes

### Part of Strategic Goal: Immutable ConfigProto

**This issue is a major architectural step** toward the overarching strategic goal of achieving immutable ConfigProto across the codebase.

**Strategic Vision:**
- ConfigProto is INPUT - should be runtime-only, never persisted
- ContextProto is OUTPUT - should only contain compilation results
- Clear separation makes serialization inherently safe
- ConfigProto created once, never mutated after construction

**This Issue's Role:**
- Remove ConfigProto field from ContextProto (make it runtime-only)
- Eliminates accidental persistence of configuration
- Enables other improvements (eliminates swapping in #012, #013)

**Related Issues Toward Immutability:**
- **Prerequisites:** #004 (remove encryption_key), #012 (remove session_configs)
- **Enabled by this:** #012, #013 (swapping eliminated), #005 (cache_key to ContextProto)
- **Other mutations:** #007 (target), #014 (pass registration), #015 (version info)

### What Should ContextProto Contain?

**Current (WRONG):**
```protobuf
message ContextProto {
  repeated MetaDefProto meta_def = 1;     // ✓ OUTPUT
  ConfigProto config = 3;                  // ✗ INPUT (should not be here)
  map<string, AnchorPointProto> origin_nodes = 4;  // ✓ OUTPUT
  repeated PassEventProto events = 5;     // ✓ OUTPUT
  repeated string cache_files = 6;        // ✓ OUTPUT
}
```

**After this issue (CORRECT):**
```protobuf
message ContextProto {
  repeated MetaDefProto meta_def = 1;     // Execution subgraphs
  // config REMOVED - belongs in PassContextImp, not persisted
  map<string, AnchorPointProto> origin_nodes = 4;  // Original graph nodes
  repeated PassEventProto events = 5;     // Pass execution history
  repeated string cache_files = 6;        // Generated cache entries
}
```

**Rule:** ContextProto = compilation results to persist to EP context

### Current Code Locations

**Proto definitions:**
- `config.proto:45` - encryption_key field (to be removed, moved to RuntimeConfig)
- `pass_context.proto:46-57` - ContextProto with config field 3 (to be removed)

**Encryption key usage:**
- `pass_context_imp.cpp:709` - clear_encryption_key() before save (to be removed)
- `pass_context_imp.cpp:1269-1271` - populate from provider options
- `morphizen_compile_model.cpp:444` - encryption usage (update to use runtime_config_)
- `morphizen_compile_model.cpp:899` - decryption usage (update to use runtime_config_)
- `pass_context_imp.cpp:1178-1183` - logging protection (masks key)

**ConfigProto access points (to be updated):**
- Search for: `context_proto.config()` and `context_proto.mutable_config()`
- All access points need to be updated to use `config_` member instead
- Initialization code needs to populate `config_` directly

**Serialization:**
- `pass_context_imp.cpp:706-740` - save_context_json() (will be simplified - no clearing needed)

### Security Best Practices

**Why separating secrets from config is critical:**
1. **Principle of Least Privilege** - Config widely shared, secrets restricted
2. **Defense in Depth** - Multiple serialization paths can't leak what they can't access
3. **Fail-Safe Defaults** - Missing clear() call doesn't leak secret
4. **Clear Intent** - RuntimeConfig name signals "never persist"

### Proto Field Reservations

**ConfigProto (config.proto):**
- encryption_key is field 6
- When removed: `reserved 6; reserved "encryption_key";`
- Prevents accidental reuse of field number
- Maintains wire-format compatibility

**ContextProto (pass_context.proto):**
- config is field 3
- When removed: `reserved 3; reserved "config";`
- Prevents accidental reuse of field number
- Breaking change for existing serialized EP contexts (acceptable for this refactoring)

**Migration note:** Existing saved EP context models (context.json) will need regeneration after this change, as the wire format for ContextProto changes.

### Related ConfigProto Mutation Issues

**Mutations resolved by this issue:**
- Issue #012: session_configs swapping (eliminated when ConfigProto not persisted)
- Issue #013: provider_options swapping (eliminated when ConfigProto not persisted)

**Mutations requiring separate fixes:**
- Issue #014: Dynamic pass registration (design flaw needing architectural redesign)
- Issue #015: Configuration initialization (version info belongs in ContextProto)
