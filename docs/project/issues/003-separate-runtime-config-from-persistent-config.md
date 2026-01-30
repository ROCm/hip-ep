<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Issue #003: Remove ConfigProto from ContextProto - Make Configuration Runtime-Only

## Metadata
- **Status:** BACKLOG
- **Priority:** HIGH
- **Type:** Architecture / Security / Refactoring
- **Owner:** TBD
- **Created:** 2026-01-30
- **Strategic Goal:** Immutable ConfigProto that is never persisted

## Description

Remove `ConfigProto` from `ContextProto` entirely and make it a runtime-only member of `PassContextImp`. Also remove `encryption_key` field from ConfigProto - read it directly from provider_options when needed.

**Architectural Issue:** ConfigProto is INPUT (configuration), ContextProto should only contain OUTPUT (compilation results). Mixing input with output violates semantic clarity.

**Security Issue:** Having encryption_key in ConfigProto creates risk of accidental leakage, even though it's never actually needed there - just read from provider_options when encrypting/decrypting.

**Design Goal:** ConfigProto should be immutable and never persisted to EP context.

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

2. **Security Risk: encryption_key Leakage**
   - ConfigProto contains `encryption_key` (field 6)
   - Currently protected by fragile pattern: `clear_encryption_key()` before serialization
   - Requires remembering to clear it (pass_context_imp.cpp:709)
   - Easy to forget in new serialization paths
   - Proto dumps/debugging could leak key

3. **Current Mitigation (Fragile):**
   ```cpp
   // pass_context_imp.cpp:706-740
   void PassContextImp::save_context_json() const {
     ContextProto proto;
     proto.CopyFrom(this->context_proto);
     proto.mutable_config()->clear_encryption_key();  // Must remember this!
     // ... serialize entire ContextProto including config ...
   }
   ```

4. **Violates User's Strategic Goal:**
   - Goal: "Immutable ConfigProto which should not be persisted"
   - Current: ConfigProto IS in ContextProto which IS persisted
   - ConfigProto should be runtime-only input, never saved to EP context

5. **Why ConfigProto Doesn't Belong in Persistent Context:**
   - EP context = compiled model state (for session reuse)
   - ConfigProto = how to compile (not part of the compiled result)
   - Persisting ConfigProto is like saving compiler flags in a .exe file
   - Session creation doesn't need original config, just the compiled MetaDefProto

## Solution

**Two-part refactoring:**
1. Remove `encryption_key` from ConfigProto (read from provider_options instead)
2. Remove `ConfigProto` from `ContextProto` (make it runtime-only)

### Part 1: Remove encryption_key from ConfigProto

**No RuntimeConfig structure needed** - encryption_key is already in provider_options, just read it directly when needed.

**Remove from ConfigProto (config.proto):**
```protobuf
message ConfigProto {
  repeated PassProto passes = 1;
  string cache_dir = 2;
  string cache_key = 3;
  AllVersionInfoProto version = 4;
  // REMOVED: string encryption_key = 6;
  reserved 6;
  reserved "encryption_key";
  map<string, string> provider_options = 8;
  // ...
}
```

**Read directly from provider_options when needed:**
```cpp
// During encryption (morphizen_compile_model.cpp:444)
auto encryption_key = context.get_provider_option("encryption_key", "");
if (!encryption_key.empty()) {
  encrypt_data(data, encryption_key);
}

// During decryption (morphizen_compile_model.cpp:899)
auto encryption_key = context.get_provider_option("encryption_key", "");
if (!encryption_key.empty()) {
  decrypt_data(data, encryption_key);
}
```

### Part 2: Remove ConfigProto from ContextProto

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
  ConfigProto config_;         // Runtime INPUT (never serialized, no encryption_key field)

  // Accessor
  const ConfigProto& get_config() const { return config_; }

  // No encryption_key member - read from provider_options when needed
};
```

### Migration Path

**Phase 1: Remove encryption_key from ConfigProto**
1. Update encryption/decryption code to read from provider_options directly:
   - `morphizen_compile_model.cpp:444` - encryption
   - `morphizen_compile_model.cpp:899` - decryption
2. Remove encryption_key field from config.proto (reserve field 6)
3. Regenerate proto C++ code
4. Remove clear_encryption_key() calls (no longer needed - field doesn't exist)

**Phase 2: Move ConfigProto Out of ContextProto**
1. Add config_ member to PassContextImp
2. Update all `context_proto.config()` calls to `get_config()` or `config_`
3. Update initialization to populate config_ (not context_proto.config)
4. Remove config field from pass_context.proto (reserve field 3)
5. Update save_context_json() - serializes ContextProto without config

**Result:**
- ContextProto only contains compilation results (immutable output)
- ConfigProto is runtime-only input (never persisted, no encryption_key field)
- encryption_key read from provider_options when needed (used only twice)
- Impossible to accidentally serialize config or encryption_key

## Benefits

**Architecture:**
- ✅ **Clear Semantic Separation**: Input (ConfigProto) vs Output (ContextProto)
- ✅ **Achieves Strategic Goal**: Immutable ConfigProto that is never persisted
- ✅ **Single Responsibility**: ContextProto only contains compilation results
- ✅ **Self-Documenting**: Structure clearly shows what gets persisted vs runtime-only

**Security:**
- ✅ **Impossible to leak encryption_key**: Not in any persistable proto
- ✅ **Impossible to leak ConfigProto**: Not in ContextProto
- ✅ **No fragile clear() calls needed**: Can't serialize what doesn't exist
- ✅ **Proto dumps/logs automatically safe**: No secrets in persistable structures

**Design Quality:**
- ✅ **Fail-Safe**: New serialization paths can't leak config or secrets
- ✅ **Less Cognitive Load**: Don't need to track what to clear before save
- ✅ **Future-Proof**: Runtime-only data has clear home (config_, runtime_config_)
- ✅ **Easier Testing**: Can verify ContextProto never contains config in unit tests

**Maintainability:**
- ✅ **Easier to Reason About**: ContextProto = output to persist, ConfigProto = input to use
- ✅ **Prevents Mistakes**: Can't accidentally include config in EP context
- ✅ **Cleaner Code**: No more clearing fields before serialization

## Acceptance Criteria

**Part 1: Remove encryption_key from ConfigProto**
- [ ] Encryption code reads from provider_options (morphizen_compile_model.cpp:444)
- [ ] Decryption code reads from provider_options (morphizen_compile_model.cpp:899)
- [ ] encryption_key removed from config.proto (field 6 reserved)
- [ ] Proto regenerated, code compiles
- [ ] clear_encryption_key() calls removed (lines 709, 142, 244)
- [ ] No regressions in encryption/decryption functionality

**Part 2: ConfigProto Out of ContextProto**
- [ ] PassContextImp has config_ member (ConfigProto, no encryption_key field)
- [ ] All config access updated to use config_ instead of context_proto.config()
- [ ] config field removed from pass_context.proto (field 3 reserved)
- [ ] save_context_json() serializes ContextProto without config
- [ ] Proto regenerated, code compiles

**Verification**
- [ ] Unit tests verify encryption_key never serialized
- [ ] Unit tests verify ConfigProto never in serialized ContextProto
- [ ] All existing tests pass (no regressions)
- [ ] EP context save/load works correctly with encryption

## Implementation Plan

### Phase 1: Remove encryption_key from ConfigProto (1 day)

1. Update encryption code to read from provider_options:
   - `morphizen_compile_model.cpp:444` - change to `context.get_provider_option("encryption_key", "")`
   - `morphizen_compile_model.cpp:899` - change to `context.get_provider_option("encryption_key", "")`
2. Remove encryption_key from config.proto (reserve field 6)
3. Regenerate proto C++ code
4. Remove clear_encryption_key() calls:
   - `pass_context_imp.cpp:709` - in save_context_json()
   - `pass_context_imp.cpp:142` - if exists
   - `pass_context_imp.cpp:244` - if exists
5. Test: encryption/decryption still works

### Phase 2: Move ConfigProto Out of ContextProto (2 days)

**Day 1: Code Migration**
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

**Day 2: Proto Changes**
1. Remove config field from pass_context.proto (reserve field 3)
2. Regenerate proto C++ code
3. Update save_context_json() to serialize ContextProto (now without config)
4. Remove clear_encryption_key() calls (lines 709, 142, 244)
5. Verify no compilation errors

### Phase 3: Testing & Validation (1 day)

1. **Unit tests:**
   - Verify encryption_key never appears in serialized ContextProto
   - Verify ConfigProto never appears in serialized ContextProto
   - Test encryption/decryption with runtime_config_

2. **Integration tests:**
   - Test EP context save (embed mode and non-embed mode)
   - Test EP context load and session creation
   - Verify loaded ContextProto has MetaDefProto but no ConfigProto

3. **Regression tests:**
   - All existing unit tests pass
   - Encrypted EP context models work correctly
   - Cache files work correctly

4. **Verification:**
   - Inspect saved context.json - should NOT contain config or encryption_key
   - Verify ContextProto only contains: meta_def, origin_nodes, events, cache_files

**Total Estimate: 4 days**

## Plans

_No plans yet._

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
1. Move encryption_key to RuntimeConfig (security)
2. Remove ConfigProto from ContextProto entirely (architecture)
3. Make ConfigProto runtime-only member of PassContextImp

**Recommendation agreed:** Remove ConfigProto from ContextProto completely (fail-safe design) rather than just excluding during serialization (fragile pattern).

### 2026-01-30: Simplified - No RuntimeConfig Needed

**While discussing Issue #004:**

**User question:** "do you think we need to create member variable for encryption key in PassContextImp?"

**Analysis:**
- encryption_key is already in provider_options (source of truth)
- Used only twice: encryption (line 444) and decryption (line 899)
- Not performance-critical
- No need to cache or duplicate

**Decision: No RuntimeConfig structure needed**
- Just remove encryption_key from ConfigProto proto (reserve field 6)
- Read directly from provider_options when needed
- Simpler solution, less code, no new structure

**User confirmed:** "yes, it is correct. come back to your question. do you think we can totally remove the `std::string encryption_key;`"

**Answer:** YES - totally remove encryption_key field from ConfigProto proto.

**Updated approach:**
- Issue #003: Remove encryption_key from proto + remove ConfigProto from ContextProto
- Issue #004: Remove copying logic + update usage to read from provider_options
- No RuntimeConfig, no member variable, just read when needed

## Related PRs

_None yet._

## Notes

### Part of Larger Refactoring Goal

**User's Strategic Vision:** Immutable ConfigProto that is never persisted
- ConfigProto is INPUT - should be runtime-only, never saved
- ContextProto is OUTPUT - should only contain compilation results
- Clear separation makes serialization inherently safe
- Prevents accidental leakage of configuration or secrets

**This Issue: Complete Implementation**
- Remove encryption_key from ConfigProto proto (read from provider_options instead)
- Move ConfigProto out of ContextProto (make it runtime-only member)
- ContextProto becomes pure output container (MetaDefProto, events, etc.)
- Achieves the full strategic goal with simplest approach (no RuntimeConfig needed)

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
