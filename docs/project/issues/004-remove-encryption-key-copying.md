<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Issue #004: Remove encryption_key Copying in update_config_proto_root_field

## Metadata
- **Status:** BACKLOG
- **Priority:** HIGH
- **Type:** Tech Debt / Refactoring / Security
- **Owner:** TBD
- **Created:** 2026-01-30
- **Updated:** 2026-01-31 (Scope expanded - remove encryption_key field from ConfigProto)
- **Dependencies:** Issues #008, #009 (need clean provider_options first)
- **Blocks:** Issue #003 (ConfigProto must be clean before architectural change)
- **Strategic Goal:** Immutable ConfigProto

## Description

Remove `encryption_key` field from ConfigProto entirely and read from provider_options directly when needed.

**Security issue:** encryption_key is a secret that shouldn't be in ConfigProto proto field.

**Current bad design:** encryption_key is copied from provider_options into ConfigProto (lines 1269-1272 in pass_context_imp.cpp), creating unnecessary duplication of sensitive data.

**Solution:** Remove encryption_key field from config.proto and read from provider_options when needed (2 locations: encryption, decryption).

## Problem

**Current code (pass_context_imp.cpp:1269-1272):**
```cpp
if (auto encryption_key =
        get_provider_option_local({"encryption_key", "encryptionKey"})) {
  context_proto.mutable_config()->set_encryption_key(*encryption_key);  // BAD!
}
```

**Why this is wrong:**
1. **Security risk** - Sensitive data shouldn't be in proto fields (risk of accidental serialization/logging)
2. **No good reason** - encryption_key is already in provider_options, just read it when needed
3. **Used infrequently** - only encryption (line 444) and decryption (line 899)
4. **Redundant copying** - creates unnecessary duplicate of the secret
5. **Blocks immutability** - ConfigProto mutation to set encryption_key after construction

## Context

**Part of strategic goal:** Achieving immutable ConfigProto across the codebase.

**This issue is a prerequisite for #003:**
- ConfigProto should be clean before removing it from ContextProto
- encryption_key field must be removed from config.proto
- Then #003 can make ConfigProto runtime-only

**Part of larger cleanup:** `update_config_proto_root_field()` copies multiple fields from provider_options into ConfigProto:
- encryption_key (this issue) - REMOVE FIELD
- session_configs (Issue #012) - REMOVE FIELD
- cache_key (Issue #005) - MOVE to ContextProto
- cache_dir (Issue #006) - REMOVE ENTIRELY (obsolete)
- target (Issue #007) - REFACTOR (two-path architecture)

**Final goal:** Remove `update_config_proto_root_field()` entirely when all copying is eliminated.

## Solution

**Prerequisites:** Complete Issues #008, #009 (clean provider_options pollution first)

**Step 1:** Update encryption/decryption code to read from provider_options
```cpp
// OLD: Read from ConfigProto
auto encryption_key = context.get_config_proto().encryption_key();

// NEW: Read from provider_options directly
auto encryption_key = context.get_provider_option("encryption_key", "");
```

**Locations to update:**
- `morphizen_compile_model.cpp:444` - encryption usage
- `morphizen_compile_model.cpp:899` - decryption usage

**Step 2:** Remove encryption_key field from config.proto
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

**Step 3:** Remove copying logic
- Remove lines 1269-1272 from `update_config_proto_root_field()`

**Step 4:** Remove clear_encryption_key() calls
- `pass_context_imp.cpp:709` - in save_context_json()
- `pass_context_imp.cpp:142, 244` - if they exist
- Field doesn't exist anymore, no need to clear

**Step 5:** Regenerate proto C++ code and verify no regressions
- Encryption/decryption still works
- All tests pass

**No RuntimeConfig or member variable needed** - encryption_key is already in provider_options, just read it when needed (only used twice).

## Acceptance Criteria

**Prerequisites:**
- [ ] Issues #008, #009 completed (provider_options is clean)

**Implementation:**
- [ ] Encryption code updated to read from provider_options (line 444)
- [ ] Decryption code updated to read from provider_options (line 899)
- [ ] encryption_key field removed from config.proto (field 6 reserved)
- [ ] Proto C++ code regenerated
- [ ] Lines 1269-1272 removed from update_config_proto_root_field()
- [ ] clear_encryption_key() calls removed (field doesn't exist)
- [ ] No regressions in encryption/decryption functionality
- [ ] All tests pass
- [ ] No RuntimeConfig structure or member variable added

## Sessions

### 2026-01-30: Issue Identified

**Discussion context:**
- User: "there is no good reason, it was bad design, but we cannot change it because of compatibility issues. now we have a new project, we can clean it up."
- Identified as part of systematic cleanup of `update_config_proto_root_field()`
- Depends on Issue #003 completion (encryption_key won't exist in ConfigProto proto)

### 2026-01-30: Simplified Approach - No RuntimeConfig

**User question:** "do you think we need to create member variable for encryption key in PassContextImp?"

**Analysis:**
- encryption_key already in provider_options (source of truth)
- Used only twice: encryption (line 444) and decryption (line 899)
- Infrequent use, not performance-critical
- No need to cache or duplicate

**Options considered:**
- A) RuntimeConfig member variable → duplicate copy
- B) No member - read from provider_options directly → simpler

**Decision: Option B - No member variable**

**User confirmed:** "yes, it is correct. come back to your question. do you think we can totally remove the `std::string encryption_key;`"

**Answer:** YES - totally remove encryption_key field from ConfigProto proto entirely.

**Updated solution:**
- Remove encryption_key from ConfigProto proto (Issue #003)
- Remove copying logic (this issue)
- Update usage to read from provider_options directly (this issue)
- No RuntimeConfig, no member variable
- Simpler, less code, cleaner design

## Notes

### Current Code Locations

**Copying logic:**
- `pass_context_imp.cpp:1251-1276` - update_config_proto_root_field() function
- `pass_context_imp.cpp:1269-1272` - encryption_key copying (to be removed)

**Called from:**
- `pass_context_imp.cpp:107` - create_pass_context(ConfigProto)
- `pass_context_imp.cpp:120` - create_pass_context(ProviderOptions)
- `pass_context_imp.cpp:1238` - pass_context_update_context_json()

### Relationship to Issue #003

Issue #003 removes encryption_key from ConfigProto proto:
```cpp
// config.proto
message ConfigProto {
  // REMOVED: string encryption_key = 6;
  reserved 6;
  reserved "encryption_key";
}

class PassContextImp {
  ConfigProto config_;  // No encryption_key field in proto
  // No RuntimeConfig member - read from provider_options when needed
};
```

This issue removes the COPYING logic and updates usage to read from provider_options.

### Part of Systematic Cleanup

```cpp
void PassContextImp::update_config_proto_root_field() {
  // Lines 1263-1264: cache_key (Issue #005)
  // Lines 1266-1267: cache_dir (Issue #006)
  // Lines 1269-1272: encryption_key (THIS ISSUE)
  // Lines 1273-1274: target (Issue #007)
}
```

Final goal: Remove entire function when all fields cleaned up.

### Why No RuntimeConfig Needed

**encryption_key usage is infrequent:**
- Used only during EP context save (encryption) and load (decryption)
- Not in hot path
- Reading from map is cheap: `get_provider_option("encryption_key", "")`

**provider_options is the source of truth:**
- User provides encryption_key via provider_options
- Already available via `get_provider_option()`
- No reason to duplicate

**Simpler design:**
- No new structure to maintain
- No member to initialize
- No synchronization needed
- Fewer lines of code

**Contrast with other fields:**
- **cache_key:** Needs persistence → goes in ContextProto (Issue #005)
- **encryption_key:** Never persisted, used twice → read from provider_options
- **cache_dir:** Obsolete → remove entirely (Issue #006)
