<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Issue #012: session_configs Swapping After Cache Load

## Metadata
- Status: BACKLOG
- Priority: HIGH
- Type: Tech Debt / Refactoring
- Updated: 2026-01-31 (Scope expanded - remove session_configs field from ConfigProto)
- Dependencies: Issues #008, #009 (need clean provider_options first)
- Blocks: Issue #003 (ConfigProto must be clean before architectural change)
- Strategic Goal: Immutable ConfigProto

## Description

Remove `session_configs` field from ConfigProto and pass session configs as separate parameter to PassContext.

**Key insight:** There are TWO distinct mechanisms for session config handling:
1. **ORT Auto EP Selection** (ACTIVE) - Provider options passed via session config with `ep.morphizenexecutionprovider.*` prefix
2. **Legacy session config access** (CAN BE REMOVED) - Workaround when EP couldn't access SessionOptions directly

**Solution:**
- Extract session configs at ort-bridge layer using modern `GetSessionConfigEntry()` API
- Pass as separate parameter to `initialize_context()` and `create_pass_context()` (similar to how provider_options is passed)
- Remove session_configs field from ConfigProto
- Remove legacy workaround (morphizen-ep.cpp:118-120)
- Clean separation: no mixing of provider_options and session_configs

**See:** `docs/technical/provider-options-session-configs-mixing.md` for detailed explanation.

## Problem

**Two problems:**

1. **Copying into ConfigProto (lines 1207):**
```cpp
for (auto& kv : config.session_configs()) {
  LOG_VERBOSE(3) << "session_config: " << kv.first << " = " << kv.second;
}
```

2. **Swapping after cache load (lines 1234-1235, 247-248):**
```cpp
auto session_configs = context->config().session_configs(); // Save user's
read_cache(context);  // Overwrites ConfigProto with cached version
context->config()->set_session_configs(session_configs);  // Restore user's
```

**Why this is wrong:**
- session_configs = SESSION configuration (session-level settings: ep.context_enable, etc.)
- ConfigProto = COMPILATION configuration (how to compile: passes, target, optimizations)
- Semantic separation of concerns violated
- Swapping exists to preserve session settings when loading cached compilation config
- Blocks ConfigProto immutability (mutation after construction)

**Design flaw:** ConfigProto mixes two different concerns (compilation vs session configuration).

## Solution

**Prerequisites:** Complete Issues #008, #009 (clean provider_options pollution first)

### Architecture: Separate Parameters

**Follow the existing pattern for provider_options** (morphizen_compile_model.cpp:362-369):

```cpp
// CURRENT API
std::shared_ptr<PassContextImp> initialize_context(
    const std::string& model_path,
    const Graph& onnx_graph,
    const std::vector<morphizen_cxx::NodeConstRef>& ep_context_nodes,
    const onnxruntime::ProviderOptions& options,
    std::unique_ptr<LoggerAdapter> logger_adapter);

// NEW API - add session_configs parameter
std::shared_ptr<PassContextImp> initialize_context(
    const std::string& model_path,
    const Graph& onnx_graph,
    const std::vector<morphizen_cxx::NodeConstRef>& ep_context_nodes,
    const onnxruntime::ProviderOptions& provider_options,
    const std::map<std::string, std::string>& session_configs,  // NEW!
    std::unique_ptr<LoggerAdapter> logger_adapter);
```

### Implementation Steps

**Step 1:** Extract session configs at ort-bridge layer
```cpp
// ort-bridge layer (morphizen-ep.cpp) - NEW function
std::map<std::string, std::string> extract_session_configs(const OrtSessionOptions& options) {
  std::map<std::string, std::string> configs;
  std::string value;

  // Extract all needed session configs using modern API
  GetSessionConfigEntryOrDefault(options, "ep.context_enable", "0", value);
  configs["ep.context_enable"] = value;

  GetSessionConfigEntryOrDefault(options, "ep.context_file_path", "", value);
  configs["ep.context_file_path"] = value;

  // ... extract other session configs as needed
  return configs;
}
```

**Step 2:** Pass session_configs separately
```cpp
// ort-bridge layer - modify compile_onnx_model_3 call
auto provider_options = /* extract provider options */;
auto session_configs = extract_session_configs(session_options);

compile_onnx_model_3(model_path, graph, provider_options, session_configs, ...);
```

**Step 3:** Update PassContextImp to accept session_configs
```cpp
// pass_context_imp.cpp
static std::unique_ptr<PassContextImp>
create_pass_context(const onnxruntime::ProviderOptions& provider_options,
                    const std::map<std::string, std::string>& session_configs);
```

**Step 4:** Remove session_configs field from config.proto
```protobuf
message ConfigProto {
  repeated PassProto passes = 1;
  // ...
  // REMOVED: map<string, string> session_configs = 13;
  reserved 13;
  reserved "session_configs";
  // ...
}
```

**Step 5:** Remove legacy workaround
```cpp
// morphizen-ep.cpp:118-120 - DELETE THIS
} else {
  provider_options_[ort_session_prefix + key] = value;
}
```

**Step 6:** Remove swapping logic
- Remove lines 1234-1235 from pass_context_imp.cpp
- Remove lines 247-248 from morphizen_compile_model.cpp
- Field doesn't exist anymore, no swapping needed

**Step 7:** Regenerate proto C++ code and verify no regressions

## Evidence

**Current session_configs source:**
- config_reader.cpp:216-234 - Extracted from provider_options (session_options ptr or ort_session_config.* entries)

**NEW session_configs source (after this issue):**
- Extracted at ort-bridge layer using `GetSessionConfigEntry()` API
- Passed as separate parameter to morphizen-core

**session_configs usage:**
- pass_context_imp.cpp:224-225 - get_session_config() lookup function
- pass_context_imp.cpp:1207 - Logging only

**session_configs swapping:**
- pass_context_imp.cpp:1234-1235 - Swap after cache load
- morphizen_compile_model.cpp:247-248 - Swap after cache read

**session_configs field:**
- config.proto:49 - map<string, string> session_configs = 13 (to be removed)

## Acceptance Criteria

**Prerequisites:**
- [ ] Issues #008, #009 completed (provider_options is clean)

**Implementation:**
- [ ] `extract_session_configs()` function implemented in ort-bridge layer
- [ ] `initialize_context()` signature updated to accept `session_configs` parameter
- [ ] `PassContextImp::create_pass_context()` signature updated to accept `session_configs` parameter
- [ ] session_configs field removed from config.proto (field 13 reserved)
- [ ] Proto C++ code regenerated
- [ ] Legacy workaround removed (morphizen-ep.cpp:118-120)
- [ ] Swapping logic removed (lines 1234-1235, 247-248)
- [ ] All tests pass

**Verification:**
- [ ] provider_options contains ONLY EP-specific settings (no "ort_session_config." entries)
- [ ] session_configs passed as separate parameter (clean separation)
- [ ] session_configs no longer in ConfigProto
- [ ] No swapping after cache load
- [ ] Layering maintained (core doesn't access OrtSessionOptions)
- [ ] Blocks Issue #003 (ConfigProto is clean)

## Sessions

### 2026-01-31: Design Discussion - Separate Parameters Solution

**Context:** After creating Issue #012, discussed how to properly handle session configs without mixing with provider_options.

**Key questions explored:**

**Q1: Can we remove the legacy workaround (morphizen-ep.cpp:118-120)?**
- A: YES - Modern ORT API v10+ provides `GetSessionConfigEntry()` for direct access

**Q2: How to pass OrtSessionOptions to PassContextImp without breaking layering?**
- Initial concern: PassContextImp is in morphizen-core layer, can't access ORT types
- Core layer should be ORT-agnostic

**Q3: What's the clean solution?**
- A: **Follow the existing pattern for provider_options!**
- morphizen_compile_model.cpp:362-369 already passes `ProviderOptions` as parameter
- `ProviderOptions` is just `std::map<std::string, std::string>` (ORT-agnostic)

**Solution decided:**
```cpp
// Add session_configs parameter alongside provider_options
initialize_context(model_path, graph, ep_context_nodes,
                  provider_options,   // EP-specific settings
                  session_configs,    // Session-level settings (NEW!)
                  logger_adapter);
```

**Benefits:**
- Clean separation - no mixing
- Maintains layering - core doesn't access ORT types
- Follows existing architectural pattern
- ORT-agnostic - both are `std::map<std::string, std::string>`

**Q4: Why doesn't session_configs belong in ConfigProto?**
- A: **Semantic separation of concerns**
- ConfigProto = COMPILATION configuration (how to compile)
- session_configs = SESSION configuration (session-level behavior)
- Even though both are INPUT, they configure different aspects
- ConfigProto should only contain compilation settings

**Related documentation:**
- `docs/technical/provider-options-session-configs-mixing.md` - Updated with this solution

## Notes

**Part of strategic goal:** Achieving immutable ConfigProto.

**This issue is a prerequisite for #003:**
- session_configs field must be removed from config.proto
- ConfigProto should be clean before removing it from ContextProto

**Clean separation approach:**
- Follows existing pattern for provider_options (already passed as separate parameter)
- Maintains layering: morphizen-core doesn't access ORT-specific types
- No mixing: provider_options and session_configs in separate containers
- Extraction at boundary: ort-bridge extracts session configs using modern API

**Semantic separation of concerns:**
- **ConfigProto** = COMPILATION configuration (passes, target, optimization settings)
- **session_configs** = SESSION configuration (ep.context_enable, ep.context_file_path, etc.)
- **provider_options** = EP-specific configuration (cache_key, target, EP settings)
- All three are different concerns that should be kept separate
- Both session_configs and provider_options passed as separate parameters to morphizen-core
- ConfigProto should contain ONLY compilation settings

**Why this is better than extraction from provider_options:**
- Avoids "ort_session_config." prefix pollution
- Clear semantic separation
- Follows established architectural pattern
- Easier to understand and maintain
