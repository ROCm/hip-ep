<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Provider Options and Session Configs Mixing: Design Rationale

**Created:** 2026-01-31
**Status:** Current as of ORT API v10+

## Overview

This document explains why `provider_options` and `session_configs` are mixed together in MorphiZen's codebase, and why this complexity exists at the ORT API boundary.

**Key Insight:** The mixing is NOT a design flaw in MorphiZen - it's a consequence of ORT API constraints and features.

---

## TL;DR - Three Reasons for Mixing

1. **ORT Auto EP Selection** - Provider options passed via session configs (CURRENT, ACTIVE)
2. **Legacy Session Config Access** - Workaround when EP couldn't access SessionOptions (LEGACY, can be removed)
3. **Bidirectional API Compatibility** - ORT automatically adds EP prefix (ORT DESIGN)

All three result in `provider_options` containing both EP-specific settings and session-level settings.

---

## Reason 1: ORT Auto EP Selection (CURRENT)

### The Problem

**ORT's auto EP selection feature:**
- ORT automatically selects the best EP based on available hardware
- User does NOT explicitly call `AppendExecutionProvider_VitisAI(provider_options)`
- **Question:** How can user specify provider options if not calling the API?

### The Solution

**Pass provider options via SessionOptions with EP prefix:**

```cpp
// User code (auto EP selection scenario)
auto session_options = Ort::SessionOptions();

// Specify provider options via session config with EP prefix
session_options.SetConfigEntry("ep.morphizenexecutionprovider.cache_key", "my_key");
session_options.SetConfigEntry("ep.morphizenexecutionprovider.target", "NPU");
session_options.SetConfigEntry("ep.morphizenexecutionprovider.encryption_key", "secret");

// No explicit AppendExecutionProvider call!
// ORT auto-selects MorphiZen EP based on detected NPU hardware
auto session = Ort::Session(env, model_path, session_options);
```

**EP receives session configs and extracts provider options:**

```cpp
// morphizen-ep.cpp:108-120
const std::string morphizen_ep_prefix = "ep.morphizenexecutionprovider.";

for (const auto& [key, value] : session_config_entries) {
  if (key.rfind(morphizen_ep_prefix, 0) == 0) {
    // Strip "ep.morphizenexecutionprovider." prefix
    std::string option_name = key.substr(morphizen_ep_prefix.length());
    provider_options_[option_name] = value;  // cache_key, target, encryption_key
  }
}
```

**Result:**
```
provider_options_["cache_key"] = "my_key"
provider_options_["target"] = "NPU"
provider_options_["encryption_key"] = "secret"
```

### Why This Is Necessary

**Auto EP selection is a real ORT feature** used in production scenarios:
- Cloud environments where hardware varies
- Cross-platform applications
- Deployment flexibility

**This is NOT legacy - it's CURRENT and ACTIVE.**

---

## Reason 2: Legacy Session Config Access (LEGACY)

### The Problem (Historical)

**Old ORT API limitation:**
- Execution Provider code could NOT access SessionOptions
- EP needed session-level settings like `"ep.context_enable"`, `"ep.context_file_path"`
- No API to read these from within EP code

### The Workaround (Legacy)

**Smuggle session configs through provider_options:**

```cpp
// morphizen-ep.cpp:118-120 (LEGACY WORKAROUND)
else {
  // If NOT an EP option, add session config to provider_options with prefix
  provider_options_["ort_session_config." + key] = value;
}
```

**Result:**
```
provider_options_["ort_session_config.ep.context_enable"] = "1"
provider_options_["ort_session_config.ep.context_file_path"] = "/path/to/context"
```

**Then read from provider_options:**
```cpp
// Old way (legacy)
auto ep_context_enable = provider_options_["ort_session_config.ep.context_enable"];
```

### The Modern API Solution

**ORT API v10+ provides direct access:**

```cpp
// morphizen-ep.cpp:806-827 - Modern API
OrtStatus* MorphiZenEP::GetSessionConfigEntryOrDefault(
    const OrtSessionOptions& session_options,
    const char* config_key,
    const std::string& default_val,
    std::string& config_val) {

  int has_config = 0;
  ort_api.HasSessionConfigEntry(&session_options, config_key, &has_config);

  if (has_config) {
    ort_api.GetSessionConfigEntry(&session_options, config_key,
                                   config_val.data(), &size);
  } else {
    config_val = default_val;
  }
  return nullptr;
}
```

**Current usage (modern):**
```cpp
// morphizen-ep.cpp:73-75
std::string ep_context_enable;
GetSessionConfigEntryOrDefault(session_options, "ep.context_enable", "0", ep_context_enable);
enable_ep_context_ = (ep_context_enable == "1");
```

### Can We Remove the Legacy Workaround?

**YES, if:**
1. All session config reads migrated to `GetSessionConfigEntry()`
2. No code depends on reading session configs from provider_options with `"ort_session_config."` prefix
3. ConfigProto is clean (Issues #004, #012)

**This is part of cleanup effort tracked in Issue #012 and #013.**

---

## Reason 3: ORT Bidirectional Conversion (ORT DESIGN)

### Different EPs Handle This Differently

**NvTensorRT (minimal approach):**

```cpp
// nv_provider_factory.cc:82-86
for (const auto& [key, value] : config_options_map) {
  if (key.rfind(key_prefix, 0) == 0) {
    provider_options[key.substr(key_prefix.size())] = value;
  }
  // NO else clause - ignores session configs!
}
```

**VitisAI/MorphiZen (captures everything):**

```cpp
// vitisai_provider_factory.cc:52-58
for (const auto& [key, value] : config_options_map) {
  if (key.rfind(key_prefix, 0) == 0) {
    provider_options[key.substr(key_prefix.size())] = value;
  } else {
    // ALSO capture session configs
    provider_options["ort_session_config." + key] = value;
  }
}
```

**MorphiZen chose to capture session configs because:**
1. Legacy workaround for SessionOptions access (Reason #2)
2. Needed session configs during compilation

**This else clause is legacy and can be removed with modern API.**

---

## Current State: provider_options Contains Both

### What's In provider_options Map Today

```cpp
std::map<std::string, std::string> provider_options_;

// EP-specific settings (legitimate)
provider_options_["cache_key"] = "..."
provider_options_["target"] = "NPU"
provider_options_["encryption_key"] = "..."

// Session configs (from auto EP selection - ACTIVE)
provider_options_["cache_key"] = "..."  // User set via ep.morphizenexecutionprovider.cache_key

// Session configs (legacy workaround - CAN BE REMOVED)
provider_options_["ort_session_config.ep.context_enable"] = "1"
provider_options_["ort_session_config.ep.context_file_path"] = "..."
```

### The Confusion

**Semantically different concepts mixed in one map:**
- `provider_options` sounds like "EP-specific settings"
- But actually contains: EP settings + session configs (from multiple sources)
- Hard to reason about "what is provider_options really?"

---

## Code Locations

### Where Mixing Happens

**1. Auto EP Selection (morphizen-ep.cpp:89-131):**
```cpp
void MorphiZenEP::update_provider_options_from_session_config(
    const OrtSessionOptions& session_options) {

  const std::string morphizen_ep_prefix = "ep.morphizenexecutionprovider.";
  const std::string ort_session_prefix = "ort_session_config.";

  for (const auto& [key, value] : session_config_entries) {
    if (key.rfind(morphizen_ep_prefix, 0) == 0) {
      // Auto EP selection: strip prefix
      provider_options_[key.substr(morphizen_ep_prefix.length())] = value;
    } else {
      // Legacy workaround: add prefix
      provider_options_[ort_session_prefix + key] = value;
    }
  }
}
```

**2. EP Metadata (morphizen-ep.cpp:206-240):**
```cpp
void MorphiZenEP::update_provider_options_from_ep_metadata(
    const OrtKeyValuePairs* const* ep_metadata) {

  // Direct provider options from explicit AppendExecutionProvider call
  for (size_t i = 0; i < num_keys; ++i) {
    provider_options_[keys[i]] = values[i];
  }
}
```

**3. Config Reader (config_reader.cpp:96-111):**
```cpp
static void set_session_config(google::protobuf::Struct& ret,
                               const std::string& key,
                               const std::string& value) {
  if (key.rfind(kEpProviderOptionPrefix, 0) == 0) {
    // Strip "ep.morphizenexecutionprovider." prefix
    auto key2 = key.substr(sizeof(kEpProviderOptionPrefix) - 1);
    set_struct_value(ret, kProviderOptions, key2, value);
  } else {
    // Session config
    set_struct_value(ret, kSessionConfig, key, value);
  }
}
```

---

## Cleanup Strategy

### Clean Solution: Separate Parameters

**Pass session_configs separately from provider_options** (similar to how provider_options is already passed):

```cpp
// Current API (morphizen_compile_model.cpp:362-369)
std::shared_ptr<PassContextImp> initialize_context(
    const std::string& model_path,
    const Graph& onnx_graph,
    const std::vector<morphizen_cxx::NodeConstRef>& ep_context_nodes,
    const onnxruntime::ProviderOptions& options,  // Provider options
    std::unique_ptr<LoggerAdapter> logger_adapter);

// NEW API - add session_configs parameter
std::shared_ptr<PassContextImp> initialize_context(
    const std::string& model_path,
    const Graph& onnx_graph,
    const std::vector<morphizen_cxx::NodeConstRef>& ep_context_nodes,
    const onnxruntime::ProviderOptions& provider_options,      // Provider options
    const std::map<std::string, std::string>& session_configs, // Session configs (NEW!)
    std::unique_ptr<LoggerAdapter> logger_adapter);
```

**Benefits:**
- ✅ Clean separation - no mixing
- ✅ Maintains layering - core layer doesn't access OrtSessionOptions
- ✅ ORT-agnostic - both are just `std::map<std::string, std::string>`
- ✅ Follows existing pattern for provider_options

### What Can Be Removed

**Legacy workaround (Reason #2):**
```cpp
// morphizen-ep.cpp:118-120 - CAN BE REMOVED
} else {
  provider_options_[ort_session_prefix + key] = value;
}
```

**Replace with:**
```cpp
// ort-bridge layer - extract session configs using modern API
std::map<std::string, std::string> extract_session_configs(const OrtSessionOptions& options) {
  std::map<std::string, std::string> configs;
  std::string config_value;
  GetSessionConfigEntryOrDefault(options, "ep.context_enable", "0", config_value);
  configs["ep.context_enable"] = config_value;
  // ... extract other session configs
  return configs;
}

// Pass separately to core layer
auto session_configs = extract_session_configs(session_options);
compile_onnx_model_3(model_path, graph, provider_options, session_configs, ...);
```

### What Must Stay

**Auto EP selection (Reason #1):**
```cpp
// morphizen-ep.cpp:114-117 - MUST STAY (active feature)
if (key_str.rfind(morphizen_ep_prefix, 0) == 0) {
  std::string option_name = key_str.substr(morphizen_ep_prefix.length());
  provider_options_[option_name] = value_str;
}
```

**This supports users who:**
- Use ORT auto EP selection
- Set provider options via `session_options.SetConfigEntry("ep.morphizenexecutionprovider.X", val)`

### Related Issues

**Issue #012: Remove session_configs from ConfigProto**
- Part of removing legacy workaround
- Extract session configs using modern `GetSessionConfigEntry()` API
- Stop storing session configs in ConfigProto or provider_options

**Issue #013: Provider Options Aggregation**
- Addresses provider_options pollution from multiple sources
- After cleanup: provider_options will contain only EP settings (from user or auto EP selection)

**Issue #008, #009: Remove MEP/Target Injection**
- Clean up provider_options pollution from non-user sources
- After cleanup: only user-provided options remain

---

## Two Valid Use Cases

### Use Case 1: Explicit EP Selection

**User controls EP selection directly:**

```cpp
// User explicitly selects VitisAI EP
std::map<std::string, std::string> provider_options;
provider_options["cache_key"] = "my_key";
provider_options["target"] = "NPU";

session_options.AppendExecutionProvider_VitisAI(provider_options);
```

**Result:** `provider_options_` contains user's EP settings.

### Use Case 2: Auto EP Selection

**ORT automatically selects best EP:**

```cpp
// User sets options via session config with EP prefix
session_options.SetConfigEntry("ep.morphizenexecutionprovider.cache_key", "my_key");
session_options.SetConfigEntry("ep.morphizenexecutionprovider.target", "NPU");

// ORT auto-selects EP based on hardware
auto session = Ort::Session(env, model_path, session_options);
```

**Result:** EP extracts provider options from session config by stripping prefix.

**Both are valid!** Code must support both entry points.

---

## Future Direction

### After Cleanup (Issues #004, #008, #009, #012, #013)

**Clean architecture with separate parameters:**

```cpp
// ort-bridge layer (morphizen-ep.cpp)
class MorphiZenEP {
  // Extract provider options (EP-specific settings)
  std::map<std::string, std::string> extract_provider_options() {
    // From auto EP selection: "ep.morphizenexecutionprovider.X"
    // From explicit API: AppendExecutionProvider
    // From config file
  }

  // Extract session configs (session-level settings)
  std::map<std::string, std::string> extract_session_configs() {
    // Use modern GetSessionConfigEntry() API
    std::string value;
    GetSessionConfigEntryOrDefault(session_options, "ep.context_enable", "0", value);
    // Extract all needed session configs
  }

  // Pass both separately to core layer
  compile_onnx_model_3(provider_options, session_configs, ...);
};

// morphizen-core layer (pass_context_imp.cpp)
class PassContextImp {
  // Separate member variables (or just parameters)
  std::map<std::string, std::string> provider_options_;  // EP settings
  std::map<std::string, std::string> session_configs_;   // Session settings (if needed)
};
```

**provider_options sources (after cleanup):**
1. ✅ User explicit (AppendExecutionProvider)
2. ✅ User via session config with EP prefix (auto EP selection)
3. ✅ Config file
4. ❌ MEP table (removed - Issue #008)
5. ❌ TargetProto (removed - Issue #009)
6. ❌ Session configs smuggled with "ort_session_config." prefix (removed - legacy workaround)

**session_configs sources:**
1. ✅ Extracted at ort-bridge layer using `GetSessionConfigEntry()`
2. ✅ Passed as separate parameter to morphizen-core
3. ❌ NOT mixed with provider_options
4. ❌ NOT stored in ConfigProto (Issue #012)

### Clean Architecture Goal

**Clear semantic separation:**
- `provider_options` = EP-specific settings (from user or auto EP selection)
- `session_configs` = Session-level settings (extracted at ort-bridge, passed separately)
- No mixing, no pollution, no confusion
- Clean layering: core layer doesn't access ORT-specific types

---

## References

**Code Locations:**
- `ort-bridge/src/morphizen-ep.cpp:89-131` - Auto EP selection extraction
- `ort-bridge/src/morphizen-ep.cpp:806-827` - Modern session config API
- `morphizen-core/src/binary/config_reader.cpp:96-111` - Bidirectional conversion
- `onnxruntime/core/providers/vitisai/vitisai_provider_factory.cc:52-58` - VitisAI pattern
- `onnxruntime/core/providers/nv_tensorrt_rtx/nv_provider_factory.cc:82-86` - NvTensorRT pattern

**Related Documentation:**
- `docs/technical/cleanup-provider-options.md` - Provider options pollution analysis
- `docs/project/issues/012-session-configs-swapping.md` - Remove session_configs from ConfigProto
- `docs/project/issues/013-provider-options-aggregation.md` - Provider options aggregation cleanup

**Related Issues:**
- Issue #004: Remove encryption_key from ConfigProto
- Issue #008: Remove MEP table injection
- Issue #009: Remove TargetProto injection
- Issue #012: Remove session_configs from ConfigProto
- Issue #013: Provider options aggregation cleanup

---

## Discussion History

This document captures insights from discussion on 2026-01-31 analyzing why session_configs and provider_options are mixed in the codebase. Key realizations:

1. **Auto EP selection is NOT legacy** - it's an active ORT feature that requires EP prefix
2. **Different EPs handle mixing differently** - NvTensorRT ignores session configs, VitisAI captures them
3. **Modern ORT API v10+ provides clean session config access** - legacy workaround can be removed
4. **The mixing serves multiple purposes** - some current (auto EP), some legacy (session config access)

Understanding this context is critical for cleanup work in Issues #012-#013.
