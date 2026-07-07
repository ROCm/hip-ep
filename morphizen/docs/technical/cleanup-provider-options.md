<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Provider Options Cleanup

**Status:** ✅ Cleanup complete (Issues #003, #008, #009, #013 all completed)

## Overview

This document describes the current complexity of provider_options aggregation from multiple sources, and the plan to simplify it.

**Original problem:** provider_options came from 4 different sources, creating pollution and inconsistent state issues.

**Current state (after cleanup):** Simplified to 3 sources - 2 user sources + target defaults.

---

## The Inconsistent State Problem (RESOLVED)

**Original problem:**

```
Cached COMPILATION done with OLD provider_options
BUT we swap in NEW provider_options after loading cache
→ INCONSISTENT STATE: compiled binary doesn't match current options!
```

**Example scenario:**
1. User compiles: `A.onnx` + `{target: "NPU"}` → `A_ctx.onnx` (compiled for NPU)
2. User deploys: Load `A_ctx.onnx` with `{target: "CPU"}`
3. Old design swapped options: binary says NPU, options say CPU
4. Can't recompile (original `A.onnx` is gone in EP context deployment)

**Why this existed:**
- ConfigProto was persisted with cache (Issue #003 - FIXED)
- Multiple injection points polluted provider_options (Issues #008, #009 - FIXED)
- Swapping tried to preserve user's current options (Issue #013 - FIXED)
- `provider_option_from_cache_` tracked this complexity (Issue #013 - REMOVED)

**Resolution:** Issues #003, #008, #009, #013 eliminated this problem.
- ✅ ConfigProto is runtime-only (not persisted)
- ✅ No more swapping logic
- ✅ `provider_option_from_cache_` removed
- ✅ MEP table injection removed
- ✅ Simplified to 3 sources (2 user + target)

---

## Current Sources of provider_options

## explicitly set by end users

```c++
Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test_onnx_runner");

// Register EP library using ORT API 2.0
const std::string kRegistrationName = "MorphiZenExecutionProvider";
auto library_path = std::filesystem::u8path("onnxruntime_vitisai_ep.dll");
auto status = Ort::GetApi().RegisterExecutionProviderLibrary(
    env, kRegistrationName.c_str(), library_path.c_str());

// Get EP devices
std::vector<Ort::ConstEpDevice> selected_devices;
for (const auto& device : env.GetEpDevices()) {
    if (device.EpName() == kRegistrationName) {
        selected_devices.emplace_back(device);
    }
}

// Set provider options and append EP
auto session_options = Ort::SessionOptions();
auto provider_options = std::unordered_map<std::string, std::string>{{"cache_key", "a-sample-cache-key"}};
session_options.AppendExecutionProvider_V2(env, selected_devices, provider_options);
```

it is saved, and now it is supported by PR #209

## set by config file

```c++
Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test_onnx_runner");

// Register EP library using ORT API 2.0
const std::string kRegistrationName = "MorphiZenExecutionProvider";
auto library_path = std::filesystem::u8path("onnxruntime_vitisai_ep.dll");
auto status = Ort::GetApi().RegisterExecutionProviderLibrary(
    env, kRegistrationName.c_str(), library_path.c_str());

// Get EP devices
std::vector<Ort::ConstEpDevice> selected_devices;
for (const auto& device : env.GetEpDevices()) {
    if (device.EpName() == kRegistrationName) {
        selected_devices.emplace_back(device);
    }
}

// Set provider options and append EP
auto session_options = Ort::SessionOptions();
auto provider_options = std::unordered_map<std::string, std::string>{{"config", "morphizen_config.json"}};
session_options.AppendExecutionProvider_V2(env, selected_devices, provider_options);
```

the content of `morphizen_config.json`

```json
{
    "provider_options" : {
        "cache_dir" : "sample_cache_dir",
        "cache_key" : "sample_cache_key"
    }
}
```

## read from `context.json`

it is removed. see jira:VAI-9685. and [pass_context_imp.cpp#L1177][s1]

## ~~set when MEP table hit~~ REMOVED

MEP table feature has been removed entirely. See Issue #008.

**Why removed:**
- Doesn't scale (maintenance burden for thousands of models)
- Fragile (MD5-based identification breaks with model modifications)
- GPU requirements unclear (too early to commit to this approach)
- High testing/validation burden for each entry

**Alternatives:**
- Use smart defaults for common scenarios
- Users set provider_options explicitly when needed via API or config file
- Consult documentation for model-specific best practices
- Runtime heuristics can adapt automatically based on model characteristics

# ~~set when TargetProto hit~~ REMOVED (Issue #009)

All NPU-specific TargetProto features removed entirely:

* `xlnx_enable_py3_round` - Xilinx quantization rounding (NPU-specific)
* `xlnx_enable_old_qdq` - Legacy QDQ handling (NPU-specific)
* `xclbin` - FPGA firmware files (NPU-specific)

Not needed for GPU project.


[s1]: ../morphizen-core/src/pass_context_imp.cpp#L1177

---

## Current State (After Issues #003, #008, #009, #013)

### Simplified Architecture

**Current provider_options sources (3 total):**

1. ✅ **User (direct API)** - `provider_option_origin_` - Explicit via AppendExecutionProvider
2. ✅ **User (config file)** - `config_.provider_options()` - Via morphizen_config.json
3. ✅ **Target defaults** - `target_proto_->provider_options()` - Target-specific defaults (still present)

**Priority:** Direct API > Config file > Target defaults (standard pattern)

**Removed sources:**
4. ❌ **From cache** (Issue #003, #013) - ConfigProto not persisted, provider_option_from_cache_ removed
5. ❌ **MEP table injection** (Issue #008) - MEP table removed

---

### Rationale: Why Two User Sources?

**Why config file is needed:**

Setting many provider options via direct API is tedious and error-prone:

```cpp
// Without config file - tedious to repeat for every session
auto opts = std::unordered_map<std::string, std::string>{
  {"cache_dir", "/path/to/cache"},
  {"target", "NPU"},
  {"xclbin", "/path/to/xclbin"},
  {"enable_profiling", "1"},
  {"log_level", "3"},
  // ... many more options
};
session_options.AppendExecutionProvider_VitisAI(opts);
```

Config file solves this:

```cpp
// With config file - convenient, reusable defaults
auto opts = std::unordered_map<std::string, std::string>{
  {"config", "morphizen_config.json"}  // All defaults in one place
};
session_options.AppendExecutionProvider_VitisAI(opts);
```

**Benefits of config file:**
- ✅ **Convenience** - Avoid repeating many options
- ✅ **Reusability** - Same config across multiple sessions/models
- ✅ **Maintainability** - Update options in one place
- ✅ **Version control** - Check config file into repository
- ✅ **Environment-specific** - Different configs for dev/test/prod

**Why direct API needs to override config file:**

Different sessions may need different settings even with same base config:

```cpp
// Base config file: morphizen_config.json
// {
//   "provider_options": {
//     "cache_dir": "/shared/cache",
//     "target": "NPU",
//     "log_level": "1"
//   }
// }

// Session 1: Use defaults from config file
auto opts1 = std::unordered_map<std::string, std::string>{
  {"config", "morphizen_config.json"}
};

// Session 2: Override specific options (e.g., enable debugging)
auto opts2 = std::unordered_map<std::string, std::string>{
  {"config", "morphizen_config.json"},
  {"log_level", "3"}  // Override for this session only - enable verbose logging
};

// Session 3: Override target (e.g., testing CPU fallback)
auto opts3 = std::unordered_map<std::string, std::string>{
  {"config", "morphizen_config.json"},
  {"target", "CPU"}  // Override for this session only - test CPU mode
};
```

**Benefits of direct API override:**
- ✅ **Flexibility** - Per-session customization without changing config file
- ✅ **Testing** - Override specific options for debugging/testing
- ✅ **Runtime decisions** - Change options based on runtime conditions
- ✅ **Temporary overrides** - Test different settings without modifying shared config
- ✅ **Standard pattern** - Like command-line args override config files

**Implementation: Priority via std::map::insert()**

```cpp
get_all_provider_options() const {
  auto ret = std::map<std::string, std::string>();
  // insert() only inserts if key doesn't exist
  // First source (direct API) has highest priority
  for (auto& kv : provider_option_origin_) {
    ret.insert(kv);  // Direct API values inserted first
  }
  for (auto& kv : config_.provider_options()) {
    ret.insert(kv);  // Config file values inserted only if key not already present
  }
  return ret;
}
```

**Debugging: All sources logged (pass_context_imp.cpp:1188-1213)**

Users can see exactly where each value came from:

```
provider_option_from_origin: log_level = 3         # Direct API
provider_options_in_config: log_level = 1          # Config file
provider_options_in_config: cache_dir = /shared    # Config file only
provider_option: log_level = 3                     # Final (direct API won)
provider_option: cache_dir = /shared               # Final (from config)
```

---

### Simplified Code

```cpp
// Current: pass_context_imp.cpp (after Issue #013)
class PassContextImp {
  ConfigProto config_;                              // Runtime-only (has provider_options from config file)
  std::map<std::string, std::string> provider_option_origin_;  // User (direct API)
  const TargetProto* target_proto_;                 // Target defaults

  // Obsolete members REMOVED:
  // std::map<std::string, std::string> provider_option_from_cache_;  // DELETED (Issue #013)

  // Simplified method - 3 sources (was 4)
  get_all_provider_options() const {
    auto ret = std::map<std::string, std::string>();
    get_all_provider_option_impl(
        ret,
        &provider_option_origin_,                                    // User (direct API) - highest priority
        &config_.provider_options(),                                 // User (config file) - medium priority
        target_proto_ ? &target_proto_->provider_options() : nullptr // Target defaults - lowest priority
    );
    return ret;
  }
};
```

### Benefits After Cleanup

1. ✅ **No inconsistent state** - No swapping, no cached options to conflict with current
2. ✅ **Clear sources** - 2 user-controlled sources (direct API + config file) + 1 target defaults
3. ✅ **Simpler code** - 3 sources instead of 4 (removed cache)
4. ✅ **Standard priority pattern** - Direct API > Config file > Target (like CLI overrides config)
5. ✅ **No pollution** - No automatic injection from MEP table or cache
6. ✅ **Well logged** - All sources printed separately (easy to debug)
7. ✅ **No cache complexity** - provider_option_from_cache_ removed entirely

### Migration Path (COMPLETED)

**Step 1:** Issue #003 - ConfigProto runtime-only ✅ DONE
- Eliminated: provider_option_from_cache_ population, context proto source
- Eliminated: Swapping after cache load

**Step 2:** Issue #008 - Remove MEP table ✅ DONE
- Eliminated: Model-specific option injection

**Step 3:** Issue #009 - Remove NPU-specific xclbin API ✅ DONE
- Eliminated: NPU-specific TargetProto features (xclbin, xlnx_* options)
- Note: target_proto_->provider_options() still exists for target defaults

**Step 4:** Issue #013 - Cleanup ✅ DONE
- Removed: provider_option_from_cache_ member variable
- Simplified: get_all_provider_options() (3 sources instead of 4)
- Updated: This documentation

### Pragmatic Note

We recognize that this is an imperfect world:
- Can't categorize every option as "compilation" vs "runtime"
- `get_provider_option()` is fundamental API used everywhere
- MEP Table and TargetProto served real purposes in the past

**The goal is simpler, not perfect:**
- One source (user)
- No swapping
- No inconsistent state
- Clear documentation

---

## Related Issues

- **Issue #003:** Remove ConfigProto from ContextProto (make it runtime-only)
- **Issue #008:** Remove MEP table injection
- **Issue #009:** Remove TargetProto injection
- **Issue #013:** Cleanup provider_options aggregation and swapping

## References

- `docs/project/issues/013-provider-options-aggregation.md` - Full problem analysis
- `docs/technical/ep_shard_context.md` - EP context design (why can't recompile)
- `docs/technical/provider-options-session-configs-mixing.md` - Related mixing problem
