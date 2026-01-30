<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Issue #007: Remove target Copying in update_config_proto_root_field

## Metadata
- **Status:** BACKLOG
- **Priority:** MEDIUM
- **Type:** Tech Debt / Refactoring / Architecture
- **Owner:** TBD
- **Created:** 2026-01-30
- **Updated:** 2026-01-30 (Solution designed after deep discussion)
- **Dependencies:** Coordinated with Issue #003 (ConfigProto immutability)

## Description

Clean up target handling by removing redundant copying and clarifying the two-path architecture for target discovery.

**Current bad design:**
- target is copied from provider_options into ConfigProto (lines 1273-1275)
- ConfigProto.target is mutable and gets overwritten
- Priority chain reads from provider_options directly anyway, making copy useless
- Auto-discovery runs for both built-in config and user config (should only run for built-in)

**Clean design (this issue):**
- Remove copying - ConfigProto.target is immutable (keeps original value from config file)
- Two clear paths: Built-in config (with auto-discovery) vs User config file (without auto-discovery)
- Change target_proto_ from `unique_ptr<TargetProto>` to `const TargetProto*` (raw pointer into ConfigProto)
- Update docs/technical/target-auto-discovery.md to match implementation

## Problem

### The Confusing Copying

**Current code (pass_context_imp.cpp:1273-1275):**
```cpp
if (auto target = get_provider_option_local({"target", "xlnx_target_name"})) {
  context_proto.mutable_config()->set_target(*target);  // Overwrites ConfigProto.target
}
```

**Why this is wrong:**

1. **Priority 1 doesn't use it** - target_auto_discovery() reads from provider_options DIRECTLY (line 1368-1370):
   ```cpp
   auto target_specified_by_end_user = get_provider_option_internal(
       kProviderOptionTarget, provider_option_origin_);
   ```
   The copied value in ConfigProto is never used by Priority 1.

2. **Mutates immutable config** - ConfigProto should be readonly (Issue #003), but copying overwrites ConfigProto.target

3. **Pollutes fallback** - Priority 4 should read config file default, but copying overwrites it with user value (redundant, since Priority 1 already won)

4. **provider_options pollution** - provider_options is overused as dumping ground (see docs/technical/cleanup-provider-options.md):
   - User explicit values
   - Config file provider_options section
   - MEP table injection (separate issue)
   - TargetProto injection (separate issue)

### The Architecture Confusion

**Three layers with unclear relationships:**
1. `provider_options["target"]` - User runtime input (map)
2. `ConfigProto.target` - Config file default (proto field)
3. `target_proto_` - Resolved TargetProto object (member variable, line 385)

**Current flow is convoluted:**
```
provider_options → Copy to ConfigProto → target_auto_discovery() reads provider_options directly (!)
                                       → Also reads ConfigProto as fallback
                                       → Resolves to target_proto_
```

**After target_proto_ is set, the target NAME (string) is never needed again** - only target_proto_ is used.

### Auto-Discovery Runs Inappropriately

**From docs/technical/target-auto-discovery.md:**
> "when the build-in config file is used, the plugin must return a valid target name"

**Current implementation:** Auto-discovery runs for BOTH built-in config and user config files.

**Problem:** If user provides their own config file, they KNOW their target. Auto-discovery should not override their explicit choice.

## Context

**Part of larger cleanup:** `update_config_proto_root_field()` copies multiple fields from provider_options into ConfigProto:
- encryption_key (Issue #004) - Read from provider_options directly, no RuntimeConfig needed
- cache_key (Issue #005) - Move to ContextProto (cache metadata, needs persistence)
- cache_dir (Issue #006) - Remove entirely (obsolete legacy system)
- target (this issue) - Remove copying, two-path architecture, immutable ConfigProto

**Final goal:** Remove `update_config_proto_root_field()` entirely when all copying is eliminated.

**Relationship to Issue #003:**
- Issue #003 makes ConfigProto runtime-only and immutable
- This issue removes target copying and respects ConfigProto immutability
- ConfigProto owns all TargetProto definitions (catalog of valid targets)
- target_proto_ becomes raw pointer into ConfigProto (no ownership, just reference)

**Target discovery has two distinct use cases:**

1. **Built-in config** (no user config file)
   - System provides default config compiled into binary
   - Auto-discovery is ESSENTIAL (detects target from ONNX model)
   - Fatal error if auto-discovery fails with built-in config

2. **User config file** via `provider_options["config_file"]`
   - User explicitly provides their own config
   - User KNOWS their target (specified in their config file)
   - Auto-discovery should NOT override user's explicit choice

**Current implementation doesn't distinguish these two paths** - auto-discovery runs for both.

## Solution

### Key Principles

1. **Remove copying** - ConfigProto.target is immutable (keeps original value from config file)
2. **Two paths** - Built-in config vs User config file have different behaviors
3. **Auto-discovery only for built-in** - Skip when user provides config file
4. **target_proto_ as raw pointer** - Points into ConfigProto's TargetProto catalog
5. **target NAME used only during discovery** - After target_proto_ is set, name is discarded

### Two-Path Architecture

**Path A: Built-in Config (no user config file)**

Priority chain:
1. `provider_options["target"]` - User explicit override (fatal error if invalid)
2. Plugin auto-discovery - REQUIRED (fatal error if fails)
3. Built-in `ConfigProto.target` - Fallback default

**Path B: User Config File**

Priority chain:
1. `provider_options["target"]` - User explicit override (fatal error if invalid)
2. User `ConfigProto.target` - Direct use (skip auto-discovery)

**Rationale for skipping auto-discovery in Path B:**
- User explicitly provided config file → they KNOW their configuration
- Config file contains target field → user's explicit choice
- Auto-discovery would override user's explicit choice (wrong!)
- Auto-discovery is for built-in config (when system needs to guess)

### Implementation Changes

**Step 1: Remove copying logic**

```cpp
// DELETE lines 1273-1275 in update_config_proto_root_field()
if (auto target = get_provider_option_local({"target", "xlnx_target_name"})) {
  context_proto.mutable_config()->set_target(*target);  // REMOVE THIS
}
```

**Step 2: Change target_proto_ to raw pointer**

```cpp
// pass_context_imp.hpp:385
// OLD:
std::unique_ptr<TargetProto> target_proto_ = nullptr;

// NEW:
const TargetProto* target_proto_ = nullptr;
```

**Rationale:** ConfigProto is immutable and owns all TargetProto definitions. target_proto_ just references one of them. No ownership needed.

**Step 3: Update find_target_proto() signature**

```cpp
// OLD:
std::unique_ptr<TargetProto> find_target_proto(const std::string& target_name);

// NEW:
const TargetProto* find_target_proto(const std::string& target_name);
```

Returns raw pointer to TargetProto inside ConfigProto (or nullptr if not found).

**Step 4: Simplify target_auto_discovery()**

```cpp
void PassContextImp::target_auto_discovery(const Model& model) {
  bool using_builtin_config = !has_user_config_file();

  // Priority 1: User explicit override (both paths)
  if (auto target = get_provider_option("target")) {
    target_proto_ = find_target_proto(*target);
    if (target_proto_ == nullptr) {
      auto valid_targets = get_valid_target_names();
      throw std::invalid_argument("Invalid target: " + *target +
                                  ", valid targets: " + valid_targets);
    }
    return;  // Success
  }

  if (using_builtin_config) {
    // Path A: Built-in config - auto-discovery REQUIRED
    auto discovered = discover_target(config_, model);
    if (discovered.has_value()) {
      target_proto_ = find_target_proto(*discovered);
      if (target_proto_ != nullptr) {
        LOG(INFO) << "Auto-discovery: detected target: " << *discovered;
        return;  // Success
      }
    }
    // Fatal error - built-in config MUST have working auto-discovery
    throw std::runtime_error(
        "Auto-discovery failed with built-in config - this is a fatal error");

  } else {
    // Path B: User config file - use config target directly (no auto-discovery)
    auto& target_name = config_.target();
    if (target_name.empty()) {
      throw std::invalid_argument("User config file must specify target field");
    }
    target_proto_ = find_target_proto(target_name);
    if (target_proto_ == nullptr) {
      auto valid_targets = get_valid_target_names();
      throw std::invalid_argument("Invalid target in config file: " + target_name +
                                  ", valid targets: " + valid_targets);
    }
    LOG(INFO) << "Using target from user config file: " << target_name;
    return;  // Success
  }
}
```

**Step 5: Add has_user_config_file() helper**

```cpp
bool PassContextImp::has_user_config_file() const {
  return get_provider_option("config_file").has_value();
}
```

**Step 6: Update all target_proto_ usage**

Change from:
```cpp
target_proto_->field()
```

To:
```cpp
target_proto_->field()  // Same syntax! (raw pointer dereferencing)
```

No changes needed to usage sites - raw pointer and unique_ptr have same dereferencing syntax.

**Step 7: Update docs/technical/target-auto-discovery.md**

Add sections:
- Two-path architecture explanation
- When auto-discovery runs (built-in config only)
- When auto-discovery is skipped (user config file)
- Rationale for this design
- Examples for each path

### Benefits

- ✅ ConfigProto is truly immutable (Issue #003 alignment)
- ✅ No redundant copying
- ✅ Clear separation: built-in config vs user config file
- ✅ Auto-discovery only where it makes sense (built-in config)
- ✅ Simpler, more predictable behavior
- ✅ Respects user's explicit choices (config file target not overridden)
- ✅ No ownership management (raw pointer simpler than unique_ptr)
- ✅ Documentation matches implementation

## Plans

_No plans yet - solution to be designed later._

## Sessions

### 2026-01-30: Issue Identified

**User guidance:**
> "we can come back to `cache_key`, `cache_dir` and `target` later. I don't have a clear plan. I need you help to write them down and clean my mind, then we have better picture and then we have a better plan."

**Purpose:** Capture the issue now, plan solution later with full context.

### 2026-01-30: Deep Discussion and Solution Design

**User:** "let's focus on 007, and have a discussion. it is more difficult"

**Initial investigation:**
- Discovered target_auto_discovery() has 4-layer priority chain
- Discovered plugin auto-discovery system (sophisticated!)
- Discovered ConfigProto.target used as fallback (Priority 4)
- Confused about when copying is actually needed

**Key documents reviewed:**
- `docs/technical/target-auto-discovery.md` - Priority chain documentation
- `docs/technical/cleanup-provider-options.md` - provider_options pollution problem

**Critical insight from user:**
> "provider_options is over used, see cleanup-provider-options.md for details."

**Questions explored:**

**Q1: Is ConfigProto.target only used during target_auto_discovery()?**
- User: "no ConfigProto.target is the last resort when target_auto_discovery() fail to find any valid target"
- Clarification: It's the fallback default, not just temporary holder

**Q2: Where does ConfigProto.target come from?**
- User: "you hit another complex topic. with provider_options["config_file"] it is possible to have another ConfigProto other than the built-in ConfigProto"
- Discovery: TWO sources - built-in config OR user config file

**Q3: What wins when both provider_options["target"] and config file target exist?**
- User: "provider_options["target"] win"
- Clear priority: user explicit > config file

**Breakthrough moment - User suggested:**
> "let's not worry about copying. let's re-design it. let's have a clear design."

**Use cases enumerated:**
1. User explicit target (highest priority)
2. User config file with target
3. User explicit + config file (override scenario)
4. Built-in config default
5. Plugin auto-discovery
6. Plugin fails, use built-in default
7. Nothing works, fatal error

**User clarification on target_proto_:**
> "if ConfigProto is readonly, I think target_proto_ could be a raw pointer instead, not a unique ptr."

**Insight:** ConfigProto owns all TargetProto definitions (catalog), target_proto_ just references one.

**Q4: Does code need target NAME (string) after discovery?**
- User: "no need to access target NAME anymore. target_proto_ is sufficient."
- Simplification: name is only for lookup during initialization

**Critical design question about auto-discovery:**
> "let's think about auto-discovery again. do you think we need disable it when ConfigProto is not builtin one."

**Analysis:**
- Docs say auto-discovery is for built-in config specifically
- If user provides config file, they KNOW their configuration
- Auto-discovery would override user's explicit choice (wrong!)

**User confirmation:**
> "EXACTLY. RIGHT! thank you."

**Final design principle:**
- Built-in config: Auto-discovery REQUIRED (fatal error if fails)
- User config file: Skip auto-discovery (use config target directly)

**User decision:**
> "yes, update issue 007 with this design"

**Documentation update:**
> User: "in 007 issue, do you think to revise docs/technical/target-auto-discovery.md after resolve it."
> "yes, update issue 007 with this design"

**Key decisions made:**
1. Remove copying entirely - ConfigProto.target is immutable
2. Two-path architecture: built-in vs user config file
3. Auto-discovery only for built-in config
4. target_proto_ as raw pointer (no ownership)
5. Update documentation as part of this issue

## Related PRs

_None yet._

## Notes

### Current Code Locations

**Copying logic (to be removed):**
- `pass_context_imp.cpp:1251-1276` - update_config_proto_root_field() function
- `pass_context_imp.cpp:1273-1275` - target copying

**target_proto_ member:**
- `pass_context_imp.hpp:385` - Currently `unique_ptr<TargetProto>`, change to `const TargetProto*`

**Target discovery:**
- `pass_context_imp.cpp:1367-1411` - target_auto_discovery() function (to be rewritten)
- `pass_context_imp.cpp:1335-1366` - discover_target() - plugin auto-discovery system
- `pass_context_imp.cpp:1315-1334` - try_initialize_target_proto() - resolve name to proto

**Priority chain implementation:**
- Line 1368-1380: Priority 1 - reads provider_options["target"] directly
- Line 1390-1399: Priority 3 - plugin auto-discovery
- Line 1402-1408: Priority 4 - reads ConfigProto.target() as fallback

**Config loading:**
- `binary/config_reader.cpp:35-43` - get_default_config() - built-in config
- `binary/config_reader.cpp:45-68` - JsonFileToMessage() - load user config file

### Three Layers of Target Representation

1. **String name** (temporary, during discovery)
   - `provider_options["target"]` - user runtime input
   - `ConfigProto.target` - config file default (immutable)

2. **TargetProto object** (working configuration)
   - `target_proto_` member variable (raw pointer into ConfigProto)
   - Resolved during target_auto_discovery()
   - Used by all code after initialization

3. **TargetProto catalog** (owned by ConfigProto)
   - ConfigProto contains all valid TargetProto definitions
   - find_target_proto() searches this catalog
   - target_proto_ points to one entry in this catalog

**Key insight:** After target_proto_ is set, the target name (string) is never needed again.

### Two ConfigProto Sources

**Built-in ConfigProto:**
- Compiled into binary (config_json_binary.hpp)
- Default configuration provided by system
- Auto-discovery is REQUIRED (fatal error if fails)
- Purpose: Works out-of-box without user configuration

**User ConfigProto:**
- Loaded from `provider_options["config_file"]`
- User explicitly provides their own config
- Auto-discovery should be SKIPPED (user knows their target)
- Purpose: Users can customize configuration

**Current problem:** Code doesn't distinguish these two sources. Auto-discovery runs for both.

### Plugin Auto-Discovery System

**How it works:**
```cpp
static std::optional<std::string> discover_target(const ConfigProto& proto,
                                                  const Model& model) {
  // Load all plugins with symbol "morphizen_target_discovery"
  auto all_plugin_functions =
      morphizen::Plugin::get_all_symbols("morphizen_target_discovery");

  // Try each plugin
  for (auto& plugin : all_plugin_functions) {
    auto target = target_discovery_func(proto, model);
    if (target.has_value()) {
      return target;  // Success
    }
  }
  return std::nullopt;  // No plugin could detect target
}
```

**Purpose:** Analyze ONNX model to automatically detect which target hardware to use.

**When to use:**
- Built-in config: Essential feature (user doesn't specify target)
- User config: Should NOT run (user already specified target in their config)

### Part of Systematic Cleanup

```cpp
void PassContextImp::update_config_proto_root_field() {
  // Lines 1263-1264: cache_key (Issue #005) - Move to ContextProto
  // Lines 1266-1267: cache_dir (Issue #006) - Remove entirely (obsolete)
  // Lines 1269-1272: encryption_key (Issue #004) - Read from provider_options
  // Lines 1273-1274: target (THIS ISSUE) - Remove, two-path architecture
}
```

Final goal: Remove entire function when all fields cleaned up.

### provider_options Pollution Problem

**From docs/technical/cleanup-provider-options.md:**

provider_options can be modified at multiple stages:
1. User explicit: `provider_options["target"] = "xyz"`
2. Config file: JSON contains `provider_options` section
3. MEP table hit: Injects model_name, model_category, etc. (separate issue)
4. TargetProto hit: Injects xlnx_enable_py3_round, xclbin, etc. (separate issue)

**Related separate issues to create later:**
- MEP table provider_options injection cleanup
- TargetProto provider_options injection cleanup (xlnx_enable_py3_round, xlnx_enable_old_qdq, xclbin)

### Documentation to Update

**docs/technical/target-auto-discovery.md:**

Current issues:
- Doesn't explain built-in config vs user config file distinction
- Doesn't explain when auto-discovery should be skipped
- Priority list incomplete (missing MEP table priority 2)
- No rationale for design decisions

Updates needed:
- Add two-path architecture section
- Explain when auto-discovery runs (built-in only)
- Explain when auto-discovery is skipped (user config)
- Add rationale section
- Add examples for each path
- Clarify that target name is temporary (only target_proto_ used after discovery)
