<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Issue #013: provider_options Aggregation and Swapping

## Metadata
- Status: BACKLOG
- Priority: MEDIUM
- Type: Tech Debt / Refactoring / Cleanup
- Updated: 2026-01-31 (Clarified as cleanup task after discussion)
- Dependencies: Issues #003 (ConfigProto runtime-only), #008 (MEP removal), #009 (TargetProto injection removal)
- Strategic Goal: Immutable ConfigProto

## Description

Remove obsolete provider_options aggregation and swapping logic after prerequisites (#003, #008, #009) are complete.

**Current problem:** provider_options merged from 4 sources (user > cache > context > target) then swapped after cache load.

**After prerequisites:** Aggregation from 4 sources and swapping become obsolete - just cleanup needed.

## Problem

**Current implementation (pass_context_imp.cpp:187-196):**

```cpp
get_all_provider_options() const {
  auto ret = std::map<std::string, std::string>();
  get_all_provider_option_impl(
      ret,
      &provider_option_origin_,                     // Source 1: User (direct API)
      &provider_option_from_cache_,                 // Source 2: Cache (obsolete after #003)
      &context_proto.config().provider_options(),   // Source 3: User (config file) - STAYS after #003
      target_proto_ ? &target_proto_->provider_options() : nullptr  // Source 4: Target (removed by #009)
  );
  return ret;
}
```

**Note:** Source 3 is provider_options from config file (human convenience). After #003, accessed as `config_.provider_options()` instead of `context_proto.config().provider_options()`, but still exists and still aggregated.

**Two mutation patterns:**

1. **Aggregation before save (lines 709-713):**
```cpp
auto all_opts = get_all_provider_options();  // Merge from 4 sources
proto.mutable_config()->mutable_provider_options()->clear();
proto.mutable_config()->mutable_provider_options()->insert(all_opts.begin(), all_opts.end());
```

2. **Swapping after load (lines 1236-1237):**
```cpp
this->context_proto.mutable_config()->mutable_provider_options()->swap(...);
```

### Root Cause: Inconsistent State Problem

**The fundamental issue:**

```
Cached COMPILATION done with OLD provider_options
BUT we swap in NEW provider_options
→ INCONSISTENT STATE: compiled binary doesn't match current options!
```

**Why this is wrong (EP context scenario):**
- User compiles: `A.onnx` + provider_options{target=NPU} → `A_ctx.onnx` (compiled for NPU)
- Later deploys: Load `A_ctx.onnx` with provider_options{target=CPU}
- Swapping allows this inconsistency: binary compiled for NPU, options say CPU
- Original `A.onnx` is GONE - can't recompile (see docs/technical/ep_shard_context.md)

**Current "solution" (complex workaround):**
- `provider_option_from_cache_` tracks cached options
- Aggregation from 4 sources tries to merge everything
- Swapping tries to preserve user's current options
- Creates complexity documented in cleanup-provider-options.md

**Better approach (after #003):**
- ConfigProto NOT persisted → no cached options to manage
- No swapping needed
- Simpler aggregation (fewer sources)

## Solution

**This issue becomes a CLEANUP task** after prerequisites are complete.

### Prerequisites (must complete first)
1. **Issue #003:** ConfigProto runtime-only (not persisted)
   - Eliminates source 2 (`provider_option_from_cache_`)
   - Changes source 3 access: `context_proto.config()` → `config_`
   - Eliminates swapping need

2. **Issue #008:** Remove MEP table injection
   - Eliminates model-specific injection

3. **Issue #009:** Remove TargetProto injection
   - Eliminates source 4 (`target_proto_->provider_options()`)

### Implementation Steps

**Step 1: Remove obsolete member variables**
```cpp
// pass_context_imp.hpp - DELETE
std::map<std::string, std::string> provider_option_from_cache_ = {};
```

**Step 2: Simplify get_all_provider_options()**
```cpp
// After cleanup: Two user sources remain
get_all_provider_options() const {
  auto ret = std::map<std::string, std::string>();
  get_all_provider_option_impl(
      ret,
      &provider_option_origin_,      // User (direct API)
      &config_.provider_options()    // User (config file)
  );
  return ret;
}
```

**Step 3: Remove swapping logic**
```cpp
// pass_context_imp.cpp:1236-1237 - DELETE
this->context_proto.mutable_config()->mutable_provider_options()->swap(...);
```

**Step 4: Remove aggregation logic**
```cpp
// pass_context_imp.cpp:709-713 - SIMPLIFY
// No need to aggregate from multiple sources anymore
```

**Step 5: Update documentation**
- Update `docs/technical/cleanup-provider-options.md` with new reality
- Document that provider_options now comes from single source (user)
- Remove references to MEP/Target injection
- Remove references to cache swapping

## Evidence

**Code locations:**
- pass_context_imp.hpp:386-387 - Member variables (provider_option_origin_, provider_option_from_cache_)
- pass_context_imp.cpp:187-196 - get_all_provider_options() aggregation
- pass_context_imp.cpp:709-713 - Aggregation before save
- pass_context_imp.cpp:1236-1237 - Swapping after cache load

**Documentation:**
- docs/technical/cleanup-provider-options.md - Current pollution analysis (needs update)
- docs/technical/ep_shard_context.md - EP context design (explains why can't recompile)

## Acceptance Criteria

**Prerequisites:**
- [ ] Issue #003 completed (ConfigProto runtime-only)
- [ ] Issue #008 completed (MEP table removed)
- [ ] Issue #009 completed (TargetProto injection removed)

**Implementation:**
- [ ] provider_option_from_cache_ removed
- [ ] get_all_provider_options() simplified (single source)
- [ ] Swapping logic removed (lines 1236-1237)
- [ ] Aggregation logic simplified (lines 709-713)
- [ ] All tests pass

**Documentation:**
- [ ] cleanup-provider-options.md updated to reflect new reality
- [ ] References to MEP/Target injection removed
- [ ] References to cache swapping removed
- [ ] Document shows provider_options from single source (user)

## Sessions

### 2026-01-31: Discussion - Understanding the Real Problem

**Context:** Discussed what Issue #013 should be after #003, #008, #009 are done.

**Key insights:**

**Q1: Why does swapping exist?**
- A: Inconsistent state problem
- Cached compilation done with OLD options, swap in NEW options
- Binary compiled with old options, but current options are new
- This is fundamentally wrong

**Q2: Why can't we just recompile with new options?**
- A: **EP context model limitation** (see docs/technical/ep_shard_context.md)
- Once compiled: `A.onnx` → `A_ctx.onnx` + binary
- Deployment: Only `A_ctx.onnx` exists (original `A.onnx` is GONE)
- Can't recompile - no source model

**Q3: Can we distinguish compilation vs runtime options?**
- Initial idea: Categorize options, validate compilation options, allow runtime options to differ
- **Reality check:** Too disruptive
- `get_provider_option()` is fundamental API used everywhere
- Can't force everyone to categorize their options
- MEP Table and TargetProto served real purposes (model-specific, target-specific)

**Q4: So what is Issue #013 really about?**
- **After #003, #008, #009, most problems go away**
- Swapping: ✅ Gone (no ConfigProto in cache)
- MEP/Target injection: ✅ Gone (#008, #009)
- provider_option_from_cache_: ✅ Obsolete
- **What remains: Cleanup + Documentation**

**Conclusion:**
- Issue #013 = cleanup task, not design problem
- Remove obsolete code
- Update documentation to reflect new reality
- Simple: provider_options from single source (user only)

**Pragmatic approach:**
- We live in an imperfect world
- Can't categorize every option as compilation vs runtime
- Just make it simpler: fewer sources, no swapping, clear documentation

**Q5: Is the priority/override design good?**
- A: **YES, the design is actually good**
- Direct API overrides config file (standard pattern - like CLI args override config)
- Priority is documented (cleanup-provider-options.md)
- Everything is logged (pass_context_imp.cpp:1188-1213):
  - `provider_option_from_origin` - Direct API
  - `provider_options_in_config` - Config file
  - `provider_option` - Final aggregated result
- Users can see which source each value came from
- Easy to debug: all sources printed separately
- Not silent: conflicting values are visible in logs

**Conclusion on aggregation:**
- Two user sources (direct API + config file) is reasonable
- Priority order makes sense (direct API wins)
- Well documented and logged
- After cleanup: simpler (2 sources instead of 4)

## Notes

**Part of strategic goal:** Achieving immutable ConfigProto.

**Why this issue exists:**
- Complex workaround for inconsistent state problem
- Aggregation from 4 sources attempts to manage conflicting options
- Swapping attempts to preserve user's current options while using cached compilation
- `provider_option_from_cache_` exists to track this complexity

**After cleanup (#003, #008, #009):**
- ConfigProto not persisted → no cached options to manage
- MEP/Target injection gone → no automatic option injection
- Two user sources remain (both intentional):
  1. Direct API: `provider_option_origin_`
  2. Config file: `config_.provider_options()`
- Priority: Direct API overrides config file (standard pattern)
- No swapping needed
- Much simpler (2 sources instead of 4)

**Key constraint we must respect:**
- `get_provider_option()` is fundamental API (pass_context.hpp:93-118)
- Used everywhere by passes
- Can't change this API or force categorization
- Just make the implementation simpler

**Related documentation:**
- docs/technical/cleanup-provider-options.md - Needs update after this cleanup
- docs/technical/ep_shard_context.md - Explains EP context model design
