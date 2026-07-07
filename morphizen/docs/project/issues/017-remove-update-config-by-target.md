<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Issue #017: Remove update_config_by_target() - Obsolete After #007 and #014

## Metadata
- **Status:** BACKLOG
- **Priority:** LOW
- **Type:** Tech Debt / Refactoring
- **Dependencies:** Issue #007, Issue #014 (must complete first)
- **Strategic Goal:** Immutable ConfigProto

## Description

Remove `update_config_by_target()` function and its call site. After Issues #007 and #014 are resolved, this function becomes completely obsolete.

## Problem

**Current design (config.cpp:130-144):**
```cpp
void update_config_by_target(ConfigProto& proto,
                             TargetProto* target_proto_in_pass_context,
                             std::shared_ptr<PassContext> ctx) {
  auto target_proto = get_target_proto(proto, target_proto_in_pass_context->name());

  std::unordered_map<std::string, PassProto> pass_map;
  remove_pass(proto, pass_map);                    // Line 141
  add_target_pass(proto, pass_map, target_proto);  // Line 142 - ISSUE #014
  *target_proto_in_pass_context = *target_proto;   // Line 143 - ISSUE #007
}
```

**Called during initialization (morphizen_compile_model.cpp:417):**
```cpp
morphizen::update_config_by_target(*context->context_proto.mutable_config(),
                                   context->target_proto_.get(), context);
```

**Why this becomes obsolete:**

1. **Lines 141-142: Dynamic pass registration (Issue #014)**
   - Current: Mutates ConfigProto.passes at runtime
   - After #014: Passes computed via `compute_effective_passes()` (no mutation)
   - These lines eliminated by #014 solution

2. **Line 143: TargetProto copying (Issue #007)**
   - Current: Copies entire TargetProto object
   - After #007: target_proto_ becomes `const TargetProto*` (raw pointer, no copying)
   - This line eliminated by #007 solution

3. **Function becomes empty:**
   - After #007 and #014, all three operations (lines 141-143) are eliminated
   - Function has no remaining work to do
   - Call site at line 417 becomes unnecessary

## Solution

Remove `update_config_by_target()` function and its call site.

**Step 1: Remove function definition**

```cpp
// DELETE config.cpp:130-144
void update_config_by_target(...) {
  // Entire function removed
}
```

**Step 2: Remove function declaration**

```cpp
// DELETE config.hpp:23
void update_config_by_target(ConfigProto& proto,
                             TargetProto* target_proto_in_pass_context,
                             std::shared_ptr<PassContext> ctx);
```

**Step 3: Remove call site**

```cpp
// DELETE morphizen_compile_model.cpp:417
morphizen::update_config_by_target(*context->context_proto.mutable_config(),
                                   context->target_proto_.get(), context);
```

**Step 4: Verify no other callers**

```bash
grep -r "update_config_by_target" morphizen-core/src/
# Should find: NONE (all removed in steps 1-3)
```

**Benefits:**
- ✅ Removes obsolete code (~15 lines)
- ✅ Eliminates function that mutates ConfigProto
- ✅ Cleaner initialization flow
- ✅ No redundant operations

## Evidence

**Function definition:**
- config.cpp:130-144 - update_config_by_target() implementation

**Function declaration:**
- config.hpp:23 - update_config_by_target() declaration

**Call site:**
- morphizen_compile_model.cpp:417 - Called during initialize_context()

**Related issues:**
- Issue #014:181 - Lists config.cpp:130+ as mutation site (update_config_by_target)
- Issue #007:373 - Lists pass_context_imp.cpp:1273-1275 as target copying

## Acceptance Criteria

**Prerequisites (MUST complete first):**
- [ ] Issue #014 completed (compute_effective_passes() implemented)
- [ ] Issue #007 completed (target_proto_ is raw pointer)

**Implementation:**
- [ ] Function removed from config.cpp
- [ ] Declaration removed from config.hpp
- [ ] Call site removed from morphizen_compile_model.cpp:417
- [ ] No other callers exist (grep verification)
- [ ] All tests pass

**Verification:**
- [ ] grep shows no references to update_config_by_target
- [ ] ConfigProto not mutated during initialization (for passes or target)
- [ ] Target auto-discovery works correctly (Issue #007 solution)
- [ ] Pass selection works correctly (Issue #014 solution)

## Notes

### Why Function Becomes Obsolete

**Current purpose:** This function does THREE things:
1. Remove passes from ConfigProto
2. Add target-based passes to ConfigProto (dynamic registration)
3. Copy resolved TargetProto into context

**After prerequisites:**

**After Issue #014 (Dynamic Pass Registration):**
- Lines 141-142 eliminated
- Passes no longer mutated in ConfigProto
- `compute_effective_passes()` computes passes on-demand
- No `add_target_pass()` or `remove_pass()` calls needed

**After Issue #007 (Target Two-Path Architecture):**
- Line 143 eliminated
- target_proto_ becomes `const TargetProto*` (raw pointer into ConfigProto)
- Pointer assignment, not object copying: `target_proto_ = target_proto;`
- No separate copying step needed

**Result:** All three operations eliminated → function is empty → remove it.

### Part of ConfigProto Immutability Goal

This function is one of the mutation sites identified during ConfigProto immutability analysis. Removing it is the final cleanup after #007 and #014 eliminate the root causes of mutation.

**ConfigProto mutations eliminated:**
- Issue #014: No more `add_passes()` or `remove_pass()` mutations
- Issue #007: No more target copying mutations
- **This issue:** Removes the function that performed those mutations

### Execution Order

**CRITICAL:** This issue MUST be done AFTER both #007 and #014 are complete.

**Correct order:**
1. Complete Issue #014 → `compute_effective_passes()` implemented
2. Complete Issue #007 → target_proto_ is raw pointer
3. **Then this issue** → Remove obsolete `update_config_by_target()`

**Why order matters:**
- Removing this function before #007/#014 would break initialization
- Function is still needed until both prerequisites eliminate its purpose
- Safe to remove only after both mutations are gone

### Simple Cleanup Issue

**Effort:** 15 minutes (simple deletion)

**Complexity:** LOW (just removing obsolete code)

**Risk:** NONE (after prerequisites complete, function does nothing)

This is a straightforward cleanup issue - no design decisions needed, just delete obsolete code.
