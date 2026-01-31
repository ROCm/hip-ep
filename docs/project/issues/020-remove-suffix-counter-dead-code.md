<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Issue #020: Remove suffix_counter Dead Code - Never Written, Always Zero

## Metadata
- **Status:** BACKLOG
- **Priority:** LOW
- **Type:** Tech Debt / Refactoring
- **Owner:** TBD
- **Created:** 2026-01-31
- **Dependencies:** None (independent)
- **Related:** Issue #019 (initialize_context refactoring)

## Description

Remove dead code that reads `suffix_counter` from model metadata. The metadata is never written, so the condition is always false and the code never executes.

## Problem

**Current code (morphizen_compile_model.cpp:421-424):**
```cpp
if (morphizen_cxx::ModelConstRef(model).has_metadata("suffix_counter")) {
  context->suffix_counter = std::stoi(
      morphizen_cxx::ModelConstRef(model).get_metadata("suffix_counter"));
}
```

**Why this is dead code:**

1. **Metadata is NEVER written:**
   - Searched entire codebase: No `model_set_meta_data("suffix_counter", ...)` exists
   - Searched entire codebase: No `set_metadata("suffix_counter", ...)` exists
   - No external code sets this metadata
   - No documentation mentions this metadata

2. **Condition is ALWAYS false:**
   - `has_metadata("suffix_counter")` always returns false
   - Lines 422-423 never execute
   - Dead code that looks like it does something but doesn't

3. **Default initialization is correct:**
   - `suffix_counter` defaults to 0 (pass_context_imp.hpp:215)
   - Each compilation session is independent
   - Starting from 0 is always correct
   - No need to "resume" from a previous value

**How suffix_counter actually works:**
```cpp
// pass_context_imp.hpp:215
mutable int suffix_counter = 0;  // Always starts at 0

// pass_context_imp.cpp:124-128
int PassContextImp::allocate_suffix() const {
  suffix_counter = suffix_counter + 1;  // 0 → 1 → 2 → 3 ...
  return suffix_counter;
}

// anchor_point.cpp:107, 143 - Used to generate unique node names
q->set_name(pair.second + get_name_suffix(context.allocate_suffix()));
// Generates: "node_1", "node_2", "node_3", etc.
```

**Why starting from 0 is always correct:**
- Each compilation is independent
- suffix_counter generates unique names within a single session
- Node naming is deterministic
- Even for cached models, we compile fresh with new anchor points

## Solution

Remove the dead code that reads suffix_counter from model metadata.

**Delete lines 421-424:**
```cpp
// DELETE this entire block (4 lines)
if (morphizen_cxx::ModelConstRef(model).has_metadata("suffix_counter")) {
  context->suffix_counter = std::stoi(
      morphizen_cxx::ModelConstRef(model).get_metadata("suffix_counter"));
}
```

**Keep the default initialization:**
```cpp
// pass_context_imp.hpp:215 - KEEP this (correct behavior)
mutable int suffix_counter = 0;  // Always starts at 0

// pass_context_imp.cpp:124-128 - KEEP this (actually used)
int PassContextImp::allocate_suffix() const {
  suffix_counter = suffix_counter + 1;
  return suffix_counter;
}
```

**Benefits:**
- ✅ Remove confusing dead code (looks functional but isn't)
- ✅ Simplify initialize_context() (4 fewer lines)
- ✅ No functional change (condition was always false)
- ✅ Clearer code (suffix_counter always starts at 0)
- ✅ Less confusion for future developers

## Evidence

**Dead code:**
- morphizen_compile_model.cpp:421-424 - Reads suffix_counter from model metadata

**Metadata is never written (verified by search):**
- No `model_set_meta_data("suffix_counter", ...)` found
- No `set_metadata("suffix_counter", ...)` found
- No code sets this metadata anywhere

**Default initialization (correct behavior):**
- pass_context_imp.hpp:215 - `mutable int suffix_counter = 0;`

**Actual usage (keep this):**
- pass_context_imp.cpp:124-128 - `allocate_suffix()` implementation
- anchor_point.cpp:107 - Used to generate unique node names
- anchor_point.cpp:143 - Used to generate unique node names

## Context

### Why This Dead Code Exists

**Likely historical reasons:**
1. Someone intended to persist suffix_counter across sessions
2. Code was written to read it from model metadata
3. But the write-side was never implemented
4. Or was removed at some point
5. Dead code remained

**No documentation or comments explain it:**
- No comments about why suffix_counter would be in metadata
- No documentation about what value it should have
- No external code sets this metadata

### Why We Don't Need It

**Each compilation is independent:**
- New PassContext created for each compilation
- suffix_counter generates unique names within that session
- Starting from 0 is deterministic and correct

**Even for cached models:**
- We compile the model fresh
- New anchor points created
- New names generated
- No need to resume from previous suffix_counter value

**If we did need to persist suffix_counter:**
- It would belong in ContextProto (OUTPUT), not Model metadata
- But we don't need it - each session is independent

### Related to Issue #019

This cleanup is related to Issue #019 (refactor initialize_context):
- Issue #019: Extract compute_cache_key() (main complexity)
- Issue #020: Remove suffix_counter dead code (4 lines)
- Together: Make initialize_context() simpler and clearer

**After both issues:**
- initialize_context() ~26 lines (was ~52 after #005, #006, #017)
- Much easier to understand and maintain

## Acceptance Criteria

**Implementation:**
- [ ] Delete lines 421-424 from morphizen_compile_model.cpp
- [ ] Verify suffix_counter = 0 default initialization exists (pass_context_imp.hpp:215)
- [ ] Verify allocate_suffix() still works (pass_context_imp.cpp:124-128)
- [ ] All tests pass

**Verification:**
- [ ] Search confirms no code writes "suffix_counter" metadata
- [ ] Existing tests verify node naming still works correctly
- [ ] No functional change (dead code removal)

**Testing:**
- [ ] Existing tests for anchor point naming still pass
- [ ] Node names still generated correctly (e.g., "node_1", "node_2")
- [ ] No regressions in compilation behavior

## Notes

### Simple Cleanup

**Effort:** 5 minutes (delete 4 lines)

**Complexity:** TRIVIAL (pure deletion, no logic changes)

**Risk:** NONE (code never executed, removing dead code)

**Can be done anytime:** Independent of other issues, no dependencies

### Testing Strategy

**No new tests needed:**
- Existing tests already verify suffix_counter works (allocate_suffix())
- Existing tests already verify node naming works (anchor points)
- Dead code removal doesn't change behavior

**Verification:**
```bash
# Verify suffix_counter metadata is never written
grep -r "set_metadata.*suffix" morphizen-core/src/
grep -r "model_set_meta_data.*suffix" morphizen-core/src/

# Both should return NOTHING (confirms it's dead code)
```

### Related Dead Code Pattern

**Similar to morphizen_log_dir (Issue #006):**
- morphizen_log_dir: SET but never read (Issue #006)
- suffix_counter: READ but never set (this issue)
- Both are dead code from obsolete features
- Both should be removed

**Pattern: Metadata that serves no purpose**
- Model metadata should have clear purpose
- If never read OR never written, it's dead code
- Remove to reduce confusion
