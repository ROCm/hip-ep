<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Immutable ConfigProto Roadmap

**Strategic Goal:** Make ConfigProto immutable after construction - never mutated at runtime.

**Created:** 2026-01-31
**Status:** Roadmap analysis - work remaining

---

## Executive Summary

**Total Issues:** 11 issues required (+ 1 optional quality improvement)
**Total Mutation Sites:** ~18 locations where ConfigProto is mutated
**Estimated Effort:** 6-8 sessions (foundational work + cleanups + final enforcement)

**Critical Path:** #003 (foundational) → #004, #005, #007, #012, #013, #014, #015, #017 → #018 (final enforcement)

**Optional:** #019 (refactor initialize_context - code quality improvement, not required for immutability)

**Final Goal:** ConfigProto immutable immediately after construction - never mutated at runtime or during initialization - enforced at compile-time

---

## Current State: ConfigProto Mutations Discovered

From the ConfigProto mutation analysis (17+ sites discovered during Issue #003 investigation):

### Mutation Categories

1. **cache_key mutations** - 5 sites (Issue #005)
2. **encryption_key mutations** - 2 sites (Issue #004)
3. **target mutations** - 1 site (Issue #007)
4. **cache_dir mutations** - 2 sites (Issue #006)
5. **session_configs swap** - 2 sites (Issue #012)
6. **provider_options aggregation/swap** - 2 sites (Issue #013)
7. **Dynamic pass registration** - 2 sites (Issue #014)
8. **Initialization (version info + swap)** - 2 sites (Issue #015)
9. **update_config_by_target cleanup** - Function removal (Issue #017)
10. **Compile-time enforcement** - Make config_ const (Issue #018)

**Total:** ~18 mutation sites + 2 cleanup/enforcement issues

---

## Issues Required for Immutable ConfigProto

### FOUNDATIONAL (Must Do First)

**#003: Remove ConfigProto from ContextProto**
- **Type:** Architectural change
- **Mutations:** 0 (enables others)
- **Complexity:** HIGH
- **Effort:** 2-3 sessions
- **Blocks:** #004, #012, #013
- **Influences:** #005, #007, #015
- **Status:** BACKLOG (highest priority)

**What it does:**
- Makes ConfigProto runtime-only member of PassContextImp
- Removes ConfigProto field from ContextProto proto
- Eliminates accidental persistence
- Enables swapping elimination (#012, #013)

**Impact:**
- Resolves #012 automatically (no swapping needed)
- Simplifies #013 (no cache swapping)
- Clean foundation for other mutations

---

### FIELD COPYING MUTATIONS (After #003)

**#004: Remove encryption_key Copying**
- **Mutations:** 2 sites
  - morphizen_compile_model.cpp:245 - restore after cache
  - pass_context_imp.cpp:1269-1271 - copy from provider_options
- **Complexity:** LOW (simple removal)
- **Effort:** 15 minutes
- **Depends on:** #003 (encryption_key won't exist in proto)
- **Status:** BACKLOG

**What it does:**
- Read encryption_key from provider_options directly
- Remove encryption_key field from ConfigProto
- No copying, no mutation

---

**#005: Move cache_key to ContextProto**
- **Mutations:** 5 sites
  - morphizen_compile_model.cpp:395 - from model metadata
  - morphizen_compile_model.cpp:402 - from file hash
  - morphizen_compile_model.cpp:407 - from memory signature
  - morphizen_compile_model.cpp:847 - from EP context node
  - pass_context_imp.cpp:1263-1264 - from provider_options
- **Complexity:** MEDIUM (move to different proto)
- **Effort:** 30-45 minutes
- **Depends on:** Should coordinate with #003
- **Status:** BACKLOG

**What it does:**
- Move cache_key from ConfigProto to ContextProto
- cache_key is cache metadata (OUTPUT), not configuration (INPUT)
- Needs persistence, belongs in ContextProto

---

**#006: Remove cache_dir Entirely**
- **Mutations:** 2 sites
  - cache_dir.cpp:67 - cache_dir computation
  - pass_context_imp.cpp:1266-1267 - copy from provider_options
- **Complexity:** MEDIUM (~200-300 LOC removal)
- **Effort:** 1 session
- **Depends on:** Independent (but related to #009)
- **Status:** BACKLOG

**What it does:**
- Remove obsolete disk-based cache system
- Remove cache_dir field from ConfigProto
- Tar_file_ system replaces it

---

**#007: Clean Up Target with Two-Path Architecture**
- **Mutations:** 1 site
  - pass_context_imp.cpp:1273-1274 - copy from provider_options
- **Complexity:** HIGH (comprehensive redesign)
- **Effort:** 1-2 sessions
- **Depends on:** Should coordinate with #003
- **Status:** BACKLOG

**What it does:**
- Two-path architecture: built-in targets vs user config
- Remove target copying from provider_options
- Make target_proto_ raw pointer (immutable reference)
- ConfigProto readonly

---

### SWAPPING MUTATIONS (Resolved by #003)

**#012: session_configs Swapping**
- **Mutations:** 2 sites
  - pass_context_imp.cpp:1234-1235 - swap after cache load
  - morphizen_compile_model.cpp:247-248 - swap after cache read
- **Complexity:** NONE (auto-resolved)
- **Effort:** 0 minutes (documentation only)
- **Resolved by:** #003
- **Status:** BACKLOG

**What it does:**
- NOTHING - automatically resolved when #003 makes ConfigProto runtime-only
- No swapping needed when ConfigProto not persisted

---

**#013: provider_options Aggregation and Swapping**
- **Mutations:** 2 sites
  - pass_context_imp.cpp:709-713 - aggregate before save
  - pass_context_imp.cpp:1236-1237 - swap after cache load
- **Complexity:** LOW (cleanup after prerequisites)
- **Effort:** 30 minutes
- **Depends on:** #003, #008, #009 (must complete first)
- **Status:** BACKLOG

**What it does:**
- Remove provider_option_from_cache_ member variable
- Simplify get_all_provider_options() (2 sources instead of 4)
- Remove swapping logic (eliminated by #003)

---

### ARCHITECTURAL MUTATIONS

**#014: Dynamic Pass Registration**
- **Mutations:** 2 sites
  - config.cpp:110-115 - target-based pass selection (add_passes)
  - pass_imp.cpp:453-456 - plugin-based anonymous pass (add_passes)
- **Complexity:** MEDIUM (architectural redesign)
- **Effort:** 1 session
- **Depends on:** Independent (related to #003 goal)
- **Status:** BACKLOG

**What it does:**
- Compute effective pass list on-demand
- ConfigProto.passes = immutable library
- compute_effective_passes() returns local variable
- No add_passes() mutations

---

**#015: Configuration Initialization**
- **Mutations:** 2 sites
  - pass_context_imp.cpp:104 - add_version_info() call
  - config.cpp:63-67 - add_version_info() implementation
- **Complexity:** LOW (simple move)
- **Effort:** 30 minutes
- **Depends on:** Independent (can coordinate with #003)
- **Status:** BACKLOG

**What it does:**
- Move version info from ConfigProto to ContextProto
- Version info = OUTPUT metadata, not INPUT configuration
- Remove add_version_info() mutation

---

**#017: Remove update_config_by_target()**
- **Mutations:** 0 (cleanup after other issues)
  - config.cpp:130-144 - Function definition (to be removed)
  - morphizen_compile_model.cpp:417 - Call site (to be removed)
- **Complexity:** LOW (simple deletion)
- **Effort:** 15 minutes
- **Depends on:** #007 AND #014 (must complete BOTH first)
- **Status:** BACKLOG

**What it does:**
- Remove obsolete update_config_by_target() function
- After #007: TargetProto copying eliminated (line 143)
- After #014: Dynamic pass registration eliminated (lines 141-142)
- Function becomes empty → simple deletion

---

**#018: Make ConfigProto const Member**
- **Mutations:** 0 (enforcement, not fixing)
  - pass_context_imp.hpp - Change config_ to const config_
- **Complexity:** LOW (one-line change)
- **Effort:** 5 minutes
- **Depends on:** #004, #005, #007, #014, #015, #017 (ALL must complete first)
- **Status:** BACKLOG

**What it does:**
- Change `ConfigProto config_` to `const ConfigProto config_`
- Compile-time enforcement of immutability
- Final step after all mutations eliminated
- Prevents future regressions (compiler catches mutations)

---

## Implementation Roadmap

### Phase 1: Foundation (Critical Path)

**Priority: CRITICAL**

1. **#003: Remove ConfigProto from ContextProto** (2-3 sessions)
   - Architectural foundation
   - Unblocks: #004, #012, #013
   - Influences: #005, #007, #015

**Outcome after Phase 1:**
- ConfigProto is runtime-only (never persisted)
- Swapping eliminated (#012 auto-resolved)
- Clean foundation for remaining mutations

---

### Phase 2: Field Copying Cleanups (Quick Wins After #003)

**Priority: HIGH**

2. **#004: Remove encryption_key** (15 min)
   - Must complete after #003
   - Simple removal

3. **#005: Move cache_key** (30-45 min)
   - Coordinate with #003
   - Medium complexity

4. **#015: Configuration Initialization** (30 min)
   - Simple move
   - Independent

5. **#014: Dynamic Pass Registration** (1 session)
   - Architectural redesign
   - Can be done in parallel with other Phase 2 work

**Outcome after Phase 2:**
- 4 more mutation categories eliminated
- ~10 mutation sites removed
- ConfigProto cleaner
- Dynamic passes computed on-demand

---

### Phase 3: Architectural Redesigns

**Priority: MEDIUM**

6. **#007: Target Two-Path Architecture** (1-2 sessions)
   - Comprehensive redesign
   - Coordinate with #003

**Outcome after Phase 3:**
- 1 architectural improvement complete
- ~1 mutation site removed
- ConfigProto closer to immutable

---

### Phase 4: Final Cleanups

**Priority: LOW (after prerequisites)**

7. **#017: Remove update_config_by_target()** (15 min)
   - After #007 AND #014
   - Simple deletion

8. **#006: Remove cache_dir** (1 session)
   - Large cleanup (~200-300 LOC)
   - Independent (but consider #009 coordination)

9. **#013: provider_options Aggregation** (30 min)
   - After #003, #008, #009
   - Simple cleanup

10. **#018: Make ConfigProto const Member** (5 min)
    - After #004, #005, #007, #014, #015, #017
    - Compile-time enforcement

**Outcome after Phase 4:**
- All mutations eliminated
- ConfigProto fully immutable (compile-time enforced)
- ~565+ LOC cleaned up

---

### Phase 5: Verification

**Final validation:**

✅ **Verify ConfigProto never mutated:**
```bash
# Search for mutations
grep -r "mutable_config()" morphizen-core/src/
grep -r "add_passes()" morphizen-core/src/
grep -r "set_.*(" morphizen-core/src/config.cpp
```

✅ **Expected results:**
- No `mutable_config()` calls (except during construction)
- No `add_passes()` calls at runtime
- No field setters on ConfigProto after construction
- All tests pass

✅ **ConfigProto characteristics:**
- Constructed once during PassContextImp initialization
- Never modified after construction
- Never persisted to disk
- Clean INPUT-only structure

---

## Effort Summary

### By Issue

| Issue | Mutations | Complexity | Effort | Dependencies |
|-------|-----------|------------|--------|--------------|
| #003 | 0 (enables) | HIGH | 2-3 sessions | None (foundational) |
| #004 | 2 | LOW | 15 min | #003 |
| #005 | 5 | MEDIUM | 30-45 min | Coordinate with #003 |
| #006 | 2 | MEDIUM | 1 session | Independent |
| #007 | 1 | HIGH | 1-2 sessions | Coordinate with #003 |
| #012 | 2 | NONE | 0 min | Auto-resolved by #003 |
| #013 | 2 | LOW | 30 min | #003, #008, #009 |
| #014 | 2 | MEDIUM | 1 session | Independent |
| #015 | 2 | LOW | 30 min | Independent |
| #017 | 0 (cleanup) | LOW | 15 min | #007, #014 |
| #018 | 0 (enforcement) | LOW | 5 min | #004, #005, #007, #014, #015, #017 |
| **Total** | **18** | **MIXED** | **6-8 sessions** | **Critical path: #003 → #018** |

### By Complexity

**HIGH Complexity (Architectural):**
- #003 - Remove ConfigProto from ContextProto (2-3 sessions)
- #007 - Target two-path architecture (1-2 sessions)

**MEDIUM Complexity:**
- #005 - Move cache_key (30-45 min)
- #006 - Remove cache_dir (1 session)
- #014 - Dynamic pass registration (1 session)

**LOW Complexity (Simple):**
- #004 - Remove encryption_key (15 min)
- #013 - provider_options cleanup (30 min)
- #015 - Configuration initialization (30 min)

**ZERO Complexity (Auto-resolved):**
- #012 - session_configs swapping (0 min)

### Total Effort Estimate

**Minimum:** 6 sessions (~24 hours)
**Maximum:** 8 sessions (~32 hours)
**Average:** 7 sessions (~28 hours)

**Breakdown:**
- Foundational: 2-3 sessions (#003)
- Architectural: 2-3 sessions (#007, #014)
- Medium cleanups: 2 sessions (#005, #006)
- Simple cleanups: ~2 hours total (#004, #013, #015)
- Auto-resolved: 0 (#012)

---

## Risk Assessment

### High Risk Items

**#003: Remove ConfigProto from ContextProto**
- **Risk:** Breaks many code paths, affects serialization
- **Mitigation:** Comprehensive testing, careful migration
- **Impact if delayed:** Blocks #004, #012, #013

**#007: Target Two-Path Architecture**
- **Risk:** Complex redesign, affects target auto-discovery
- **Mitigation:** Thorough testing, gradual rollout
- **Impact if delayed:** One mutation remains

### Medium Risk Items

**#014: Dynamic Pass Registration**
- **Risk:** Changes pass selection logic
- **Mitigation:** Test all pass configurations
- **Impact if delayed:** Two mutations remain

**#006: Remove cache_dir**
- **Risk:** Large code removal (~200-300 LOC)
- **Mitigation:** Search all get_log_dir() callers first
- **Impact if delayed:** Two mutations remain

### Low Risk Items

All other issues (#004, #005, #013, #015) are low risk:
- Simple refactorings
- Well-scoped changes
- Clear implementations

---

## Success Criteria

### Definition of "Immutable ConfigProto"

**After all issues complete:**

✅ **Construction-time only:**
- ConfigProto created once during PassContextImp initialization
- Populated from config file + provider_options
- Never modified after construction

✅ **No runtime mutations:**
- No field setters called after construction
- No add_passes() at runtime
- No copying from provider_options
- No swapping after cache load

✅ **Runtime-only (not persisted):**
- ConfigProto is member variable, not proto field
- Never serialized to disk
- Never appears in context.json

✅ **Clear INPUT semantics:**
- ConfigProto = configuration INPUT only
- ContextProto = compilation OUTPUT only
- Clean separation

### Verification Methods

1. **Code search:**
   ```bash
   # Should find NO runtime mutations
   grep -r "mutable_config()" morphizen-core/src/
   grep -r "add_passes()" morphizen-core/src/
   ```

2. **Proto inspection:**
   ```bash
   # ConfigProto should NOT be in ContextProto
   grep "ConfigProto" morphizen-core/src/pass_context.proto
   # Should be: field 3 reserved
   ```

3. **Runtime validation:**
   - Compile a model, inspect context.json
   - Should NOT contain "config" field
   - Should only contain: meta_def, origin_nodes, events

4. **Test coverage:**
   - All unit tests pass
   - Cache functionality works
   - EP context models work
   - No regression in model compilation

---

## Parallel Work Opportunities

**Can work in parallel:**

**Track 1 (Critical Path):**
- #003 → #004, #005, #013, #015

**Track 2 (Independent):**
- #006 (cache_dir)
- #014 (dynamic passes)

**Track 3 (Coordination with #003):**
- #007 (target architecture)

**Strategy for maximum throughput:**
1. Start #003 immediately (blocks others)
2. Simultaneously start #006, #014 (independent)
3. After #003 completes → #004, #005, #015 (quick wins)
4. Then #007 (coordinate with #003 results)
5. Finally #013 (after #003 + #008 + #009)

---

## Conclusion

**Work Remaining:** 11 issues required (+ 1 optional), ~18 mutation sites, 6-8 sessions

**Critical Path:** #003 is foundational → #004-#017 eliminate mutations → #018 enforces immutability

**Optional Quality Improvement:** #019 (refactor initialize_context - after #005 and #017)

**Quick Wins After #003:**
- #004 (15 min) - encryption_key
- #015 (30 min) - version info
- #005 (30-45 min) - cache_key

**Major Redesigns:**
- #007 (1-2 sessions) - target architecture
- #014 (1 session) - dynamic passes

**Auto-Resolved:**
- #012 (0 min) - resolved by #003

**Final Enforcement:**
- #018 (5 min) - make config_ const (compile-time verification)

**Recommendation:**
- **Start with #003** (highest priority, unblocks others)
- **Parallel work:** #006, #014, #016 (independent)
- **After #003:** Quick wins (#004, #015, #005)
- **Then:** Architectural (#007, #014)
- **Then:** Cleanup (#017, #013)
- **Finally:** Enforcement (#018 - makes config_ const)

**Timeline estimate:** 4-6 weeks of focused work

**Final state:** ConfigProto constructed once, never mutated, never persisted - fully immutable and enforced at compile-time.

---

## Optional Quality Improvements

**Issue #019: Refactor initialize_context()**
- **Type:** Code quality / Refactoring
- **Mutations:** 0 (no ConfigProto mutations)
- **Complexity:** MEDIUM
- **Effort:** 2-3 hours (one session)
- **Dependencies:** Should complete after #005 and #017 (cleaner function to refactor)
- **Status:** BACKLOG (optional)

**What it does:**
- Refactors initialize_context() god function (~75 lines, too many responsibilities)
- Extracts compute_cache_key() as pure function (testable in isolation)
- Extracts setup_model_metadata() with clear responsibility
- Simplifies main function to ~20 lines
- Eliminates or documents const-cast

**Benefits:**
- ✅ Better testability (unit test cache_key computation)
- ✅ Clearer separation of concerns
- ✅ Easier maintenance
- ✅ Reduced complexity

**Why Optional:**
- Not required for immutable ConfigProto (goal achieved with #003-#018)
- Improves code quality but doesn't change ConfigProto behavior
- Can be done anytime after #005 and #017 complete

**Why Worth Doing:**
- initialize_context() will still be complex after #005 and #017
- Prevents future maintenance debt
- Extracted functions may be useful elsewhere
- Better testability helps prevent regressions

**When to do:**
- After #005 (cache_key to ContextProto) - clearer logic
- After #017 (remove update_config_by_target) - function is shorter
- Then #019 - extract functions, reduce complexity

---

## When Does ConfigProto Become Immutable?

**Answer:** Immediately after construction.

### Current Flow (BEFORE all fixes)

```cpp
// Line 103: Construction
auto config_proto = ConfigProto(config_proto1);  // ← ConfigProto created

// Line 104: MUTATION #1 (Issue #015)
Config::add_version_info(config_proto);  // ← Mutates ConfigProto

// Line 106: Move into ContextProto
ret->context_proto.mutable_config()->Swap(&config_proto);

// Lines 395, 402, 407: MUTATION #2 (Issue #005)
*context->context_proto.mutable_config()->mutable_cache_key() = new_cache_key;  // ← Mutates ConfigProto

// Line 417: MUTATION #3 (Issues #007, #014)
update_config_by_target(*context->mutable_config(), ...);  // ← Mutates ConfigProto
```

**Problem:** ConfigProto mutated during initialization (after construction, before compilation).

### Final Flow (AFTER all fixes)

```cpp
// Line 103: Construction
auto config_proto = ConfigProto(config_proto1);  // ← ConfigProto created

// Line 104: NO MUTATION (Issue #015 fixed)
Config::add_version_info(context_proto);  // ✓ Mutates ContextProto, NOT ConfigProto

// Line 106: Move into ContextProto
ret->context_proto.mutable_config()->Swap(&config_proto);  // ← ConfigProto still unchanged

// Lines 395, 402, 407: NO MUTATION (Issue #005 fixed)
*context->context_proto.mutable_cache_key() = new_cache_key;  // ✓ Mutates ContextProto, NOT ConfigProto

// Line 417: DELETED (Issue #017)
// update_config_by_target() removed entirely  // ✓ No mutation
```

**Result:** ConfigProto has ZERO mutations after construction.

### Immutability Timeline

**Construction (line 103):**
- ConfigProto created from input
- **ConfigProto becomes immutable HERE** ✓

**Initialization (lines 104-417):**
- AFTER fixes: Only ContextProto mutated
- ConfigProto never touched
- ConfigProto remains immutable ✓

**Compilation:**
- ConfigProto used as INPUT (readonly)
- ContextProto populated as OUTPUT
- ConfigProto remains immutable ✓

**Final verification:**
```bash
# After all fixes, search for ConfigProto mutations:
grep -r "mutable_config()" morphizen-core/src/
# Should find: NONE (except during construction at line 106 Swap)
```

### Success Criteria Met

✅ **ConfigProto constructed once** (line 103)
✅ **Never mutated after construction** (all mutations eliminated)
✅ **Never persisted** (Issue #003 - runtime-only)
✅ **Clean INPUT semantics** (readonly configuration)
✅ **Enforced at compile-time** (Issue #018 - const member)

**ConfigProto is immutable immediately after construction.**

### Final Enforcement: Issue #018

**After all mutations eliminated:**

```cpp
class PassContextImp {
  ContextProto context_proto;
  const ConfigProto config_;  // ← CONST enforces immutability at compile-time
};
```

**Issue #018 makes config_ const:**
- **Depends on:** #004, #005, #007, #014, #015, #017 (ALL complete)
- **Changes:** `ConfigProto config_` → `const ConfigProto config_`
- **Effort:** 5 minutes (one-line change)
- **Benefit:** Compiler enforces immutability (prevents future regressions)

**Verification:**
```bash
# Try to mutate (should NOT compile after #018):
context->config_.set_target("xyz");  // ERROR: cannot call on const
context->config_.add_passes();        // ERROR: cannot call on const
```

**If compilation succeeds after making const:** ConfigProto is truly immutable ✓

**Why this matters:**
- Prevents accidental mutations (compile-time safety)
- Self-documenting code (type shows immutability)
- Future-proof (new code cannot mutate)
- Completes strategic goal (immutable by design AND by enforcement)
