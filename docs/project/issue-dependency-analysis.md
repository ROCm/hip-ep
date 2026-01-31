<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Issue Dependency Analysis

## Overview

This document analyzes all 20 backlog issues, identifies their groupings, dependencies, and blocking relationships.

**Created:** 2026-01-30
**Updated:** 2026-01-31 (added issues #012-#020)
**Status:** Current as of issue #020

---

## Issue Groups

### Group A: Core Config/Context Refactoring
*Foundational changes to ConfigProto and ContextProto separation*

- **#003**: Remove ConfigProto from ContextProto
- **#004**: Remove encryption_key Copying
- **#005**: Move cache_key to ContextProto
- **#007**: Clean Up Target with Two-Path Architecture
- **#012**: session_configs Swapping (resolved by #003)
- **#013**: provider_options Aggregation (cleanup after #003, #008, #009)
- **#014**: Dynamic Pass Registration (ConfigProto immutability)
- **#015**: Configuration Initialization (move version info to ContextProto)
- **#017**: Remove update_config_by_target() (cleanup after #007 and #014)
- **#018**: Make ConfigProto const Member (final enforcement after all mutations eliminated)
- **#019**: Refactor initialize_context() (god function cleanup - optional quality improvement)

### Group B: Cache System Cleanup
*Removing obsolete cache_dir system and dead code*

- **#002**: Remove mem_files_ - Always Create tar_file_
- **#006**: Remove cache_dir Entirely
- **#010**: Remove cache_files - Dead Code
- **#011**: Update PassContext Header Documentation

### Group C: Provider Options Cleanup
*Removing NPU-specific and obsolete provider_options injection*

- **#008**: MEP Table Cleanup
- **#009**: TargetProto Provider Options Injection Cleanup

### Group D: Global State Cleanup
*Eliminating global state mutations and improving testability*

- **#016**: Remove dirty_hack_for_model_clone_external_data_threshold

### Group E: Feature Addition
*New functionality*

- **#001**: Add mmap Support for Embed Mode

### Group F: General Code Cleanup
*Independent cleanup - dead code removal, code quality improvements*

- **#020**: Remove suffix_counter Dead Code (never written, always false condition)

---

## Dependency Graph

```
┌──────────────────────────────────────────────────────────────┐
│                      FOUNDATIONAL                            │
│                                                              │
│  ┌──────────────────────────────────────────────────┐       │
│  │ #003: Remove ConfigProto from ContextProto       │       │
│  │       (Make ConfigProto runtime-only)            │       │
│  └──────────────┬───────────────────────────────────┘       │
│                 │                                            │
│                 │ BLOCKS (must complete first)               │
│                 │                                            │
│  ┌──────────────▼───────────────────────────────────┐       │
│  │ #004: Remove encryption_key Copying              │       │
│  │       (encryption_key won't exist in proto)      │       │
│  └──────────────────────────────────────────────────┘       │
│                                                              │
│                 │ INFLUENCES (should coordinate)             │
│                 │                                            │
│  ┌──────────────▼───────────────────────────────────┐       │
│  │ #005: Move cache_key to ContextProto             │       │
│  │       (cache_key needs persistence)              │       │
│  └──────────────────────────────────────────────────┘       │
│                                                              │
│                 │ INFLUENCES (ConfigProto immutability)      │
│                 │                                            │
│  ┌──────────────▼───────────────────────────────────┐       │
│  │ #007: Target Two-Path Architecture               │       │
│  │       (ConfigProto readonly design)              │       │
│  └──────────────────────────────────────────────────┘       │
│                                                              │
│                 │ RESOLVES (ConfigProto runtime-only)        │
│                 │                                            │
│  ┌──────────────▼───────────────────────────────────┐       │
│  │ #012: session_configs Swapping                   │       │
│  │       (no separate work - resolved by #003)      │       │
│  └──────────────────────────────────────────────────┘       │
│                                                              │
│                 │ RELATED (ConfigProto immutability)         │
│                 │                                            │
│  ┌──────────────▼───────────────────────────────────┐       │
│  │ #014: Dynamic Pass Registration                  │       │
│  │       (compute passes on-demand)                 │       │
│  └──────────────────────────────────────────────────┘       │
│                                                              │
│  ┌──────────────────────────────────────────────────┐       │
│  │ #015: Configuration Initialization               │       │
│  │       (move version info to ContextProto)        │       │
│  └──────────────────────────────────────────────────┘       │
│                                                              │
│                 │ DEPENDS ON (#007 AND #014)                 │
│                 │                                            │
│  ┌──────────────▼───────────────────────────────────┐       │
│  │ #017: Remove update_config_by_target()           │       │
│  │       (cleanup after #007 and #014)              │       │
│  └──────────────┬───────────────────────────────────┘       │
│                 │                                            │
│                 │ DEPENDS ON (ALL mutations eliminated)      │
│                 │ (#004, #005, #007, #014, #015, #017)      │
│                 │                                            │
│  ┌──────────────▼───────────────────────────────────┐       │
│  │ #018: Make ConfigProto const Member              │       │
│  │       (final enforcement - compile-time)         │       │
│  └──────────────────────────────────────────────────┘       │
└──────────────────────────────────────────────────────────────┘


┌──────────────────────────────────────────────────────────────┐
│                    CACHE CLEANUP                             │
│                                                              │
│  ┌──────────────────────────────────────────────────┐       │
│  │ #006: Remove cache_dir Entirely                  │       │
│  │       (~200-300 LOC obsolete system)             │       │
│  └──────────────┬───────────────────────────────────┘       │
│                 │                                            │
│                 │ RELATED (uses get_log_dir from cache_dir)  │
│                 │                                            │
│  ┌──────────────▼───────────────────────────────────┐       │
│  │ #009: Remove xclbin API                          │       │
│  │       (calls get_log_dir, will break after #006) │       │
│  └──────────────────────────────────────────────────┘       │
│                                                              │
│                 │ RELATED (cache_files from old system)      │
│                 │                                            │
│  ┌──────────────▼───────────────────────────────────┐       │
│  │ #010: Remove cache_files Proto Field             │       │
│  │       (dead code from cache_dir era)             │       │
│  └──────────────────────────────────────────────────┘       │
│                                                              │
│                 │ RELATED (docs reference cache_dir)         │
│                 │                                            │
│  ┌──────────────▼───────────────────────────────────┐       │
│  │ #011: Update PassContext Documentation           │       │
│  │       (remove outdated ASCII art)                │       │
│  └──────────────────────────────────────────────────┘       │
│                                                              │
│  ┌──────────────────────────────────────────────────┐       │
│  │ #002: Remove mem_files_                          │       │
│  │       (always create tar_file_)                  │       │
│  └──────────────────────────────────────────────────┘       │
│                 │                                            │
│                 │ ENABLES (better tar_file_ foundation)      │
│                 │                                            │
│  ┌──────────────▼───────────────────────────────────┐       │
│  │ #001: Add mmap Support for Embed Mode            │       │
│  │       (enhance tar_file_ with mmap)              │       │
│  └──────────────────────────────────────────────────┘       │
└──────────────────────────────────────────────────────────────┘


┌──────────────────────────────────────────────────────────────┐
│              PROVIDER OPTIONS CLEANUP                        │
│                                                              │
│  ┌──────────────────────────────────────────────────┐       │
│  │ #008: Remove MEP Table                           │       │
│  │       (NPU-specific, doesn't scale)              │       │
│  └──────────────┬───────────────────────────────────┘       │
│                 │                                            │
│  ┌──────────────▼───────────────────────────────────┐       │
│  │ #009: Remove xclbin API                          │       │
│  │       (NPU-specific, dead code)                  │       │
│  └──────────────┬───────────────────────────────────┘       │
│                 │                                            │
│                 │ DEPENDS ON (#003, #008, #009)              │
│                 │                                            │
│  ┌──────────────▼───────────────────────────────────┐       │
│  │ #013: provider_options Aggregation               │       │
│  │       (cleanup after prerequisites complete)     │       │
│  └──────────────────────────────────────────────────┘       │
└──────────────────────────────────────────────────────────────┘


┌──────────────────────────────────────────────────────────────┐
│              GLOBAL STATE CLEANUP                            │
│                                                              │
│  ┌──────────────────────────────────────────────────┐       │
│  │ #016: Remove dirty_hack_for_model_clone_threshold│       │
│  │       (eliminate global state mutation)          │       │
│  └──────────────────────────────────────────────────┘       │
│                                                              │
│  INDEPENDENT - can be done anytime                           │
│  RELATED to #014 (both involve pass-dependent config)        │
└──────────────────────────────────────────────────────────────┘


┌──────────────────────────────────────────────────────────────┐
│              GENERAL CODE CLEANUP                            │
│                                                              │
│  ┌──────────────────────────────────────────────────┐       │
│  │ #020: Remove suffix_counter Dead Code            │       │
│  │       (read metadata never written)              │       │
│  └──────────────────────────────────────────────────┘       │
│                                                              │
│  INDEPENDENT - can be done anytime                           │
│  RELATED to #019 (both simplify initialize_context)          │
└──────────────────────────────────────────────────────────────┘
```

---

## Blocking Relationships

### Critical Path (Must Complete in Order)

**#003 → #004** (Hard Block)
- **Issue #004 BLOCKED by #003**
- Reason: encryption_key field won't exist in ConfigProto proto after #003
- #004 explicitly states: "Dependencies: Issue #003 (must complete first)"
- Cannot remove encryption_key copying until ConfigProto is removed from ContextProto

### Coordination Required (Should Work Together)

**#003 ⇄ #005** (Soft Dependency)
- **Should coordinate or do #005 after #003**
- Reason: cache_key currently in ConfigProto, needs to move to ContextProto
- #005 states: "Dependencies: Should be done after or with Issue #003"
- After #003, ConfigProto won't be persisted → cache_key loses persistence → must move

**#003 ⇄ #007** (Design Coordination)
- **Should coordinate** (ConfigProto immutability)
- Reason: Both issues assume ConfigProto becomes immutable/runtime-only
- #007 states: "Dependencies: Coordinated with Issue #003 (ConfigProto immutability)"
- #007's design depends on ConfigProto being readonly

**#003 → #012** (Resolves)
- **#012 automatically resolved by #003**
- Reason: session_configs swapping exists because ConfigProto is persisted
- After #003, ConfigProto is runtime-only → no swapping needed
- #012 is a design discussion issue, not a separate implementation

**#003, #008, #009 → #013** (Prerequisites)
- **#013 depends on #003, #008, #009 completing first**
- Reason: #013 is cleanup task after prerequisites eliminate sources
- After #003: No swapping needed
- After #008: MEP table source gone
- After #009: TargetProto injection source gone
- Then #013: Remove obsolete code (provider_option_from_cache_, etc.)

**#007, #014 → #017** (Prerequisites)
- **#017 depends on BOTH #007 AND #014 completing first**
- Reason: update_config_by_target() does two things resolved by these issues
- After #007: TargetProto copying eliminated (raw pointer)
- After #014: Dynamic pass registration eliminated (compute_effective_passes)
- Then #017: Remove obsolete update_config_by_target() function (becomes empty)

**#004, #005, #007, #014, #015, #017 → #018** (Final Enforcement)
- **#018 depends on ALL mutation-fixing issues completing first**
- Reason: Making config_ const requires zero mutations to ConfigProto
- After #004: encryption_key mutations eliminated
- After #005: cache_key mutations eliminated
- After #007: target mutations eliminated
- After #014: pass mutations eliminated
- After #015: version info mutations eliminated
- After #017: update_config_by_target() removed
- Then #018: Make config_ const (compile-time enforcement of immutability)

### Related (No Blocking, But Connected)

**#006 → #009** (Related, Not Blocking)
- **#009 related to #006** (xclbin uses cache_dir)
- Reason: xclbin functions call `get_log_dir()` from obsolete cache_dir system
- #009 states: "Dependencies: Related to Issue #006 (cache_dir removal)"
- #009 notes: "After Issue #006, get_log_dir() will be removed"
- **Not blocking** - #009 can be done before #006 (removes dead code anyway)
- **Cleaner together** - Both remove obsolete cache_dir remnants

**#006 → #010** (Related)
- **#010 related to #006** (cache_files from old system)
- Reason: cache_files proto field is dead code from cache_dir era
- Both remove remnants of obsolete disk-based cache system

**#006 → #011** (Related)
- **#011 related to #006** (docs reference cache_dir)
- Reason: ASCII art shows cache_files_to_dir (doesn't exist, legacy from cache_dir)
- Documentation cleanup after cache_dir removal

**#002 → #001** (Enables)
- **#001 enabled by #002** (better tar_file_ foundation)
- Reason: #002 ensures tar_file_ always created, #001 adds mmap to tar_file_
- #001 works on tar_file_ system that #002 improves
- **Not blocking** - #001 can be done without #002
- **Better with #002** - Simpler tar_file_ logic makes mmap easier

**#003 ⇄ #014** (Related)
- **#014 related to #003** (ConfigProto immutability goal)
- Reason: #014 eliminates ConfigProto mutations from dynamic pass registration
- Both work toward immutable ConfigProto
- **Not blocking** - #014 can be done independently

**#003 ⇄ #015** (Related)
- **#015 related to #003** (moving things out of ConfigProto)
- Reason: #015 moves version info from ConfigProto to ContextProto
- Similar pattern to other fields moving out of ConfigProto
- **Not blocking** - #015 can be done independently

**#014 ⇄ #016** (Related)
- **#016 related to #014** (pass-dependent configuration)
- Reason: Both involve configuration that depends on passes
- #014: Pass selection at runtime (compute effective passes)
- #016: Threshold based on passes (compute threshold)
- **Pattern:** Configuration depending on passes should be computed, not mutated globally

**#007, #014 → #017** (Cleanup After)
- **#017 is cleanup after both #007 and #014** (removes obsolete function)
- Reason: update_config_by_target() becomes empty after prerequisites
- #007 eliminates TargetProto copying (line 143)
- #014 eliminates dynamic pass registration (lines 141-142)
- #017 removes the now-empty function (simple deletion)

**#005, #017 → #019** (Optional Quality Improvement)
- **#019 related to #005 and #017** (code quality refactoring)
- Reason: initialize_context() is cleaner after #005 and #017 complete
- After #005: cache_key mutations move to ContextProto (clearer logic)
- After #017: update_config_by_target() removed (~15 lines shorter)
- Then #019: Extract cache_key computation, reduce complexity
- **Not blocking** - #019 is optional quality improvement
- **Not required for immutable ConfigProto** - ConfigProto will be immutable without #019
- **Benefits:** Better testability, clearer code, easier maintenance

**#019 ⇄ #020** (Related - Both Simplify initialize_context)
- **#020 related to #019** (both cleanup initialize_context)
- Reason: Both issues simplify initialize_context() function
- #019: Extract cache_key computation (~23 lines → separate function)
- #020: Remove suffix_counter dead code (4 lines → delete)
- Together: initialize_context() becomes ~26 lines (was ~52 after #005, #006, #017)
- **Independent** - Can be done in any order or separately
- **Benefits:** Cleaner, simpler, more maintainable initialize_context()

---

## Recommended Implementation Order

### Phase 1: Foundational Refactoring (Config/Context Separation)

**Priority: HIGH** - Unblocks other issues

1. **#003: Remove ConfigProto from ContextProto** ⚠️ **START HERE**
   - Foundational change
   - Blocks #004, influences #005 and #007
   - Medium complexity

2. **#005: Move cache_key to ContextProto** (immediately after #003)
   - Must move cache_key before it loses persistence
   - Related to #003 changes

3. **#004: Remove encryption_key Copying** (after #003)
   - Blocked by #003
   - Simple cleanup once #003 done

4. **#007: Clean Up Target with Two-Path Architecture** (after #003)
   - Benefits from ConfigProto immutability from #003
   - Medium complexity, comprehensive redesign

5. **#012: session_configs Swapping** (automatically resolved by #003)
   - No separate implementation needed
   - Resolved when #003 makes ConfigProto runtime-only

6. **#015: Configuration Initialization** (can be done with or after #003)
   - Move version info from ConfigProto to ContextProto
   - Simple refactoring, related to #003 cleanup

7. **#014: Dynamic Pass Registration** (independent, related to #003 goal)
   - Architectural redesign
   - Compute effective passes on-demand
   - Can be done in parallel with other Phase 1 issues

8. **#017: Remove update_config_by_target()** (after #007 AND #014)
   - Simple cleanup (15 minutes)
   - Remove obsolete function
   - Must wait for both #007 and #014 to complete

9. **#018: Make ConfigProto const Member** (after ALL mutations eliminated)
   - Final enforcement (5 minutes)
   - Change config_ to const config_
   - Must wait for #004, #005, #007, #014, #015, #017 to complete
   - Compile-time verification of immutability

### Phase 2: Cache System Cleanup

**Priority: MEDIUM** - Tech debt cleanup, no blockers

Can be done in parallel with Phase 1 (except items coordinating with #003):

10. **#006: Remove cache_dir Entirely** (independent)
   - Large cleanup (~200-300 LOC)
   - Enables cleaner #009, #010, #011

11. **#009: Remove xclbin API** (after or with #006)
    - Related to #006 (uses get_log_dir)
    - Can be done before #006, but cleaner after

12. **#010: Remove cache_files Proto Field** (after or with #006)
    - Dead code from cache_dir era
    - Simple removal

13. **#011: Update PassContext Documentation** (after #006)
    - Documentation cleanup
    - References cache_dir in ASCII art

14. **#002: Remove mem_files_** (independent)
    - Improve tar_file_ system
    - Medium complexity (needs TarFile error handling fix)

### Phase 3: Provider Options Cleanup

**Priority: MEDIUM** - After Phase 1 prerequisites

15. **#008: Remove MEP Table** (independent)
    - NPU-specific, doesn't scale
    - Simple removal, docs update

16. **#009: Remove xclbin API** (if not done in Phase 2)
    - NPU-specific, dead code
    - Simple removal

17. **#013: provider_options Aggregation Cleanup** (after #003, #008, #009)
    - Remove obsolete code after prerequisites
    - Remove provider_option_from_cache_
    - Simplify get_all_provider_options()

### Phase 4: Global State Cleanup

**Priority: MEDIUM** - Independent refactoring

18. **#016: Remove dirty_hack_for_model_clone_external_data_threshold** (independent)
    - Eliminate global state mutation
    - Compute threshold as pure function
    - Related to #014 (pass-dependent configuration)

### Phase 5: Feature Addition

**Priority: LOW** - Enhancement, not cleanup

19. **#001: Add mmap Support for Embed Mode** (optional, after #002)
    - New feature, not tech debt
    - Benefits from #002 (better tar_file_ foundation)
    - Lower priority than cleanups

---

## Summary of Blocking Relationships

### Hard Blocks (Must Complete First)

- **#003 blocks #004** - encryption_key won't exist in proto after #003

### Soft Dependencies (Should Coordinate)

- **#003 influences #005** - cache_key needs new home after ConfigProto removal
- **#003 influences #007** - target design assumes ConfigProto immutability
- **#003 resolves #012** - session_configs swapping eliminated when ConfigProto runtime-only
- **#003, #008, #009 → #013** - provider_options cleanup depends on these completing first
- **#007, #014 → #017** - update_config_by_target() removal depends on both completing first

### Related (Cleaner Together, Not Blocking)

- **#006 relates to #009** - xclbin uses cache_dir's get_log_dir()
- **#006 relates to #010** - cache_files from cache_dir era
- **#006 relates to #011** - docs reference cache_dir functions
- **#002 enables #001** - better tar_file_ makes mmap easier
- **#003 relates to #014** - both work toward ConfigProto immutability
- **#003 relates to #015** - moving things out of ConfigProto
- **#014 relates to #016** - both involve pass-dependent configuration
- **#007, #014 → #017** - update_config_by_target() cleanup after prerequisites

### Independent (No Dependencies)

- **#008** - MEP table removal (independent)
- **#002** - mem_files_ removal (independent, but enables #001)
- **#006** - cache_dir removal (independent, but cleanups follow)
- **#001** - mmap feature (independent, but benefits from #002)
- **#014** - Dynamic pass registration (independent, related to #003 goal)
- **#015** - Configuration initialization (independent, related to #003)
- **#016** - Model clone threshold hack (independent, related to #014)

---

## Risk Analysis

### High Risk (Breaks Others if Not Coordinated)

- **#003** - Foundational change, affects #004, #005, #007
  - Risk: Breaking encryption_key, cache_key, target logic
  - Mitigation: Complete #003 first, then do #004, #005, #007

### Medium Risk (Large Changes)

- **#006** - Removes ~200-300 LOC
  - Risk: Breaking get_log_dir() callers
  - Mitigation: Search for all get_log_dir() usage before removal

- **#007** - Major redesign (two-path architecture)
  - Risk: Breaking target auto-discovery
  - Mitigation: Comprehensive testing, update docs

- **#002** - Fixing TarFile error handling
  - Risk: tmpfile() failures not handled
  - Mitigation: Add proper error handling before removing mem_files_

- **#014** - Dynamic pass registration redesign
  - Risk: Breaking pass selection logic
  - Mitigation: Comprehensive testing, ensure compute_effective_passes() handles all cases

### Low Risk (Simple Removals)

- **#004** - Simple removal (after #003)
- **#008** - Simple removal (MEP table)
- **#009** - Simple removal (xclbin dead code)
- **#010** - Simple removal (cache_files dead code)
- **#011** - Documentation only
- **#012** - No implementation (resolved by #003)
- **#013** - Simple cleanup after prerequisites (remove obsolete code)
- **#015** - Simple move (version info to ContextProto)
- **#016** - Refactor function to pure function (testable, low risk)

---

## Parallel Work Opportunities

These groups can be worked on in parallel:

**Track 1: Config/Context Refactoring** (sequential within track)
- #003 → #004, #005, #007, #012 (resolved), #014, #015
- #007, #014 → #017 (cleanup after both complete)
- #004, #005, #007, #014, #015, #017 → #018 (final enforcement after ALL mutations eliminated)
- #013 (cleanup after #003, #008, #009)

**Track 2: Cache System Cleanup** (mostly parallel)
- #006, #002 (independent)
- #009, #010, #011 (after or with #006)

**Track 3: Provider Options Cleanup** (parallel)
- #008 (independent)
- #009 (if not in Track 2)

**Track 4: Global State Cleanup** (parallel)
- #016 (independent)

**Track 5: Feature** (independent)
- #001 (anytime, better after #002)

**Maximum parallelism:**
- Start #003 (Track 1 - foundational)
- Simultaneously start #006 + #002 (Track 2 - independent)
- Simultaneously start #008 (Track 3 - independent)
- Simultaneously start #016 (Track 4 - independent)
- After #003 completes → #004, #005, #007, #014, #015 (Track 1 continues)
- After #007 + #014 complete → #017 (Track 1 cleanup)
- After #004 + #005 + #007 + #014 + #015 + #017 complete → #018 (Track 1 final enforcement)
- After #003 + #008 + #009 complete → #013 (Track 1 cleanup)
- After #006 completes → #009, #010, #011 (Track 2 continues)
- After #002 completes → #001 (Track 5 - optional)

---

## Completion Metrics

### By Phase

**Phase 1 Complete:** 9 issues (Config/Context separation done)
- #003, #004, #005, #007, #012 (resolved), #014, #015, #017, #018
- Unblocks: #004, #013, #017, #018, enables cleaner #005, #007, #015
- Final result: ConfigProto immutable (compile-time enforced)

**Phase 2 Complete:** 5 issues (Cache system cleaned up)
- #006, #009, #010, #011, #002
- Benefits: ~300+ LOC removed, cleaner tar_file_ system

**Phase 3 Complete:** 3 issues (Provider options cleaned up)
- #008, #009, #013
- Benefits: Simpler provider_options flow

**Phase 4 Complete:** 1 issue (Global state cleanup)
- #016
- Benefits: No global state mutations, better testability

**Phase 5 Complete:** 1 issue (Feature added)
- #001
- Benefits: Better memory efficiency for embed mode

**Total:** 18 issues (includes #012 which is resolved by #003)

### By Lines of Code Removed

- **#006**: ~200-300 LOC (cache_dir)
- **#002**: ~200 LOC (mem_files_)
- **#009**: ~40 LOC (xclbin API)
- **#010**: ~15 LOC (cache_files + restore function)
- **#004**: ~10 LOC (encryption_key copying)
- **#008**: Variable (MEP table, docs update)
- **#011**: ASCII art removal (docs only)
- **#012**: No code changes (resolved by #003)
- **#013**: ~20 LOC (provider_option_from_cache_, swapping logic)
- **#014**: Refactoring (not deletion - replace mutations with compute)
- **#015**: ~10 LOC (move, not delete - version info to ContextProto)
- **#016**: ~10 LOC (replace dirty hack with pure function)
- **#017**: ~15 LOC (remove obsolete update_config_by_target function)
- **#018**: ~1 LOC (add const keyword - enforcement, not cleanup)

**Estimated total cleanup:** ~565+ LOC removed/refactored + compile-time enforcement

---

## Next Steps

1. **Confirm priority**: Should we start with #003 (foundational) or work on independent issues?
2. **Resource allocation**: Can we work on multiple tracks in parallel?
3. **Create plans**: Enter plan mode for chosen issue(s)?

**Recommendations:**

**Foundational approach (sequential):**
- Start with **#003** (Remove ConfigProto from ContextProto)
- Unblocks: #004, #012 (auto-resolved), #013
- Influences: #005, #007, #015

**Parallel approach (maximum throughput):**
- **Track 1:** #003 (foundational)
- **Track 2:** #006 (largest cleanup ~200-300 LOC) + #002
- **Track 3:** #008 (independent)
- **Track 4:** #014 + #016 (independent, architectural improvements)

**Quick wins (independent, low risk):**
- #015 (Configuration Initialization - simple move)
- #016 (Model Clone Threshold - refactor to pure function)
- #008 (MEP Table - simple removal)
