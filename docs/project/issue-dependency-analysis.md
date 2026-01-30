<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Issue Dependency Analysis

## Overview

This document analyzes all 11 backlog issues, identifies their groupings, dependencies, and blocking relationships.

**Created:** 2026-01-30
**Status:** Current as of issue #011

---

## Issue Groups

### Group A: Core Config/Context Refactoring
*Foundational changes to ConfigProto and ContextProto separation*

- **#003**: Remove ConfigProto from ContextProto
- **#004**: Remove encryption_key Copying
- **#005**: Move cache_key to ContextProto
- **#007**: Clean Up Target with Two-Path Architecture

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

### Group D: Feature Addition
*New functionality*

- **#001**: Add mmap Support for Embed Mode

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
│  └──────────────────────────────────────────────────┘       │
│                                                              │
│  ┌──────────────────────────────────────────────────┐       │
│  │ #009: Remove xclbin API                          │       │
│  │       (NPU-specific, dead code)                  │       │
│  └──────────────────────────────────────────────────┘       │
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

### Phase 2: Cache System Cleanup

**Priority: MEDIUM** - Tech debt cleanup, no blockers

Can be done in parallel with Phase 1 (except items coordinating with #003):

5. **#006: Remove cache_dir Entirely** (independent)
   - Large cleanup (~200-300 LOC)
   - Enables cleaner #009, #010, #011

6. **#009: Remove xclbin API** (after or with #006)
   - Related to #006 (uses get_log_dir)
   - Can be done before #006, but cleaner after

7. **#010: Remove cache_files Proto Field** (after or with #006)
   - Dead code from cache_dir era
   - Simple removal

8. **#011: Update PassContext Documentation** (after #006)
   - Documentation cleanup
   - References cache_dir in ASCII art

9. **#002: Remove mem_files_** (independent)
   - Improve tar_file_ system
   - Medium complexity (needs TarFile error handling fix)

### Phase 3: Provider Options Cleanup

**Priority: LOW** - Independent cleanups, no blockers

Can be done anytime:

10. **#008: Remove MEP Table** (independent)
    - NPU-specific, doesn't scale
    - Simple removal, docs update

11. **#009: Remove xclbin API** (if not done in Phase 2)
    - NPU-specific, dead code
    - Simple removal

### Phase 4: Feature Addition

**Priority: LOW** - Enhancement, not cleanup

12. **#001: Add mmap Support for Embed Mode** (optional, after #002)
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

### Related (Cleaner Together, Not Blocking)

- **#006 relates to #009** - xclbin uses cache_dir's get_log_dir()
- **#006 relates to #010** - cache_files from cache_dir era
- **#006 relates to #011** - docs reference cache_dir functions
- **#002 enables #001** - better tar_file_ makes mmap easier

### Independent (No Dependencies)

- **#008** - MEP table removal (independent)
- **#002** - mem_files_ removal (independent, but enables #001)
- **#006** - cache_dir removal (independent, but cleanups follow)
- **#001** - mmap feature (independent, but benefits from #002)

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

### Low Risk (Simple Removals)

- **#004** - Simple removal (after #003)
- **#008** - Simple removal (MEP table)
- **#009** - Simple removal (xclbin dead code)
- **#010** - Simple removal (cache_files dead code)
- **#011** - Documentation only

---

## Parallel Work Opportunities

These groups can be worked on in parallel:

**Track 1: Config/Context Refactoring** (sequential within track)
- #003 → #004, #005, #007

**Track 2: Cache System Cleanup** (mostly parallel)
- #006, #002 (independent)
- #009, #010, #011 (after or with #006)

**Track 3: Provider Options Cleanup** (parallel)
- #008 (independent)
- #009 (if not in Track 2)

**Track 4: Feature** (independent)
- #001 (anytime, better after #002)

**Maximum parallelism:**
- Start #003 (Track 1 - foundational)
- Simultaneously start #006 + #002 (Track 2 - independent)
- Simultaneously start #008 (Track 3 - independent)
- After #003 completes → #004, #005, #007 (Track 1 continues)
- After #006 completes → #009, #010, #011 (Track 2 continues)
- After #002 completes → #001 (Track 4 - optional)

---

## Completion Metrics

### By Phase

**Phase 1 Complete:** 4 issues (Config/Context separation done)
- Unblocks: #004, enables cleaner #005 and #007

**Phase 2 Complete:** 5 issues (Cache system cleaned up)
- Benefits: ~300+ LOC removed, cleaner tar_file_ system

**Phase 3 Complete:** 2 issues (Provider options cleaned up)
- Benefits: Simpler provider_options flow

**Phase 4 Complete:** 1 issue (Feature added)
- Benefits: Better memory efficiency for embed mode

**Total:** 11 issues

### By Lines of Code Removed

- **#006**: ~200-300 LOC (cache_dir)
- **#002**: ~200 LOC (mem_files_)
- **#009**: ~40 LOC (xclbin API)
- **#010**: ~15 LOC (cache_files + restore function)
- **#004**: ~10 LOC (encryption_key copying)
- **#008**: Variable (MEP table, docs update)
- **#011**: ASCII art removal (docs only)

**Estimated total cleanup:** ~500+ LOC removed

---

## Next Steps

1. **Confirm priority**: Should we start with #003 (foundational) or #006/#002 (independent cleanups)?
2. **Resource allocation**: Can we work on multiple tracks in parallel?
3. **Create plans**: Enter plan mode for #003 (foundational) or #006 (largest cleanup)?

**Recommendation:** Start with **#003** (Remove ConfigProto from ContextProto) as it unblocks #004 and influences #005 and #007.
