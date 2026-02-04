<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Issue #046: Update Dependencies After Each Issue in /create-issue

## Metadata
- **Type:** Skill Improvement
- **Priority:** MEDIUM
- **Created:** 2026-02-03
- **Dependencies:** Related to #043, #045 (all improve /create-issue skill)

## Description

Add dependency tracking to `/create-issue` skill so it automatically updates issue dependencies after creating each issue. Currently the skill doesn't track relationships between issues, losing valuable context about which issues are related, blocking, or should be coordinated.

## Problem

**Current behavior:**
When `/create-issue` creates a new issue, it:
1. ✅ Creates issue file
2. ✅ Creates plan file (if needed)
3. ✅ Updates backlog.md (adds issue entry)
4. ❌ Does NOT update "Quick dependencies" section
5. ❌ Does NOT add dependency metadata to issue file
6. ✅ Commits

**Missing dependency information:**

The `backlog.md` has a "Quick dependencies" section:
```markdown
**Quick dependencies:**
- **#003 blocks #004** ⚠️ - Must complete #003 first
- **#003 influences #005, #007** - Should coordinate
- **#006 relates to #009, #010, #011** - Cache cleanup group
```

And issues can have dependency metadata:
```markdown
## Metadata
- **Dependencies:** Issue #003 (must complete first)
```

But `/create-issue` doesn't populate either of these.

**Why this is problematic:**

1. **Lost context:** After creating 5-10 issues, relationships are forgotten
2. **Manual work:** User must manually add dependencies later
3. **Fresh context wasted:** When issue just created, AI knows how it relates to previous ones
4. **Session interruption:** If session interrupted, dependency info is lost forever

**Example:**
During this session, created:
- Issue #043: Fix /create-issue implementation during discussion
- Issue #044: Use erase-remove idiom in TarFile
- Issue #045: Add approval workflow to /create-issue

Issues #043 and #045 are clearly related (both improve same skill), but this relationship isn't documented anywhere.

## Solution

After creating issue and plan files (before committing), AI should:

### Step 1: Analyze Relationships
For the newly created issue, determine how it relates to previously created issues in this session and existing backlog issues.

**Relationship types:**
- **Blocks:** New issue must complete before another can start
- **Blocked by:** Another issue must complete before new issue can start
- **Influences:** Should coordinate (shared component, might conflict)
- **Related:** Work on same feature/component (can work independently)

### Step 2: Update "Quick dependencies" Section
Add entries to the dependencies section in backlog.md:

```markdown
**Quick dependencies:**
[existing entries...]
- **#046 relates to #043, #045** - All improve /create-issue skill
```

### Step 3: Update Issue File Metadata
Add dependency info to the new issue's metadata:

```markdown
## Metadata
- **Type:** Skill Improvement
- **Priority:** MEDIUM
- **Created:** 2026-02-03
- **Dependencies:** Related to #043, #045 (all improve /create-issue skill)
```

### Step 4: Commit Together
Include all updates in the same commit:
- Issue file (with dependency metadata)
- Plan file
- Backlog.md (issue entry + dependency update)

**Benefits:**
- ✅ Dependencies captured while context is fresh
- ✅ No manual dependency tracking needed
- ✅ Clear relationships between issues
- ✅ Helps prioritize work (know which issues to group together)
- ✅ Single commit keeps changes atomic

## Plans

- [046-update-dependencies-after-each-issue-in-create-issue-plan.md](../plans/046-update-dependencies-after-each-issue-in-create-issue-plan.md) - Created 2026-02-03

## Notes

**Discovery:** While creating issues #043, #044, #045 for TarFile organizational improvements and `/create-issue` skill fixes, realized the skill wasn't tracking that #043 and #045 are related (both improve same skill).

**Related to:**
- Issue #043: Prevent implementation during discussion
- Issue #045: Add approval workflow

All three (#043, #045, #046) improve the `/create-issue` skill workflow.
