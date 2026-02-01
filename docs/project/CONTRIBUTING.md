<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Contributing to Project Issues

Guidelines for creating and maintaining high-quality issues in the backlog.

---

## Issue Granularity

Split complex tasks into focused, actionable issues.

**When to create a new issue:**
- Problem is complex and has multiple sub-problems
- Solution requires multiple design decisions
- Different parts can be implemented independently
- Implementation would take multiple sessions

**When to expand existing issue:**
- Direct extension of existing scope
- Same root cause, same solution approach
- Cannot be separated without losing context

**Example:**
- ❌ BAD: One issue "Fix ConfigProto Mutability" (too broad)
- ✅ GOOD: Issues #003-#015 each addressing specific mutation categories

---

## Issue Quality

Keep issues lean and meaningful - avoid "empty talking".

### Only Include Sections with Actual Content

- ❌ Don't add "Plans: _No plans yet_"
- ❌ Don't add "Sessions: _To be filled in_"
- ❌ Don't add "Related PRs: _None yet_"
- ✅ Add Plans/Sessions/PRs sections ONLY when they have real content

### Make Notes Section Add Value

- ❌ "Part of ConfigProto mutation analysis" (obvious from title)
- ❌ "This is a design flaw" (already stated in Description)
- ✅ Explain root cause, relationships to other issues, or design trade-offs
- ✅ Provide context NOT already in Description/Problem/Solution

### Write Specific, Concrete Content

- ❌ Vague: "This needs to be fixed"
- ✅ Specific: "Swapping exists because ConfigProto is INPUT but persisted in ContextProto (OUTPUT)"
- ❌ Generic: "Should be discussed"
- ✅ Actionable: "Three possible solutions: (1) X, (2) Y, (3) Z"

### Use Evidence Section Effectively

- ✅ List exact code locations: `file.cpp:123 - What happens here`
- ✅ Helps reviewers verify the problem exists
- ✅ Makes implementation easier to scope

---

## Template Usage

When creating a new issue from TEMPLATE.md:

1. Copy the template to `NNN-name.md`
2. Fill in required sections: Metadata, Description, Problem, Solution, Evidence
3. **Delete optional sections** that you don't have content for yet:
   - If no plans exist, delete Plans section
   - If no sessions yet, delete Sessions section
   - If no PRs yet, delete Related PRs section
   - If Notes would just repeat Description, delete Notes section
4. Add optional sections later when you have meaningful content

**Result:** Clean, focused issues with only valuable information.

---

## Examples

### Bad Issue (empty talking)

```markdown
## Description
Need to fix the cache system.

## Plans
_No plans yet._

## Sessions
_To be filled in._

## Notes
Part of cache cleanup.
```

### Good Issue (meaningful content)

```markdown
## Description
Remove cache_dir entirely - obsolete disk-based cache system (~200-300 LOC).

## Problem
cache_dir was designed for disk-based caching, but new tar_file_ system
makes it obsolete. All cache operations now use tar_file_ directly.

## Solution
Remove all cache_dir code: get_log_dir(), cache_dir proto field, copying logic.

## Evidence
- cache_dir.cpp:67 - cache_dir computation (to be removed)
- pass_context_imp.cpp:1266 - cache_dir copying (to be removed)

## Notes
After removal, xclbin functions (Issue #009) will break - they call get_log_dir().
Coordinate removal or do #009 first.
```

---

## Issue Lifecycle

1. **Create:** Copy `issues/TEMPLATE.md` to `issues/NNN-name.md`
2. **Track:** Update issue file with plans, sessions, progress
3. **Complete:** Move to "Recently Completed" section, DELETE the detailed file
4. **Archive:** Keep max 5 in "Recently Completed", delete older ones

**Why delete detailed files?**
- Completed work is documented in merged PRs
- Git history preserves deleted file if needed
- Keeps backlog focused on current/future work
- Reduces file clutter

**To recover deleted issue file:**
```bash
git log --all --full-history -- "docs/project/issues/NNN-*.md"
git show <commit-hash>:docs/project/issues/NNN-name.md
```

---

## Claude Code Automation

When asked to complete an issue, Claude will automatically:

1. Verify on feature branch (not main)
2. Update backlog.md (move to Recently Completed)
3. Delete the detailed issue file: `git rm docs/project/issues/NNN-*.md`
4. Commit: `git commit -m "docs: complete issue #NNN"`
5. Push to fork

When creating new issues, Claude will:
1. Determine next issue number
2. Copy TEMPLATE.md to new numbered file
3. Fill in all required sections
4. Delete empty optional sections
5. Only include Notes if it adds value
6. Add link to backlog.md

**Quality checklist:**
- ✅ Concrete problem statement with code evidence
- ✅ Clear solution with benefits and approach
- ✅ No "No plans yet" or "To be filled in" placeholders
- ✅ Notes add context not found elsewhere, or section deleted
- ✅ Only meaningful content, no empty talking
