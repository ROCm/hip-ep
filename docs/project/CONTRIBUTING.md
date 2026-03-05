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

---

## Issue Quality

Keep issues lean and meaningful - avoid "empty talking".

### Only Include Sections with Actual Content

- ❌ Don't add "Plans: _No plans yet_"
- ❌ Don't add "Sessions: _To be filled in_"
- ❌ Don't add "Related PRs: _None yet_"
- ✅ Add Plans/Sessions/PRs sections ONLY when they have real content

### Write Specific, Concrete Content

- ❌ Vague: "This needs to be fixed"
- ✅ Specific: "buffersInterfere() unconditionally returns true at line 281"
- ❌ Generic: "Should be discussed"
- ✅ Actionable: "Replace return true with interval overlap check using sequential op indices"

### Use Evidence Section Effectively

- ✅ List exact code locations: `file.cpp:123 - What happens here`
- ✅ Helps reviewers verify the problem exists
- ✅ Makes implementation easier to scope

---

## Issue Lifecycle

1. **Create:** Add `issues/NNN-name.md` and optional `plans/NNN-name-plan.md`
2. **Track:** Update issue file with plans, sessions, progress
3. **Complete:** Move to "Recently Completed" section in backlog, DELETE the detailed file
4. **Archive:** Keep max 5 in "Recently Completed", delete older ones

**Why delete detailed files?**
- Completed work is documented in merged PRs
- Git history preserves deleted file if needed
- Keeps backlog focused on current/future work

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
2. Create `issues/NNN-name.md` with all required sections
3. Create `plans/NNN-name-plan.md` if detailed plan requested
4. Add link to backlog.md
5. Commit and push to fork
