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
- ❌ Don't add "Notes: _To be filled in_"
- ✅ Add Plans/Notes sections ONLY when they have real content

### Write Specific, Concrete Content

- ❌ Vague: "This needs to be fixed"
- ✅ Specific: "lib/Dialect/IR/*.td duplicates include/hip/Dialect/IR/*.td but with diverged content"
- ❌ Generic: "Should be discussed"
- ✅ Actionable: "Delete lib/*.td, merge content improvements into include/*.td"

### Use Evidence Section Effectively

- ✅ List exact code locations: `file.cpp:123 - What happens here`
- ✅ Helps reviewers verify the problem exists
- ✅ Makes implementation easier to scope

---

## Template Usage

When creating a new issue from TEMPLATE.md:

1. Copy the template to `NNN-name.md`
2. Fill in required sections: Metadata, Description, Problem, Solution
3. **Delete optional sections** that you don't have content for yet
4. Add optional sections later when you have meaningful content

---

## Issue Lifecycle

1. **Create:** Copy `issues/TEMPLATE.md` to `issues/NNN-name.md`
2. **Track:** Update issue file with plans, sessions, progress
3. **Complete:** Move to completed-issues.md, DELETE the detailed file
4. **Archive:** Keep record in completed-issues.md

**Why delete detailed files?**
- Completed work is documented in merged PRs
- Git history preserves deleted file if needed
- Keeps backlog focused on current/future work

**To recover deleted issue file:**
```bash
git log --all --full-history -- "docs/project/issues/NNN-*.md"
git show <commit-hash>:docs/project/issues/NNN-name.md
```
