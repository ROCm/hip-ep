<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->

# Issue Resolution Workflow

Complete lifecycle for identifying, implementing, and merging improvements to the codebase using automated skills.

## Overview

The issue resolution workflow integrates three skills to automate the complete journey from discovering problems to merged PRs:

```
/create-issue → /fix-issue → Author Review → /resolve-ci → Merged
```

**Automation benefits:**
- Token-efficient: Skills load only when invoked
- Enforces git workflow (feature branches, draft PRs, pre-commit)
- Reduces manual repetition (backlog updates, PR formatting, CI monitoring)
- Clear separation: exploration, implementation, finalization

---

## Phase 1: Issue Discovery & Documentation

**Skill:** `/create-issue`

**Purpose:** Explore codebase interactively and create issues incrementally.

**Workflow:**
1. Start exploration session: `/create-issue`
2. Identify high-level topics (e.g., "TarFile cleanup", "API improvements")
3. Drill down into sub-topics hierarchically
4. For each finding:
   - Explore code thoroughly
   - Discuss with user (ask questions, confirm understanding)
   - Create issue file + optional plan file
   - Update backlog.md
5. Single branch holds all issues created in session
6. Draft PR tracks all documentation additions

**Output:**
- Issue files: `docs/project/issues/NNN-name.md`
- Plan files (optional): `docs/project/plans/NNN-name-plan.md`
- Updated backlog: `docs/project/backlog.md`
- Draft PR with all issue documentation

**When done:** Issues are documented and ready for implementation.

---

## Phase 2: Implementation

**Skill:** `/fix-issue`

**Purpose:** Implement a backlog issue from selection to code completion.

**Workflow:**

### Step 1: Select issue
- Skill recommends highest-priority, unblocked issue
- User can request alternatives or select specific issue

### Step 2: Review and confirm
- Show full issue content (problem, solution, files)
- Show plan status: None / Single plan / Multiple plans
- Ask user: Proceed / Cancel
- **Exit here if user cancels** (no workspace created yet)

### Step 3: Create workspace
- Create feature branch: `feature/issue-NNN-name`
- Add "Started:" timestamp to issue file
- Commit and push to fork
- Create draft PR

### Step 4: Implement
- Offer: Auto-implement / Manual / Cancel
- **Auto-implement:** Read plan, execute steps, build, test, commit, push
- **Manual:** User implements, runs `/fix-issue --finalize #NNN` later

**Output:**
- Feature branch with implementation commits
- Draft PR (not ready for review yet)

**When done:** Implementation is complete, PR is DRAFT. Author review needed before finalization.

---

## Phase 3: Author Review

**Not automated** - Critical author decision point.

**Steps:**
1. Review the implementation in draft PR
2. Check code quality, logic, adherence to plan
3. Test locally if needed
4. Decide:
   - ✅ Approve → Proceed to Phase 4 (`/resolve-ci`)
   - ❌ Request changes → Continue working on branch, push updates
   - ⏸️  Defer → Leave draft PR for later

**Why human review matters:**
- Verify implementation correctness
- Ensure code quality and maintainability
- Catch issues before CI runs
- Final check before marking PR ready

---

## Phase 4: Finalization & CI

**Skill:** `/resolve-ci`

**Purpose:** Finalize PR, monitor CI, handle failures, merge when ready.

**Workflow:**

### Phase 0: PR Finalization
1. Read issue file for context
2. Craft PR title: `Issue #NNN: <type>: <description>`
3. Write PR description (problem, solution, changes)
4. Update `completed-issues.md` (add to current month)
5. Update `backlog.md` (move to "Recent" list, remove from active)
6. Delete issue/plan files
7. Commit and push documentation updates
8. Mark PR ready for review

### Phase 1-2: Monitor CI
- Script polls CI status every 30 seconds (no token cost)
- Auto-fix pre-commit failures (run formatter, commit, push)
- Return to AI only when intervention needed:
  - **STATUS:FORK_CONFLICT** - Conflict with fork branch
  - **STATUS:BASE_CONFLICT** - Conflict with origin/main
  - **STATUS:NEEDS_FIX_CI** - Build/test failures
  - **STATUS:AUTO_MERGE_FAILED** - Permission issue

### Phase 3: Enable auto-merge
- When all CI checks pass, enable auto-merge (squash)
- GitHub will merge automatically when approved

### Phase 4: Cleanup
- After merge detected:
  - Switch to main branch
  - Pull latest from origin/main
  - Delete local feature branch
  - Delete remote branch on fork
- **STATUS:CLEANUP_COMPLETE** - Done!

**Output:**
- PR merged to main
- Branches cleaned up
- Backlog updated
- Ready for next issue

---

## Complete Example Session

```bash
# Phase 1: Discover and document issues
/create-issue
> "Let's explore the TarFile code"
# ... interactive exploration ...
# Creates issues #050, #051, #052 in draft PR #123

# Phase 2: Implement first issue
/fix-issue
> Recommends issue #050
> Shows full issue content and plan
> User confirms: Proceed
# ... creates workspace, implements, commits, pushes ...
# Draft PR #124 created

# Phase 3: Author review
# Review PR #124 code on GitHub
# Verify implementation looks good

# Phase 4: Finalize and merge
/resolve-ci
# ... finalizes PR, marks ready ...
# ... monitors CI, auto-fixes pre-commit ...
# ... CI passes, auto-merge enabled ...
# ... PR merged, branches cleaned up ...
# STATUS:CLEANUP_COMPLETE

# Repeat for issues #051, #052...
```

---

## Quick Reference

| Skill | Purpose | When to use |
|-------|---------|-------------|
| `/create-issue` | Explore codebase, create issues | Brainstorming, discovering improvements |
| `/fix-issue` | Select and implement issue | Ready to work on backlog issue |
| `/resolve-ci` | Finalize PR, monitor CI, merge | After implementation and human review |

**Decision points:**
- After `/fix-issue` implementation → **Author reviews** before `/resolve-ci`
- During `/resolve-ci` CI failures → **AI analyzes logs** and fixes, or asks for help

**Resumption:**
- `/fix-issue --finalize #NNN` - Skip to finalization if manually implemented
- `/resolve-ci` - Can retry after fixing conflicts/failures externally

---

## Workflow Principles

1. **Automation where it matters:** Mechanical tasks (backlog updates, CI polling, formatting) automated; intelligent tasks (code review, complex fixes) require human/AI judgment

2. **Token efficiency:** Skills only loaded when invoked, large scripts run without token cost, documentation separated from execution

3. **Git workflow enforcement:** All changes through feature branches and PRs, pre-commit hooks enforced, no direct commits to main

4. **Single-PR per issue:** One issue = one branch = one PR (multi-PR plans automatically revised)

5. **Draft by default:** PRs start as drafts, marked ready only after finalization and human approval

6. **Autonomous CI monitoring:** Python script polls GitHub (no token waste), returns to AI only when intelligent intervention needed

---

## Related Documentation

- **Git Workflow:** `docs/workflows/git-workflow.md`
- **PR Workflow:** `docs/workflows/pr-workflow.md`
- **Build Workflow:** `docs/workflows/build-workflow.md`
- **Project Backlog:** `docs/project/backlog.md`
- **Contributing Guidelines:** `docs/project/CONTRIBUTING.md`
