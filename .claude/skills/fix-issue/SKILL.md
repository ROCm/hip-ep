<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
---
name: fix-issue
description: Complete workflow automation for fixing backlog issues (selection, workspace setup, implementation)
allowed-tools: [Bash, Read, Grep, Glob, Edit, Write, AskUserQuestion]
---

# /fix-issue - Complete Workflow Automation for Backlog Issues

## Purpose

Automate issue selection, workspace setup, and implementation for backlog issues, eliminating repetitive manual steps while enforcing git-workflow.md rules.

**Scope:** Phase 2 of issue resolution workflow (selection → implementation). After /fix-issue completes, author reviews the implementation, then runs /resolve-ci for finalization and CI monitoring.

---

## Phase 0: Pre-Selection Sync

Ensure on main branch with no uncommitted changes, then sync local main with origin/main via `git pull origin main`.

---

## Phase 1: Issue Selection

### Step 1.1: Load backlog and open PRs

Read backlog table and get open PRs to detect file conflicts.

### Step 1.2: Smart recommendation

Parse backlog table, filter out blocked/conflicting issues, rank by priority (C > H > M > L), apply tie-breakers (blockers first, quick wins, sequential within group).

Display top recommendation with metadata. User can request more options or select specific issue.

---

## Phase 2: Review and Confirm

### Step 2.1: Read issue and plans

Read issue file and find plans: `docs/project/plans/${ISSUE_NUM}-*.md`.

### Step 2.2: Display full context

Show complete issue content:
- Issue metadata (number, title, type, priority)
- Problem description
- Proposed solution
- Files to be changed
- Plans status: None / Single plan (show filename and summary) / Multiple plans (list all with summaries)

### Step 2.3: Ask user to proceed

After showing full context, offer: Proceed / Cancel.

If user cancels, exit without creating workspace.

---

## Phase 3: Workspace Setup

Run `python .claude/skills/fix-issue/setup-workspace.py <issue_num>` to:
- Check prerequisites (on main, no uncommitted changes)
- Sync main branch
- Create feature branch
- Add "Started:" date to issue file
- Initial commit with pre-commit enforcement
- Push with `-u` to fork
- Create draft PR
- Return workspace info (branch, PR number, issue details)

---

## Phase 3.5: Plan Revision

If multi-PR patterns detected in plans, revise to single-PR approach, commit, and push.

---

## Phase 4: Implementation

**CRITICAL: Single-PR Enforcement**

All work happens in the branch and PR created in Phase 3.

❌ NEVER run: `gh pr create`, `git checkout -b`
✅ ALLOWED: Edit/Write files, build/test, `git add/commit/push fork <branch>`

If plan mentions creating PRs: SKIP that instruction and continue.

---

### Step 4.1: Ask user

Offer: Auto-implement / Manual / Cancel.

### Step 4.2: If YES - Auto-implement

Read plan, execute implementation steps using Edit/Write tools.

Build and test. If failures occur, offer: fix attempt, continue anyway, or cancel.

Commit with validation (no AI mentions), run pre-commit, push to fork.

### Step 4.3: If NO - Manual implementation

Exit skill. User implements manually on the feature branch.

---

## Next Steps: Author Review (Phase 3)

After /fix-issue completes (auto or manual implementation):

1. **Author reviews the draft PR:**
   - Check code quality, logic, adherence to plan
   - Test locally if needed
   - Verify implementation correctness

2. **Author marks PR ready:**
   ```bash
   gh pr ready <PR_NUMBER>
   ```

3. **Run /resolve-ci for finalization and CI monitoring:**
   - Finalizes PR (updates backlog, deletes issue files, crafts PR description)
   - Monitors CI, handles conflicts/failures
   - Auto-merges when approved + CI passes

See `docs/workflows/issue-resolution-workflow.md` for complete workflow.
