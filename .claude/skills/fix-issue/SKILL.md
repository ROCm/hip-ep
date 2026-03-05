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

## Phase 5: Finalization

After implementation is complete (auto or manual), finalize the PR and backlog:

### Step 5.1: Craft PR title and body

1. **Read issue and plan files** for context
2. **Update PR title** - Remove `[WIP]`, add proper type:
   - Format: `Issue #NNN: <type>: <description>`
   - Example: `Issue #038: docs: document lazy symlink resolution const_cast`
   - Use `gh pr edit {PR_NUM} --title "..."`

3. **Write comprehensive PR body** from issue context:
   - Read issue file and plan files
   - Create sections: Summary, Problem, Context, Solution, Benefits, Changes
   - Include implementation details and rationale
   - Use `gh pr edit {PR_NUM} --body "..."`

### Step 5.2: Update backlog

1. **Update completed-issues.md:**
   - Add entry to top of current month's table
   - Format: `| #NNN | {AUTHOR} | #PR | TBD | {DATE} | {TITLE} |`
   - Use `TBD` for commit hash (updated after merge)

2. **Update backlog.md:**
   - Add to "Recent (last 5)" list (prepend, remove oldest if >5)
   - Remove from active backlog table
   - Remove from "Quick dependencies" section if referenced

3. **Delete files:**
   - `git rm docs/project/issues/{NNN}-*.md`
   - `git rm docs/project/plans/{NNN}-*.md` (if exists)

4. **Commit and push:**
   - `git add docs/project/backlog.md docs/project/completed-issues.md`
   - `git commit -m "docs: complete issue #NNN"`
   - `git push fork {BRANCH}`

### Step 5.3: Exit with reminder

**PR remains DRAFT** - User must review before marking ready.

Remind user:
- Review implementation in PR #NNN
- When satisfied, mark ready: `gh pr ready {PR_NUM}`
- Then run `/resolve-ci` to monitor CI and auto-merge

---

## Next Steps: Author Review (Phase 3)

After /fix-issue completes:

1. **Author reviews the draft PR:**
   - Check code quality, logic, adherence to plan
   - Verify PR title/body are accurate
   - Test locally if needed

2. **Author marks PR ready:**
   ```bash
   gh pr ready <PR_NUMBER>
   ```

3. **Run /resolve-ci for CI monitoring:**
   - Monitors CI, handles conflicts/failures
   - Auto-merges when approved + CI passes

See `docs/workflows/issue-resolution-workflow.md` for complete workflow.
