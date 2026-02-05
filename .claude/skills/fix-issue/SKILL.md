<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
---
name: fix-issue
description: Complete workflow automation for fixing backlog issues (selection, implementation, finalization)
allowed-tools: [Bash, Read, Grep, Glob, Edit, Write, AskUserQuestion]
---

# /fix-issue - Complete Workflow Automation for Backlog Issues

## Purpose

Automate the complete workflow for fixing backlog issues from selection to finalization, eliminating repetitive manual steps while enforcing git-workflow.md rules.

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

Run `setup-workspace.py <issue_num>` to:
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

Exit skill. User implements manually, then runs `/fix-issue --finalize #NNN` to complete.

---

## Phase 5-8: Finalization

### Step 4.1: Generate PR body

Read issue and plan files. Extract Description, Problem, Solution sections. Collect changed files and commit history. Generate comprehensive PR body combining issue context with implementation results.

### Step 4.2: Update PR

Update PR title (format: `Issue #NNN: <type>: <description>`) and body using `gh pr edit`.

### Step 4.3: Update backlog and cleanup

Read backlog.md, move issue to "Recently Completed" section with PR number (keep max 5).

Run `finalize-issue.sh <issue_num> <pr_number> <updated_backlog_content>` to:
- Update backlog.md
- Delete issue and plan files
- Commit with pre-commit enforcement
- Push to fork

### Step 4.4: Show summary

Display completed work summary with changed files, commits, and PR status. Remind user PR is DRAFT and needs review before marking ready.

---

## Resumption Support

Use `/fix-issue --finalize #NNN` to skip to Phase 4-8 (finalization only).
