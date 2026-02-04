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

## Phase 1: Issue Selection

### Step 1.1: Load backlog and open PRs

Read backlog table and get open PRs to detect file conflicts.

### Step 1.2: Smart recommendation

Parse backlog table, filter out blocked/conflicting issues, rank by priority (C > H > M > L), apply tie-breakers (blockers first, quick wins, sequential within group).

Display top recommendation with metadata. User can request more options or select specific issue.

---

## Phase 2: Workspace Setup

Run `setup-workspace.py <issue_num>` to:
- Check prerequisites (on main, no uncommitted changes)
- Sync main branch
- Create feature branch
- Add "Started:" date to issue file
- Initial commit with pre-commit enforcement
- Push with `-u` to fork
- Create draft PR
- Return workspace info (branch, PR number, issue details)

Read issue and plan files, display context to user.

---

## Phase 2.5: Plan Detection & Revision

### Step 2.5.1: Detect plan files

Find all plan files for the issue:
```bash
PLAN_FILES=$(ls docs/project/plans/${ISSUE_NUM}-*.md 2>/dev/null || echo "")
```

Count plans and read each one.

### Step 2.5.2: Display plan information

Show plan status to user:

**If no plans found:**
```
⚠️ Implementation Plan: No detailed plan found
   You'll need to design the implementation approach yourself
```

**If one plan found:**
```
📋 Implementation Plan: ✅ Detailed plan available
   Location: docs/project/plans/027-eliminate-c-style-apis-migration-plan.md
   Steps: [count from reading file]
```

**If multiple plans found:**
```
📋 Implementation Plans: 2 plans found
   - docs/project/plans/027-migration-plan.md
   - docs/project/plans/027-alternative-approach.md
```

### Step 2.5.3: Check for multi-PR patterns

Read each plan file and search for patterns indicating multiple PRs:
- "PR 1:", "PR 2:", "PR #1", "PR #2"
- "create new PR", "create another PR", "create draft PR" (but not in setup context)
- "git checkout -b" (creating new branches)
- "gh pr create" (after initial setup)
- "Phase 1 PR", "Phase 2 PR", "subsequent PR", "follow-up PR"
- "first pull request", "second pull request"

### Step 2.5.4: Revise multi-PR plans

**If multi-PR patterns detected:**

1. Show user:
   ```
   ⚠️ Plan suggests multiple PRs. Revising to single-PR approach...
   ```

2. For each plan file:
   - Read entire content
   - Edit to consolidate:
     - Remove "create new PR" instructions
     - Change "PR 1: Do X, PR 2: Do Y" to "Step 1: Do X, Step 2: Do Y"
     - Add note at top: "NOTE: All work completed in single PR #[PR_NUMBER]"
     - Convert multi-PR phases to sequential steps
   - Use Edit tool to update plan file

3. Commit revised plans:
   ```bash
   git add docs/project/plans/${ISSUE_NUM}-*.md
   git commit -m "docs: revise plans for single-PR workflow (issue #${ISSUE_NUM})"
   git push fork "$BRANCH_NAME"
   ```

4. Show user:
   ```
   ✅ Plans revised for single-PR execution
      All implementation will happen in PR #[PR_NUMBER]
   ```

**If no multi-PR patterns:**
   ```
   ℹ️  Plan follows single-PR approach
   ```

---

## Phase 3: Implementation

**CRITICAL: Single-PR Enforcement**

All work happens in the branch and PR created in Phase 2.

❌ NEVER run: `gh pr create`, `git checkout -b`
✅ ALLOWED: Edit/Write files, build/test, `git add/commit/push fork <branch>`

If plan mentions creating PRs: SKIP that instruction and continue.

---

### Step 3.1: Ask user

Display summary:
```
📋 Ready for Implementation
Plans: X file(s) | PR #XXX | Branch: feature/issue-XXX-...
```

Offer: auto-implement, manual, or cancel.

### Step 3.2: If YES - Auto-implement

Read plan, execute implementation steps using Edit/Write tools.

Build and test. If failures occur, offer: fix attempt, continue anyway, or cancel.

Commit with validation (no AI mentions), run pre-commit, push to fork.

### Step 3.3: If NO - Manual implementation

Exit skill. User implements manually, then runs `/fix-issue --finalize #NNN` to complete.

---

## Phase 4-8: Finalization

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
