<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
---
name: resolve-ci
description: Autonomous loop to resolve merge conflicts and CI failures until PR merges
allowed-tools: [Bash, Read, Grep, Glob, Edit, Write, AskUserQuestion]
---

# /resolve-ci - Autonomous Conflict and CI Failure Resolution

## Purpose

Monitor CI and handle failures until PR merges, then clean up workspace.

**Does not do finalization** - PR title/body/backlog updates are done by /fix-issue.

---

## Prerequisites

Before running /resolve-ci:
1. ✅ Implementation complete (code done, committed, pushed)
2. ✅ **Finalization complete** (PR title/body updated, backlog updated) - done by /fix-issue
3. ✅ PR marked ready for review: `gh pr ready {PR_NUM}`

## Workflow

Call monitor-pr.py. It validates prerequisites then monitors CI.

### Validation by monitor-pr.py

The script checks:

1. **PR is marked ready** (not draft)
   - If draft → `STATUS:NEEDS_READY`
   - Exit with message: "Mark ready first: `gh pr ready {PR_NUM}`"

2. **PR is finalized** (title doesn't contain `[WIP]`)
   - If still has `[WIP]` → `STATUS:NEEDS_FINALIZATION`
   - Exit with message: "Run /fix-issue to complete finalization first"

If both checks pass, proceed to monitoring loop.

---

## Main Loop

**Script runs autonomously, returns status when intervention needed:**

```bash
python .claude/skills/resolve-ci/monitor-pr.py --attempt 1
```

**Timeout:** 10 min. Retry up to 6 attempts (1 hour). After attempt 6: inform user "CI still running, check GitHub".

**Status codes:**

**STATUS:FORK_CONFLICT** - conflict with fork/branch, resolve and re-run

**STATUS:BASE_CONFLICT** - conflict with origin/main, resolve and re-run

**STATUS:NEEDS_FIX_CI** - complex CI failure, fix and re-run

**STATUS:AUTO_MERGE_FAILED** - inform user "Check PR settings"

**STATUS:CLEANUP_COMPLETE** - done!

---

## Conflict Resolution

When `STATUS:FORK_CONFLICT` or `STATUS:BASE_CONFLICT`:

1. **List conflicts:**
   ```bash
   git diff --name-only --diff-filter=U
   ```

2. **Read conflicting files** - use Read tool

3. **Resolve conflicts:**
   - Simple (whitespace/imports) → auto-merge with Edit tool
   - Complex → analyze and resolve intelligently
   - Too complex → AskUserQuestion

4. **Complete rebase:**
   ```bash
   git add <conflicting_file>
   git rebase --continue
   git push fork "{BRANCH_NAME}" --force-with-lease
   ```

---

## CI Failure Resolution

When `STATUS:NEEDS_FIX_CI`:

1. **Get CI logs:**
   ```bash
   RUN_ID=$(gh run list --branch "{BRANCH_NAME}" --limit 1 --json databaseId | python -c "import sys, json; print(json.load(sys.stdin)[0]['databaseId'])")
   gh run view $RUN_ID --log-failed > /tmp/ci_logs.txt
   ```

2. **Read logs** - use Read tool on /tmp/ci_logs.txt

3. **Categorize and fix:**
   - Build errors → fix syntax/types/includes
   - Test failures → fix logic bugs
   - Other → analyze and resolve

4. **Verify fix (if build error):**
   ```bash
   cmake --build ../../build/$(basename $PWD) --config Debug --parallel
   ```

5. **Commit and push:**
   ```bash
   git add <file> && git commit -m "fix: ..." && git push fork "{BRANCH_NAME}"
   ```
