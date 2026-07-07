<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->

# /resolve-ci - Autonomous PR Lifecycle Management

## Overview

Autonomous skill that shepherds a PR from "ready for review" to merged, handling conflicts and CI failures along the way. Designed to minimize AI token usage by delegating mechanical work to Python scripts.

## Architecture

### Design Principle: Token Efficiency

**Problem:** Polling GitHub for CI status every 30 seconds with AI would waste ~180,000 tokens per 10 minutes.

**Solution:** Python script polls autonomously, returns to AI only when intelligent intervention needed.

### Components

**monitor-pr.py (Autonomous Orchestrator)**
- Updates branch (sync fork, rebase main) - detects conflicts upfront
- Monitors CI status every 30 seconds - auto-fixes pre-commit failures
- Enables auto-merge when CI passes
- Cleans up after merge (switch to main, delete branches)
- **Token cost:** Zero (runs as subprocess)

**AI Intervention (Intelligent Problem Solving)**
- Conflict resolution - read both sides, understand intent, merge correctly
- CI failure analysis - read logs, understand errors, fix code
- PR documentation - craft quality titles and descriptions
- **Token cost:** Only when actually needed (~5% of lifecycle)

### Value Proposition

**95% automated, 5% intelligent intervention where it matters**

The script handles all mechanical operations. AI only engages for tasks requiring judgment and understanding.

## Workflow

### Phase 0: PR Finalization (AI)
1. Read issue file
2. Craft PR title and description
3. Update project documentation (backlog, completed issues)
4. Delete issue files
5. Mark PR ready

### Phase 1: Update Branch (Script)
- Fetch and rebase onto fork/branch (detects upstream conflicts)
- Fetch and rebase onto origin/main (detects base conflicts)
- Push with --force-with-lease
- Returns status if conflicts found

### Phase 2: Monitor CI (Script)
- Poll every 30 seconds (no token cost)
- Auto-fix pre-commit failures (run formatter, commit, push)
- Enable auto-merge when all checks pass
- Wait for GitHub to merge
- Returns status if complex CI failure

### Phase 3: Cleanup (Script)
- Switch to main branch
- Pull latest from origin/main
- Delete local feature branch
- Delete remote branch on fork

## Why This Design?

### Git vs GitHub Complexity

**Git reality (simple):**
- Two types of conflicts: fork/branch vs origin/main
- Detect by attempting rebase

**GitHub adds complexity:**
- `mergeable: CONFLICTING` - prediction
- `mergeStateStatus: BEHIND` - state
- We ignore these, rely on actual git operations

### Status Detection

**Check with git (simple):**
```bash
git log origin/main..HEAD  # Empty = merged
git rebase origin/main     # Exit code tells us about conflicts
```

**Check with GitHub (only for CI):**
```bash
gh pr view --json statusCheckRollup  # Can't get this from git
```

We use GitHub ONLY for what git can't provide: CI status.

### Retry Logic

**Why timeout handling:** CI can take 15-30 minutes, Bash tool times out at 10 minutes.

**Solution:** AI retries up to 6 times (1 hour total). Script shows attempt number for user visibility.

**Token cost:** Retry decision made by AI, but actual polling done by script (no token waste).

## Status Codes

Script returns these when AI intervention needed:

- `STATUS:FORK_CONFLICT` - Conflict with fork/branch-name (concurrent push to same branch)
- `STATUS:BASE_CONFLICT` - Conflict with origin/main (branch diverged from base)
- `STATUS:NEEDS_FIX_CI` - Complex CI failure (not pre-commit)
- `STATUS:AUTO_MERGE_FAILED` - Couldn't enable auto-merge (permissions issue)
- `STATUS:CLEANUP_COMPLETE` - Success! PR merged, workspace cleaned up

## Development History

**Original design:** Multiple scripts (update-branch.py, cleanup-after-merge.py, monitor-pr.py)

**Current design:** Single script handles complete lifecycle

**Why consolidate:** Simpler mental model, fewer files, easier to maintain. All mechanical operations in one place.

**What stayed separate:** AI-guided work (conflict resolution, CI debugging) remains in SKILL.md as execution instructions.

## Token Optimization Journey

**SKILL.md evolution:**
- Started: 253 lines with embedded bash and detailed explanations
- Phase 1: Extract bash to Python scripts
- Phase 2: Remove redundant sections
- Phase 3: Consolidate scripts into one
- Final: 119 lines of pure execution instructions

**Token savings per invocation:** ~5,700 tokens (12,650 → 6,950)

**Where we spent tokens wisely:**
- Architecture documentation → moved to README.md (this file)
- Bash scripts → converted to Python (executed, not loaded)
- Explanations → removed from SKILL.md, kept here for humans

## Usage

```bash
# After Phase 0 (PR finalization by AI)
python .claude/skills/resolve-ci/monitor-pr.py --attempt 1

# Script runs autonomously, returns status when intervention needed
# AI resolves issues, re-runs script
# Repeat until STATUS:CLEANUP_COMPLETE
```

## Future Enhancements

Potential improvements:
- Parallel CI execution (if multiple independent jobs)
- Smart conflict resolution suggestions (ML-based)
- Automatic PR description generation from commits (might sacrifice quality)
- Integration with code review tools

But remember: **95% automation is already excellent**. Don't over-engineer.
