<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Claude Code Configuration

This directory contains Claude Code configuration files for the MorphiZen project.

## Files

- **`settings.json`** - Team-shared hook configuration (committed to git)
- **`settings.local.json`** - Personal local settings (gitignored, optional)

## Skills

Team-shared Claude Code skills are located in `skills/`:

- **`/go`** - Execute approved plans with `.clinerules/git-rules.md` enforcement
  - Automatically invoked when user says "go", "/go", or "execute the plan"
  - Ensures feature branch creation, conventional commits, draft PR creation, and CI monitoring
  - See `skills/go/SKILL.md` for details

- **`/status`** - Quick health check of git and PR status
  - Automatically invoked when user says "status", "/status", or asks about current state
  - Shows current branch, uncommitted changes, unpushed commits, and PR status
  - Read-only operation, safe to run anytime
  - See `skills/status/SKILL.md` for details

- **`/update-branch`** - Update feature branch with latest upstream changes
  - Automatically invoked when user says "update-branch", "/update-branch", or wants to sync with main
  - Fetches from origin, rebases onto origin/main, force-pushes to fork
  - Handles conflicts intelligently (auto-resolve safe cases, guide for complex conflicts)
  - Safety checks: blocks on main branch, requires clean working directory
  - See `skills/update-branch/SKILL.md` for details

## Hook Enforcement System

The `settings.json` file contains PreToolUse hooks that enforce the git workflow rules defined in `.clinerules/git-rules.md`. These hooks run automatically before Claude Code executes certain tool operations.

### Hook Overview

| Workflow Rule | Hook Enforcement | Type |
|---------------|------------------|------|
| Feature branch FIRST | Block edits on main | 🚫 BLOCK |
| Never push to main | Block push on main | 🚫 BLOCK |
| Push to fork (not origin) | Ask confirmation | ⚠️ ASK |
| Create draft PR early | Remind after push | 💡 REMIND |
| Commit frequently | Remind before edits | 💡 REMIND |
| Push frequently | Remind before commit | 💡 REMIND |

### Hook Types

**🚫 BLOCK** - Hard blocking (exit 1)
- Prevents the operation from proceeding
- Shows structured error message with:
  - `STATUS: BLOCKED`
  - `REASON`: Why the operation was blocked
  - `REQUIRED_ACTION`: Exact command to fix the issue
  - `EXPLANATION`: Why this rule exists
  - `REFERENCE`: Link to relevant documentation

**⚠️ ASK** - User confirmation required
- Pauses and asks for user approval
- Shows `permissionDecision: ask` with reason
- Operation proceeds only if user confirms

**💡 REMIND** - Non-blocking reminder
- Shows informational message
- Operation proceeds automatically
- Encourages best practices without blocking work

### Detailed Hook Behavior

#### 1. Block Edits on Main Branch
**Triggers on**: Write, Edit tools
**Condition**: Current branch is `main`
**Action**: BLOCK - Prevents file modifications
**Required Action**: `git checkout -b feature/<descriptive-name>`
**Also**: Reminds to commit frequently if >5 uncommitted changes

#### 2. Block Push on Main Branch
**Triggers on**: `git push` commands
**Condition**: Current branch is `main`
**Action**: BLOCK - Prevents push operation
**Required Action**: `git checkout -b feature/<descriptive-name>`
**Also**: Reminds to create draft PR if no PR exists for current branch

#### 3. Ask Before Pushing to Origin
**Triggers on**: `git push origin ...` commands
**Condition**: Pushing to `origin` instead of `fork`
**Action**: ASK - Request user confirmation
**Reason**: Fork-based workflow prefers `git push fork <branch>`

#### 4. Remind to Push Frequently
**Triggers on**: `git commit` commands
**Condition**: Unpushed commits exist on current branch
**Action**: REMIND - Show count of unpushed commits
**Suggested Action**: `git push fork <branch>`

#### 5. Remind to Commit Frequently
**Triggers on**: Write, Edit tools
**Condition**: >5 uncommitted file changes exist
**Action**: REMIND - Show count of uncommitted changes
**Suggested Action**: `git add <files> && git commit -m "<message>"`

### Self-Correction for Claude

The structured error messages are designed to help Claude Code self-correct when blocked:

```
=== CLAUDE CODE HOOK ERROR ===
STATUS: BLOCKED
REASON: Attempting to modify files while on main branch
REQUIRED_ACTION: git checkout -b feature/<descriptive-name>
EXPLANATION: Create a feature branch before making any file changes
REFERENCE: .clinerules/git-rules.md - Section: CRITICAL: Feature Branch FIRST
=============================
```

Claude can parse these fields and:
1. Recognize it's blocked by a workflow rule
2. Understand why (REASON)
3. Execute the corrective action (REQUIRED_ACTION)
4. Learn the underlying principle (EXPLANATION)
5. Reference detailed documentation (REFERENCE)

This enables Claude to autonomously correct workflow violations without requiring user intervention.

### Team Sharing

Since `settings.json` is committed to git:
- All team members get the same hook enforcement automatically
- Hooks are version-controlled alongside code and documentation
- Changes to workflow rules can be reviewed and approved via PR

Personal preferences can be set in `settings.local.json` (gitignored), which takes precedence over `settings.json`.
