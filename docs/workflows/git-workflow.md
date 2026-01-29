<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Git Workflow Rules

## CRITICAL: Feature Branch FIRST

**Create feature branch AS EARLY AS POSSIBLE - immediately when starting work on a task:**

1. Check current branch: `git branch --show-current`
2. If on `main`, create feature branch **RIGHT NOW**: `git checkout -b feature/<name>`
3. **ONLY THEN** start any work (reading code, planning, making changes, etc.)

**Timing**: Don't wait until you're about to make the first file change. Create the branch immediately when you receive or start thinking about a task.

**Enforcement**: The `.claude/settings.json` hooks will block file modifications on `main` branch, but create your branch FIRST to avoid interruptions.

---

## Resuming Work: Check Context First

**Before starting work (especially when resuming from previous sessions):**

1. **Check current branch:**
   ```bash
   git branch --show-current
   ```

2. **Look for existing PR branches:**
   ```bash
   git branch -a | grep feature/
   ```
   - Verify if a PR branch already exists for this task
   - Check if there's an active PR: `gh pr list --repo ROCm/MorphiZen`

3. **Read existing plan files:**
   - Check `~/.claude/plans/` for active plans related to current work
   - Understand the context and which branch should be used

4. **When uncertain, ask user:**
   - If multiple branches exist or context is unclear
   - Ask: "Which branch should I use for this task?"
   - Prevents creating duplicate branches for the same work

**Why this matters:**
- Prevents accidentally working on wrong branch
- Avoids creating duplicate branches for the same task
- Ensures continuity when resuming work across sessions

---

## Quick Checklist

- [ ] Check branch: `git branch --show-current`
- [ ] Create feature branch if on main: `git checkout -b feature/<name>`
- [ ] Make initial changes (even small - plan, TODO, skeleton)
- [ ] Stage specific files: `git add <file>` (avoid `git add -A`)
- [ ] Verify no binaries: `git diff --cached --numstat`
- [ ] Initial commit: `feat:`, `fix:`, `docs:`, etc.
- [ ] Push to fork: `git push fork <branch>`
- [ ] **Create DRAFT PR immediately**: `gh pr create --draft`
- [ ] Continue working, commit and push frequently
- [ ] Mark as ready for review when complete

## Branch Protection

**NEVER** commit or push directly to `main` branch. Always use Pull Requests.

## Required Workflow

1. Create a feature branch from `main`
2. Make an initial commit (even small - plan, TODO, skeleton code)
3. Push to fork: `git push fork <branch>`
4. **Create DRAFT PR immediately** - `gh pr create --draft`
   - Makes work visible to team early
   - Allows early feedback and discussion
   - Shows progress and intent
5. Continue working: **Commit and push frequently** - Small, incremental commits are preferred
6. Mark PR as "Ready for review" when complete
7. Wait for review and approval before merging

## Commit and PR Content Guidelines

**DO NOT mention tools/AI in commits or PRs** - keep all messages professional and tool-agnostic.

**Prohibited content:**
- ❌ No "🤖 Generated with Claude Code" or similar footers in PR descriptions
- ❌ No "Co-Authored-By: Claude" in commit messages
- ❌ No mentions of AI assistants, LLMs, or automation tools
- ❌ No tool-specific references in any commit or PR content

**Correct approach:**
- ✅ Write all commits and PRs as if authored by a human developer
- ✅ Use standard professional language and formatting
- ✅ Focus on what changed and why, not how it was created

## Before Any Git Push

- Verify current branch: `git branch --show-current`
- If on `main`, switch to a feature branch first
- Never run `git push origin main`

## After PR Merge

1. Switch to main: `git checkout main && git pull origin main`
2. Delete branches: `git branch -D <feature> && git push fork --delete <feature>`
3. Clean working directory: `git status` (must show "nothing to commit, working tree clean")

## CI Checks Monitoring

**CRITICAL: After PR is ready for review, actively monitor CI checks until all are green.**

1. Check CI status: `gh pr view <number> --repo ROCm/MorphiZen --json statusCheckRollup`
2. Monitor builds actively - don't leave PR with failing checks
3. If any check fails:
   - Review the failure logs
   - Fix the issue locally
   - Commit and push the fix
   - Continue monitoring until all checks pass
4. **Do not request review or expect merge until all CI checks are green**

### Common CI Checks:
- `pre-commit` - Code formatting and linting
- `build-and-test-lnx` - Linux build and tests
- `build-and-test-win` - Windows build and tests

## Pull Request Description Updates

Update PR description/title **when the scope, approach, or status of the work changes meaningfully** - not for every push.

### When to Update PR Description/Title:
- **Scope changes** - The PR now covers more/different functionality than originally described
- **Significant refactoring** - The approach has changed meaningfully
- **Checklist updates** - Mark items as done or add new requirements
- **Address review feedback** - Document how you responded to reviewer comments
- **Ready for review** - When moving from Draft to Ready state

### When NOT to Update:
- Small bug fixes within the same scope
- Code style/formatting changes
- Adding tests for existing functionality
- Minor refactoring that doesn't change intent

---

## Switching to Main Branch

Switch to the main branch and update it with the latest remote changes.

### Step 1: Stash Local Changes (if any)

If the working directory is dirty (has uncommitted changes), stash them first:

```bash
git stash push -m "WIP: stashed before switching to main"
```

### Step 2: Switch to Main Branch

```bash
git checkout main
```

### Step 3: Sync with Remote

Fetch the latest changes and reset to match origin/main exactly:

```bash
git fetch origin main
git reset --hard origin/main
```

This ensures your local main branch and working directory are in perfect sync with the remote, discarding any local-only commits on main.

### Quick Command

For stashing, switching, and syncing in one operation (PowerShell):

```powershell
git stash push -m "WIP: auto-stash"; git checkout main; git fetch origin main; git reset --hard origin/main
```

### When to Use

- Before starting a new feature branch (to branch from latest main)
- When you need to review the current state of main
- After completing work on a feature branch and merging via PR

### Restoring Stashed Changes

After returning to your feature branch, restore stashed changes:

```bash
git checkout <your-branch>
git stash pop
```

To list all stashes:

```bash
git stash list
```

### Notes

- This workflow is for updating your local main branch only
- Do not make commits directly on main - use feature branches for changes
- Always stash uncommitted changes before switching branches to avoid conflicts
- Use descriptive stash messages to identify changes later

---

## Committing and Syncing Changes

Commit local changes and synchronize with the remote repository.

### Prerequisites (One-time Setup)

Before using this workflow, ensure pre-commit is set up:

#### Install Dependencies

```bash
pip install -r requirements-lintrunner.txt
```

#### Install Pre-commit Hooks

```bash
pre-commit install
```

#### Initialize Lintrunner

```bash
lintrunner init
```

For more details, see [pre-commit.md](../technical/pre-commit.md).

### Pre-flight Checks

#### Step 1: Verify Current Branch

**CRITICAL:** Never commit directly to `main` branch.

```bash
git branch --show-current
```

If on `main`, create or switch to a feature branch first:
```bash
git checkout -b feature/<descriptive-name>
# or switch to existing branch
git checkout <branch-name>
```

#### Step 2: Check Repository Status

```bash
git status
```

Review:
- Staged changes (green)
- Unstaged changes (red)
- Untracked files

### Committing Changes

#### Step 3: Stage Changes

Stage specific files:
```bash
git add <file1> <file2>
```

Stage all changes (use with caution):
```bash
git add -A
```

Interactive staging (recommended for selective commits):
```bash
git add -p
```

#### Step 4: Review Staged Changes

```bash
git diff --staged
```

Verify only intended changes are staged before committing.

#### Step 5: Commit with Message

```bash
git commit -m "<type>: <brief description>"
```

Commit message conventions:
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `refactor:` - Code refactoring
- `test:` - Adding or updating tests
- `chore:` - Maintenance tasks
- `style:` - Code style/formatting changes

Examples:
```bash
git commit -m "feat: add new graph optimization pass"
git commit -m "fix: resolve memory leak in plugin loader"
git commit -m "docs: update README with build instructions"
```

For detailed commit messages:
```bash
git commit
# Opens editor for multi-line message
```

### Syncing with Remote

#### Step 6: Fetch Remote Changes

```bash
git fetch origin
```

Check if there are upstream changes:
```bash
git log HEAD..origin/main --oneline
```

#### Step 7: Sync with Main Branch

Option A - Rebase (preferred for cleaner history):
```bash
git fetch origin main
git rebase origin/main
```

Option B - Merge:
```bash
git fetch origin main
git merge origin/main
```

#### Step 8: Handle Conflicts (if any)

If conflicts occur during rebase/merge:

1. Check conflicting files:
   ```bash
   git status
   ```

2. Resolve conflicts in each file (look for `<<<<<<<`, `=======`, `>>>>>>>` markers)

3. After resolving, stage the fixed files:
   ```bash
   git add <resolved-file>
   ```

4. Continue rebase/merge:
   ```bash
   git rebase --continue
   # or
   git merge --continue
   ```

5. If needed, abort and start over:
   ```bash
   git rebase --abort
   # or
   git merge --abort
   ```

#### Step 9: Push Changes

Push to remote (feature branch):
```bash
git push fork <branch-name>
```

If branch doesn't exist on remote:
```bash
git push -u fork <branch-name>
```

If rebase changed history (force push with lease for safety):
```bash
git push --force-with-lease fork <branch-name>
```

#### Step 10: Create or Update Pull Request

If no PR exists for this branch, create one. **On Windows, use `--body-file` to avoid quoting issues:**

1. Create a temporary file `pr_body.md` with the PR description:
```markdown
# Summary of Changes

* <High-level description of what this request adds/changes/improves/fixes>

Closes #<issue-number> (if applicable)

# Motivation

<Why this change is needed/useful>

# Implementation

<Any implementation details or design choices worth noting>
```

2. Create the PR using the file:
```bash
gh pr create --title "<type>: <brief description>" --body-file pr_body.md
```

3. Delete the temporary file after PR creation.

To update an existing PR description:
```bash
gh pr edit <PR-number> --body-file pr_body.md
```

If a PR already exists, the push will automatically update it (no need to edit unless scope changed).

### Quick Workflow Summary

For a typical commit and sync cycle:

```bash
# 1. Check branch (must NOT be main)
git branch --show-current

# 2. Stage changes
git add -A

# 3. Commit
git commit -m "feat: your change description"

# 4. Sync with main
git fetch origin main
git rebase origin/main

# 5. Push
git push fork <branch-name>

# 6. Create PR (if not exists) - use --body-file on Windows to avoid quoting issues
# First write PR body to pr_body.md, then:
gh pr create --title "feat: your change description" --body-file pr_body.md
```

### After Pushing

If PR already exists:
- Update PR description if scope changed significantly
- Request review if ready: `gh pr ready`

### Pre-commit Hook Behavior

When you run `git commit`, pre-commit hooks run automatically and may:
- Fix trailing whitespace and line endings
- Add license headers to new files
- Run lintrunner to apply code formatting patches

**If hooks modify files**, the commit will fail. Simply re-stage the modified files and commit again:

```bash
git add -A
git commit -m "your message"
```

#### Manual Pre-commit Commands

Run hooks on all files (useful for first-time setup):
```bash
pre-commit run --all-files
```

Skip hooks if absolutely needed (use sparingly):
```bash
git commit --no-verify -m "message"
```

### Commit and Sync Notes

- **Small, frequent commits** are preferred over large, infrequent ones
- Always verify you're on a feature branch before committing
- Use `--force-with-lease` instead of `--force` for safer force pushes
- Run tests before pushing when possible
- Pre-commit hooks help maintain code quality - don't skip them without good reason
