<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Commit Changes and Sync Git Repository

Commit local changes and synchronize with the remote repository.

## Prerequisites (One-time Setup)

Before using this workflow, ensure pre-commit is set up:

### Install Dependencies

```bash
pip install -r requirements-lintrunner.txt
```

### Install Pre-commit Hooks

```bash
pre-commit install
```

### Initialize Lintrunner

```bash
lintrunner init
```

For more details, see [doc/pre-commit.md](../../doc/pre-commit.md).

## Pre-flight Checks

### Step 1: Verify Current Branch

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

### Step 2: Check Repository Status

```bash
git status
```

Review:
- Staged changes (green)
- Unstaged changes (red)
- Untracked files

## Commit Changes

### Step 3: Stage Changes

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

### Step 4: Review Staged Changes

```bash
git diff --staged
```

Verify only intended changes are staged before committing.

### Step 5: Commit with Message

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

## Sync with Remote

### Step 6: Fetch Remote Changes

```bash
git fetch origin
```

Check if there are upstream changes:
```bash
git log HEAD..origin/main --oneline
```

### Step 7: Sync with Main Branch

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

### Step 8: Handle Conflicts (if any)

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

### Step 9: Push Changes

Push to remote (feature branch):
```bash
git push origin <branch-name>
```

If branch doesn't exist on remote:
```bash
git push -u origin <branch-name>
```

If rebase changed history (force push with lease for safety):
```bash
git push --force-with-lease origin <branch-name>
```

## Quick Workflow Summary

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
git push origin <branch-name>
```

## After Pushing

Consider:
- Create a Pull Request if not already created: `gh pr create`
- Update PR description if scope changed significantly
- Request review if ready: `gh pr ready`

## Pre-commit Hook Behavior

When you run `git commit`, pre-commit hooks run automatically and may:
- Fix trailing whitespace and line endings
- Add license headers to new files
- Run lintrunner to apply code formatting patches

**If hooks modify files**, the commit will fail. Simply re-stage the modified files and commit again:

```bash
git add -A
git commit -m "your message"
```

### Manual Pre-commit Commands

Run hooks on all files (useful for first-time setup):
```bash
pre-commit run --all-files
```

Skip hooks if absolutely needed (use sparingly):
```bash
git commit --no-verify -m "message"
```

## Notes

- **Small, frequent commits** are preferred over large, infrequent ones
- Always verify you're on a feature branch before committing
- Use `--force-with-lease` instead of `--force` for safer force pushes
- Run tests before pushing when possible
- Pre-commit hooks help maintain code quality - don't skip them without good reason
