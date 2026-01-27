<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Git Workflow Rules

## Quick Checklist

- [ ] Check branch: `git branch --show-current`
- [ ] Create feature branch if on main: `git checkout -b feature/<name>`
- [ ] Make changes (no binaries, no temp files)
- [ ] Stage specific files: `git add <file>` (avoid `git add -A`)
- [ ] Verify no binaries: `git diff --cached --numstat`
- [ ] Commit: `feat:`, `fix:`, `docs:`, etc.
- [ ] Push to fork: `git push fork <branch>`
- [ ] Create PR: `gh pr create`

## Branch Protection

**NEVER** commit or push directly to `main` branch. Always use Pull Requests.

## Required Workflow

1. Create a feature branch from `main`
2. Make changes on the feature branch
3. **Commit and push frequently** - Small, incremental commits are preferred over large, infrequent ones
4. Push the feature branch and create a Pull Request
5. Wait for review and approval before merging

## Before Any Git Push

- Verify current branch: `git branch --show-current`
- If on `main`, switch to a feature branch first
- Never run `git push origin main`

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
