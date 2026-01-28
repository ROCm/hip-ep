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
