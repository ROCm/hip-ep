<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Git Workflow Rules

## CRITICAL: Feature Branch FIRST

**ALWAYS sync main before creating feature branch:**

1. **Sync local main with origin/main FIRST:**
   ```bash
   git checkout main
   git pull origin main
   # Or: git fetch origin main && git reset --hard origin/main
   ```

2. **THEN create feature branch:**
   ```bash
   git checkout -b feature/<descriptive-name>
   ```

3. **ONLY THEN start any work** (reading code, planning, making changes)

**Why sync first?**
- Ensures branch starts from latest code
- Reduces merge conflicts later
- Avoids rebasing large divergences

**Timing**: Create branch immediately when starting a task, not when making first file change.

**Enforcement**: `.claude/settings.json` hooks block modifications on `main` branch.

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

## Required Workflow

**Before starting:**
- [ ] Sync main: `git checkout main && git pull origin main`
- [ ] Create feature branch: `git checkout -b feature/<name>`
- [ ] Verify branch: `git branch --show-current`

**During development:**
- [ ] Make initial commit (even small - plan, TODO, skeleton)
- [ ] Stage specific files: `git add <file>` (avoid `git add -A`)
- [ ] Verify no binaries: `git diff --cached --numstat`
- [ ] Commit with convention: `feat:`, `fix:`, `docs:`, etc.
- [ ] Push to fork: `git push fork <branch>`
- [ ] **Create DRAFT PR immediately**: `gh pr create --draft`
  - Makes work visible to team early
  - Allows early feedback and discussion
  - Shows progress and intent
- [ ] Continue working, commit and push frequently (small, incremental commits preferred)

**After completion:**
- [ ] Mark PR as ready for review
- [ ] Wait for review and approval before merging

See [git-workflow-reference.md](git-workflow-reference.md) for detailed commit/sync procedures.

## Branch Protection

**NEVER** commit or push directly to `main` branch. Always use Pull Requests.

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

**Note**: If checks fail due to main branch advancing, see "Keeping PR Updated with Main" below.

## Keeping PR Updated with Main

**When origin/main advances while your PR is in review:**

1. Fetch and check:
   ```bash
   git fetch origin main
   git log HEAD..origin/main --oneline
   ```

2. Update branch:
   ```bash
   git rebase origin/main
   ```

3. Resolve conflicts (see Step 8 in Committing workflow below)

4. Force push:
   ```bash
   git push --force-with-lease fork <branch-name>
   ```

5. Monitor CI checks - they will re-run

**When to update:**
- When GitHub shows branch is out-of-date
- Before requesting final review if main has advanced
- When reviewers request it
- If CI checks fail due to main branch changes
- Before merge (if required by policy)

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
