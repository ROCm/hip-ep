<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
---
name: go
description: Execute approved plan with git-rules.md enforcement. Use immediately when user says "go", "/go", or "execute the plan" after plan approval.
---

# /go - Execute Plan with Git Rules Enforcement

## CRITICAL: Read Git Rules FIRST

Before executing ANY approved plan, you MUST:

1. **Read `.clinerules/git-rules.md`** from the repository root
2. **Understand ALL requirements** - don't skip any sections
3. **Follow them throughout execution** - not just at the start

## Branch Check and Creation

Before making ANY code changes or commits:

1. Check current branch:
   ```bash
   git branch --show-current
   ```

2. If on `main` AND the plan involves code changes:
   - **Stop immediately**
   - Create feature branch: `git checkout -b feature/<descriptive-name>`
   - Use kebab-case for branch names
   - Name should describe the change (e.g., `feature/add-caching`, `fix/memory-leak`)

3. **TIMING**: Reading code and planning is OK to help choose a good branch name, but create the feature branch BEFORE making any file modifications

## During Plan Execution

### Commit Requirements
- **Conventional commits**: Use `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, etc.
- **No AI mentions**: Never include "Co-Authored-By: Claude" or similar
- **No tool references**: No "Generated with Claude Code" or automation mentions
- **Stage specific files**: Use `git add <file>`, NEVER `git add -A` or `git add .`
- **Verify no binaries**: Run `git diff --cached --numstat` before committing

### PR Requirements
- **Create DRAFT PR immediately** after first push: `gh pr create --draft`
- Makes work visible early
- Allows early feedback
- Shows progress and intent
- **Push to fork**: Use `git push fork <branch>`, not origin

### PR Content
- **No AI footers**: Don't add "🤖 Generated with Claude Code"
- **Professional language**: Write as a human developer would
- **Focus on what/why**: Describe changes and reasoning, not tools used

## After Execution

### Verification Checklist
- [ ] Feature branch used (not main)
- [ ] Specific files staged (no `git add -A`)
- [ ] No binaries committed
- [ ] Draft PR created immediately
- [ ] No AI mentions in commits/PRs
- [ ] Professional, tool-agnostic language

## Enforcement

If you violate ANY rule from `.clinerules/git-rules.md`:
1. **Stop execution immediately**
2. **Fix the violation** (e.g., amend commit, update PR description)
3. **Continue with corrected approach**

## Summary

Read `.clinerules/git-rules.md` → Check/create feature branch → Execute plan → Follow git rules → Create draft PR → Verify compliance
