<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Quick Reference: PR Review Commands

## 🚀 Quick Start
```powershell
# Use the automated script
.\review_pr.ps1 -PRNumber 437
```

---

## 📋 Manual Review Steps

### 1. Fetch the PR
```bash
git fetch origin pull/437/head:pr-437
```

### 2. Update origin/main
```bash
git fetch origin main
```

### 3. Find merge-base (IMPORTANT!)
```bash
git merge-base pr-437 origin/main
```

### 4. Get PR changes (THE CORRECT WAY)
```bash
# Statistics
git diff $(git merge-base pr-437 origin/main) pr-437 --stat

# Full diff
git diff $(git merge-base pr-437 origin/main) pr-437

# Specific file
git diff $(git merge-base pr-437 origin/main) pr-437 -- path/to/file
```

### 5. View commit history
```bash
# Commits in PR
git log $(git merge-base pr-437 origin/main)..pr-437 --oneline

# With files
git log $(git merge-base pr-437 origin/main)..pr-437 --oneline --name-only

# What's on main but not in PR (check if rebase needed)
git log $(git merge-base pr-437 origin/main)..origin/main --oneline
```

### 6. Visual inspection
```bash
# Graph view
git log --graph --oneline --all --decorate -20

# With PR filter
git log --graph --oneline --all --decorate | Select-String -Pattern "pr-437|main" -Context 3,3
```

---

## 🔍 Detailed Code Review

### Checkout the PR branch
```bash
git stash                    # Save local changes
git checkout pr-437          # Switch to PR branch
# Review code...
git checkout main            # Back to main
git stash pop               # Restore changes
```

### View specific commits
```bash
# Single commit
git show 5eae1f3

# With stats
git show 5eae1f3 --stat

# Specific file in commit
git show 5eae1f3:path/to/file
```

### Search in PR changes
```bash
# Find specific pattern in diff
git diff $(git merge-base pr-437 origin/main) pr-437 | grep "pattern"

# Find in specific file type
git diff $(git merge-base pr-437 origin/main) pr-437 -- "*.cpp"
```

---

## 📝 Submit Review via GitHub CLI

### 1. Create review file
```bash
# Edit pr_review_template.md or create new file
notepad review.md
```

### 2. Submit review
```bash
# Approve
gh pr review 437 --approve --body-file review.md

# Request changes
gh pr review 437 --request-changes --body-file review.md

# Comment only
gh pr review 437 --comment --body-file review.md

# Inline approval (short message)
gh pr review 437 --approve --body "LGTM! Great work!"
```

### 3. Add comment to specific file/line
```bash
gh pr review 437 --comment --body "Consider adding error handling here" \
  --file mlir-imp/src/mlir-node-arg.cpp --line 45
```

### 4. View PR
```bash
# Summary
gh pr view 437

# Full details
gh pr view 437 --web
```

---

## 🛠️ Useful Git Aliases (Optional)

Add to `~/.gitconfig`:

```ini
[alias]
    pr-fetch = "!f() { git fetch origin pull/$1/head:pr-$1; }; f"
    pr-diff = "!f() { git diff $(git merge-base pr-$1 origin/main) pr-$1; }; f"
    pr-stat = "!f() { git diff $(git merge-base pr-$1 origin/main) pr-$1 --stat; }; f"
    pr-log = "!f() { git log $(git merge-base pr-$1 origin/main)..pr-$1 --oneline; }; f"
    pr-base = "!f() { git merge-base pr-$1 origin/main; }; f"
```

Usage:
```bash
git pr-fetch 437      # Fetch PR #437
git pr-stat 437       # Show stats
git pr-diff 437       # Show full diff
git pr-log 437        # Show commits
git pr-base 437       # Show merge-base
```

---

## 📊 Check PR Status

### PR information
```bash
# View PR
gh pr view 437

# List all open PRs
gh pr list

# List PRs by state
gh pr list --state open
gh pr list --state closed

# Search PRs
gh pr list --search "MLIR"
```

### PR checks
```bash
# View CI/CD status
gh pr checks 437

# View specific check
gh pr checks 437 --watch
```

---

## ⚠️ Common Mistakes to Avoid

### ❌ DON'T DO THIS:
```bash
# This compares two diverged branches, not the PR changes!
git diff origin/main pr-437
git diff origin/main...pr-437
```

### ✅ ALWAYS DO THIS:
```bash
# This shows only the changes introduced by the PR
git diff $(git merge-base pr-437 origin/main) pr-437
```

---

## 🎯 PR Review Checklist

- [ ] Fetched latest PR and origin/main
- [ ] Checked merge-base to understand branch point
- [ ] Reviewed actual PR changes using merge-base
- [ ] Checked if PR needs rebasing (main ahead?)
- [ ] Reviewed all commits in PR
- [ ] Checked code quality (readability, maintainability)
- [ ] Verified no breaking changes
- [ ] Checked test coverage
- [ ] Verified PR description follows template
- [ ] Submitted review via `gh pr review`
- [ ] Cleaned up local PR branch (if checked out)

---

## 🔄 After Review

### If approved and ready to merge
```bash
# Merge via CLI (if you have permissions)
gh pr merge 437 --merge
gh pr merge 437 --squash
gh pr merge 437 --rebase
```

### Clean up local PR branch
```bash
git branch -D pr-437
```

---

## 📚 Resources

- GitHub CLI: https://cli.github.com/manual/
- Git Diff: https://git-scm.com/docs/git-diff
- Git Log: https://git-scm.com/docs/git-log
- Git Merge-Base: https://git-scm.com/docs/git-merge-base

---

## 💡 Pro Tips

1. **Always use merge-base** for accurate PR diffs
2. **Check both PR commits AND main commits** to see if rebase needed
3. **Use `--stat` first** to get overview, then full diff
4. **Create review file** for complex reviews (easier to edit)
5. **Use visual graph** to understand branch relationships
6. **Automate with scripts** for consistent reviews
