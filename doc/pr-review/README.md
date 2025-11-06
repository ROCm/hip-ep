<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# PR Review Toolkit

A comprehensive set of tools and documentation for conducting thorough pull request reviews.

---

## 📁 Files in This Toolkit

| File | Purpose | When to Use |
|------|---------|-------------|
| **review_pr.ps1** | Automated review analysis | Start here - analyzes any PR |
| **submit_review.ps1** | Submit review helper | After writing review |
| **pr_review_template.md** | Review template | When writing detailed reviews |
| **QUICK_REFERENCE_PR_REVIEW.md** | Command cheat sheet | Quick lookup for specific commands |
| **Example_PR437_Review.md** | Example complete review | Reference for quality reviews |
| **README_PR_REVIEW.md** | This file - complete guide | Learning the toolkit |

---

## 🚀 Quick Start

### Option 1: Automated (Recommended)
```powershell
# Review any PR with one command
.\review_pr.ps1 437

# Or without specifying number (will prompt)
.\review_pr.ps1
```

### Option 2: Manual Review
Follow the commands in `QUICK_REFERENCE_PR_REVIEW.md`

---

## 📋 Complete Review Workflow

### Step 1: Analyze the PR
```powershell
# Run the automated script
.\review_pr.ps1 <PR_NUMBER>
```

This will show you:
- ✅ Commits in the PR
- ✅ Files changed with statistics
- ✅ Whether PR needs rebasing
- ✅ Visual commit graph
- ✅ PR details and status

### Step 2: Deep Dive (Optional)
```powershell
# The script will ask if you want to checkout the branch
# Choose 'y' to checkout and review code locally
```

### Step 3: Write Your Review
```powershell
# Option A: Manual
cp pr_review_template.md pr999_review.md
notepad pr999_review.md

# Option B: Use the submit script (will help you create it)
.\submit_review.ps1 999
```

### Step 4: Submit Review
```powershell
# Easy way - Use the helper script
.\submit_review.ps1 999 approve    # Approve
.\submit_review.ps1 999 changes    # Request changes
.\submit_review.ps1 999 comment    # Comment only

# Manual way - Direct GitHub CLI
gh pr review 999 --approve --body-file pr999_review.md
gh pr review 999 --request-changes --body-file pr999_review.md
gh pr review 999 --comment --body-file pr999_review.md

# Quick approve (one-liner)
gh pr review 999 --approve --body "LGTM! Great work! ✅"
```

### Step 5: Clean Up (If you checked out the branch)
```powershell
git checkout main
git stash pop  # Restore your local changes
git branch -D pr-999  # Delete local PR branch
```

---

## 🎯 Best Practices

### ✅ DO:
- **Always use merge-base** to compare PR changes
- **Check if rebase is needed** (script does this automatically)
- **Review all commits** in the PR, not just the final diff
- **Follow the PR template** when suggesting description improvements
- **Be constructive** in your feedback
- **Test locally** for complex changes

### ❌ DON'T:
- Don't compare PR directly with main: `git diff origin/main pr-999` ❌
- Don't skip checking the merge-base
- Don't forget to check if tests are included
- Don't approve without understanding the changes
- Don't leave vague comments like "fix this"

---

## 📊 Review Quality Checklist

Use this for every review:

- [ ] **Understand the change**: What problem does it solve?
- [ ] **Check correctness**: Does the code do what it's supposed to?
- [ ] **Verify tests**: Are there tests for new functionality?
- [ ] **Code quality**: Is it readable and maintainable?
- [ ] **Breaking changes**: Any backward compatibility issues?
- [ ] **Performance**: Any performance implications?
- [ ] **Security**: Any security concerns?
- [ ] **Documentation**: Is PR description complete and accurate?
- [ ] **Rebase needed**: Is PR based on latest main?

---

## 🛠️ Customization

### Add Git Aliases (Optional but Recommended)

Edit `~/.gitconfig`:

```ini
[alias]
    # PR Review aliases
    pr-fetch = "!f() { git fetch origin pull/$1/head:pr-$1; }; f"
    pr-diff = "!f() { git diff $(git merge-base pr-$1 origin/main) pr-$1; }; f"
    pr-stat = "!f() { git diff $(git merge-base pr-$1 origin/main) pr-$1 --stat; }; f"
    pr-log = "!f() { git log $(git merge-base pr-$1 origin/main)..pr-$1 --oneline; }; f"
    pr-base = "!f() { git merge-base pr-$1 origin/main; }; f"
    pr-cleanup = "!f() { git checkout main && git branch -D pr-$1; }; f"
```

Then use:
```bash
git pr-fetch 999
git pr-stat 999
git pr-diff 999
git pr-cleanup 999
```

### Create Review Shortcuts

Add to your PowerShell profile (`$PROFILE`):

```powershell
# Quick PR review
function Review-PR {
    param([int]$Number)
    & "$PSScriptRoot\review_pr.ps1" $Number
}

# Quick approve
function Approve-PR {
    param([int]$Number, [string]$Message = "LGTM! ✅")
    gh pr review $Number --approve --body $Message
}

Set-Alias rpr Review-PR
Set-Alias apr Approve-PR
```

Then use:
```powershell
rpr 999          # Review PR 999
apr 999          # Quick approve
apr 999 "Great work on the refactoring!"  # Approve with custom message
```

---

## 📚 Learning Resources

### Understanding the Tools

**Merge-Base**: The common ancestor between two branches
```bash
git merge-base pr-999 origin/main
# Returns: abc1234 (the commit where the PR branched off)
```

**Why Merge-Base Matters**:
- Shows ONLY what the PR introduces
- Ignores commits that happened on main after PR creation
- Essential for accurate reviews

**Diagram**:
```
         origin/main
              |
              E (latest main)
              |
              D (commit after PR branched)
             /
    merge-base C (where PR branched from)
            |   \
            |    F (PR commit 1)
            |    |
            B    G (PR commit 2)
            |
            A
```

Comparing:
- ❌ `git diff E G` - Shows D, E, F, G (confusing!)
- ✅ `git diff C G` - Shows only F, G (PR changes only!)

### GitHub CLI Basics

```bash
# View PR
gh pr view 999

# List PRs
gh pr list
gh pr list --state open
gh pr list --author username

# Check status
gh pr checks 999

# View in browser
gh pr view 999 --web
```

---

## 🎓 Example Reviews

See `PR437_Review_Summary.md` for a complete example of:
- Thorough code analysis
- Clear categorization of changes
- Constructive recommendations
- Proposed PR description improvements
- Professional review tone

---

## 🔧 Troubleshooting

### Problem: Script won't run
**Solution**:
```powershell
# Enable script execution (run as Administrator)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Problem: `gh` command not found
**Solution**: Install GitHub CLI
```powershell
winget install --id GitHub.cli
# Or download from: https://cli.github.com/
```

### Problem: PR already checked out locally
**Solution**:
```bash
git checkout main
git branch -D pr-999
# Then run review_pr.ps1 again
```

### Problem: Can't fetch PR
**Solution**:
```bash
# Ensure you have access to the repository
git remote -v

# Update remote if needed
git fetch origin

# Check if PR exists
gh pr view 999
```

---

## 📈 Review Metrics (Optional)

Track your review effectiveness:

```powershell
# Create a review log
$review = @{
    PRNumber = 999
    Date = Get-Date
    TimeSpent = "30 minutes"
    IssuesFound = 2
    Status = "Approved"
}
$review | Export-Csv -Path review_log.csv -Append
```

---

## 🤝 Contributing to This Toolkit

Have improvements? Update the files:

1. **review_pr.ps1** - Add new automation features
2. **pr_review_template.md** - Improve the template
3. **QUICK_REFERENCE_PR_REVIEW.md** - Add new commands
4. **README_PR_REVIEW.md** - Update documentation

---

## 📞 Quick Help

| Need | Command |
|------|---------|
| Review a PR | `.\review_pr.ps1 999` |
| View PR | `gh pr view 999` |
| Approve quickly | `gh pr review 999 --approve --body "LGTM"` |
| Request changes | `gh pr review 999 --request-changes --body-file review.md` |
| Check my reviews | `gh pr list --search "reviewed-by:@me"` |
| Clean up PR branch | `git checkout main && git branch -D pr-999` |

---

## 🎯 Pro Tips

1. **Use the script for every PR** - Consistency is key
2. **Template saves time** - Copy and fill in, don't start from scratch
3. **Test locally when unsure** - Checkout the branch and run tests
4. **Be specific** - Don't say "fix this", explain what and why
5. **Approve with confidence** - Only approve what you understand
6. **Link to standards** - Reference coding guidelines in your reviews
7. **Suggest, don't demand** - Use "Consider..." instead of "You must..."
8. **Balance criticism** - Mention good things too, not just problems

---

## 📝 Version History

- **v1.0** - Initial toolkit creation (Nov 2025)
  - Automated review script
  - Review template
  - Quick reference guide
  - Example review (PR #437)

---

**Happy Reviewing! 🚀**

For questions or improvements, update this toolkit and share with your team.
