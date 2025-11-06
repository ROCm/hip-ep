<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# 🚀 PR Review Toolkit - START HERE

Welcome! This toolkit helps you conduct thorough, consistent pull request reviews.

---

## ⚡ Quick Start (3 Commands)

```powershell
# 1. Analyze the PR (run from project root)
tools/pr-review/review_pr.ps1 999

# 2. Write and submit your review
tools/pr-review/submit_review.ps1 999 approve

# Done! 🎉
```

---

## 📚 What's Included

### 🔧 **Scripts** (Use These)
1. **review_pr.ps1** - Analyzes any PR automatically
   - Shows commits, diffs, files changed
   - Checks if rebase needed
   - Provides review guidance

2. **submit_review.ps1** - Submits your review
   - Helps create review from template
   - Supports approve/changes/comment
   - Cleans up after submission

### 📖 **Documentation** (Read These)
1. **README_PR_REVIEW.md** - Complete guide with best practices
2. **QUICK_REFERENCE_PR_REVIEW.md** - Command cheat sheet
3. **pr_review_template.md** - Template for writing reviews
4. **Example_PR437_Review.md** - Real review example

---

## 🎯 Common Scenarios

### Scenario 1: Quick Review & Approve
```powershell
tools/pr-review/review_pr.ps1 999                    # Analyze
gh pr review 999 --approve --body "LGTM! ✅"        # Quick approve
```

### Scenario 2: Detailed Review
```powershell
tools/pr-review/review_pr.ps1 999                    # Analyze
# Review code locally if needed (script will ask)
tools/pr-review/submit_review.ps1 999 approve        # Submit detailed review
```

### Scenario 3: Request Changes
```powershell
tools/pr-review/review_pr.ps1 999                    # Analyze
tools/pr-review/submit_review.ps1 999 changes        # Submit with change requests
```

### Scenario 4: Just Comment
```powershell
tools/pr-review/review_pr.ps1 999                    # Analyze
tools/pr-review/submit_review.ps1 999 comment        # Comment without approval
```

---

## 📋 Typical Workflow

```
┌────────────────────────────────────────────────────┐
│  1. Run tools/pr-review/review_pr.ps1 <PR_NUMBER> │
│     - Fetches PR                                   │
│     - Shows diffs and commits                      │
│     - Checks if rebase needed                      │
└────────────────────────────────────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────────────────┐
│  2. Analyze the Changes                            │
│     - Read the diffs                               │
│     - Checkout branch if needed                    │
│     - Test locally if complex                      │
└────────────────────────────────────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────────────────┐
│  3. Write Review                                   │
│     - Use doc/pr-review/review_template.md         │
│     - Fill in your findings                        │
│     - Be specific and constructive                 │
└────────────────────────────────────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────────────────┐
│  4. Run tools/pr-review/submit_review.ps1 <STATUS> │
│     - Submits your review                          │
│     - Choose: approve/changes/comment              │
└────────────────────────────────────────────────────┘
```

---

## 🎓 Learn More

- **First time?** Read [README.md](README.md) for complete guide
- **Need commands?** Check [QUICK_REFERENCE.md](QUICK_REFERENCE.md)
- **Want example?** See [example_review_PR437.md](example_review_PR437.md)
- **Writing review?** Use [review_template.md](review_template.md)

---

## ⚙️ Prerequisites

Make sure you have:
- ✅ Git installed
- ✅ GitHub CLI (`gh`) installed - https://cli.github.com/
- ✅ PowerShell (Windows has it by default)
- ✅ Access to the repository

### Install GitHub CLI (if needed)
```powershell
winget install --id GitHub.cli
# Or download from: https://cli.github.com/
```

### Enable Scripts (if needed)
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

## 💡 Pro Tips

1. **Use tools/pr-review/review_pr.ps1 for EVERY review** - Consistency matters
2. **Always check the merge-base** - Script does this automatically
3. **Be specific in feedback** - Don't just say "fix this"
4. **Test locally for complex changes** - Checkout the branch
5. **Follow the template** - In doc/pr-review/review_template.md
6. **Check if rebase needed** - Script warns you
7. **Run from project root** - All paths are relative to root

---

## 🎯 One-Liner Reference

```powershell
# Complete review in one go (run from project root)
tools/pr-review/review_pr.ps1 999; tools/pr-review/submit_review.ps1 999 approve

# Quick approve without detailed review (use sparingly!)
gh pr view 999; gh pr review 999 --approve --body "LGTM! ✅"

# View all your pending reviews
gh pr list --search "review-requested:@me"
```

---

## 🆘 Need Help?

| Problem | Solution |
|---------|----------|
| Script won't run | `Set-ExecutionPolicy RemoteSigned` |
| `gh` not found | Install GitHub CLI |
| Can't fetch PR | Check repository access |
| Merge-base error | Update origin/main: `git fetch origin main` |
| PR already exists locally | `git branch -D pr-999` and try again |

---

## 📁 File Guide

```
MorphiZen/  (Project Root)
├── .cursor/commands/
│   ├── review-pr.mdc                ← Cursor command: review
│   └── submit-pr-review.mdc         ← Cursor command: submit
├── tools/pr-review/
│   ├── review_pr.ps1                ← Main analysis script
│   └── submit_review.ps1            ← Review submission helper
└── doc/pr-review/
    ├── START_HERE.md                ← You are here!
    ├── README.md                    ← Complete documentation
    ├── QUICK_REFERENCE.md           ← Command cheat sheet
    ├── review_template.md           ← Template for reviews
    └── example_review_PR437.md      ← Real review example
```

---

## 🚦 Your First Review (Step by Step)

1. **Open PowerShell in the project root**
   ```powershell
   cd path\to\your\project
   ```

2. **Get a PR number to review** (e.g., #999)

3. **Run the analysis**
   ```powershell
   tools/pr-review/review_pr.ps1 999
   ```

4. **Review the output carefully**
   - Check what files changed
   - Look at the commit messages
   - Note if rebase is needed

5. **Checkout branch if you want to test locally**
   - Script will ask you
   - Say 'y' to checkout

6. **Write your review**
   ```powershell
   # Use the submit script - it will help you create the review
   tools/pr-review/submit_review.ps1 999 approve

   # Or manually:
   cp doc/pr-review/review_template.md pr999_review.md
   notepad pr999_review.md
   gh pr review 999 --approve --body-file pr999_review.md
   ```

7. **Done!** 🎉

---

## ✨ Best Practices Summary

### ✅ DO:
- Use the scripts - they ensure consistency
- Check the merge-base (scripts do this)
- Be constructive and specific
- Test complex changes locally
- Follow the review template

### ❌ DON'T:
- Skip using the scripts
- Compare PR directly with main (use merge-base!)
- Leave vague feedback
- Approve without understanding
- Forget to check for tests

---

## 🎉 You're Ready!

Start reviewing PRs with confidence. The scripts handle the complexity, you focus on the code quality.

**Next Steps:**
1. Try reviewing a PR: `tools/pr-review/review_pr.ps1 <PR_NUMBER>`
2. Read the complete guide: [README.md](README.md)
3. Bookmark the quick reference: [QUICK_REFERENCE.md](QUICK_REFERENCE.md)

Happy reviewing! 🚀

---

**Version:** 1.0
**Last Updated:** November 2025
**Questions?** Check `README_PR_REVIEW.md` or update this toolkit
