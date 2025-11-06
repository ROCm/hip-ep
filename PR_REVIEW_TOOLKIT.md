<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# 🚀 PR Review Toolkit

A comprehensive, automated toolkit for conducting thorough and consistent GitHub pull request reviews.

---

## 📁 Directory Structure

```
MorphiZen/
├── .cursor/
│   └── commands/
│       ├── review-pr.mdc           # Cursor command: review PR
│       └── submit-pr-review.mdc    # Cursor command: submit review
│
├── tools/
│   └── pr-review/
│       ├── review_pr.ps1           # Main analysis script
│       └── submit_review.ps1       # Review submission helper
│
└── doc/
    └── pr-review/
        ├── START_HERE.md           # Quick start guide
        ├── README.md               # Complete documentation
        ├── QUICK_REFERENCE.md      # Command cheat sheet
        ├── review_template.md      # Review template
        └── example_review_PR437.md # Example review
```

---

## ⚡ Quick Start

### Step 1: Analyze a PR
```powershell
tools/pr-review/review_pr.ps1 437
```

### Step 2: Submit Your Review
```powershell
tools/pr-review/submit_review.ps1 437 approve
```

**That's it!** Works for any PR number.

---

## 📚 Documentation

| Document | Purpose | Link |
|----------|---------|------|
| **Quick Start** | Get started in 5 minutes | [doc/pr-review/START_HERE.md](doc/pr-review/START_HERE.md) |
| **Complete Guide** | Comprehensive documentation | [doc/pr-review/README.md](doc/pr-review/README.md) |
| **Command Reference** | All commands in one place | [doc/pr-review/QUICK_REFERENCE.md](doc/pr-review/QUICK_REFERENCE.md) |
| **Review Template** | Template for writing reviews | [doc/pr-review/review_template.md](doc/pr-review/review_template.md) |
| **Example Review** | See a complete review example | [doc/pr-review/example_review_PR437.md](doc/pr-review/example_review_PR437.md) |

---

## 🔧 Tools

### 1. Review Analysis Script
**Location:** `tools/pr-review/review_pr.ps1`

Automatically analyzes any PR:
- Fetches PR and updates main
- Finds merge-base (correct way to compare!)
- Shows commits, diffs, and statistics
- Checks if rebase is needed
- Provides next-step commands

**Usage:**
```powershell
tools/pr-review/review_pr.ps1 <PR_NUMBER>
```

### 2. Review Submission Script
**Location:** `tools/pr-review/submit_review.ps1`

Helps you write and submit reviews:
- Creates review from template
- Supports approve/request-changes/comment
- Submits via GitHub CLI
- Cleans up after submission

**Usage:**
```powershell
tools/pr-review/submit_review.ps1 <PR_NUMBER> <approve|changes|comment>
```

---

## 🎯 Common Scenarios

### Quick Approval
```powershell
# Analyze
tools/pr-review/review_pr.ps1 999

# Quick approve
gh pr review 999 --approve --body "LGTM! ✅"
```

### Detailed Review
```powershell
# Analyze
tools/pr-review/review_pr.ps1 999

# Submit detailed review
tools/pr-review/submit_review.ps1 999 approve
```

### Request Changes
```powershell
# Analyze
tools/pr-review/review_pr.ps1 999

# Request changes
tools/pr-review/submit_review.ps1 999 changes
```

---

## 💡 Key Features

### ✅ **Correct Git Comparison**
Uses merge-base to show ONLY the PR changes, not divergence between branches.

### ✅ **Completely Generic**
Works with any PR number in any repository.

### ✅ **Standardized Reviews**
Template ensures consistent, thorough reviews every time.

### ✅ **Time Saving**
Automated analysis and template-based reviews save hours.

### ✅ **Best Practices Built-In**
Follows Git and GitHub best practices automatically.

---

## 🎓 Learn More

1. **First time?** Start here: [doc/pr-review/START_HERE.md](doc/pr-review/START_HERE.md)
2. **Need details?** Read: [doc/pr-review/README.md](doc/pr-review/README.md)
3. **Quick lookup?** Use: [doc/pr-review/QUICK_REFERENCE.md](doc/pr-review/QUICK_REFERENCE.md)

---

## ⚙️ Prerequisites

- ✅ Git
- ✅ GitHub CLI (`gh`) - [Install](https://cli.github.com/)
- ✅ PowerShell
- ✅ Repository access

### Install GitHub CLI
```powershell
winget install --id GitHub.cli
```

---

## 🆘 Troubleshooting

| Issue | Solution |
|-------|----------|
| Script won't run | `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser` |
| `gh` not found | Install GitHub CLI |
| Can't fetch PR | Check repository access with `git remote -v` |
| Template not found | Ensure you're running from project root |

---

## 📊 Workflow Diagram

```
┌──────────────────────────────────────────┐
│  1. Run: tools/pr-review/review_pr.ps1  │
│     • Fetches PR                         │
│     • Shows diffs and commits            │
│     • Checks rebase status               │
└──────────────────────────────────────────┘
                  │
                  ▼
┌──────────────────────────────────────────┐
│  2. Analyze Changes                      │
│     • Read diffs                         │
│     • Check code quality                 │
│     • Verify tests                       │
└──────────────────────────────────────────┘
                  │
                  ▼
┌──────────────────────────────────────────┐
│  3. Write Review                         │
│     • Use template                       │
│     • Be specific and constructive       │
│     • Include recommendations            │
└──────────────────────────────────────────┘
                  │
                  ▼
┌──────────────────────────────────────────┐
│  4. Run: tools/pr-review/submit_review   │
│     • Submits your review                │
│     • Choose status: approve/changes     │
└──────────────────────────────────────────┘
```

---

## 🎯 One-Liners

```powershell
# Complete review workflow
tools/pr-review/review_pr.ps1 999 && tools/pr-review/submit_review.ps1 999 approve

# Quick approve
gh pr review 999 --approve --body "LGTM! ✅"

# View pending reviews
gh pr list --search "review-requested:@me"

# Check PR status
gh pr view 999
```

---

## 🔍 What Makes This Different?

### ❌ Common Mistakes (This Toolkit Prevents)
- Comparing PR directly with main: `git diff origin/main pr-999`
- Forgetting to check merge-base
- Inconsistent review quality
- Missing important checks

### ✅ This Toolkit Does It Right
- Always uses merge-base for accurate comparison
- Standardized template ensures completeness
- Automated checks for common issues
- Best practices built into workflows

---

## 📈 Example

See a complete, real review: [doc/pr-review/example_review_PR437.md](doc/pr-review/example_review_PR437.md)

This example shows:
- Thorough code analysis
- Clear categorization of changes
- Constructive recommendations
- PR description improvements
- Professional review tone

---

## 🤝 Contributing

To improve this toolkit:

1. Update scripts: `tools/pr-review/`
2. Improve docs: `doc/pr-review/`
3. Add commands: `.cursor/commands/`

---

## 📝 Version

**Version:** 1.0
**Created:** November 2025
**Based on:** PR #437 review experience

---

## 🎉 Get Started Now!

```powershell
# Read the quick start guide
code doc/pr-review/START_HERE.md

# Or dive right in - review your first PR
tools/pr-review/review_pr.ps1 <PR_NUMBER>
```

Happy reviewing! 🚀
