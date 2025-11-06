<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# PR Review Toolkit - File Organization

This document explains the organized structure of the PR Review Toolkit.

---

## 📁 Complete Directory Structure

```
MorphiZen/ (Project Root)
│
├── PR_REVIEW_TOOLKIT.md              # Main toolkit overview (start here from root)
│
├── .cursor/
│   └── commands/
│       ├── review-pr.mdc             # Cursor command for reviewing PRs
│       └── submit-pr-review.mdc      # Cursor command for submitting reviews
│
├── tools/
│   └── pr-review/
│       ├── review_pr.ps1             # Main PR analysis script
│       └── submit_review.ps1         # Review submission helper script
│
└── doc/
    └── pr-review/
        ├── START_HERE.md             # Quick start guide (5-min read)
        ├── README.md                 # Complete documentation
        ├── QUICK_REFERENCE.md        # Command cheat sheet
        ├── review_template.md        # Standard review template
        ├── example_review_PR437.md   # Real review example
        └── ORGANIZATION.md           # This file
```

---

## 📂 Directory Purposes

### `.cursor/commands/` - Cursor IDE Commands
Contains Cursor-specific command files for easy access within the IDE.

**Files:**
- `review-pr.mdc` - Quick command to start PR review
- `submit-pr-review.mdc` - Quick command to submit review

**Usage in Cursor:**
- Type: `review pr 437`
- Type: `submit review 437 approve`

---

### `tools/pr-review/` - Executable Scripts
Contains all PowerShell scripts for PR review automation.

**Files:**
- `review_pr.ps1` - Main analysis script
  - Fetches PR
  - Shows diffs using merge-base
  - Displays statistics
  - Checks rebase status

- `submit_review.ps1` - Submission helper
  - Creates review from template
  - Submits to GitHub
  - Cleans up files

**Run from project root:**
```powershell
tools/pr-review/review_pr.ps1 437
tools/pr-review/submit_review.ps1 437 approve
```

---

### `doc/pr-review/` - Documentation
Contains all documentation and templates.

**Files:**

1. **START_HERE.md** - First stop for new users
   - Quick start (3 commands)
   - Common scenarios
   - First review walkthrough

2. **README.md** - Comprehensive guide
   - Complete workflows
   - Best practices
   - Troubleshooting
   - Customization options

3. **QUICK_REFERENCE.md** - Command reference
   - All git commands
   - GitHub CLI commands
   - Quick lookup table
   - Git aliases

4. **review_template.md** - Review template
   - Standardized format
   - Ensures completeness
   - Copy and fill in

5. **example_review_PR437.md** - Example review
   - Real review from PR #437
   - Shows quality standards
   - Reference for tone

6. **ORGANIZATION.md** - This file
   - Explains structure
   - Directory purposes
   - File locations

---

## 🎯 Usage Patterns

### Pattern 1: Quick Access from Root
```powershell
# From project root
tools/pr-review/review_pr.ps1 999
```

### Pattern 2: Using Cursor Commands
```
# In Cursor IDE command palette
> review pr 999
> submit review 999 approve
```

### Pattern 3: Direct Documentation
```powershell
# Open documentation
code doc/pr-review/START_HERE.md
code doc/pr-review/README.md
```

---

## 🔄 Workflow with New Structure

### Step 1: Read Overview (First Time Only)
```powershell
# From project root
code PR_REVIEW_TOOLKIT.md
```

### Step 2: Start Reviewing
```powershell
# Run analysis
tools/pr-review/review_pr.ps1 437

# Submit review
tools/pr-review/submit_review.ps1 437 approve
```

### Step 3: Reference Documentation As Needed
```powershell
# Quick reference
code doc/pr-review/QUICK_REFERENCE.md

# See example
code doc/pr-review/example_review_PR437.md
```

---

## 📝 Why This Organization?

### ✅ **Clear Separation of Concerns**
- **Scripts** in `tools/` - executable code
- **Documentation** in `doc/` - reference materials
- **IDE integration** in `.cursor/` - editor-specific

### ✅ **Standard Conventions**
- `tools/` - Common convention for utility scripts
- `doc/` - Standard documentation directory
- `.cursor/` - IDE-specific hidden directory

### ✅ **Easy to Find**
- All PR review files under clear prefixes
- No clutter in project root
- Organized by function

### ✅ **Scalable**
- Easy to add more tools
- Easy to add more documentation
- Clear namespace (pr-review/)

---

## 🔍 Finding Files

### From Project Root:
```powershell
# Scripts
tools/pr-review/review_pr.ps1
tools/pr-review/submit_review.ps1

# Documentation
doc/pr-review/START_HERE.md
doc/pr-review/README.md
doc/pr-review/QUICK_REFERENCE.md
doc/pr-review/review_template.md
doc/pr-review/example_review_PR437.md

# Cursor commands
.cursor/commands/review-pr.mdc
.cursor/commands/submit-pr-review.mdc

# Overview
PR_REVIEW_TOOLKIT.md
```

---

## 🎓 Where to Start?

### Brand New User:
1. Read: `PR_REVIEW_TOOLKIT.md` (from root)
2. Read: `doc/pr-review/START_HERE.md`
3. Try: `tools/pr-review/review_pr.ps1 <PR_NUMBER>`

### Quick Reference Needed:
- Open: `doc/pr-review/QUICK_REFERENCE.md`

### Need Examples:
- Read: `doc/pr-review/example_review_PR437.md`

### Writing Review:
- Copy: `doc/pr-review/review_template.md`

---

## 🛠️ Path References in Scripts

Scripts automatically calculate paths relative to project root:

```powershell
# In review_pr.ps1 and submit_review.ps1
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent (Split-Path -Parent $scriptDir)
$templatePath = Join-Path $projectRoot "doc/pr-review/review_template.md"
```

This means:
- ✅ Scripts work from project root
- ✅ Scripts work when called from any directory
- ✅ No hardcoded paths

---

## 📊 File Dependencies

```
review_pr.ps1
    ├── Uses: git (external)
    ├── Uses: gh (external)
    └── References: No internal files

submit_review.ps1
    ├── Uses: gh (external)
    ├── Reads: doc/pr-review/review_template.md
    └── Creates: pr<NUMBER>_review.md (temporary, in pwd)

Cursor Commands
    ├── .cursor/commands/review-pr.mdc
    │   └── References: tools/pr-review/review_pr.ps1
    └── .cursor/commands/submit-pr-review.mdc
        └── References: tools/pr-review/submit_review.ps1

Documentation
    ├── All self-contained
    └── Internal links between docs
```

---

## 🔄 Migration from Old Structure

Old (unorganized):
```
review_pr.ps1
submit_review.ps1
START_HERE.md
README_PR_REVIEW.md
QUICK_REFERENCE_PR_REVIEW.md
pr_review_template.md
PR437_Review_Summary.md
```

New (organized):
```
tools/pr-review/review_pr.ps1
tools/pr-review/submit_review.ps1
doc/pr-review/START_HERE.md
doc/pr-review/README.md
doc/pr-review/QUICK_REFERENCE.md
doc/pr-review/review_template.md
doc/pr-review/example_review_PR437.md
.cursor/commands/review-pr.mdc
.cursor/commands/submit-pr-review.mdc
PR_REVIEW_TOOLKIT.md (new overview)
```

---

## ✨ Benefits of New Structure

1. **Cleaner root directory** - Only one overview file
2. **Logical grouping** - Scripts with scripts, docs with docs
3. **IDE integration** - Cursor commands in standard location
4. **Easier maintenance** - Clear where to add new files
5. **Professional structure** - Follows common conventions
6. **Better discoverability** - Clear naming and organization

---

**Last Updated:** November 2025
**Version:** 1.0 (Reorganized Structure)
