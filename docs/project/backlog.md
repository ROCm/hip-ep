<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Project Backlog

Issue tracking: `backlog.md` (index) + `issues/NNN-name.md` (detailed files)

Active work tracked in GitHub PRs: `gh pr list --repo ROCm/onnx-hipdnn-ep`

See [CONTRIBUTING.md](CONTRIBUTING.md) for issue quality guidelines.

---

## Backlog

**Quick dependencies:**

| # | Title | Pri | Group | Blocked | Files |
|---|-------|-----|-------|---------|-------|
| [#001](issues/001-consolidate-hipdialect-td-files.md) | Consolidate HipDialect .td files into include/ | H | HipDialect | - | lib/Dialect/IR/*.td, include/hip/Dialect/IR/*.td, lib/Dialect/IR/CMakeLists.txt |

**Legend:**
- **Pri**: Priority (C=Critical, H=High, M=Medium, L=Low)
- **Group**: Feature area grouping
- **Blocked**: Issue numbers that must complete first (`-` if none)
- **Files**: Key files affected (see issue file for complete list)

---

## Completed Issues

See **[completed-issues.md](completed-issues.md)** for full archive of all completed issues.

---

## Completing an Issue

When implementation is done (code complete, tests pass), update the backlog before marking PR ready:

1. Add to completed-issues.md - Add entry to top of current month's table with: #, Author, PR, Commit, Date, Title
2. Update backlog.md - Remove from active backlog table
3. Delete issue file: `git rm docs/project/issues/NNN-*.md`
4. Commit: `git commit -m "docs: complete issue #NNN"`
5. Push
6. Mark PR ready for review

**CRITICAL: PR Title Must Include Issue Number**

When creating PRs for backlog issues, ALWAYS include the issue number in the PR title:

- ✅ CORRECT: `Issue #001: fix: consolidate HipDialect .td files into include/`
- ❌ WRONG: `Fix HipDialect .td files`

---

## Creating an Issue

1. Copy `issues/TEMPLATE.md` to `issues/NNN-name.md`
2. Fill required sections: Description, Problem, Solution, Evidence
3. Delete empty optional sections (Plans/Notes)
4. Add link to backlog.md

Numbering: 001, 002, 003, etc.

See [CONTRIBUTING.md](CONTRIBUTING.md) for quality guidelines.
