<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Project Backlog

Issue tracking: `backlog.md` (index) + `issues/NNN-name.md` (detailed files)

Active work tracked in GitHub PRs: `gh pr list --repo ROCm/MorphiZen`

See [CONTRIBUTING.md](CONTRIBUTING.md) for issue quality guidelines.

---

## Backlog

**Quick dependencies:**
- **#052 relates to #050, #051** - All improve TarFile implementation quality
- **#053 relates to #052** - Documents architecture that #052 will modify
- **#056 relates to #051** - Both improve TarFile documentation quality

See [issue-dependency-analysis.md](issue-dependency-analysis.md) for details.

| # | Title | Pri | Est | Group | Blocked | Files |
|---|-------|-----|-----|-------|---------|-------|
| [#019](issues/019-refactor-initialize-context.md) | Refactor initialize_context | L | 2h | Config/Context | - | morphizen_compile_model.cpp |
| [#026](issues/026-mlir-model-export-api.md) | MLIR Model Export API | M | 1h | MLIR Features | - | morphizen-ort-api-ext.hpp/mlir-model.cpp |
| [#029](issues/029-passcontext-tarfile-initialization.md) | PassContext tarfile Init | M | 1h | Test Infrastructure | - | test_pass_context.cpp/pass_context_imp.cpp |
| [#030](issues/030-morphizen-pass-init-plugin-loading.md) | morphizen-pass_init Plugin | M | 1h | Test Infrastructure | - | pass_imp.cpp/CMakeLists.txt |
| [#031](issues/031-target-auto-discovery-failure.md) | Target Auto-Discovery Fail | M | 1h | Test Infrastructure | - | pass_context_imp.cpp/test-hello-ep.cpp |
| [#032](issues/032-ep-context-model-generation-failure.md) | EP Context Model Gen Fail | M | 2h | Test Infrastructure | #030/#031 | graph_partitioner/session.cpp |
| [#034](issues/034-mlir-shape-nullptr-check-failure.md) | MLIR Shape Nullptr Check | M | 1h | MLIR Testing | - | mlir-graph.cpp/Casting.h |
| [#035](issues/035-constdatatest-boost-dependency.md) | ConstDataTest Boost Dep | L | 1h | Test Infrastructure | - | test_const_data.cpp/CMakeLists.txt |
| [#036](issues/036-boost-dependent-tests-skipped.md) | Boost-Dependent Tests | L | 1h | Test Infrastructure | - | test_graph.cpp/test_tar_file.cpp |
| [#051](issues/051-document-complex-tarfile-factory-methods.md) | Document Factory Methods | M | 1h | TarFile Docs | - | tar_file.cpp |
| [#052](issues/052-eliminate-redundant-cached-stream-pointers.md) | Eliminate Redundant Cached Stream Pointers | L | 1h | TarFile Refactor | - | tar_file.hpp/.cpp/tar_entry.hpp/.cpp |
| [#053](issues/053-document-tar-streaming-architecture.md) | Document TAR Streaming | M | 2h | TarFile Docs | - | tar-streaming-architecture.md |
| [#056](issues/056-add-comprehensive-documentation-to-tarfile.md) | Add TarFile Docs | M | 2h | TarFile Docs | - | tar_file.hpp |

**Legend:**
- **Pri**: Priority (C=Critical, H=High, M=Medium, L=Low)
- **Est**: Estimated time (15m=15 minutes, 1h=1 hour, etc.)
- **Group**: Feature area grouping
- **Blocked**: Issue numbers that must complete first (`-` if none)
- **Files**: Key files affected (see issue file for complete list)

---

## Completed Issues

See **[completed-issues.md](completed-issues.md)** for full archive of all completed issues.

**Recent (last 5):**
- #060 (PR #179) - Create morphizen-foundation Library
- #059 (PR #179) - Component Organization Guidelines
- #061 (PR #173) - Windows CI MSBuild Generator Compatibility
- #064 (PR #170) - Enable LLVM/MLIR Smart Fallback with Auto-Fetch
- #061 (PR #168) - Remove Misleading Shape Inference Code

---

## Completing an Issue

When implementation is done (code complete, tests pass), update the backlog before marking PR ready:

1. Add to completed-issues.md - Add entry to top of current month's table with: #, Author, PR, Commit, Date, Title
2. Update backlog.md - Add to "Recent (last 5)" list in compact format, remove oldest if >5
3. Remove from backlog.md - Delete from active backlog table
4. Delete issue file: `git rm docs/project/issues/042-*.md`
5. Remove from dependencies - Clean up any references in "Quick dependencies" section
6. Commit: `git commit -m "docs: complete issue #042"`
7. Push
8. Mark PR ready for review

**CRITICAL: PR Title Must Include Issue Number**

When creating PRs for backlog issues, ALWAYS include the issue number in the PR title:

- ✅ CORRECT: `Issue #006: Remove legacy cache_dir system (~220-280 LOC)`
- ❌ WRONG: `Remove legacy cache_dir system (~220-280 LOC)`

This ensures PR is easily linked to the issue and makes tracking easier.

---

## Creating an Issue

1. Copy `issues/TEMPLATE.md` to `issues/NNN-name.md`
2. Fill required sections: Description, Problem, Solution, Evidence
3. Delete empty optional sections (Plans/Sessions/PRs/Notes)
4. Add link to backlog.md

Numbering: 001, 002, 003, etc.

See [CONTRIBUTING.md](CONTRIBUTING.md) for quality guidelines.
