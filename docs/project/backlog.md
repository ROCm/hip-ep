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
- **#003 blocks #004** ⚠️ - Must complete #003 first
- **#003 influences #005, #007** - Should coordinate
- **#006 relates to #009, #010, #011** - Cache cleanup group
- **#008 is independent**
- **#047 relates to #044** - Both improve TarFile deduplication code
- **#052 relates to #050, #051** - All improve TarFile implementation quality
- **#053 relates to #052** - Documents architecture that #052 will modify
- **#054 relates to #050, #055** - All reduce TarFile public API surface
- **#055 relates to #054, #050** - All reduce TarFile public API surface
- **#056 relates to #051** - Both improve TarFile documentation quality
- **#054 relates to #050** - Both reduce TarFile public API surface
- **#057 relates to /pick-issue skill** - Backlog format optimization enables efficient AI parsing
- **#058 relates to #057** - Uses optimized backlog format for efficient issue selection
- **#059 relates to #058** - Handles Phase 9 (post-review conflict/CI resolution) after /fix-issue completes

See [issue-dependency-analysis.md](issue-dependency-analysis.md) for details.

| # | Title | Pri | Est | Group | Blocked | Files |
|---|-------|-----|-----|-------|---------|-------|
| [#001](issues/001-mmap-support-for-embed-mode.md) | Add mmap Support | M | 3h | Feature | - | mmap_file.cpp/tar_file.cpp |
| [#009](issues/009-targetproto-provider-options-injection-cleanup.md) | TargetProto Provider Opts | L | 30m | Provider Options | - | pass_context.hpp/config.cpp |
| [#012](issues/012-session-configs-swapping.md) | session_configs Swapping | H | 2h | Config/Context | #008/#009 | pass_context_imp.cpp/config.proto |
| [#013](issues/013-provider-options-aggregation.md) | provider_options Aggregation | M | 1h | Provider Options | #003/#008/#009 | pass_context_imp.cpp |
| [#014](issues/014-dynamic-pass-registration.md) | Dynamic Pass Registration | M | 1h | Config/Context | - | pass_imp.cpp/config.cpp |
| [#017](issues/017-remove-update-config-by-target.md) | Remove update_config_by_target | L | 15m | Config/Context | #007/#014 | config.cpp |
| [#018](issues/018-make-configproto-const.md) | Make ConfigProto const | L | 15m | Config/Context | #004-#017 | pass_context_imp.hpp |
| [#019](issues/019-refactor-initialize-context.md) | Refactor initialize_context | L | 2h | Config/Context | #005/#006/#017 | morphizen_compile_model.cpp |
| [#024](issues/024-remove-legacy-compile-entry-points.md) | Remove Legacy Compile API | M | 1h | Legacy Cleanup | - | onnxruntime_morphizen_ep.hpp/.cpp |
| [#026](issues/026-mlir-model-export-api.md) | MLIR Model Export API | M | 1h | MLIR Features | - | morphizen-ort-api-ext.hpp/mlir-model.cpp |
| [#027](issues/027-eliminate-c-style-apis-from-morphizen-graph.md) | Eliminate C-Style APIs | M | 3h+ | Legacy Cleanup | - | graph.hpp/pass.cpp/node_builder.cpp |
| [#028](issues/028-mlir-backend-test-model-support.md) | ResNet50 MLIR Test Model | H | 1h | MLIR Testing | - | test_environment.hpp/mlir-model.cpp |
| [#029](issues/029-passcontext-tarfile-initialization.md) | PassContext tarfile Init | M | 1h | Test Infrastructure | - | test_pass_context.cpp/pass_context_imp.cpp |
| [#030](issues/030-morphizen-pass-init-plugin-loading.md) | morphizen-pass_init Plugin | M | 1h | Test Infrastructure | - | pass_imp.cpp/CMakeLists.txt |
| [#031](issues/031-target-auto-discovery-failure.md) | Target Auto-Discovery Fail | M | 1h | Test Infrastructure | - | pass_context_imp.cpp/test-hello-ep.cpp |
| [#032](issues/032-ep-context-model-generation-failure.md) | EP Context Model Gen Fail | M | 2h | Test Infrastructure | #028/#030/#031 | graph_partitioner/session.cpp |
| [#033](issues/033-ep-duplicate-registration.md) | EP Duplicate Registration | M | 30m | Test Infrastructure | - | morphizen_unit_test_main.cpp/env.cpp |
| [#034](issues/034-mlir-shape-nullptr-check-failure.md) | MLIR Shape Nullptr Check | M | 1h | MLIR Testing | - | mlir-graph.cpp/Casting.h |
| [#035](issues/035-constdatatest-boost-dependency.md) | ConstDataTest Boost Dep | L | 1h | Test Infrastructure | - | test_const_data.cpp/CMakeLists.txt |
| [#036](issues/036-boost-dependent-tests-skipped.md) | Boost-Dependent Tests | L | 1h | Test Infrastructure | - | test_graph.cpp/test_tar_file.cpp |
| [#038](issues/038-document-lazy-symlink-const-cast.md) | Document Lazy Symlink | L | 30m | TarFile Docs | - | tar_file.cpp/tar_entry.hpp |
| [#039](issues/039-fix-tarfile-typos-and-outdated-comments.md) | Fix TarFile Typos | L | 15m | TarFile Docs | - | tar_file.hpp/tar_entry.cpp |
| [#040](issues/040-remove-legacy-tar-ball-reserialization.md) | Remove tar_ball Reserial | M | 2h | TarFile Refactor | - | tar_ball.cpp/.hpp/util.cpp |
| [#042](issues/042-document-privatetag-factory-pattern.md) | Document PrivateTag | L | 30m | TarFile Docs | - | tar_file.hpp/privatetag-doc.md |
| [#044](issues/044-use-erase-remove-idiom-in-tarfile.md) | Use Erase-Remove Idiom | L | 15m | TarFile Cleanup | - | tar_file.cpp |
| [#047](issues/047-extract-duplicate-entry-deduplication-logic.md) | Extract Entry Dedup Logic | M | 30m | TarFile Refactor | #044 | tar_file.cpp/.hpp |
| [#048](issues/048-extract-platform-specific-tmpfile-helper.md) | Extract tmpfile Helper | L | 30m | TarFile Refactor | - | util.hpp/tar_file.cpp/temp_file_stream.cpp |
| [#050](issues/050-standardize-tarfile-factory-method-naming.md) | Standardize Factory Naming | M | 1h | TarFile Refactor | - | tar_file.hpp/.cpp |
| [#051](issues/051-document-complex-tarfile-factory-methods.md) | Document Factory Methods | M | 1h | TarFile Docs | - | tar_file.cpp |
| [#052](issues/052-replace-tarfile-memstream-member-with-helper.md) | Eliminate Cached Pointers | L | 1h | TarFile Refactor | - | tar_file.hpp/.cpp/tar_entry.hpp/.cpp |
| [#053](issues/053-document-tar-streaming-architecture.md) | Document TAR Streaming | M | 2h | TarFile Docs | - | tar-streaming-architecture.md |
| [#054](issues/054-eliminate-dangerous-unowned-buffer-factory-method.md) | Eliminate Unowned Buffer | L | 15m | TarFile Cleanup | - | tar_file.hpp/.cpp/test_tar_file.cpp |
| [#055](issues/055-remove-non-const-entries-overload.md) | Remove Non-Const entries | L | 15m | TarFile Cleanup | - | tar_file.hpp |
| [#056](issues/056-add-comprehensive-documentation-to-tarfile.md) | Add TarFile Docs | M | 2h | TarFile Docs | - | tar_file.hpp |

**Legend:**
- **Pri**: Priority (C=Critical, H=High, M=Medium, L=Low)
- **Est**: Estimated time (15m=15 minutes, 1h=1 hour, etc.)
- **Group**: Feature area grouping
- **Blocked**: Issue numbers that must complete first (`-` if none)
- **Files**: Key files affected (see issue file for complete list)

---

## Recently Completed

_(Max 5 items, deleted files deleted)_

- **Issue #025: MLIR Graph Binary Serialization** - PR #112 - Changed MLIR graph save format from text (.mlir) to binary bytecode using mlir::writeBytecodeToFile()
- **Issue #057: Optimize backlog.md Format for AI Parsing** - PR #115 - Converted backlog to table format (52% token reduction), added metadata columns for efficient issue selection
- **Issue #058: Create /fix-issue Skill - Complete Workflow Automation** - PR #115 - Added skill for complete workflow automation (Phases 1-8: selection, workspace setup, implementation, finalization)
- **Issue #059: Create /resolve-ci Skill - Autonomous Conflict and CI Failure Resolution** - PR #115 - Added autonomous monitoring loop (every 30s) to resolve conflicts and CI failures until PR merges
- **Issue #037: Remove Dead Code from TarFile** - PR #113 - Removed unused rename_symlink/rename_existing_entry functions (~60 LOC, eliminated 4 const_cast violations)

---

## Completing an Issue

When implementation is done (code complete, tests pass), update the backlog before marking PR ready:

1. Edit backlog.md - move issue to "Recently Completed" with PR number
2. Delete issue file: `git rm docs/project/issues/042-*.md`
3. Commit: `git commit -m "docs: complete issue #042"`
4. Push
5. Mark PR ready for review

Keep max 5 in "Recently Completed". When adding #6, delete oldest.

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
