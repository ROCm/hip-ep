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

See [issue-dependency-analysis.md](issue-dependency-analysis.md) for details.

- [Issue #001: Add mmap Support for Embed Mode](issues/001-mmap-support-for-embed-mode.md) - Feature to enable memory-mapped file access for EP context embed mode
- [Issue #009: TargetProto Provider Options Injection Cleanup](issues/009-targetproto-provider-options-injection-cleanup.md) - Remove xclbin API functions
- [Issue #012: session_configs Swapping](issues/012-session-configs-swapping.md) - Resolved by #003
- [Issue #013: provider_options Aggregation](issues/013-provider-options-aggregation.md) - Related to #003, #008, #009
- [Issue #014: Dynamic Pass Registration](issues/014-dynamic-pass-registration.md) - Design flaw needing architectural redesign
- [Issue #017: Remove update_config_by_target()](issues/017-remove-update-config-by-target.md) - Remove obsolete function after #007 and #014
- [Issue #018: Make ConfigProto const Member](issues/018-make-configproto-const.md) - Enforce immutability at compile-time
- [Issue #019: Refactor initialize_context()](issues/019-refactor-initialize-context.md) - God function cleanup
- [Issue #024: Remove Legacy Compile Entry Points](issues/024-remove-legacy-compile-entry-points.md) - Remove unused compile_onnx_model_morphizen_ep_with_options and with_error_handling (~50 LOC)
- [Issue #025: MLIR Graph Binary Serialization](issues/025-mlir-graph-binary-serialization.md) - Change MLIR graph save format from text (.mlir) to binary bytecode (.mlirbc)
- [Issue #026: MLIR Model Export API](issues/026-mlir-model-export-api.md) - Add model_save_mlir() API to export Model pointer to MLIR file (MLIR backend only)
- [Issue #027: Eliminate C-Style APIs from morphizen-graph](issues/027-eliminate-c-style-apis-from-morphizen-graph.md) - Complete C++ wrapper migration by eliminating 28 C-style free functions
- [Issue #028: MLIR Backend Test Model Support](issues/028-mlir-backend-test-model-support.md) - Provide MLIR format test models for 23 unit tests (ModelTest, GraphTest, PatternTest, PassContextConfigTest)
- [Issue #029: PassContext tar_file_ Initialization Missing](issues/029-passcontext-tarfile-initialization.md) - Fix tar_file_ nullptr in PassContextTest (3 tests)
- [Issue #030: morphizen-pass_init Plugin Loading Failure](issues/030-morphizen-pass-init-plugin-loading.md) - Fix plugin loading failure affecting 7 tests (TestAnchorPoint, MorphizenOrtApiTest)
- [Issue #031: Target Auto-Discovery Failure Causes Compilation Errors](issues/031-target-auto-discovery-failure.md) - Fix empty target string causing compilation failures (2 tests)
- [Issue #032: EP Context Model Generation Failure in E2E Tests](issues/032-ep-context-model-generation-failure.md) - Fix "Unable to compile any nodes" causing 4 E2E tests to fail
- [Issue #033: EP Duplicate Registration in V2 API Test](issues/033-ep-duplicate-registration.md) - Handle duplicate EP registration in V2 API test
- [Issue #034: MLIR Backend Shape Nullptr Check Failure](issues/034-mlir-shape-nullptr-check-failure.md) - Fix shape nullptr and LLVM casting failures (2 tests)
- [Issue #035: ConstDataTest Suite Skipped - Missing Boost::Process Support](issues/035-constdatatest-boost-dependency.md) - Enable 16 ConstDataTest tests without Boost dependency
- [Issue #036: Other Boost-Dependent Tests Skipped](issues/036-boost-dependent-tests-skipped.md) - Enable GraphTest.NewConstantInitializer and TarFileTest.WriteTo without Boost
- [Issue #037: Remove Dead Code from TarFile](issues/037-remove-tarfile-dead-code.md) - Remove unused rename_symlink/rename_existing_entry functions (dead since April 2025)
- [Issue #038: Document Lazy Symlink Resolution const_cast](issues/038-document-lazy-symlink-const-cast.md) - Add comprehensive documentation explaining intentional const_cast for write-once lazy initialization
- [Issue #039: Fix Typos and Outdated Comments in TarFile](issues/039-fix-tarfile-typos-and-outdated-comments.md) - Fix outdated renaming comment, typos, incorrect FIXME, and remove unused parameter
- [Issue #040: Remove Legacy tar_ball.cpp/hpp Re-Serialization](issues/040-remove-legacy-tar-ball-reserialization.md) - Remove ~750 lines of legacy TarWriter/TarReader code that wastefully re-serializes tar data
- [Issue #041: Remove Duplicate friend Declaration in TarEntryInputStream](issues/041-remove-duplicate-friend-declaration.md) - Remove duplicate friend class declaration (trivial 1-line fix)
- [Issue #042: Document PrivateTag Factory Pattern](issues/042-document-privatetag-factory-pattern.md) - Add technical documentation and update code comment for PrivateTag pattern
- [Issue #044: Use Standard Erase-Remove Idiom in TarFile](issues/044-use-erase-remove-idiom-in-tarfile.md) - Replace verbose erase-remove pattern with standard C++ idiom (remove unnecessary if check)
- [Issue #047: Extract Duplicate Entry Deduplication Logic in TarFile](issues/047-extract-duplicate-entry-deduplication-logic.md) - Extract duplicated deduplication logic from add_regular_entry/add_symlink_entry into helper function
- [Issue #048: Extract Platform-Specific tmpfile Helper Function](issues/048-extract-platform-specific-tmpfile-helper.md) - Extract duplicated platform-specific tmpfile creation code into reusable helper function
- [Issue #050: Standardize TarFile Factory Method Naming](issues/050-standardize-tarfile-factory-method-naming.md) - Rename factory methods to follow consistent "create_from_X" pattern for better discoverability
- [Issue #051: Document Complex TarFile Factory Methods](issues/051-document-complex-tarfile-factory-methods.md) - Add comprehensive comments explaining mmap strategy, platform differences, and fallback logic in complex factory methods
- [Issue #052: Replace TarFile mem_stream_ Member with Helper Function](issues/052-replace-tarfile-memstream-member-with-helper.md) - Replace redundant mem_stream_ cached member with inline get_mem_stream() helper function
- [Issue #053: Document TAR Streaming Architecture](issues/053-document-tar-streaming-architecture.md) - Create technical documentation explaining stream ownership, buffering mechanism, mmap access, and thread-safety
- [Issue #054: Eliminate Dangerous Un-Owned Buffer Factory Method](issues/054-eliminate-dangerous-unowned-buffer-factory-method.md) - Remove test-only create(const char*, size_t) factory method that requires caller-managed lifetime
- [Issue #055: Remove Non-Const entries() Overload from TarFile](issues/055-remove-non-const-entries-overload.md) - Remove unused non-const entries() accessor to enforce read-only access
- [Issue #056: Add Comprehensive Documentation to TarFile Class](issues/056-add-comprehensive-documentation-to-tarfile.md) - Add class-level docs, document 4 undocumented methods, fix outdated open_for_write() comments

---

## Recently Completed

_(Max 5 items, deleted files deleted)_

- **Issue #043: Fix /create-issue Skill - Prevent Implementation During Discussion** - PR #107 - Added explicit rules to prevent AI from implementing work during issue creation (should only create documentation)
- **Issue #045: Add Approval Workflow to /create-issue Skill** - PR #107 - Added 5-step workflow (Explore→Discuss→Summarize→Approve→Create) with plan detail options
- **Issue #046: Update Dependencies After Each Issue in /create-issue** - PR #107 - Added automatic dependency tracking to update Quick dependencies and issue metadata after creating each issue
- **Issue #049: Add Sub-topic Breakdown Workflow to /create-issue** - PR #107 - Added structured workflow for breaking down complex topics into manageable sub-topics with automatic task creation
- **Issue #015: Move version info to ContextProto** - PR #105 - Moved AllVersionInfoProto from ConfigProto to ContextProto (version metadata belongs in persisted ContextProto, not runtime-only ConfigProto)

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
