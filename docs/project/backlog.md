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

See [issue-dependency-analysis.md](issue-dependency-analysis.md) for details.

- [Issue #001: Add mmap Support for Embed Mode](issues/001-mmap-support-for-embed-mode.md) - Feature to enable memory-mapped file access for EP context embed mode
- [Issue #005: Move cache_key to ContextProto](issues/005-remove-cache-key-copying.md) - Move cache_key from ConfigProto to ContextProto
- [Issue #007: Clean Up Target with Two-Path Architecture](issues/007-remove-target-copying.md) - Remove target copying, two-path architecture
- [Issue #009: TargetProto Provider Options Injection Cleanup](issues/009-targetproto-provider-options-injection-cleanup.md) - Remove xclbin API functions
- [Issue #012: session_configs Swapping](issues/012-session-configs-swapping.md) - Resolved by #003
- [Issue #013: provider_options Aggregation](issues/013-provider-options-aggregation.md) - Related to #003, #008, #009
- [Issue #014: Dynamic Pass Registration](issues/014-dynamic-pass-registration.md) - Design flaw needing architectural redesign
- [Issue #015: Configuration Initialization](issues/015-configuration-initialization.md) - Move version info to ContextProto
- [Issue #017: Remove update_config_by_target()](issues/017-remove-update-config-by-target.md) - Remove obsolete function after #007 and #014
- [Issue #018: Make ConfigProto const Member](issues/018-make-configproto-const.md) - Enforce immutability at compile-time
- [Issue #019: Refactor initialize_context()](issues/019-refactor-initialize-context.md) - God function cleanup
- [Issue #024: Remove Legacy Compile Entry Points](issues/024-remove-legacy-compile-entry-points.md) - Remove unused compile_onnx_model_morphizen_ep_with_options and with_error_handling (~50 LOC)
- [Issue #025: MLIR Graph Binary Serialization](issues/025-mlir-graph-binary-serialization.md) - Change MLIR graph save format from text (.mlir) to binary bytecode (.mlirbc)
- [Issue #026: MLIR Model Export API](issues/026-mlir-model-export-api.md) - Add model_save_mlir() API to export Model pointer to MLIR file (MLIR backend only)
- [Issue #027: Eliminate C-Style APIs from morphizen-graph](issues/027-eliminate-c-style-apis-from-morphizen-graph.md) - Complete C++ wrapper migration by eliminating 28 C-style free functions

---

## Recently Completed

_(Max 5 items, deleted files deleted)_

- **Issue #023: Migrate v1 to v2 Execution Provider API** - PR #TBD - Migrated from AppendExecutionProvider_VitisAI to AppendExecutionProvider_V2, deleted unused common/initialize_morphizen.hpp
- **Issue #021: Remove cache_file_use_cache_key_prefix_** - PR #87 - Always use cache_key prefix, remove flag (~20-25 LOC)
- **Issue #016: Remove dirty_hack_for_model_clone_external_data_threshold** - PR #85
- **Issue #004: Remove encryption_key Copying** - PR #82 - Removed encryption_key from ConfigProto, read from provider_options (security improvement)
- **Issue #010: Remove cache_files Dead Code** - PR #84 - Remove unused cache_files proto field and restore_cache_files() no-op function (~17 LOC)
- **Issue #011: Update PassContext Header Documentation** - PR #83 - Remove outdated ASCII art, replace with accurate tar_file_ API documentation

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
