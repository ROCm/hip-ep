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
- [Issue #004: Remove encryption_key Copying](issues/004-remove-encryption-key-copying.md) - Remove encryption_key copying, read from provider_options directly
- [Issue #005: Move cache_key to ContextProto](issues/005-remove-cache-key-copying.md) - Move cache_key from ConfigProto to ContextProto
- [Issue #007: Clean Up Target with Two-Path Architecture](issues/007-remove-target-copying.md) - Remove target copying, two-path architecture
- [Issue #009: TargetProto Provider Options Injection Cleanup](issues/009-targetproto-provider-options-injection-cleanup.md) - Remove xclbin API functions
- [Issue #010: Remove cache_files - Dead Code](issues/010-cache-files-investigation.md) - Remove cache_files proto field and restore_cache_files()
- [Issue #011: Update PassContext Header Documentation](issues/011-update-passcontext-documentation.md) - Remove outdated ASCII art diagram
- [Issue #012: session_configs Swapping](issues/012-session-configs-swapping.md) - Resolved by #003
- [Issue #013: provider_options Aggregation](issues/013-provider-options-aggregation.md) - Related to #003, #008, #009
- [Issue #014: Dynamic Pass Registration](issues/014-dynamic-pass-registration.md) - Design flaw needing architectural redesign
- [Issue #015: Configuration Initialization](issues/015-configuration-initialization.md) - Move version info to ContextProto
- [Issue #016: Remove dirty_hack_for_model_clone_external_data_threshold](issues/016-model-clone-threshold-hack.md) - Eliminate global state mutation
- [Issue #017: Remove update_config_by_target()](issues/017-remove-update-config-by-target.md) - Remove obsolete function after #007 and #014
- [Issue #018: Make ConfigProto const Member](issues/018-make-configproto-const.md) - Enforce immutability at compile-time
- [Issue #019: Refactor initialize_context()](issues/019-refactor-initialize-context.md) - God function cleanup
- [Issue #021: Remove cache_file_use_cache_key_prefix_](issues/021-clean-up-cache-file-use-cache-key-prefix.md) - Always use prefix, remove flag
- [Issue #023: Migrate v1 to v2 Execution Provider API](issues/023-migrate-v1-to-v2-execution-provider-api.md) - **CRITICAL** - Migrate from AppendExecutionProvider_VitisAI to AppendExecutionProvider_V2 for MLIR backend

---

## Recently Completed

_(Max 5 items, detailed files deleted)_

- **Issue #022: Remove fix_info Dead Code** - PR #81 - Remove unused fix_info proto field and API methods (~73 LOC)
- **Issue #008: MEP Table Cleanup** - PR #76 - Removed MEP table runtime code, documentation, and test code
- **Issue #006:** Remove cache_dir Entirely - PR #80 - Remove legacy disk-based cache system (~200 LOC)
- **Issue #003:** Remove ConfigProto from ContextProto - PR #77
- **Issue #002:** Remove mem_files_ - Always Create tar_file_ for Cache - PR #67
- **Issue #020: Remove suffix_counter Dead Code** - PR #75 - Remove dead code that reads suffix_counter from model metadata (never written)

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
