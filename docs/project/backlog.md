<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Project Backlog

## How to Use This Backlog

This backlog tracks project issues (features, bugs, tech debt, refactorings).

**Structure:**
- Each issue gets a detailed file: `docs/project/issues/NNN-name.md`
- This file (backlog.md) is the index/navigation
- Issues move: Backlog → Planned → Active → Recently Completed

**Issue Lifecycle:**
1. **Create:** Copy `issues/TEMPLATE.md` to `issues/NNN-name.md`
2. **Track:** Update issue file with plans, sessions, progress
3. **Complete:** Move to "Recently Completed" section, DELETE the detailed file
4. **Archive:** Keep max 5 in "Recently Completed", delete older ones

**Numbering:** Use sequential numbers: 001, 002, 003, etc.

**Why delete detailed files?**
- Completed work is documented in merged PRs
- Git history preserves deleted file if needed
- Keeps backlog focused on current/future work
- Reduces file clutter

**To recover deleted issue file:**
```bash
git log --all --full-history -- "docs/project/issues/NNN-*.md"
git show <commit-hash>:docs/project/issues/NNN-name.md
```

---

## Active Issues

_(Issues currently being worked on - link to detailed files)_

_No active issues._

---

## Planned Issues

_(Approved for next - link to detailed files)_

_No planned issues._

---

## Backlog

_(Lower priority, not scheduled - link to detailed files)_

**Quick dependency summary:**
- **#003 blocks #004** ⚠️ - Must complete #003 first (encryption_key won't exist in proto)
- **#003 influences #005, #007** - Should coordinate (ConfigProto immutability, cache_key needs new home)
- **#006 relates to #009, #010, #011** - Cache cleanup group (all remove cache_dir remnants)
- **#002, #008 are independent** - Can be done anytime
- **Recommended start: #003** (foundational) or **#006** (largest cleanup ~200-300 LOC)

**See [Issue Dependency Analysis](issue-dependency-analysis.md) for full details: dependency graphs, implementation phases, risk analysis, and parallel work opportunities.**

- [Issue #001: Add mmap Support for Embed Mode](issues/001-mmap-support-for-embed-mode.md) - Feature to enable memory-mapped file access for EP context embed mode
- [Issue #002: Remove mem_files_ - Always Create tar_file_ for Cache](issues/002-evaluate-mem-files-necessity.md) - Remove mem_files_ by fixing tar_file_ creation logic
- [Issue #003: Remove ConfigProto from ContextProto](issues/003-separate-runtime-config-from-persistent-config.md) - Remove ConfigProto from ContextProto, make it runtime-only immutable member (no RuntimeConfig needed)
- [Issue #004: Remove encryption_key Copying](issues/004-remove-encryption-key-copying.md) - Remove encryption_key copying, read from provider_options directly (depends on #003)
- [Issue #005: Move cache_key to ContextProto](issues/005-remove-cache-key-copying.md) - Move cache_key from ConfigProto to ContextProto (cache metadata, needs persistence)
- [Issue #006: Remove cache_dir Entirely](issues/006-remove-cache-dir-copying.md) - Remove legacy disk-based cache system (~200-300 LOC dead code)
- [Issue #007: Clean Up Target with Two-Path Architecture](issues/007-remove-target-copying.md) - Remove target copying, two-path architecture (built-in vs user config), target_proto_ as raw pointer
- [Issue #008: MEP Table Cleanup](issues/008-mep-table-cleanup.md) - Remove MEP table entirely (doesn't scale, NPU-specific)
- [Issue #009: TargetProto Provider Options Injection Cleanup](issues/009-targetproto-provider-options-injection-cleanup.md) - Remove xclbin API functions (dead code, NPU-specific)
- [Issue #010: Remove cache_files - Dead Code](issues/010-cache-files-investigation.md) - Remove cache_files proto field and restore_cache_files() (dead code, never read)
- [Issue #011: Update PassContext Header Documentation](issues/011-update-passcontext-documentation.md) - Remove outdated ASCII art diagram (references non-existent cache_files_to_dir and dir_to_cache_files)

---

## Recently Completed

_(Max 5 items, summary only, detailed files DELETED)_

_No completed issues yet._

---

## Completing an Issue

When an issue is resolved:

1. **Update backlog.md:**
   - Remove issue link from Active/Planned/Backlog section
   - Add to "Recently Completed" section:
     ```
     - ~~Issue #NNN: Title~~ - Completed YYYY-MM-DD, PR #NN
       Brief summary of what was accomplished
     ```

2. **Delete the detailed file:**
   ```bash
   git rm docs/project/issues/NNN-name.md
   git commit -m "docs: archive completed issue #NNN"
   ```

3. **Cleanup old completed items:**
   - Keep max 5 in "Recently Completed"
   - Delete oldest entry when adding #6

---

## Claude Automation

When asked to complete an issue, Claude will automatically:

1. Update backlog.md (move to Recently Completed)
2. Delete the detailed issue file: `docs/project/issues/NNN-*.md`
3. Stage both changes: `git add docs/project/backlog.md docs/project/issues/NNN-*.md`
4. Commit: `git commit -m "docs: complete issue #NNN - [title]"`

When creating new issues, Claude will:
1. Determine next issue number
2. Copy TEMPLATE.md to new numbered file
3. Fill in metadata and description
4. Add link to backlog.md

---

## Integration with Git Workflow

This backlog system complements the existing [git workflow](../workflows/git-workflow.md):

**Workflow Integration:**
1. **Issue Creation:** Create issue file in backlog before starting work
2. **Branch Creation:** Follow [git workflow](../workflows/git-workflow.md) - sync main, create feature branch
3. **Development:** Update issue file with session notes, plans, progress
4. **Pull Request:** Reference issue number in PR description
5. **Completion:** When PR merges, complete the issue and delete detailed file

**Example Flow:**
```
Create Issue #042 → Create feature/implement-feature-x branch → Work + commit + push →
Create Draft PR → Update issue with PR link → Mark PR ready → PR approved and merged →
Complete Issue #042 → Delete docs/project/issues/042-implement-feature-x.md
```

**Hybrid Workflow Benefits:**
- **Issue files (markdown):** Long-term issue tracking with rich context
- **Backlog.md (markdown):** Quick roadmap overview
- **Tasks (JSON):** Short-term session execution
- **Git workflow:** Branch/PR enforcement
- **Plan mode:** Design approval before implementation

---

## Benefits

This backlog system provides:
- ✅ Scalable structure (detailed files per issue)
- ✅ Rich documentation (extensive session notes, multiple plans per issue)
- ✅ Git-tracked (team visible, fine-grained history per issue)
- ✅ Automatic cleanup (completed issues archived, not cluttered)
- ✅ Quick navigation (backlog.md index for scanning)
- ✅ Complements existing git/PR workflows
- ✅ Human and Claude can both maintain it
- ✅ Clean starting point (templates, no pre-filled content)
