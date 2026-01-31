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

## Issue Granularity Philosophy

**Why split complex tasks into smaller issues?**

Complex tasks should be decomposed into focused, actionable issues. This approach:

- ✅ **Reduces cognitive load** - Each issue has clear, bounded scope
- ✅ **Enables efficient implementation** - Complete in one or few sessions
- ✅ **Facilitates better discussions** - Focus on one design problem at a time
- ✅ **Manages risk** - Small changes, easier to review and rollback
- ✅ **Clarifies dependencies** - Real dependencies emerge as you work through issues
- ✅ **Shows progress** - Check off issues one by one
- ✅ **Enables collaboration** - Other developers can pick up specific issues
- ✅ **Works better with AI assistants** - Claude Code can focus deeply on one problem

**When to create a new issue:**
- Problem is complex and has multiple sub-problems
- Solution requires multiple design decisions
- Different parts can be implemented independently
- Implementation would take multiple sessions

**When to expand existing issue:**
- Direct extension of existing scope
- Same root cause, same solution approach
- Cannot be separated without losing context

**Example decomposition:**
- ❌ BAD: One issue "Fix ConfigProto Mutability" (too broad)
- ✅ GOOD: Issues #003-#015 each addressing specific mutation categories

**This granularity makes working with Claude Code more effective:**
- Claude can create focused plans for each issue
- Less chance of missing edge cases in large plans
- Implementation steps are manageable
- You maintain control of overall architecture

---

## Issue Quality Best Practices

**Keep issues lean and meaningful - avoid "empty talking".**

### ✅ DO: Write High-Quality Issues

**1. Only include sections with actual content:**
- ❌ Don't add "Plans: _No plans yet_"
- ❌ Don't add "Sessions: _To be filled in_"
- ❌ Don't add "Related PRs: _None yet_"
- ✅ Add Plans/Sessions/PRs sections ONLY when they have real content

**2. Make Notes section add value:**
- ❌ "Part of ConfigProto mutation analysis" (obvious from title)
- ❌ "This is a design flaw" (already stated in Description)
- ✅ Explain root cause, relationships to other issues, or design trade-offs
- ✅ Provide context NOT already in Description/Problem/Solution

**3. Write specific, concrete content:**
- ❌ Vague: "This needs to be fixed"
- ✅ Specific: "Swapping exists because ConfigProto is INPUT but persisted in ContextProto (OUTPUT)"
- ❌ Generic: "Should be discussed"
- ✅ Actionable: "Three possible solutions: (1) X, (2) Y, (3) Z"

**4. Use Evidence section effectively:**
- ✅ List exact code locations: `file.cpp:123 - What happens here`
- ✅ Helps reviewers verify the problem exists
- ✅ Makes implementation easier to scope

### ❌ DON'T: Add Noise

**Empty placeholders waste space:**
```markdown
## Plans
_No plans yet._

## Sessions
_To be filled in during implementation discussions._

## Related PRs
_None yet._
```
**Just delete these sections!** Add them later when you have content.

**Obvious statements waste time:**
```markdown
## Notes
This is part of the cache cleanup effort.
```
If the issue title says "cache cleanup", don't repeat it in Notes.

### Template Usage

**When creating a new issue from TEMPLATE.md:**

1. **Copy the template** to `NNN-name.md`
2. **Fill in required sections:** Metadata, Description, Problem, Solution, Evidence
3. **Delete optional sections** that you don't have content for yet:
   - If no plans exist, delete Plans section
   - If no sessions yet, delete Sessions section
   - If no PRs yet, delete Related PRs section
   - If Notes would just repeat Description, delete Notes section
4. **Add optional sections later** when you have meaningful content

**Result:** Clean, focused issues with only valuable information.

### Examples

**Bad Issue (empty talking):**
```markdown
## Description
Need to fix the cache system.

## Plans
_No plans yet._

## Sessions
_To be filled in._

## Notes
Part of cache cleanup.
```

**Good Issue (meaningful content):**
```markdown
## Description
Remove cache_dir entirely - obsolete disk-based cache system (~200-300 LOC).

## Problem
cache_dir was designed for disk-based caching, but new tar_file_ system
makes it obsolete. All cache operations now use tar_file_ directly.

## Solution
Remove all cache_dir code: get_log_dir(), cache_dir proto field, copying logic.

## Evidence
- cache_dir.cpp:67 - cache_dir computation (to be removed)
- pass_context_imp.cpp:1266 - cache_dir copying (to be removed)

## Notes
After removal, xclbin functions (Issue #009) will break - they call get_log_dir().
Coordinate removal or do #009 first.
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
- [Issue #012: session_configs Swapping](issues/012-session-configs-swapping.md) - Resolved by #003 (ConfigProto runtime-only)
- [Issue #013: provider_options Aggregation](issues/013-provider-options-aggregation.md) - Related to #003, #008, #009
- [Issue #014: Dynamic Pass Registration](issues/014-dynamic-pass-registration.md) - Design flaw needing architectural redesign
- [Issue #015: Configuration Initialization](issues/015-configuration-initialization.md) - Move version info to ContextProto
- [Issue #016: Remove dirty_hack_for_model_clone_external_data_threshold](issues/016-model-clone-threshold-hack.md) - Eliminate global state mutation for model clone threshold
- [Issue #017: Remove update_config_by_target()](issues/017-remove-update-config-by-target.md) - Remove obsolete function after #007 and #014 complete
- [Issue #018: Make ConfigProto const Member](issues/018-make-configproto-const.md) - Enforce immutability at compile-time (final step after all mutations eliminated)
- [Issue #019: Refactor initialize_context()](issues/019-refactor-initialize-context.md) - God function cleanup - extract cache_key computation, reduce complexity
- [Issue #020: Remove suffix_counter Dead Code](issues/020-remove-suffix-counter-dead-code.md) - Remove dead code reading metadata that's never written (4 lines)

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
3. Fill in all required sections: Metadata, Description, Problem, Solution, Evidence
4. **Delete empty optional sections** (Plans/Sessions/PRs if no content yet)
5. Only include Notes section if it adds value beyond Description/Problem
6. Add link to backlog.md

**Quality checklist for new issues:**
- ✅ Concrete problem statement with code evidence
- ✅ Clear solution with benefits and approach
- ✅ No "No plans yet" or "To be filled in" placeholders
- ✅ Notes add context not found elsewhere, or section deleted
- ✅ Only meaningful content, no empty talking

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
- ✅ High quality standards (no empty talking, only meaningful content)
