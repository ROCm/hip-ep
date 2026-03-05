<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# /fix-issue Design

## Why AI-Driven Instead of Bash Scripts

### Original Problem

`finalize-issue.sh` attempted to automate backlog updates but failed because:
- Expected full `backlog.md` content as parameter but received only partial content
- Overwrote entire file, destroying backlog table and dependencies
- Silent failure - no validation or error handling

### Why Bash Scripts Don't Work

Backlog manipulation requires:
1. **Parsing markdown tables** - extracting issue numbers from `[#NNN](path)` links
2. **Understanding relationships** - bidirectional dependencies between issues
3. **Context-aware updates** - updating "Blocked" columns in OTHER issues when one completes
4. **Conditional logic** - keep/remove dependency lines based on whether related issues still exist
5. **Table formatting** - maintaining proper markdown alignment

Bash/awk can handle simple text replacement but not semantic understanding.

### Token Efficiency

**AI-driven approach is MORE token efficient:**
- No redundant file reads (bash reads → AI reads again to verify)
- Direct edits with full context understanding
- Single pass through files instead of multi-stage pipelines
- Error recovery without retry loops

**Bash script approach:**
- Requires AI to read backlog.md
- Generate complete updated content
- Pass to script
- Script writes blindly
- AI reads again to verify
- = 2x reads + intermediate content generation

### Design Decision

**AI handles everything directly:**
- `Edit` tool for `completed-issues.md` and `backlog.md`
- `Bash` tool for `git rm`, pre-commit, commit, push
- Full visibility and error handling
- Adaptable to edge cases

This eliminates abstraction layers that caused the original bug.
