<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
---
name: create-issue
description: Create backlog issue with implementation plan, update backlog, create PR
allowed-tools: [Bash, Read, Grep, Edit, Write]
---

# /create-issue - Document Planned Work in Project Backlog

## Purpose

Automate creating backlog issues with detailed implementation plans. Extracts plan from conversation context, creates issue/plan files, updates backlog, and creates draft PR.

**Prerequisite**: User must have discussed/created a detailed implementation plan in the current conversation.

## When to Use

- After planning a feature/refactoring/fix in conversation
- User says "let's create an issue for this" or "save this as issue #NNN"
- When you have a complete implementation plan ready to document
- To make planned work available for future Claude sessions

## Workflow Overview

1. **Verify Prerequisites**: Check on main branch, plan exists in context
2. **Extract Metadata**: Get issue title, priority, type from user/context
3. **Analyze Dependencies**: Detect and confirm issue dependencies from conversation/backlog
4. **Generate Issue Number**: Find next available number
5. **Create Branch**: Feature branch for documentation changes
6. **Create Plan File**: Save detailed plan to docs/project/plans/
7. **Create Issue File**: Fill template and save to docs/project/issues/
8. **Update Backlog**: Add entry to backlog.md
9. **Commit & Push**: Git workflow
10. **Create Draft PR**: Immediately after push (mandatory)
11. **Report**: Show created files and PR URL

---

## Phase 1: Verify Prerequisites

### Step 1: Check Current Branch

```bash
git branch --show-current
```

**Expected**: Should be on `main` branch

**If not on main**:
- Display: "❌ Must be on main branch to create issue. Current branch: <branch-name>"
- Stop execution

### Step 2: Verify Plan Exists in Context

**Check conversation history for plan content**:
- Look for structured plan with sections like: Objective, Implementation Steps, Critical Files
- Plan should have enough detail for future Claude session to execute

**If no plan found in context**:
- Display: "❌ No implementation plan found in conversation. Please discuss and plan the work first, then invoke /create-issue."
- Stop execution

---

## Phase 2: Extract Issue Metadata

### Step 1: Extract from Conversation Context

From user's messages and plan discussion, extract:
- **Title**: Brief description (e.g., "Eliminate C-style APIs from morphizen-graph")
- **Priority**: CRITICAL | HIGH | MEDIUM | LOW
- **Type**: Feature | Bug | Tech Debt / Refactoring | Architecture | Documentation
- **Brief Description**: One-line summary for backlog

### Step 2: Confirm with User (if unclear)

If any field is ambiguous, ask:
- "What should the issue title be?"
- "What priority: CRITICAL, HIGH, MEDIUM, or LOW?"
- "What type: Feature, Bug, Tech Debt/Refactoring, Architecture, or Documentation?"

---

## Phase 3: Analyze Issue Dependencies

### Step 1: Scan Conversation for Dependency Indicators

**Automatic detection** - Look for:
- **Explicit issue references**: "#022", "Issue #015", etc.
- **Dependency phrases**: "after", "requires", "depends on", "prerequisite", "blocked by", "blocker"
- **Sequential work**: "once X is done", "following the completion of"
- **Technical blockers**: References to code/files that other planned issues will modify

**Example patterns to detect**:
```
"This requires #022 to be done first"
"After we eliminate C-style APIs, we can..."
"Depends on the graph refactor in #015"
"Blocked by the ONNX Runtime upgrade"
```

### Step 2: List Current Backlog Issues

```bash
# Read backlog to show available issues
Read docs/project/backlog.md
```

Parse and display current backlog issues for user reference:
```
Current backlog issues:
- Issue #022: Eliminate C-style APIs
- Issue #015: Refactor graph structure
- Issue #018: Update ONNX Runtime integration
...
```

### Step 3: Present Findings and Confirm with User

**If dependencies detected**:
```
Dependency analysis for this issue:

Detected potential dependencies:
- Issue #022: Eliminate C-style APIs (mentioned as prerequisite)
- Issue #015: Refactor graph structure (same files: morphizen-graph/)

Are these correct? Any others to add?
```

**If no dependencies detected**:
```
No dependencies detected in conversation.

Does this issue have any dependencies from the backlog above? (or None)
```

**User can respond with**:
- Issue numbers: "#022, #015"
- "None"
- Additional context: "Needs #022 completed first"

### Step 4: Store Dependencies

Store confirmed dependencies for use in Phase 6 (Create Issue File):
```bash
DEPENDENCIES="#022, #015"  # or "None"
```

---

## Phase 5: Generate Next Issue Number

### Step 1: Find Highest Issue Number

```bash
HIGHEST=$(ls docs/project/issues/ | grep -E '^[0-9]+-' | sed 's/-.*//' | sort -n | tail -1)
NEXT_ISSUE=$((HIGHEST + 1))
ISSUE_NUM=$(printf "%03d" $NEXT_ISSUE)
echo "Next issue number: $ISSUE_NUM"
```

**Expected output**: Three-digit padded number like `024`, `025`

### Step 2: Generate Kebab-Case Filename

Convert title to kebab-case:
- Lowercase all characters
- Replace spaces with hyphens
- Remove special characters

**Example**: "Eliminate C-style APIs" → "eliminate-c-style-apis"

Store as: `ISSUE_FILENAME="${ISSUE_NUM}-${KEBAB_TITLE}"`

---

## Phase 6: Create Feature Branch

```bash
git checkout -b feature/add-issue-${ISSUE_NUM}-${BRIEF_NAME}
```

**Example**: `feature/add-issue-024-c-style-api-elimination`

**Validation**:
- Verify branch created successfully
- Display: "Created branch: feature/add-issue-${ISSUE_NUM}-${BRIEF_NAME}"

---

## Phase 7: Create Plan File

### Step 1: Extract Plan Content from Context

From conversation, extract the complete implementation plan including:
- Objective/Goal
- Context/Background
- Implementation steps/phases
- Critical files to modify
- Testing/verification strategy
- Success criteria

### Step 2: Write Plan File

**File**: `docs/project/plans/${ISSUE_NUM}-${KEBAB_TITLE}-plan.md`

**Content structure**:
```markdown
<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# [Plan Title]

**Issue:** #${ISSUE_NUM}
**Created:** $(date +%Y-%m-%d)
**Status:** READY

[Rest of plan content extracted from conversation]
```

Use Write tool to create the file.

**Validation**:
- Verify file created
- Display: "✓ Created plan: docs/project/plans/${ISSUE_NUM}-${KEBAB_TITLE}-plan.md"

---

## Phase 8: Create Issue File

### Step 1: Read Template

```bash
# Read to understand structure
Read docs/project/issues/TEMPLATE.md
```

### Step 2: Fill Template

**File**: `docs/project/issues/${ISSUE_NUM}-${KEBAB_TITLE}.md`

**Content** (follow TEMPLATE.md structure):
```markdown
<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Issue #${ISSUE_NUM}: ${TITLE}

## Metadata
- **Type:** ${TYPE}
- **Priority:** ${PRIORITY}
- **Created:** $(date +%Y-%m-%d)
- **Dependencies:** ${DEPENDENCIES}

## Description

[1-3 sentences: Extract from conversation/plan - what needs to be done and why]

## Problem

**Current design/code:**
```cpp
// [Extract from conversation/plan - show current problematic code or design]
```

**Why this is problematic:**
1. [Specific problem with explanation]
2. [Impact or consequence]
3. [Root cause]

**Code locations:**
- `file.cpp:123` - [What happens here - extract from conversation if available]
- `file.cpp:456` - [What happens here - extract from conversation if available]

## Solution

**Proposed design:**
```cpp
// [Extract from conversation/plan - show proposed solution]
```

**Approach:**
1. [Step 1 - extract from conversation/plan]
2. [Step 2 - extract from conversation/plan]
3. [Expected outcome]

**Benefits:**
- ✅ [Benefit 1]
- ✅ [Benefit 2]

## Plans

- [${ISSUE_NUM}-${KEBAB_TITLE}-plan.md](../plans/${ISSUE_NUM}-${KEBAB_TITLE}-plan.md) - Created $(date +%Y-%m-%d)
```

Use Write tool to create the file.

**Validation**:
- Verify file created
- Display: "✓ Created issue: docs/project/issues/${ISSUE_NUM}-${KEBAB_TITLE}.md"

---

## Phase 9: Update Backlog

### Step 1: Read Current Backlog

```bash
Read docs/project/backlog.md
```

### Step 2: Add Entry to Backlog Section

Find the "## Backlog" section and add new entry:

```markdown
- [Issue #${ISSUE_NUM}: ${TITLE}](issues/${ISSUE_NUM}-${KEBAB_TITLE}.md) - ${BRIEF_DESCRIPTION}
```

**Placement**: Add in numeric order by issue number

Use Edit tool to insert the line.

**Validation**:
- Verify backlog.md updated
- Display: "✓ Updated backlog.md with issue #${ISSUE_NUM}"

---

## Phase 10: Commit and Push

### Step 1: Review Changes

```bash
git status
```

**Expected files**:
- `docs/project/issues/${ISSUE_NUM}-${KEBAB_TITLE}.md`
- `docs/project/plans/${ISSUE_NUM}-${KEBAB_TITLE}-plan.md`
- `docs/project/backlog.md`

### Step 2: Stage Files

```bash
git add docs/project/issues/${ISSUE_NUM}-${KEBAB_TITLE}.md
git add docs/project/plans/${ISSUE_NUM}-${KEBAB_TITLE}-plan.md
git add docs/project/backlog.md
```

### Step 3: Commit

```bash
git commit -m "docs: add issue #${ISSUE_NUM} - ${TITLE}

- Create detailed plan in docs/project/plans/
- Create issue file with metadata, problem/solution analysis
- Update backlog.md to reference new issue
- Plan is ready for future implementation in clean Claude session"
```

### Step 4: Push with Upstream Tracking

```bash
git push -u fork feature/add-issue-${ISSUE_NUM}-${BRIEF_NAME}
```

**Validation**:
- Verify push successful
- Capture remote branch info

---

## Phase 11: Create Draft PR (MANDATORY)

**Critical**: Per `docs/workflows/git-workflow.md`, create draft PR IMMEDIATELY after first push.

### Step 1: Create Draft PR

```bash
gh pr create --draft --title "docs: add issue #${ISSUE_NUM} - ${TITLE}" --body "$(cat <<'EOF'
## Summary
- Add Issue #${ISSUE_NUM} to project backlog documenting [work description]
- Create detailed implementation plan in `docs/project/plans/${ISSUE_NUM}-${KEBAB_TITLE}-plan.md`
- Update backlog.md to reference new issue

## Details

This PR adds documentation for a future [feature/fix/refactoring] effort.

**Issue #${ISSUE_NUM}** includes:
- Metadata (status: BACKLOG, priority: ${PRIORITY}, type: ${TYPE})
- Problem analysis
- Solution overview
- Code location evidence

**Implementation Plan** includes:
- Detailed step-by-step approach
- Critical files to modify
- Testing/verification strategy
- Success criteria

## Purpose

This allows future Claude sessions to pick up Issue #${ISSUE_NUM}, read the saved plan from `docs/project/plans/`, and implement the work directly without re-planning from scratch.

## Test plan
- [x] Issue file follows template format
- [x] Plan file is complete and actionable
- [x] Backlog updated with new issue reference
- [x] No code changes, documentation only
EOF
)"
```

**Validation**:
- Capture PR number and URL
- Display: "✓ Created draft PR #<number>: <url>"

---

## Phase 12: Report Results

Display final summary:

```
✅ Created Issue #${ISSUE_NUM} with implementation plan

Files created:
- docs/project/plans/${ISSUE_NUM}-${KEBAB_TITLE}-plan.md
- docs/project/issues/${ISSUE_NUM}-${KEBAB_TITLE}.md

Backlog updated with new issue entry.

Branch: feature/add-issue-${ISSUE_NUM}-${BRIEF_NAME}
PR: <url> (draft)

Next steps:
- Review and refine the issue/plan as needed
- When ready to implement, a fresh Claude session can read the plan and execute directly
- Mark PR as ready when satisfied with documentation
```

---

## Error Handling

### Scenario 1: Not on Main Branch

**Error**: Current branch is not `main`

**Action**:
- Display: "❌ Must be on main branch. Current: <branch>. Switch to main first: `git checkout main`"
- Stop execution

### Scenario 2: No Plan in Context

**Error**: Cannot find implementation plan in conversation

**Action**:
- Display: "❌ No plan found. Please discuss and plan the work first, then invoke /create-issue"
- Suggest: "Use plan mode or outline the implementation approach"
- Stop execution

### Scenario 3: Uncommitted Changes

**Error**: `git status` shows uncommitted changes

**Action**:
- Display: "⚠️ Warning: Uncommitted changes detected. Commit or stash them first."
- List files with changes
- Ask user: "Continue anyway? (changes will be on new branch)"

### Scenario 4: Issue Number Collision

**Error**: Issue file already exists with generated number

**Action**:
- Regenerate next number (highest + 2, +3, etc.)
- Display: "Issue #${ISSUE_NUM} exists, using #${NEXT_ISSUE}"

### Scenario 5: GitHub CLI Error

**Error**: `gh pr create` fails

**Action**:
- Display: "⚠️ PR creation failed. You can create manually:"
- Show command: `gh pr create --draft --title "..."`
- Continue (don't fail entire skill)

---

## Notes

**Design Decisions**:
- Plan must exist in context before invoking skill (forces proper planning)
- Always starts on main branch (clean state)
- Draft PR mandatory per git-workflow.md
- Issue number auto-generated (prevents conflicts)
- Three-digit padding (001, 024) for consistent sorting
