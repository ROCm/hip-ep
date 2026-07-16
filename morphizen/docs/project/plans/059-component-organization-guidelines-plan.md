<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Implementation Plan: Issue #059 - Component Organization Guidelines

## Overview

Create `docs/technical/component-organization-guidelines.md` to establish clear decision criteria for when to create standalone components versus consolidating into existing libraries.

## Implementation Steps

### Step 1: Create docs/technical/ Directory

**Action:**
```bash
mkdir -p docs/technical
```

**Verification:**
```bash
ls -la docs/technical/
```

### Step 2: Create Component Organization Guidelines Document

**File:** `docs/technical/component-organization-guidelines.md`

**Document Structure:**

```markdown
# Component Organization Guidelines

## Overview
Brief introduction explaining purpose of this document and when to consult it.

## 1. When to Create a Standalone Component

### Criteria
List of criteria with clear thresholds:

1. **Size threshold**: Component has >500 LOC OR >5 public APIs
2. **Architectural layer**: Component represents a distinct architectural layer
3. **Strong conceptual identity**: Clear, well-defined purpose (e.g., "pattern matching system", "graph manipulation")
4. **External reusability**: Designed to be used outside MorphiZen
5. **Complex build requirements**: Needs custom build logic, code generation, or unique dependencies

### Generic Examples (✅ Should be standalone)
- "A graph manipulation layer with 4000+ LOC and 50+ APIs"
- "A pattern matching system with custom DSL and code generation"
- "A backend abstraction layer with 111 function pointers"
- "A protocol buffer schema with code generation"

## 2. When to Consolidate into Existing Library

### Criteria
List of criteria for consolidation:

1. **Size threshold**: Component has <200 LOC AND <5 public APIs
2. **Utility nature**: General-purpose helper, not a core system component
3. **Single-concern**: Provides 1-3 related functions/classes
4. **Shared dependencies**: Uses same dependencies as existing utility library
5. **No unique build requirements**: Standard compilation, no special steps

### Generic Examples (❌ Should be consolidated)
- "A 3-function string formatting helper (50 LOC)"
- "A simple caching utility (100 LOC, 2 functions)"
- "Platform-specific file path normalization (75 LOC)"
- "A single-purpose validation function"

## 3. Decision Matrix

Provide table for quick reference:

| Criteria | Standalone Component | Consolidated into Library |
|----------|---------------------|---------------------------|
| **Size** | >500 LOC | <200 LOC |
| **Public APIs** | >5 functions/classes | <5 functions |
| **Purpose** | Architectural layer or system | Utility/helper |
| **Build complexity** | Custom (code gen, protobuf, etc.) | Standard compilation |
| **Dependencies** | Unique dependencies | Shares dependencies with existing lib |
| **Conceptual cohesion** | Strong identity (e.g., "pattern matching") | Single-purpose helper |
| **Reusability** | Designed for external use | Internal utility |

## 4. Edge Cases and Exceptions

### Code Generation Components
Components with build-time code generation MAY justify standalone status even if small (<200 LOC), if:
- Generation logic is complex (>100 LOC Python/script)
- Generated code is substantial
- Build process is non-trivial

**Generic Example:**
- "Embedded resource system: 111 LOC runtime + 103 LOC Python generator" → Could be standalone OR consolidated depending on other factors

### Platform-Specific Components
Platform-specific utilities should be consolidated unless they form a complete abstraction layer.

### Experimental/Prototype Components
Early-stage experimental features may start as standalone for faster iteration, then be consolidated or removed based on adoption.

## 5. How to Apply These Guidelines

### For New Components
1. Review criteria in Section 1 and Section 2
2. Count LOC and public APIs
3. Identify dependencies and build requirements
4. Check decision matrix (Section 3)
5. Document decision rationale in PR description

### For Existing Components
These guidelines apply to future decisions. Reorganizing existing components requires:
- Separate issue/PR with analysis
- Migration plan for dependencies
- Backwards compatibility considerations

## 6. Relationship to Architecture

This document focuses on component **organization** (directory structure, build targets).

For architectural **layering** (foundation → backend abstraction → graph → core → applications), see `docs/architecture.md`.

## References

- Project structure: `docs/architecture.md`
- Developer guide: `docs/developer-guide.md`
```

**Content Guidelines:**
- Use clear, objective thresholds (e.g., ">500 LOC" not "large")
- Provide generic examples, not analysis of specific existing components (mem_binary, encryption)
- Focus on decision framework, not prescriptive reorganization
- Keep examples illustrative but generic ("A pattern matching system" not "morphizen-pattern")

### Step 3: Update Backlog

**File:** `docs/project/backlog.md`

**Add to "Active issues" section:**
```markdown
- **#059**: Create component organization guidelines
  - **Status**: Open
  - **Priority**: MEDIUM
  - **Type**: Documentation / Architecture
  - **Description**: Establish clear criteria for standalone vs consolidated components
```

### Step 4: Commit and Push

**Commit message:**
```bash
git add docs/technical/component-organization-guidelines.md \
        docs/project/issues/059-component-organization-guidelines.md \
        docs/project/plans/059-component-organization-guidelines-plan.md \
        docs/project/backlog.md

git commit -m "$(cat <<'EOF'
docs: add issue #059 - Create component organization guidelines

Establishes decision framework for when to create standalone components
vs consolidating into existing libraries. Addresses architectural
consistency and reduces decision ambiguity for future development.

Issue details:
- Problem: No documented guidelines for component organization decisions
- Impact: Inconsistent patterns, review friction, contributor confusion
- Solution: Create docs/technical/component-organization-guidelines.md
- Deliverable: Decision matrix with clear criteria and generic examples

Related discussions:
- mem_binary component organization
- Consistency between small standalone components (mem_binary, encryption)
  and consolidated utilities (morphizen-utils)
EOF
)"

git push -u fork feature/create-component-organization-guidelines
```

### Step 5: Create Draft PR

**Command:**
```bash
gh pr create --draft --title "docs: add issue #059 - Create component organization guidelines" --body "$(cat <<'EOF'
## Summary

Creates technical documentation for component organization decision-making.

Establishes clear, objective criteria for deciding when to create standalone components versus consolidating into existing libraries.

## Problem

The MorphiZen project currently has:
- 25+ top-level directories
- Mix of small standalone components (mem_binary ~111 LOC, encryption ~184 LOC)
- Larger consolidated libraries (morphizen-utils ~500+ LOC)
- No documented rationale for organization decisions

This creates:
- Review friction (questions about component organization)
- Contributor confusion (where to place new code)
- Risk of further fragmentation
- Potential bikeshedding on subjective decisions

## Solution

Create `docs/technical/component-organization-guidelines.md` with:

1. **Decision criteria**: When to create standalone vs consolidate
2. **Decision matrix**: Quick reference table with thresholds
3. **Generic examples**: Illustrative examples (not specific to current components)
4. **Edge cases**: Code generation, platform-specific components

## Design Rationale

### Why a Separate Technical Document?

- **Keep architecture.md focused**: architecture.md describes the actual structure; this doc describes how to decide structure
- **Timeless guidelines**: Focuses on reusable decision framework, not point-in-time analysis
- **Actionable reference**: Provides objective criteria for code reviews and development decisions

### Why Generic Examples?

- **Avoid staleness**: Specific component analysis belongs in issues/PRs, not guidelines
- **Git history as decision log**: Reorganization decisions captured in commit messages and issues
- **Focus on framework**: Document teaches the decision process, not prescribes specific outcomes

### Decision Criteria Design

**Objective thresholds:**
- Size: >500 LOC for standalone, <200 LOC for consolidation
- APIs: >5 public functions for standalone, <5 for consolidation
- Clear criteria reduce subjective debates

**Multi-dimensional:**
- Size alone isn't enough (code generation component may be small but complex)
- Considers: purpose, build complexity, dependencies, cohesion, reusability

## Files Changed

- **Created**: `docs/technical/` directory (new)
- **Created**: `docs/technical/component-organization-guidelines.md` (new technical doc)
- **Updated**: `docs/project/backlog.md` (issue entry)

## Testing

N/A - Documentation only

## Next Steps

After this PR merges, these guidelines can be:
1. Referenced in code reviews when questioning component organization
2. Applied to new component decisions
3. Used as foundation for future reorganization proposals (if needed)

## Related Issues

This documentation arose from discussion about mem_binary component organization.
Future reorganization decisions (if any) will be separate issues that reference these guidelines.
EOF
)"
```

## Verification Steps

After implementation:

1. **Check directory structure:**
   ```bash
   ls -la docs/technical/
   # Should show: component-organization-guidelines.md
   ```

2. **Verify document completeness:**
   ```bash
   grep -E "^## [0-9]\.|^# " docs/technical/component-organization-guidelines.md
   # Should show all major sections
   ```

3. **Check markdown formatting:**
   - Use markdown preview or `mdl` linter
   - Verify tables render correctly
   - Check links work

4. **Verify backlog updated:**
   ```bash
   grep "#059" docs/project/backlog.md
   # Should show issue entry
   ```

## Success Criteria

- [ ] `docs/technical/component-organization-guidelines.md` created with all sections
- [ ] Decision matrix table clearly formatted
- [ ] Generic examples provided (no specific component analysis)
- [ ] Edge cases documented (code generation, platform-specific)
- [ ] Backlog updated with issue #059
- [ ] Branch pushed to fork
- [ ] Draft PR created with comprehensive description
- [ ] All markdown properly formatted

## Notes

**Document maintenance:**
- This doc should remain stable over time (timeless guidelines)
- Updates only needed if decision criteria fundamentally change
- Specific reorganization decisions belong in separate issues/PRs

**Relationship to existing docs:**
- `docs/architecture.md`: Describes current architecture layers
- This doc: Describes how to decide component organization
- Complementary, not overlapping

**PR description importance:**
- Comprehensive PR description captures design rationale in git history
- Future readers understand WHY these guidelines exist
- Git history serves as decision log for this architectural choice
