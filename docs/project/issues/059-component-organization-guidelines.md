<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Issue #059: Create Component Organization Guidelines

## Problem

The MorphiZen project lacks clear, documented guidelines for deciding when to create standalone components versus consolidating into existing libraries. This creates several issues:

1. **Architectural inconsistency**: Mix of small standalone components (mem_binary ~111 LOC, encryption ~184 LOC) alongside larger consolidated libraries (morphizen-utils ~500+ LOC)

2. **Decision ambiguity**: No framework for evaluating whether new functionality should be:
   - A new top-level directory/component
   - Added to an existing library (e.g., morphizen-utils)
   - Part of a specific layer (e.g., morphizen-core)

3. **Review friction**: Code reviewers may question component organization decisions without objective criteria to reference

4. **Contributor confusion**: New developers lack guidance on where to place new utilities or features

5. **Potential bikeshedding**: Subjective debates about component organization without clear decision criteria

## Impact

**Current state:**
- 25+ top-level directories in the project
- No documented rationale for when components should be standalone vs consolidated
- Inconsistent patterns (some small utilities are standalone, others are not)

**Without guidelines:**
- Future PRs may face organizational questions/debates
- Risk of further fragmentation (more tiny top-level components)
- Harder to onboard new contributors
- Architectural drift over time

## Solution

Create a technical document that establishes clear, objective criteria for component organization decisions.

**Document location:** `docs/technical/component-organization-guidelines.md`

**Document purpose:**
- Provide decision framework for standalone vs consolidated components
- Establish consistent architectural patterns
- Reduce subjective debates with objective criteria
- Guide future development decisions

**Key principles:**
- **Timeless guidelines**: Focus on reusable decision criteria, not specific current decisions
- **Generic examples**: Use illustrative examples rather than analyzing specific existing components
- **Git history for decisions**: Specific reorganization decisions belong in issues/PRs, not the guidelines doc
- **Actionable criteria**: Clear thresholds and decision matrix

## Metadata

- **Type:** Documentation / Architecture
- **Priority:** MEDIUM
- **Created:** 2026-02-06
- **Component:** Project organization
- **Dependencies:** None (foundational documentation, arose from mem_binary organization discussion)
