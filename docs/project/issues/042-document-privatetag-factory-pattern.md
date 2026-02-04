<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Issue #042: Document PrivateTag Factory Pattern

## Metadata
- **Type:** Documentation
- **Priority:** LOW
- **Created:** 2026-02-03
- **Dependencies:** None

## Description

Add comprehensive technical documentation for the PrivateTag pattern used in TarFile and update the code comment to reference it. This addresses developer confusion, improves maintainability, and provides a learning resource for this C++ idiom.

## Problem

**Current state:**
- `tar_file.hpp:98` has minimal comment: `// for std::make_unique`
- Doesn't explain WHY the pattern exists or HOW it works
- Developers unfamiliar with this idiom may be confused
- No documentation of the trade-offs vs alternatives

**Why this matters:**
1. **Developer confusion** - Pattern is non-obvious to those unfamiliar with it
2. **Maintainability** - Future developers need to understand the rationale
3. **Learning opportunity** - Documents a useful C++ idiom for the team

## Solution

**Two deliverables:**

1. **Technical documentation** - Create `docs/technical/privatetag-factory-pattern.md` explaining:
   - The problem: Factory pattern + std::make_unique conflict
   - Why std::make_unique can't access private constructors (it's external code)
   - How PrivateTag solves it (access control on the type, not the constructor)
   - Alternative approaches and trade-offs
   - When to use this pattern

2. **Code comment update** - Update `tar_file.hpp:98`:
   ```cpp
   // Before:
   struct PrivateTag {}; // for std::make_unique

   // After:
   struct PrivateTag {}; // PrivateTag pattern - see docs/technical/privatetag-factory-pattern.md
   ```

## Implementation

1. Add `docs/technical/privatetag-factory-pattern.md` (already created)
2. Update comment in `morphizen-core/src/tar_file.hpp:98`
3. Build and verify no changes to functionality

**Files modified:**
- `docs/technical/privatetag-factory-pattern.md` (new)
- `morphizen-core/src/tar_file.hpp` (1-line comment change)

## Notes

Documentation-only change with trivial code comment update. No functional changes.
