<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Issue #062: Clean Up Root Configuration Files

## Problem

Two minor configuration file issues exist at the repository root:

1. **Empty pyproject.toml**: 0-byte file with unclear purpose
   - Python projects use pyproject.toml for package metadata and build configuration
   - Currently empty - either needs content or should be removed
   - No clear indication if placeholder for future use

2. **.gitignore.pr-review documentation**: Example file that could be simplified
   - Contains example patterns for PR review toolkit
   - Pattern `pr*_review.md` should be in main `.gitignore`, not a separate example file
   - Naming doesn't follow `.example` convention

## Current State

```
Repository root:
├── pyproject.toml              (0 bytes - EMPTY)
├── .gitignore                  (51 lines)
└── .gitignore.pr-review        (11 lines - documentation/example)
```

**pyproject.toml:**
- Size: 0 bytes
- No content

**.gitignore.pr-review:**
```gitignore
# PR Review Toolkit - Add these to your .gitignore

# Generated review files (created by submit_review.ps1)
pr*_review.md

# Temporary PR branches (optional - if you prefer to keep PR branches local)
# pr-*

# Example:
# pr437_review.md  <- Will be ignored
# pr999_review.md  <- Will be ignored
```

**Main .gitignore:**
- Does NOT contain `pr*_review.md` pattern
- Pattern should be in main file, not example

## Impact

**Low impact issues:**
- Empty pyproject.toml may confuse contributors about Python packaging intent
- `.gitignore.pr-review` pattern is not active (needs to be in main `.gitignore`)
- Unclear whether `.gitignore.pr-review` is documentation or active config

## Solution

### Option A: Minimal Cleanup (Recommended)

1. **Delete pyproject.toml**
   - Not needed - project uses CMake/Bazel, not Python packaging
   - `requirements.txt` is sufficient for Python dependencies

2. **Move PR review pattern to main .gitignore**
   - Add `pr*_review.md` to main `.gitignore`
   - Delete `.gitignore.pr-review` (pattern is now active)

### Option B: Keep as Placeholders

1. **Add comment to pyproject.toml**
   ```toml
   # Placeholder for future Python packaging configuration
   # Currently using requirements.txt for dependencies only
   ```

2. **Rename .gitignore.pr-review**
   - Rename to `.gitignore.pr-review.example`
   - Makes it clear it's documentation/example, not active config

## Deliverables

**Option A (Recommended):**
1. Delete `pyproject.toml`
2. Add `pr*_review.md` to main `.gitignore`
3. Delete `.gitignore.pr-review`

**Option B:**
1. Add comment to `pyproject.toml` explaining placeholder status
2. Rename `.gitignore.pr-review` to `.gitignore.pr-review.example`

## Acceptance Criteria

**Option A:**
- [ ] pyproject.toml deleted
- [ ] `pr*_review.md` pattern in main .gitignore
- [ ] .gitignore.pr-review deleted
- [ ] No Python build tools complain about missing pyproject.toml

**Option B:**
- [ ] pyproject.toml contains explanatory comment
- [ ] .gitignore.pr-review renamed to .gitignore.pr-review.example

## Notes

- requirements.txt contains `patch==1.16` for Bazel - this is legitimate and should remain
- All other configuration files (BUILD.bazel, CMakeLists.txt, .bazelrc, etc.) are properly organized and follow tool conventions

## Metadata

- **Type:** Cleanup / Documentation
- **Priority:** LOW
- **Created:** 2026-02-06
- **Estimated Effort:** 5-10 minutes
- **Component:** Project configuration
- **Dependencies:** None
