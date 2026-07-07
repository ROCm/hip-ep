<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Issue #051: Document Complex TarFile Factory Methods

## Metadata
- **Type:** Documentation
- **Priority:** MEDIUM
- **Dependencies:** None

## Description

Add comprehensive comments to the two complex TarFile factory methods (`create_from_buffer(string, bool)` and `create_from_path()`) to document the mmap strategy, platform differences, fallback logic, and troubleshooting context.

## Problem

**Current behavior:**

Two complex factory methods were recently added (Feb 1-3, 2026) to support mmap optimization:
- `create(string, bool)` (78 lines) - Will be renamed to `create_from_buffer()` in Issue #050
- `create_from_path()` (34 lines)

Both methods have complex fallback logic with multiple code paths, but lack comprehensive comments explaining:
- **Why fallbacks are needed** - Different environments (Windows, Linux, WebNN/Browser)
- **Platform differences** - Windows has mmap on tmpfile, Linux doesn't yet
- **Troubleshooting context** - Why mmap can be disabled
- **User vs developer controls** - Provider option vs environment variable

**Example of inadequate documentation:**

Current comments in `create(string, bool)`:
```cpp
// by default, the stream will be from a tmp file to decrease memory,
// but if access to tmp is restricted, like web sandbox condition,
// the stream should be from a memory buffer
```

This doesn't explain:
- Why mmap is attempted
- Platform differences (Windows-only)
- How to disable mmap for troubleshooting
- Reference to Issue #001 (where mmap was added)

**Why this is problematic:**

1. **Fresh code** - Added 1-3 days ago, knowledge is fresh but will fade
2. **Complex logic** - 78 lines with multiple fallback paths, hard to understand
3. **Critical context missing** - WebNN compatibility, troubleshooting needs not documented
4. **Maintenance burden** - Future developers won't understand the design decisions
5. **High-stakes code** - Affects performance and compatibility across platforms

## Solution

Add comprehensive comments to both complex factory methods documenting:

### 1. High-level Strategy Comment

Add method-level comment explaining the overall approach:

```cpp
/* Creates TarFile from string buffer.
 *
 * Strategy: Optimize for different deployment environments:
 * - Windows: tmpfile + optional mmap (best performance, Issue #001)
 * - Linux: tmpfile + FileStream (mmap not implemented yet, TODO)
 * - WebNN/Browser: In-memory buffer (sandboxed, no tmpfile access)
 *
 * Mmap can be disabled for troubleshooting hard-to-debug system failures:
 * - Production: enable_mmap=false (provider option, user-configurable)
 * - Development: MORPHIZEN_ENABLE_TAR_MMAP=0 (env var, developer-only)
 *
 * Falls back gracefully: tmpfile+mmap → tmpfile+FileStream → memory buffer
 */
```

### 2. Platform-Specific Logic Comments

Document platform differences:

```cpp
#ifdef _WIN32
  if (use_mmap) {
    // Windows supports mmap on tmpfile via MemFileTmpHandle (Issue #001)
    // Linux support is TODO - no customer request yet
    try {
      ...
    } catch {
      // Exception during mmap - fallback to regular FileStream
      // Common in troubleshooting scenarios
      ...
    }
  }
#else
  // Non-Windows platforms: tmpfile mmap not implemented yet (TODO)
  stream = std::make_unique<FileStream>(file);
#endif
```

### 3. Fallback Path Comments

Document each fallback and why it exists:

```cpp
if (file) {
  // tmpfile created successfully - use disk-backed stream (lower memory)
  ...
} else {
  // tmpfile creation failed - use in-memory buffer
  // Common in WebNN/Browser environments (sandboxed, no filesystem access)
  auto buff_owner = std::make_unique<std::string>(std::move(buffer0));
  ...
}
```

### 4. Control Mechanism Comments

Explain the two-level mmap control:

```cpp
// Two-level mmap control (intentional, do not simplify):
// 1. enable_mmap: User preference via provider option (production)
// 2. ENV_PARAM: Global override for developer debugging only
// This allows disabling mmap to troubleshoot hard-to-debug failures
bool use_mmap = enable_mmap && (ENV_PARAM(MORPHIZEN_ENABLE_TAR_MMAP) != 0);
```

### 5. Cross-reference Related Work

Link to related issues for context:

- Issue #001: Where mmap support was implemented
- Issue #048: Platform-specific tmpfile creation (will fix duplication)
- Issue #050: Factory method renaming (this method will be renamed)

## Plans

- [051-document-complex-tarfile-factory-methods-plan.md](../plans/051-document-complex-tarfile-factory-methods-plan.md) - Created 2026-02-03

## Notes

**Discovery:** While analyzing factory method proliferation, found that the 78-line `create(string, bool)` method is too complex to refactor safely (added 1-3 days ago). Better to document comprehensively first, then consider refactoring later after code stabilizes.

**Why document now:**
- Code is fresh (Feb 1-3, 2026) - your knowledge is current
- Complex mmap logic with platform differences
- Critical context (WebNN, troubleshooting) not documented
- Low-risk way to improve maintainability

**Alternative considered:**
Extract complex logic into helper functions - rejected because code is too fresh. Let it stabilize first.

**Git history:**
- 2026-02-03: Issue #001 - Add mmap support for embed mode (#88)
- 2026-02-02: Enable mmap support for embed mode tar files
- 2026-02-01: Add graceful tmpfile fallback to TarFile::create()
