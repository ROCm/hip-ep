<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Issue #038: Document Lazy Symlink Resolution const_cast

## Metadata
- **Type:** Documentation
- **Priority:** LOW
- **Created:** 2026-02-02
- **Dependencies:** None

## Description

Add comprehensive inline documentation to explain the intentional use of `const_cast` for write-once lazy initialization in symlink resolution. This distinguishes it from the dead code const_cast violations being removed in Issue #037.

## Problem

**Current code uses const_cast without explanation:**

```cpp
// tar_file.cpp:214-219
if (entry->data_begin_pos() == std::streampos(-1)) {
  auto real_entry = find_real_entry(filename);
  if (!real_entry) {
    return nullptr;
  }
  // No documentation explaining why const_cast is acceptable here
  const_cast<std::streampos&>(entry->buf_->data_begin_pos_) =
      real_entry->data_begin_pos();
  const_cast<std::streampos&>(entry->buf_->data_end_pos_) =
      real_entry->data_end_pos();
  const_cast<std::streampos&>(entry->buf_->buffer_pos_) =
      real_entry->data_begin_pos();
  return entry.get();
}
```

**Why this is problematic:**

1. **Looks like code smell**: Without context, appears similar to dead code in Issue #037
2. **Unclear intent**: Why is const_cast used? Why not `mutable`?
3. **Safety unclear**: Is this undefined behavior? Why is it acceptable?
4. **No alternatives documented**: Were other approaches considered?

**Context - Why const_cast Exists:**

TAR files can have forward symlinks (symlink appears before target):
```
Entry 1: a.txt -> _data/hash123  (symlink, target unknown yet)
Entry 2: _data/hash123           (target appears later)
```

When reading sequentially:
1. Symlink entry created with `data_begin_pos = -1` (sentinel)
2. On first `open_for_read()`, resolve target and populate positions
3. This is **lazy initialization** for performance

**Design Decision Made:**

After discussion, chose to:
- ✅ Keep members `const` (compile-time protection)
- ✅ Keep `const_cast` (write-once lazy init)
- ✅ Add documentation (explain safety and trade-offs)

Rejected alternatives:
- ❌ `mutable` members: Loses compiler protection, any function could modify
- ❌ Eager resolution: Performance cost, fails on forward symlinks
- ❌ Encapsulation class: Extra complexity

**Code locations:**
- `tar_file.cpp:214-219` - const_cast location needing documentation
- `tar_entry.hpp:64-65` - Member declarations needing clarification

## Solution

**Add comprehensive documentation:**

```cpp
// tar_file.cpp:~214
if (entry->data_begin_pos() == std::streampos(-1)) {
  // ================================================================
  // LAZY SYMLINK RESOLUTION - Write-Once Initialization
  // ================================================================
  //
  // CONTEXT: Symlinks can appear in TAR before their targets:
  //   Entry 1: a.txt -> _data/hash123  (symlink, target unknown)
  //   Entry 2: _data/hash123           (target appears later)
  //
  // DESIGN: Entries created with sentinel -1, resolved on first access.
  //
  // SAFETY: This const_cast is intentional and safe because:
  // 1. Write-once: Only modifies from sentinel -1 to real value
  // 2. Happens exactly once per entry (checked before modification)
  // 3. Members declared const prevents accidental modification elsewhere
  // 4. Alternative (mutable) would lose compiler protection
  //
  // WHY LAZY? Performance - avoid resolving all symlinks upfront.
  // Eager resolution would require second pass or fail on forward refs.
  //
  // TRADE-OFF: Technically undefined behavior per C++ standard, but:
  // - Controlled: Only one code path modifies
  // - Necessary: Design requires forward symlink support
  // - Safe in practice: No observed issues, write-once semantics
  //
  // See Issue #038 for full discussion and alternatives.
  // ================================================================
  auto real_entry = find_real_entry(filename);
  // ... rest
}
```

**Approach:**
1. Add comprehensive comment block explaining context, safety, trade-offs
2. Update member declaration comments to clarify write-once semantics
3. Link to Issue #029 for full discussion

**Benefits:**
- ✅ Clear intent - developers understand why const_cast exists
- ✅ Safety rationale - explains why this is acceptable
- ✅ Alternatives documented - shows deliberate choice
- ✅ Distinguishes from dead code - unlike Issue #028, this is intentional

## Plans

- [038-document-lazy-symlink-const-cast-plan.md](../plans/038-document-lazy-symlink-const-cast-plan.md) - Created 2026-02-02

## Notes

**This is documentation-only** - no code behavior changes. The const_cast stays, but with full justification.

**Contrast with Issue #037:**
- **Issue #037**: Removes dead code with const_cast violations
- **Issue #038**: Documents active code with intentional const_cast

The key difference: #037 is about code that should never have existed. #038 is about code that's correct but needs explanation.
