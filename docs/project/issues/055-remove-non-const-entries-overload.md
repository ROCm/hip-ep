<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Issue #055: Remove Non-Const entries() Overload from TarFile

## Metadata
- **Type:** Refactoring
- **Priority:** LOW
- **Created:** 2026-02-04
- **Dependencies:** None

## Description

Remove the unused non-const `entries()` overload from TarFile. This method is never used for mutation - even in tests, it's immediately bound to const reference. Keeping only the const overload enforces read-only access and reduces API surface.

## Problem

**Current API:**

```cpp
// tar_file.hpp:106-108
MORPHIZEN_DLL_SPEC
std::vector<std::unique_ptr<TarEntryInputStream>>& entries();  // Non-const

MORPHIZEN_DLL_SPEC
const std::vector<std::unique_ptr<TarEntryInputStream>>& entries() const;  // Const
```

**Why this is problematic:**

1. **Unused for mutation** - No caller actually modifies the returned vector
2. **Misleading API** - Non-const overload suggests mutation is intended, but it never happens
3. **Encapsulation violation** - Allows accidental mutation of internal entries vector
4. **API bloat** - Unnecessary overload increases API surface

**Usage analysis:**

**Test usage (test_tar_file.cpp:290):**
```cpp
static void check_abc(morphizen::TarFile& tar_file_obj) {
  check_entries(tar_file_obj.entries());  // Calls non-const version
  // ...
}

static void check_entries(
    const std::vector<std::unique_ptr<morphizen::TarEntryInputStream>>& entries) {
  // Immediately binds to const reference - no mutation
}
```

**Production usage (pass_context_imp.cpp:553):**
```cpp
std::vector<std::string> PassContextImp::get_cache_file_names() const {
  const auto& entries = tar_file_->entries();  // Calls const version (const context)
  // Read-only iteration
}
```

**Key insight:** Even when the non-const overload is called, the result is immediately bound to a const reference. No caller needs mutable access.

## Solution

Remove the non-const overload, keeping only the const version.

**After removal:**
```cpp
// tar_file.hpp:106-108 (after removal)
MORPHIZEN_DLL_SPEC
const std::vector<std::unique_ptr<TarEntryInputStream>>& entries() const;
```

**Migration:** Zero changes needed - const member functions can be called on non-const objects. The compiler automatically uses the const version.

## Benefits

- ✅ Enforces read-only access to entries vector (better encapsulation)
- ✅ Reduces public API surface (14 → 13 methods)
- ✅ Prevents accidental mutation of internal state
- ✅ Zero code changes required (automatic overload resolution)

## Plans

- [055-remove-non-const-entries-overload-plan.md](../plans/055-remove-non-const-entries-overload-plan.md) - Created 2026-02-04

## Notes

**Discovery:** Found during Task #14 (Public API Surface Too Large) while analyzing TarFile's public methods.

**C++ overload resolution:** When only the const overload exists, non-const objects can still call it. This is why no test changes are needed.

**API count after this change:**
- Factory methods: 6 (unchanged)
- Instance methods: 6 (was 7)
- Total public API: 13 (was 14)
