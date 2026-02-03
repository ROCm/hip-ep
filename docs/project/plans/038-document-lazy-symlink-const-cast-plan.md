<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Document Lazy Symlink Resolution const_cast

**Issue:** #038
**Created:** 2026-02-02
**Status:** READY

## Objective

Add comprehensive documentation to explain the intentional use of `const_cast` for write-once lazy initialization in symlink resolution. This is different from the dead code being removed in Issue #037.

## Background

### Why const_cast Exists

In TAR files, symlinks can appear **before** their targets (forward references):
```
Entry 1: a.txt -> _data/hash123  (symlink appears first)
Entry 2: _data/hash123           (target appears later)
```

When reading the TAR file sequentially:
1. Symlink entry `a.txt` is created with `data_begin_pos = -1` (sentinel value)
2. Later when `open_for_read("a.txt")` is called, resolve the target and populate positions

This is **lazy initialization** - only resolve symlinks that are actually accessed.

### Design Decision: Keep const with Documented const_cast

After discussion, we chose to keep members `const` and document the const_cast rather than using `mutable`:

**Rationale:**
1. ✅ **Compiler protection**: Members stay `const`, preventing accidental modification elsewhere
2. ✅ **Performance**: Lazy resolution - only resolve accessed symlinks
3. ✅ **Simplicity**: No new abstraction classes needed
4. ⚠️ **Trade-off**: Technically undefined behavior, but controlled and well-documented

**Alternatives considered and rejected:**
- **`mutable` members**: Loses compile-time protection - any function could modify by accident
- **Eager resolution**: Performance cost, fails on forward symlinks
- **Encapsulation class**: Extra complexity for minimal benefit

## Implementation Steps

### Step 1: Add Comprehensive Safety Documentation

Add detailed comment block before the const_cast in `morphizen-core/src/tar_file.cpp:203-236`:

**Location:** `TarFile::open_for_read()` at line ~214

**Add this comment block:**
```cpp
TarEntryInputStream* TarFile::open_for_read(const std::string& filename) {
  MY_LOG(1) << " open_for_read: search for file \"" << filename << "\"";
  for (auto& entry : entries_) {
    if (entry->path() == filename) {
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
        if (!real_entry) {
          MY_LOG(1) << " open_for_read: entry \"" << entry->to_string()
                    << "\" not found in the tar file";
          return nullptr;
        }
        const_cast<std::streampos&>(entry->buf_->data_begin_pos_) =
            real_entry->data_begin_pos();
        const_cast<std::streampos&>(entry->buf_->data_end_pos_) =
            real_entry->data_end_pos();
        const_cast<std::streampos&>(entry->buf_->buffer_pos_) =
            real_entry->data_begin_pos();
        return entry.get();
      }
      // ... rest of function
    }
  }
  return nullptr;
}
```

### Step 2: Add Comment to Member Declarations

Update member comments in `morphizen-core/src/tar_entry.hpp:64-69`:

**OLD:**
```cpp
const pos_type data_begin_pos_;  // beginning of the data
const pos_type data_end_pos_;    // end of the data, not including the padding.
```

**NEW:**
```cpp
const pos_type data_begin_pos_;  // beginning of data (write-once: -1 until first access)
const pos_type data_end_pos_;    // end of data, no padding (write-once: -1 until first access)
```

## Critical Files

- `morphizen-core/src/tar_file.cpp` - Add documentation block to `open_for_read()`
- `morphizen-core/src/tar_entry.hpp` - Update member declaration comments

## Benefits

1. ✅ **Clear intent**: Future developers understand why const_cast exists
2. ✅ **Safety rationale**: Documents why this is acceptable vs general const_cast abuse
3. ✅ **Alternatives documented**: Shows we considered other approaches
4. ✅ **Issue tracking**: Links to Issue #038 for full context
5. ✅ **Distinguishes from dead code**: Clear this is intentional, unlike Issue #037

## Success Criteria

- [ ] Comprehensive comment block added to `tar_file.cpp:~214`
- [ ] Member declaration comments updated in `tar_entry.hpp:64-65`
- [ ] Comments explain: context, safety, trade-offs, alternatives
- [ ] Reference to Issue #038 included

## Notes

This is **documentation-only** - no code behavior changes. The const_cast pattern stays as-is, but now with full justification documented inline.

**Different from Issue #037**: Issue #037 removes dead code with const_cast. This issue **documents** active code with const_cast that's intentional and necessary.
