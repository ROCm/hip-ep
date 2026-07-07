<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# TODO: Add Circular Symlink Protection to TarFile

**Priority**: Low
**Category**: Security / Robustness
**Created**: 2026-02-02

## Summary

Add protection against infinite recursion in `find_real_entry()` when resolving symlinks. Currently vulnerable to stack overflow if TAR file contains circular symlink chains.

## The Issue

**Current code** (`tar_file.cpp:277-304`):
```cpp
TarEntryInputStream* TarFile::find_real_entry(const std::string& real_path) {
  auto it = std::find_if(...);
  if (it != entries_.rend()) {
    if ((*it)->real_path()) {
      // ⚠️ RECURSIVE CALL - no cycle detection!
      return find_real_entry((*it)->real_path().value());
    }
    return (*it).get();
  }
  return nullptr;
}
```

**Attack scenario**: Malicious TAR file with circular symlinks:
```
a.txt -> b.txt
b.txt -> c.txt
c.txt -> a.txt  // Loop: a → b → c → a → ...
```

**Result**: Infinite recursion → stack overflow → crash

## Why Not Fixed Yet

- TAR files currently come from trusted sources only
- No known exploits in our use case
- Low priority compared to other improvements

## Proposed Solution (When Needed)

**Option A: Simple max depth limit** (Recommended)
```cpp
TarEntryInputStream* find_real_entry(const std::string& real_path,
                                     int depth = 0) {
  static constexpr int MAX_SYMLINK_DEPTH = 32;  // Linux uses 40
  if (depth >= MAX_SYMLINK_DEPTH) {
    LOG(WARNING) << "Symlink depth limit exceeded: " << real_path;
    return nullptr;
  }

  auto it = std::find_if(...);
  if (it != entries_.rend()) {
    if ((*it)->real_path()) {
      return find_real_entry((*it)->real_path().value(), depth + 1);
    }
    return (*it).get();
  }
  return nullptr;
}
```

**Pros:**
- Simple: 3 lines of code
- Fast: O(1) check, no allocations
- Proven: Linux kernel uses same approach (40 level limit)

**Option B: Full cycle detection with visited set**
- More complex, requires hash set allocation
- Still needs max depth as backup
- Overkill for this use case

## When to Implement

Consider implementing if:
- TAR files will come from untrusted/user-provided sources
- Security audit requires defense-in-depth
- We encounter actual circular symlink issues in production

## References

- Linux symlink limit: `/include/linux/namei.h` defines `SYMLOOP_MAX = 40`
- Related discussion: See plan file `dynamic-whistling-badger.md` Topic #3

## Files to Modify

- `morphizen-core/src/tar_file.hpp` - Add depth parameter to `find_real_entry()`
- `morphizen-core/src/tar_file.cpp` - Add depth check logic
- `unit-test/morphizen/test_tar_file.cpp` - Add test case for circular symlinks
