<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Remove Dead Code from TarFile Implementation

**Issue:** #037
**Created:** 2026-02-02
**Status:** READY

## Objective

Remove `rename_symlink()` and `rename_existing_entry()` functions that have been dead code since April 2025 when TarFile design shifted to append-only semantics.

## Background

### Git History - Why This Code Is Dead

**April 9, 2025** - Commit `87d108d` "refactor tar entry (#32)"
- Initial TarFile implementation with `rename_existing_entry()` and `rename_symlink()`
- Original Case 4 logic (file overwrite with different data):
  ```cpp
  auto& new_data_entry = add_entry_for_new_data(md5.value(), false);
  add_1024_padding();
  rename_existing_entry(new_data_entry, *prev_entry_for_path); // ✅ WAS CALLED
  ```
- Approach: Modify existing tar header in-place, convert old entry to symlink
- Related JIRA: VAI-10873 (weight sharing), VAI-10864 (offline caching)

**April 14, 2025** - Commit `9081edc` "Br verify write same content twice (#74)"
- Changed Case 4 logic - **removed the call to `rename_existing_entry()`**
- New approach (still current):
  ```cpp
  // TAR is append only, append the new entry to the end of the file.
  // this is as same as the first case.
  add_entry_for_new_data(md5.value());
  add_1024_padding();
  // ❌ NO LONGER calls rename_existing_entry()
  ```
- **Reason**: "TAR is append only" - modifying headers violates this principle
- Simpler, more correct: Just append, last entry wins (standard TAR semantics)

**Result**: Functions abandoned **9 months ago** but never deleted (oversight)

**Verification**: Confirmed no callers exist anywhere in codebase:
```bash
git grep -n "rename_existing_entry" -- "*.cpp" "*.hpp"
# Only finds: declaration, definition

git grep -n "\.rename_symlink" -- "*.cpp" "*.hpp"
# Only finds: internal call within rename_existing_entry()
```

## Implementation Steps

### Step 1: Remove Dead Functions

**Delete implementations** from `morphizen-core/src/tar_entry.cpp`:

1. **Line 65-81**: `TarEntryInputStreamBuffer::rename_symlink()`
   ```cpp
   bool TarEntryInputStreamBuffer::rename_symlink(const std::string& new_name,
                                                  pos_type data_begin_pos,
                                                  pos_type data_end_pos) {
     if (real_path_) {
       const_cast<std::optional<std::string>&>(real_path_) = new_name;  // ❌ const_cast
       const_cast<pos_type&>(data_begin_pos_) = data_begin_pos;         // ❌ const_cast
       const_cast<pos_type&>(data_end_pos_) = data_end_pos;             // ❌ const_cast
       const_cast<pos_type&>(buffer_pos_) = data_begin_pos;             // ❌ const_cast
       setg(buffer_.data(), buffer_.data(), buffer_.data());
       return true;
     }
     MY_LOG(1) << " rename_symlink failed. entry " << this->to_string()
               << " is not a symblink. new_name=" << new_name;
     return false;
   }
   ```

2. **Line 222-226**: `TarEntryInputStream::rename_symlink()`
   ```cpp
   bool TarEntryInputStream::rename_symlink(
       const std::string& new_name, pos_type data_begin_pos,
       pos_type data_end_pos) {
     return buf_->rename_symlink(new_name, data_begin_pos, data_end_pos);
   }
   ```

3. **Line 475-496**: `TarEntryOutputStream::rename_existing_entry()`
   ```cpp
   void TarEntryOutputStream::rename_existing_entry(
       TarEntryInputStream& data_entry, TarEntryInputStream& prev_entry) {
     MY_LOG(1) << " rename existing entry " << prev_entry.to_string() << " to "
               << data_entry.to_string();
     auto old_entry = prev_entry.to_string();
     prev_entry.rename_symlink(data_entry.path(), data_entry.data_begin_pos(),
                               data_entry.data_end_pos());
     auto header = TarHeader(prev_entry.path(), 0);
     header.set_link_name(data_entry.path());
     auto original_pos = tellp();
     CHECK(seekp(prev_entry.block_begin_pos()).good());
     header.write_header(*this);
     CHECK(seekp(original_pos).good())
         << "seekp failed. original_pos=" << original_pos
         << " prev_entry.block_begin_pos()=" << prev_entry.block_begin_pos();
     if (!this->good()) {
       MY_LOG(1) << "write symbol header failed. name=" << name_;
     } else {
       MY_LOG(1) << " old entry " << old_entry << " renamed to "
                 << prev_entry.to_string() << " stream_pos=" << tellp();
     }
   }
   ```

**Delete declarations** from `morphizen-core/src/tar_entry.hpp`:

1. **Line 28-29**: `TarEntryInputStreamBuffer::rename_symlink()` declaration
2. **Line 105-106**: `TarEntryInputStream::rename_symlink()` declaration
3. **Line 155-156**: `TarEntryOutputStream::rename_existing_entry()` declaration

### Step 2: Fix Misleading Documentation

**Update** `morphizen-core/src/tar_file.hpp:103-105`:

**OLD (incorrect)**:
```cpp
  // if the filename already exists, the orinal entry will be renamed to a wired
  // invisiable name,
  // TODO: check if tar file header support delete flag.
```

**NEW (correct)**:
```cpp
  // If the filename already exists, a new entry is appended to the tar file.
  // When reading, the last entry with a given name takes precedence (standard
  // TAR semantics). The tar file is append-only; existing entries are never
  // modified.
```

### Step 3: Verify No Remaining References

```bash
# Verify functions are gone
git grep -n "rename_symlink" -- morphizen-core/src/
git grep -n "rename_existing_entry" -- morphizen-core/src/

# Should find no results
```

### Step 4: Run Tests

```bash
# Build
cmake --build ../../build/morphizen-core --config Debug --parallel

# Run tar-related tests
../../build/morphizen-core/bin/morphizen-unit-tests.exe --gtest_filter=*Tar*
```

**Expected**: All tests pass (no behavioral changes)

## Critical Files

- `morphizen-core/src/tar_entry.cpp` - Delete 3 function implementations
- `morphizen-core/src/tar_entry.hpp` - Delete 3 function declarations
- `morphizen-core/src/tar_file.hpp` - Fix misleading comment
- `unit-test/morphizen/test_tar_file.cpp` - Verify tests still pass

## Benefits

1. ✅ **Eliminate const_cast violations** - Removes 4 const_cast usages (security/correctness)
2. ✅ **Remove misleading code** - Code suggests features that don't exist
3. ✅ **Reduce maintenance burden** - ~60 lines of dead code removed
4. ✅ **Clarify append-only design** - Documentation now matches implementation
5. ✅ **No behavioral changes** - Code was already unused

## Success Criteria

- [ ] All 3 function implementations deleted
- [ ] All 3 function declarations deleted
- [ ] tar_file.hpp comment updated to reflect append-only semantics
- [ ] No references to `rename_symlink` or `rename_existing_entry` remain
- [ ] All existing tests pass
- [ ] No const_cast violations remain in removed code

## Notes

This is pure code removal with no functional changes. The functions have been dead for 9 months since the design shifted from "modify-in-place" to "append-only" for TAR file operations.
