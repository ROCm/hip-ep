<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Standardize TarFile Factory Method Naming

**Issue:** #050
**Created:** 2026-02-03
**Status:** READY

## Objective

Rename TarFile factory methods to follow consistent "create_from_X" pattern for better discoverability and API clarity.

## Background

**Problem discovered:** While analyzing TarFile God Class, found that only `create_from_path()` has explicit naming, while 6 other factories use overloaded `create()`. This makes the API confusing.

**Why rename:** Consistent naming improves discoverability, makes code self-documenting, and helps reveal duplication patterns.

## Implementation Steps

### Step 1: Rename Method Declarations in Header

**File:** `morphizen-core/src/tar_file.hpp`

**Find and rename:**

```cpp
// Line ~45: Rename create() → create_from_tmpfile()
MORPHIZEN_DLL_SPEC static std::unique_ptr<TarFile> create();
// TO:
MORPHIZEN_DLL_SPEC static std::unique_ptr<TarFile> create_from_tmpfile();

// Line ~55: Rename create(vector) → create_from_buffer(vector)
static std::unique_ptr<TarFile> create(std::vector<char>&& buffer);
// TO:
static std::unique_ptr<TarFile> create_from_buffer(std::vector<char>&& buffer);

// Line ~67-80: Remove create(string) and rename create(string, bool)
static std::unique_ptr<TarFile> create(std::string&& buffer);  // DELETE THIS
static std::unique_ptr<TarFile> create(std::string&& buffer, bool enable_mmap);
// TO:
static std::unique_ptr<TarFile> create_from_buffer(std::string&& buffer,
                                                    bool enable_mmap = true);

// Line ~95: Rename create(char*, size_t) → create_from_data(char*, size_t)
static std::unique_ptr<TarFile> create(const char* data, size_t size);
// TO:
static std::unique_ptr<TarFile> create_from_data(const char* data, size_t size);
```

### Step 2: Rename Method Implementations

**File:** `morphizen-core/src/tar_file.cpp`

**Rename implementations:**

```cpp
// Line 62: create() → create_from_tmpfile()
std::unique_ptr<TarFile> TarFile::create() {
// TO:
std::unique_ptr<TarFile> TarFile::create_from_tmpfile() {

// Line 91: create(vector) → create_from_buffer(vector)
std::unique_ptr<TarFile> TarFile::create(std::vector<char>&& buffer) {
// TO:
std::unique_ptr<TarFile> TarFile::create_from_buffer(std::vector<char>&& buffer) {

// Line 108: create(string, bool) → create_from_buffer(string, bool = true)
std::unique_ptr<TarFile> TarFile::create(std::string&& buffer0,
                                         bool enable_mmap) {
// TO:
std::unique_ptr<TarFile> TarFile::create_from_buffer(std::string&& buffer0,
                                                      bool enable_mmap) {

// Line 187-190: DELETE this wrapper (consolidated into above)
std::unique_ptr<TarFile> TarFile::create(std::string&& buffer0) {
  return create(std::move(buffer0), true);
}
// DELETE ENTIRE METHOD

// Line 101: create(char*, size_t) → create_from_data(char*, size_t)
std::unique_ptr<TarFile> TarFile::create(const char* base, size_t size) {
// TO:
std::unique_ptr<TarFile> TarFile::create_from_data(const char* base, size_t size) {
```

### Step 3: Find All Callers

**Search for usages:**

```bash
# Find all TarFile::create() calls
grep -rn "TarFile::create(" morphizen-core/src/ unit-test/

# Find all .create() calls on TarFile objects
grep -rn "\.create(" morphizen-core/src/ unit-test/ | grep -i tar
```

**Expected callers to update:**
- Unit tests in `unit-test/morphizen/test_tar_file.cpp`
- PassContext implementations
- Any other code creating TarFile instances

### Step 4: Update All Callers

For each caller found, update to use new names:

```cpp
// Old → New
TarFile::create()                       → TarFile::create_from_tmpfile()
TarFile::create(my_vector)              → TarFile::create_from_buffer(my_vector)
TarFile::create(my_string)              → TarFile::create_from_buffer(my_string)
TarFile::create(my_string, true)        → TarFile::create_from_buffer(my_string, true)
TarFile::create(my_string, false)       → TarFile::create_from_buffer(my_string, false)
TarFile::create(ptr, size)              → TarFile::create_from_data(ptr, size)
```

**Note:** `create(stream)` and `create_from_path()` remain unchanged.

### Step 5: Update Documentation Comments

**File:** `morphizen-core/src/tar_file.hpp`

Update doc comments to reflect new names:

```cpp
/**
 * @brief Creates a TarFile instance from a temporary file.
 *
 * Creates an empty tar file using tmpfile() (or tmpfile_with_posix_delete()
 * on Windows). Falls back to in-memory stringstream if tmpfile creation fails.
 *
 * @return A unique pointer to the created TarFile instance.
 */
static std::unique_ptr<TarFile> create_from_tmpfile();

/**
 * @brief Creates a TarFile instance from a buffer.
 *
 * Takes ownership of the buffer and wraps it in a TarFile. For string buffers,
 * optionally writes to tmpfile and uses memory mapping for better performance.
 *
 * @param buffer The buffer containing tar file data (vector or string)
 * @param enable_mmap (string only) If true, attempt to use memory mapping
 * @return A unique pointer to the created TarFile instance.
 */
static std::unique_ptr<TarFile> create_from_buffer(std::vector<char>&& buffer);
static std::unique_ptr<TarFile> create_from_buffer(std::string&& buffer,
                                                    bool enable_mmap = true);

/**
 * @brief Creates a TarFile instance from raw data.
 *
 * Creates a non-owning view of the provided data. Caller must ensure the
 * data pointer remains valid for the lifetime of the TarFile.
 *
 * @param data Pointer to the raw data buffer
 * @param size The size of the data buffer
 * @return A unique pointer to the created TarFile instance.
 */
static std::unique_ptr<TarFile> create_from_data(const char* data, size_t size);
```

## Verification

### Build
```bash
cmake --build ../../build/morphizen-core --config Debug --parallel
```

### Test
```bash
../../build/morphizen-core/bin/morphizen-unit-tests.exe --gtest_filter=*Tar*
```

**Expected:** All tests pass (same functionality, just renamed)

### Code Review Checklist

- [ ] All 5 method names updated in header
- [ ] All 5 method names updated in implementation
- [ ] Wrapper method `create(string)` deleted
- [ ] Default parameter added to `create_from_buffer(string, bool = true)`
- [ ] All callers updated to use new names
- [ ] Documentation comments updated
- [ ] Build succeeds
- [ ] All tests pass

## Success Criteria

- [ ] Consistent naming pattern: `create_from_X()` for all high-level factories
- [ ] 7 methods → 6 methods (consolidated string variants)
- [ ] All callers updated
- [ ] No compilation errors
- [ ] All tests pass
- [ ] API is more discoverable and self-documenting

## Files Modified

- `morphizen-core/src/tar_file.hpp` - Rename declarations, update docs
- `morphizen-core/src/tar_file.cpp` - Rename implementations, delete wrapper
- All caller files - Update to use new method names

## Notes

**Why this order:**
Rename all methods together in one PR to avoid confusion and keep changes atomic.

**Breaking change:**
This is intentional - we want clean API with no deprecated methods cluttering the interface.

**Follow-up:**
After this naming cleanup, Issue #051 will extract complex logic from the long factory methods (especially the 77-line `create_from_buffer(string, bool)`).
