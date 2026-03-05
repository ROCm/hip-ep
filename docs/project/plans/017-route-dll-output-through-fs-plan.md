<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Plan #017 — Route model.dll output through FileSystem; rename `--constants-dir` to `--output-dir`

## Overview

Seven files to change. Steps 1–3 are the core fix; steps 4–7 are the rename
and test cleanup. All changes are straightforward mechanical edits once the
core wiring is understood.

---

## Step 1: `lib/Compiler/CompilerDriver.cpp`

### 1a. Thread `FileSystem*` into `compileImpl`

`CompilerDriver` already stores `fs_` (set via `setFileSystem()`). The private
`compileImpl` method needs to use it when writing the DLL.

Locate `linkToDLL` call in `compileImpl` (around line 175):

**Before:**
```cpp
if (!linkToDLL(obj_path, output_path, libraries, library_paths,
               export_symbols, error_message))
  return false;
```

**After:**
```cpp
if (!linkToDLL(obj_path, output_path, libraries, library_paths,
               export_symbols, fs_, error_message))
  return false;
```

### 1b. Update `linkToDLL` signature and implementation

**Before (around line 264):**
```cpp
bool CompilerDriver::linkToDLL(const std::string& objPath,
                               const std::string& dllPath,
                               const std::vector<std::string>& libraries,
                               const std::vector<std::string>& library_paths,
                               const std::vector<std::string>& export_symbols,
                               std::string& error_message) {
  hipdnn::DLLLinker linker;

  if (!linker.linkDLL(objPath, dllPath, libraries, library_paths,
                      export_symbols)) {
    error_message = "Failed to link DLL";
    return false;
  }

  return true;
}
```

**After:**
```cpp
bool CompilerDriver::linkToDLL(const std::string& objPath,
                               const std::string& dllName,
                               const std::vector<std::string>& libraries,
                               const std::vector<std::string>& library_paths,
                               const std::vector<std::string>& export_symbols,
                               morphizen::FileSystem* fs,
                               std::string& error_message) {
  hipdnn::DLLLinker linker;

  // TODO: linkDLLInMemory loads the entire DLL into memory, which doubles
  // peak memory for large models. Add a compilation option in
  // proto/compilation_options.fbs (e.g. `use_tmp_file_for_linking: bool`)
  // to allow keeping the DLL in a temp file and streaming it to fs, avoiding
  // the memory spike.
  std::vector<uint8_t> dllBytes;
  if (!linker.linkDLLInMemory(objPath, dllBytes, libraries, library_paths,
                               export_symbols)) {
    error_message = "Failed to link DLL";
    return false;
  }

  // Write DLL through fs so all compiler outputs go to the same cache
  // (EP tar cache in production, DiskFileSystem for CLI / tests).
  auto writer = fs->create_writer(dllName.c_str());
  if (!writer) {
    error_message = "Failed to create writer for: " + dllName;
    return false;
  }
  writer->write(dllBytes.data(), dllBytes.size());

  return true;
}
```

Note: `linkDLLInMemory` currently takes `const std::vector<uint8_t>&` for
object bytes. Check its signature — you may need a variant that accepts a file
path (or read the obj file into a vector first). If `linkDLLInMemory` only
accepts bytes, add a small helper to read `objPath` into a vector before
calling it.

### 1c. Update `CompilerDriver.h`

Update the private `linkToDLL` declaration to match the new signature
(add `morphizen::FileSystem* fs` parameter).

---

## Step 2: `tools/hip-compile/hip-compile.cpp`

### 2a. Rename field

```cpp
// Before:
std::string constantsDir;

// After:
std::string outputDir;
```

### 2b. Rename CLI flag parsing (around line 56)

```cpp
// Before:
} else if (arg == "--constants-dir" && i + 1 < argc) {
  constantsDir = argv[++i];

// After:
} else if (arg == "--output-dir" && i + 1 < argc) {
  outputDir = argv[++i];
```

### 2c. Update help text (around line 86)

```cpp
// Before:
<< "  --constants-dir <dir>    Directory to write the constants file "
   "into;\n"
<< "                           created if it does not exist (default: "
   "current dir)\n"

// After:
<< "  --output-dir <dir>       Directory for all compiler outputs "
   "(model.dll, constants file);\n"
<< "                           created if it does not exist (default: "
   "current dir)\n"
```

### 2d. Rename all uses of `constantsDir` → `outputDir`

Three locations: `makeFileSystem(opts.constantsDir)` (×2) and verbose output
`"Constants dir: " << opts.constantsDir`. Update the verbose label too:

```cpp
// Before:
if (!opts.constantsDir.empty())
  std::cout << "Constants dir: " << opts.constantsDir << "\n";

// After:
if (!opts.outputDir.empty())
  std::cout << "Output dir: " << opts.outputDir << "\n";
```

Also rename `makeFileSystem`'s parameter for clarity:
```cpp
// Before:
static hip::DiskFileSystem makeFileSystem(const std::string& constantsDir)

// After:
static hip::DiskFileSystem makeFileSystem(const std::string& outputDir)
```

---

## Step 3: `tools/hip-opt/hip-opt.cpp`

### 3a. Rename `PipelineOptions` field and option string

```cpp
// Before:
Option<std::string> constantsDir{
    *this, "constants-dir",
    llvm::cl::desc("Directory to write constants file into (default: .)"),
    llvm::cl::init("")};

// After:
Option<std::string> outputDir{
    *this, "output-dir",
    llvm::cl::desc("Directory for all compiler outputs (default: .)"),
    llvm::cl::init("")};
```

### 3b. Update usage in the pipeline registration lambda

```cpp
// Before:
const std::string& dir = opts.constantsDir;

// After:
const std::string& dir = opts.outputDir;
```

---

## Step 4: `test/e2e/CMakeLists.txt`

Single line change (around line 39):

```cmake
# Before:
--constants-dir "${test_dir}"

# After:
--output-dir "${test_dir}"
```

The comment on line 32 also references `--constants-dir`; update it:

```cmake
# Before:
# --constants-dir routes the constants file into the per-test directory.

# After:
# --output-dir routes all compiler outputs (model.dll, constants) into the per-test directory.
```

---

## Step 5: `test/lit/lit.site.cfg.py.in`

Add one line to route `%T` into the build tree:

```python
# Before (no test_exec_root line):
config.hip_build_dir = r"@CMAKE_BINARY_DIR@"

# After:
config.hip_build_dir = r"@CMAKE_BINARY_DIR@"
config.test_exec_root = r"@CMAKE_CURRENT_BINARY_DIR@"
```

This makes `%T` resolve to `<build>/test/lit/Output/` instead of
`test/lit/e2e/Output/` (inside the source tree).

---

## Step 6: `test/lit/lit.cfg.py`

Remove the line that overrides `test_exec_root`:

```python
# Before:
config.test_exec_root = config.test_source_root

# After:
# (line removed — test_exec_root is now set in lit.site.cfg.py.in)
```

---

## Step 7: `test/lit/e2e/*.mlir`

All files using `--morphizen-pipeline` need `output-dir` and `constants-file`
added to the pipeline options. There are ~14 files.

**Pattern (applies to every file):**

```
// Before:
// RUN: hip-opt %s --morphizen-pipeline 2>&1 | FileCheck %s

// After:
// RUN: hip-opt %s --morphizen-pipeline{output-dir=%T,constants-file=%basename_t.bin} 2>&1 | FileCheck %s
```

For files that pipe without `2>&1`:
```
// Before:
// RUN: hip-opt %s --morphizen-pipeline | FileCheck %s

// After:
// RUN: hip-opt %s --morphizen-pipeline{output-dir=%T,constants-file=%basename_t.bin} | FileCheck %s
```

**Why `%basename_t.bin`**: each test gets a unique `%t`
(e.g. `test_conv_double_layer.mlir.tmp`), so `%basename_t.bin` is unique per
test. Combined with `%T` (build-tree temp dir), there is no cross-test
interference and no source tree pollution.

---

## Verification

```bash
# Build
cmake --build ../build/$(basename $PWD) --config Debug --parallel

# Run all tests
ctest --test-dir ../build/$(basename $PWD) --verbose

# Confirm no stray constants.bin in source tree
git status  # should show no untracked constants.bin files
```

## Success Criteria

- [ ] `ctest` passes with no failures
- [ ] `git status` shows no untracked `constants.bin` in source tree
- [ ] `hip-compile --help` shows `--output-dir` (not `--constants-dir`)
- [ ] `hip-opt --help` lists `output-dir` in morphizen-pipeline options
- [ ] EP tar cache flow: `model.dll` bytes go through `fs->create_writer`
