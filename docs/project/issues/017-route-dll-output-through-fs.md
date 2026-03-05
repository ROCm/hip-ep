<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Issue #017 — Route model.dll output through FileSystem; rename `--constants-dir` to `--output-dir`

## Problem

`CompilerDriver` writes `model.dll` directly to a real disk path via
`DLLLinker::linkDLL(objPath, dllPath, ...)`, bypassing the `FileSystem*`
abstraction entirely. Constants are correctly written through `fs` (via
`fs->create_writer(constants_filename)`), but the final DLL is not.

This breaks the Contract 2 invariant documented in
`docs/design/MORPHIZEN-EP-INTEGRATION.md`: `hip-compiler.dll` must write
**all** outputs through `fs`, with no knowledge of where the cache lives. In
the EP case, `fs` points to the EP tar cache; writing the DLL to a hardcoded
disk path means it never lands in the cache.

Additionally, the CLI flag `--constants-dir` is misleadingly named — it roots
the `DiskFileSystem` used for all compiler outputs, not just constants. After
this fix it should be renamed `--output-dir`. The `--constants-file` flag
(logical name of the constants blob within the fs) is unchanged.

A secondary problem: `test/lit/lit.cfg.py` sets
`config.test_exec_root = config.test_source_root`, causing LIT to write
temporary files (including `constants.bin`) into the source tree. The e2e LIT
tests also do not pass `--output-dir`, so constants land in the source
directory, producing stray untracked files.

## Solution

1. **`CompilerDriver`**: use `linkDLLInMemory` to obtain the DLL bytes in
   memory, then write through `fs->create_writer(output_path)`. Add a `TODO`
   comment noting that in-memory linking doubles peak memory for large DLLs and
   should later be made configurable via a new field in
   `proto/compilation_options.fbs`.

2. **`--constants-dir` → `--output-dir`**: rename the flag in
   `hip-compile` and `hip-opt` (including `PipelineOptions::constantsDir`
   and its registered option string).

3. **LIT config**: set `config.test_exec_root` to the build directory in
   `lit.site.cfg.py.in` so `%T` resolves inside the build tree, not the
   source tree.

4. **LIT e2e tests**: add `{output-dir=%T,constants-file=%basename_t.bin}` to
   every `--morphizen-pipeline` RUN line, giving each test a unique constants
   filename and routing all output to the build-tree temp directory.

5. **`test/e2e/CMakeLists.txt`**: rename `--constants-dir` to `--output-dir`.

## Files Affected

- `lib/Compiler/CompilerDriver.cpp`
- `tools/hip-compile/hip-compile.cpp`
- `tools/hip-opt/hip-opt.cpp`
- `test/e2e/CMakeLists.txt`
- `test/lit/e2e/*.mlir` (all files using `--morphizen-pipeline`)
- `test/lit/lit.site.cfg.py.in`
- `test/lit/lit.cfg.py`

## Metadata

- **Type:** Bug / Architectural Fix
- **Priority:** HIGH
- **Created:** 2026-03-04
- **Dependencies:** Related to #014 (constant handling design doc)
