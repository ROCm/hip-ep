<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->

# Fixing udna-compile Linker Errors

**Date:** 2025-02-22
**Document Type:** Tech Note
**Status:** Draft
**Related:** N/A

## Overview

After refactoring Backend → Target/LLVM, udna-compile failed to link with 46 unresolved LLVM symbols. Root cause: CMakeLists.txt dropped library dependencies required by lldCOFF.

## Analysis

### Unresolved Symbols (46 total)

| Category | Count | Examples |
|----------|-------|----------|
| LTO | 12 | `llvm::lto::LTO::run`, `llvm::lto::InputFile::create` |
| Option parsing | 8 | `llvm::opt::OptTable::ParseArgs`, `llvm::opt::ArgList` |
| Windows manifest | 4 | `llvm::windows_manifest::WindowsManifestMerger` |
| Windows SDK | 6 | `llvm::findVCToolChainViaEnvironment` |
| Code generation | 16 | `llvm::createMCCodeEmitter`, `llvm::SelectionDAGBuilder` |

### Root Cause: Missing LLVM Dependencies

CMakeLists.txt refactoring dropped 14 LLVM components required by lldCOFF:

**Missing components:**
- CodeGen, SelectionDAG, AsmPrinter (instruction selection, assembly output)
- BinaryFormat, MC, MCParser, TargetParser (object file handling)
- LTO, Option (link-time optimization, command-line parsing)
- WindowsDriver, WindowsManifest, LibDriver (MSVC SDK, manifest merging, import libs)

**Why lldCOFF needs these:** lldCOFF performs link-time optimization (merges bitcode, generates machine code) and handles Windows-specific features (.manifest merging, import library generation). With static linking (`BUILD_SHARED_LIBS=OFF`), transitive dependencies must be explicit.

**Before (worked):**
```cmake
target_link_libraries(UdnaBackend PUBLIC
    LLVMCore LLVMSupport LLVMCodeGen LLVMSelectionDAG
    LLVMAsmPrinter LLVMMC LLVMOption LLVMLTO
    LLVMWindowsDriver LLVMWindowsManifest LLVMLibDriver
    lldCommon lldCOFF
)
```

**After refactoring (broken):**
```cmake
llvm_map_components_to_libnames(llvm_libs
  Support Core Target X86CodeGen Linker
)
target_link_libraries(UdnaTargetLLVM PUBLIC
  ${llvm_libs} ${LLD_COFF_LIB}
)
```

## Solution

### Restore LLVM Components

Add missing components to `llvm_map_components_to_libnames`:

```cmake
llvm_map_components_to_libnames(llvm_libs
  # Core
  Support Core IRReader BitWriter Passes Target Linker

  # Code generation (for lldCOFF LTO)
  CodeGen SelectionDAG AsmPrinter BinaryFormat MC MCParser TargetParser

  # X86 backend
  X86CodeGen X86AsmParser X86Desc X86Info

  # LTO and option parsing (for lldCOFF)
  LTO Option

  # Windows-specific (for lldCOFF)
  WindowsDriver WindowsManifest LibDriver
)
```

**Component purposes:**

| Component | Used For |
|-----------|----------|
| CodeGen, SelectionDAG, AsmPrinter | Instruction selection, assembly generation |
| BinaryFormat, MC, MCParser, TargetParser | Object file parsing, machine code emission |
| LTO, Option | Link-time optimization, command-line parsing |
| WindowsDriver, WindowsManifest, LibDriver | SDK detection, manifest merging, import libs |

## Results

**Commit:** 1f132057 - Added 14 LLVM components, resolved all 46 linker errors

**Build verification:**
- udna-compile.exe: builds successfully, 45 MB (Debug)
- udna-opt.exe: builds successfully, 42 MB (Debug)

## Conclusion

**Recommendations for refactoring CMakeLists.txt:**

1. **LLVM static builds require explicit dependencies:** Don't assume high-level libraries (lld*) pull in dependencies. Link the full stack: CodeGen, MC, LTO, Option, Windows*.

2. **Migrate custom build steps:** During refactoring, verify:
   - `add_custom_command` / `add_custom_target` migrated
   - `DEPENDS` clauses preserved
   - `add_dependencies` relationships maintained
   - Test with clean build (`rm -rf build`)

3. **Verify relative paths:** `CMAKE_CURRENT_BINARY_DIR` mirrors source tree. Moving directories changes path semantics. Use absolute paths or PARENT_SCOPE variables.

4. **Clean builds catch missing dependencies:** Incremental builds mask missing custom commands if generated files exist from previous builds.

## Related Documents

N/A
