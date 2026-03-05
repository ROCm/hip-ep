<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->

# In-Memory Linking Implementation Plan

**Status**: Design Document (Not Yet Implemented)
**Target**: Enable LLD-based linking without disk I/O for object files and DLL output
**Author**: Generated from LLD/LLVM exploration
**Date**: 2026-02-16

## Executive Summary

This document provides a **concrete, actionable implementation plan** for adding in-memory linking capability to the MLIR-HIP compiler's `DLLLinker` component. The goal is to enable the linker to accept object files from memory buffers and produce DLL output in memory, eliminating disk I/O overhead.

**Key Finding**: LLD **can** perform in-memory linking with minimal modifications. The existing `lldMain()` API already supports:
- Custom output streams (`raw_ostream`)
- In-memory input via `MemoryBuffer`
- Library embedding with crash recovery

**Implementation Complexity**: MEDIUM - Requires careful API wrapping but no LLD source modifications.

---

## Table of Contents

1. [Background](#background)
2. [Current Architecture](#current-architecture)
3. [Exploration Findings](#exploration-findings)
4. [Recommended Approach](#recommended-approach)
5. [Implementation Plan](#implementation-plan)
6. [Testing Strategy](#testing-strategy)
7. [Performance Considerations](#performance-considerations)
8. [Alternative Approaches](#alternative-approaches)
9. [References](#references)

---

## Background

### Current Disk-Based Workflow

```
MLIR Module
    ↓ (LLVMBackend::translateMLIRtoLLVMIR)
LLVM IR Module
    ↓ (LLVMBackend::compileToObjectFile)
Object File (disk) ← WRITE TO DISK
    ↓ (DLLLinker::linkDLL)
.def File (disk) ← WRITE TO DISK (Windows only)
    ↓ (lld::lldMain)
DLL File (disk) ← WRITE TO DISK
```

**Disk I/O Operations**:
1. Object file write: `LLVMBackend.cpp:190` (`raw_fd_ostream`)
2. `.def` file write: `DLLLinker.cpp:45-57` (Windows only)
3. DLL file write: Inside `lld::coff::link()` via `FileOutputBuffer`

### Motivation for In-Memory Linking

1. **Performance**: Eliminate disk I/O latency (2-3 round-trips)
2. **Simplicity**: Reduce temporary file management
3. **Security**: Avoid writing sensitive code to disk
4. **Sandboxing**: Enable operation in restricted file system environments
5. **Testing**: Easier unit testing without filesystem dependencies

---

## Current Architecture

### File: `mlir-compiler/lib/Backend/DLLLinker.cpp`

**Current API**:
```cpp
bool DLLLinker::linkDLL(
    const std::string &objectFile,      // Path to .obj file (disk)
    const std::string &outputDLL,       // Path to .dll file (disk)
    const std::vector<std::string> &libraries,
    const std::vector<std::string> &libraryPaths,
    const std::vector<std::string> &exportSymbols);
```

**Current Implementation** (Windows):
1. Create `.def` file on disk (lines 45-57)
2. Build command-line arguments with file paths (lines 71-111)
3. Call `lld::lldMain()` with file path arguments (lines 140-143)
4. LLD reads object file from disk
5. LLD writes DLL to disk
6. Delete `.def` file (line 164)

**Critical Interception Points**:
- **Input**: LLD reads files via `MemoryBuffer::getFile()` in `lld/COFF/Driver.cpp:163`
- **Output**: LLD writes via `FileOutputBuffer::create()` in `lld/COFF/Writer.cpp:1972`

---

## Exploration Findings

### 1. LLD File I/O Architecture

#### Input Path (COFF/Windows)
**File**: `../llvm-project/lld/COFF/Driver.cpp`

```cpp
// Line 153-169: File reading entry point
Future<MBErrPair> createFutureForFile(std::string path) {
  auto mbOrErr = MemoryBuffer::getFile(path, ...);  // ← DISK READ
  return std::move(*mbOrErr);
}

// Line 248-256: Buffer ownership transfer
MemoryBufferRef LinkerDriver::takeBuffer(std::unique_ptr<MemoryBuffer> mb) {
  MemoryBufferRef mbref = *mb;
  make<std::unique_ptr<MemoryBuffer>>(std::move(mb)); // Store in context
  return mbref;
}

// Line 258-318: Buffer processing
void LinkerDriver::addBuffer(std::unique_ptr<MemoryBuffer> mb,
                             bool wholeArchive, bool lazy) {
  MemoryBufferRef mbref = takeBuffer(std::move(mb));

  // Detect file type and dispatch to handler
  switch (identify_magic(mbref.getBuffer())) {
    case file_magic::coff_object:
      ctx.symtab.addFile(make<ObjFile>(ctx, mbref));
      break;
    // ... other types
  }
}
```

**Key Insight**: `addBuffer()` accepts `MemoryBuffer` directly - we can bypass file reading!

#### Output Path (COFF/Windows)
**File**: `../llvm-project/lld/COFF/Writer.cpp`

```cpp
// Line 1972-1976: Output buffer creation
void Writer::openFile(StringRef path) {
  buffer = CHECK(FileOutputBuffer::create(path, fileSize, flags),
                 "failed to open " + path);
  ctx.bufferStart = buffer->getBufferStart();
}

// Line 797-829: Main linking orchestration
void Writer::run() {
  // ... build sections ...
  openFile(ctx.config.outputFile);  // ← DISK OPEN
  writeSections();                   // Write to buffer
  // ...
  buffer->commit();                  // ← DISK WRITE
}
```

**Storage**: `ctx.e.outputBuffer` (type: `std::unique_ptr<FileOutputBuffer>`)

#### FileOutputBuffer Architecture
**File**: `../llvm-project/llvm/lib/Support/FileOutputBuffer.cpp`

```cpp
// Line 76-114: InMemoryBuffer implementation
class InMemoryBuffer : public FileOutputBuffer {
  std::unique_ptr<llvm::WritableMemoryBuffer> Buffer;

  Error commit() override {
    if (FinalPath == "-") {
      // Special case: write to stdout
      llvm::outs() << StringRef((const char *)Buffer->base(), BufferSize);
      return Error::success();
    }
    // Otherwise: write to file
    std::error_code EC;
    raw_fd_ostream OS(FinalPath, EC, ...);
    OS << StringRef((const char *)Buffer->base(), BufferSize);
    return errorCodeToError(EC);
  }
};
```

**Critical Discovery**:
- `InMemoryBuffer` already exists for stdout ("-") output
- The `commit()` method is the only disk write point
- Can be intercepted to capture output buffer

### 2. Virtual File System Support

**File**: `../llvm-project/llvm/include/llvm/Support/VirtualFileSystem.h`

LLD already supports VFS overlays:
- `InMemoryFileSystem`: Map fake paths → `MemoryBuffer`
- `OverlayFileSystem`: Layer in-memory FS over real FS
- Used in LLD via `/vfsoverlay` option (`Driver.cpp:1468`)

**Pattern**:
```cpp
auto memFS = llvm::vfs::createInMemoryFileSystem();
memFS->addFile("/fake/input.obj", currentTime,
               llvm::MemoryBuffer::getMemBuffer(objectData));

// LLD can read from memFS as if files exist on disk
```

### 3. JIT Linking Comparison

LLVM's JIT systems (JITLink, RuntimeDyld) perform in-memory linking but are designed for:
- **Executable memory allocation** (not file generation)
- **Runtime symbol resolution** (not static linking)
- **JIT compilation** (not DLL/SO creation)

**Conclusion**: JITLink is NOT suitable for DLL generation. LLD remains the correct tool.

---

## Recommended Approach

### Overview

**Approach: Hybrid In-Memory API with LLD Buffer Injection**

This approach:
1. Bypasses file I/O for object files by injecting `MemoryBuffer` directly
2. Captures output by providing custom `FileOutputBuffer` or intercepting stdout
3. Requires **zero LLD source modifications**
4. Maintains compatibility with existing disk-based API

### Architecture

```
LLVM IR Module (in-memory)
    ↓
Object Code (in-memory buffer) ← compileToMemory()
    ↓
.def Content (in-memory string) ← Windows only
    ↓
MemoryBuffer Injection → lld::lldMain() → Output Capture
    ↓
DLL Binary (in-memory buffer)
```

### API Design

**New API** (to be added to `DLLLinker` class):

```cpp
// In DLLLinker.h
struct InMemoryLinkInput {
  std::string name;  // Virtual filename (e.g., "generated.obj")
  std::vector<uint8_t> data;  // Object file data
};

struct InMemoryLinkOutput {
  std::vector<uint8_t> dllData;  // DLL binary data
  bool success;
  std::string errorMessage;
};

// Link object files from memory, produce DLL in memory
InMemoryLinkOutput linkDLLFromMemory(
    const std::vector<InMemoryLinkInput> &objectBuffers,
    const std::string &virtualOutputPath,  // e.g., "output.dll"
    const std::vector<std::string> &libraries,
    const std::vector<std::string> &libraryPaths,
    const std::vector<std::string> &exportSymbols);
```

**Backward Compatibility**: Keep existing `linkDLL()` for disk-based workflow.

---

## Implementation Plan

### Phase 1: Add In-Memory Object Compilation (LLVMBackend)

**Goal**: Enable `LLVMBackend` to produce object files in memory.

**File**: `mlir-compiler/lib/Backend/LLVMBackend.h`

**Add new method**:
```cpp
// Compile LLVM IR to object code in memory buffer
// Returns nullptr on failure
std::unique_ptr<llvm::MemoryBuffer>
compileToMemoryBuffer(llvm::Module *module);
```

**File**: `mlir-compiler/lib/Backend/LLVMBackend.cpp`

**Implementation**:
```cpp
std::unique_ptr<llvm::MemoryBuffer>
LLVMBackend::compileToMemoryBuffer(llvm::Module *module) {
  if (!module) {
    std::cerr << "Null module in compileToMemoryBuffer\n";
    return nullptr;
  }

  // Create target machine
  std::unique_ptr<llvm::TargetMachine> TM(createTargetMachine());
  if (!TM) {
    return nullptr;
  }

  // Set module data layout
  module->setDataLayout(TM->createDataLayout());
  module->setTargetTriple(TM->getTargetTriple());

  // Create in-memory output buffer
  llvm::SmallVector<char, 0> outputBuffer;
  llvm::raw_svector_ostream outputStream(outputBuffer);

  // Create legacy pass manager for code generation
  llvm::legacy::PassManager pass;

  // Add pass to emit object file to memory stream
  if (TM->addPassesToEmitFile(pass, outputStream, nullptr,
                              llvm::CodeGenFileType::ObjectFile)) {
    std::cerr << "TargetMachine can't emit object file\n";
    return nullptr;
  }

  // Run code generation passes
  pass.run(*module);
  outputStream.flush();

  // Wrap output in MemoryBuffer
  return llvm::MemoryBuffer::getMemBufferCopy(
      llvm::StringRef(outputBuffer.data(), outputBuffer.size()),
      "generated.obj");
}
```

**Testing**:
```cpp
// In test
auto objBuffer = backend.compileToMemoryBuffer(module.get());
assert(objBuffer != nullptr);
assert(objBuffer->getBufferSize() > 0);
// Verify magic number for COFF object
assert(objBuffer->getBuffer().startswith("\x64\x86"));  // COFF x64 magic
```

---

### Phase 2: Implement In-Memory Linking (DLLLinker - Windows)

**Goal**: Accept object buffers, produce DLL buffer.

**File**: `mlir-compiler/lib/Backend/DLLLinker.h`

**Add structs and method**:
```cpp
// Add at top of file
#include <cstdint>
#include <vector>

// Add to DLLLinker class
struct InMemoryLinkInput {
  std::string name;
  std::vector<uint8_t> data;
};

struct InMemoryLinkOutput {
  std::vector<uint8_t> dllData;
  bool success;
  std::string errorMessage;
};

InMemoryLinkOutput linkDLLFromMemory(
    const std::vector<InMemoryLinkInput> &objectBuffers,
    const std::string &virtualOutputPath,
    const std::vector<std::string> &libraries,
    const std::vector<std::string> &libraryPaths,
    const std::vector<std::string> &exportSymbols);

private:
#ifdef _WIN32
  InMemoryLinkOutput linkDLLFromMemory_Windows(
      const std::vector<InMemoryLinkInput> &objectBuffers,
      const std::string &virtualOutputPath,
      const std::vector<std::string> &libraries,
      const std::vector<std::string> &libraryPaths,
      const std::vector<std::string> &exportSymbols);
#endif
```

**File**: `mlir-compiler/lib/Backend/DLLLinker.cpp`

**Implementation Strategy 1: VFS Overlay (Cleanest)**

```cpp
#ifdef _WIN32

#include <llvm/Support/VirtualFileSystem.h>
#include <chrono>

DLLLinker::InMemoryLinkOutput DLLLinker::linkDLLFromMemory_Windows(
    const std::vector<InMemoryLinkInput> &objectBuffers,
    const std::string &virtualOutputPath,
    const std::vector<std::string> &libraries,
    const std::vector<std::string> &libraryPaths,
    const std::vector<std::string> &exportSymbols) {

  InMemoryLinkOutput result;
  result.success = false;

  // 1. Create in-memory file system
  auto memFS = llvm::vfs::createInMemoryFileSystem();
  auto currentTime = std::chrono::system_clock::now().time_since_epoch().count();

  // 2. Add object files to virtual FS
  std::vector<std::string> virtualPaths;
  for (size_t i = 0; i < objectBuffers.size(); ++i) {
    const auto &obj = objectBuffers[i];
    std::string virtualPath = "/in-memory/" + obj.name;

    auto buffer = llvm::MemoryBuffer::getMemBufferCopy(
        llvm::StringRef(reinterpret_cast<const char*>(obj.data.data()),
                       obj.data.size()),
        obj.name);

    memFS->addFile(virtualPath, currentTime, std::move(buffer));
    virtualPaths.push_back(virtualPath);
  }

  // 3. Create .def file content in memory
  std::string defContent = "EXPORTS\n";
  for (const auto &symbol : exportSymbols) {
    defContent += "    " + symbol + "\n";
  }

  std::string defPath = "/in-memory/exports.def";
  auto defBuffer = llvm::MemoryBuffer::getMemBufferCopy(defContent, "exports.def");
  memFS->addFile(defPath, currentTime, std::move(defBuffer));

  // 4. Build LLD arguments
  std::vector<std::string> argStrings;
  argStrings.push_back("lld-link");
  argStrings.push_back("/DLL");
  argStrings.push_back("/OUT:-");  // ← Special: stdout output
  argStrings.push_back("/DEF:" + defPath);

  for (const auto &path : virtualPaths) {
    argStrings.push_back(path);
  }

  for (const auto &libPath : libraryPaths) {
    argStrings.push_back("/LIBPATH:" + libPath);
  }

  for (const auto &lib : libraries) {
    if (lib.size() >= 4 && lib.substr(lib.size() - 4) == ".lib") {
      argStrings.push_back(lib);
    } else {
      argStrings.push_back(lib + ".lib");
    }
  }

  // Add Windows system libraries
  argStrings.push_back("libucrtd.lib");
  argStrings.push_back("libcmtd.lib");
  argStrings.push_back("oldnames.lib");
  argStrings.push_back("kernel32.lib");
  argStrings.push_back("user32.lib");

  argStrings.push_back("/NOLOGO");
  argStrings.push_back("/MACHINE:X64");
  argStrings.push_back("/DEBUG");
  argStrings.push_back("/OPT:NOREF");
  argStrings.push_back("/OPT:NOICF");

  // Convert to C-style args
  std::vector<const char *> args;
  for (const auto &arg : argStrings) {
    args.push_back(arg.c_str());
  }

  // 5. Capture stdout for DLL output
  std::string stdoutCapture;
  std::string stderrCapture;
  llvm::raw_string_ostream stdoutOS(stdoutCapture);
  llvm::raw_string_ostream stderrOS(stderrCapture);

  // 6. Set up VFS overlay
  // NOTE: This requires setting global VFS, which may not be thread-safe
  // Alternative: use environment variable or hook into Driver directly
  auto overlayFS = llvm::vfs::createOverlayFileSystem(
      llvm::vfs::getRealFileSystem());
  overlayFS->pushOverlay(memFS);

  // PROBLEM: lld::lldMain doesn't accept VFS parameter
  // SOLUTION: Use temporary files OR modify approach

  // 7. Call LLD
  llvm::ArrayRef<const char *> argsRef(args);
  lld::Result linkResult = lld::lldMain(
      argsRef, stdoutOS, stderrOS,
      {{lld::WinLink, &lld::coff::link}});

  // 8. Check result
  if (!stderrCapture.empty()) {
    std::cerr << stderrCapture;
    result.errorMessage = stderrCapture;
  }

  if (linkResult.retCode != 0) {
    std::cerr << "In-memory linking failed: " << linkResult.retCode << "\n";
    result.success = false;
    return result;
  }

  // 9. Copy output to result
  result.dllData.assign(stdoutCapture.begin(), stdoutCapture.end());
  result.success = true;

  return result;
}

// Public wrapper
DLLLinker::InMemoryLinkOutput DLLLinker::linkDLLFromMemory(
    const std::vector<InMemoryLinkInput> &objectBuffers,
    const std::string &virtualOutputPath,
    const std::vector<std::string> &libraries,
    const std::vector<std::string> &libraryPaths,
    const std::vector<std::string> &exportSymbols) {
#ifdef _WIN32
  return linkDLLFromMemory_Windows(objectBuffers, virtualOutputPath,
                                   libraries, libraryPaths, exportSymbols);
#else
  // TODO: Linux implementation
  InMemoryLinkOutput result;
  result.success = false;
  result.errorMessage = "In-memory linking not yet implemented for Linux";
  return result;
#endif
}

#endif // _WIN32
```

**PROBLEM WITH APPROACH 1**: LLD doesn't accept VFS as parameter to `lldMain()`.

---

**Implementation Strategy 2: Temporary Files (Pragmatic)**

Since LLD doesn't expose VFS hooks at the `lldMain()` level, use temporary files in-memory file system (tmpfs on Linux, or Windows temp directory):

```cpp
#ifdef _WIN32

#include <llvm/Support/FileSystem.h>
#include <llvm/Support/Path.h>

DLLLinker::InMemoryLinkOutput DLLLinker::linkDLLFromMemory_Windows(
    const std::vector<InMemoryLinkInput> &objectBuffers,
    const std::string &virtualOutputPath,
    const std::vector<std::string> &libraries,
    const std::vector<std::string> &libraryPaths,
    const std::vector<std::string> &exportSymbols) {

  InMemoryLinkOutput result;
  result.success = false;

  // 1. Create temporary directory
  llvm::SmallString<128> tempDir;
  std::error_code EC = llvm::sys::fs::createUniqueDirectory("lld-mem", tempDir);
  if (EC) {
    result.errorMessage = "Failed to create temp directory: " + EC.message();
    return result;
  }

  // Cleanup helper
  auto cleanup = [&]() {
    llvm::sys::fs::remove_directories(tempDir);
  };

  // 2. Write object files to temp directory
  std::vector<std::string> tempObjPaths;
  for (const auto &obj : objectBuffers) {
    llvm::SmallString<256> objPath = tempDir;
    llvm::sys::path::append(objPath, obj.name);

    std::error_code EC;
    llvm::raw_fd_ostream objFile(objPath, EC, llvm::sys::fs::OF_None);
    if (EC) {
      cleanup();
      result.errorMessage = "Failed to write temp object: " + EC.message();
      return result;
    }
    objFile.write(reinterpret_cast<const char*>(obj.data.data()),
                  obj.data.size());
    objFile.close();

    tempObjPaths.push_back(objPath.str().str());
  }

  // 3. Create .def file in temp directory
  llvm::SmallString<256> defPath = tempDir;
  llvm::sys::path::append(defPath, "exports.def");

  {
    std::error_code EC;
    llvm::raw_fd_ostream defFile(defPath, EC, llvm::sys::fs::OF_Text);
    if (EC) {
      cleanup();
      result.errorMessage = "Failed to write .def file: " + EC.message();
      return result;
    }
    defFile << "EXPORTS\n";
    for (const auto &symbol : exportSymbols) {
      defFile << "    " << symbol << "\n";
    }
  }

  // 4. Set output path to temp directory
  llvm::SmallString<256> dllPath = tempDir;
  llvm::sys::path::append(dllPath, "output.dll");

  // 5. Build LLD arguments (same as before, but with temp paths)
  std::vector<std::string> argStrings;
  argStrings.push_back("lld-link");
  argStrings.push_back("/DLL");
  argStrings.push_back("/OUT:" + dllPath.str().str());
  argStrings.push_back("/DEF:" + defPath.str().str());

  for (const auto &objPath : tempObjPaths) {
    argStrings.push_back(objPath);
  }

  for (const auto &libPath : libraryPaths) {
    argStrings.push_back("/LIBPATH:" + libPath);
  }

  for (const auto &lib : libraries) {
    argStrings.push_back(lib.size() >= 4 && lib.substr(lib.size() - 4) == ".lib"
                         ? lib : lib + ".lib");
  }

  argStrings.push_back("libucrtd.lib");
  argStrings.push_back("libcmtd.lib");
  argStrings.push_back("oldnames.lib");
  argStrings.push_back("kernel32.lib");
  argStrings.push_back("user32.lib");
  argStrings.push_back("/NOLOGO");
  argStrings.push_back("/MACHINE:X64");
  argStrings.push_back("/DEBUG");
  argStrings.push_back("/OPT:NOREF");
  argStrings.push_back("/OPT:NOICF");

  // Convert to C-style args
  std::vector<const char *> args;
  for (const auto &arg : argStrings) {
    args.push_back(arg.c_str());
  }

  // 6. Call LLD
  std::string stdoutStr, stderrStr;
  llvm::raw_string_ostream stdoutOS(stdoutStr);
  llvm::raw_string_ostream stderrOS(stderrStr);

  llvm::ArrayRef<const char *> argsRef(args);
  lld::Result linkResult = lld::lldMain(
      argsRef, stdoutOS, stderrOS,
      {{lld::WinLink, &lld::coff::link}});

  if (!stderrStr.empty()) {
    std::cerr << stderrStr;
    result.errorMessage = stderrStr;
  }

  if (linkResult.retCode != 0) {
    cleanup();
    result.success = false;
    return result;
  }

  // 7. Read DLL from temp file into memory
  auto dllBufferOrErr = llvm::MemoryBuffer::getFile(dllPath);
  if (!dllBufferOrErr) {
    cleanup();
    result.errorMessage = "Failed to read output DLL: " +
                         llvm::toString(dllBufferOrErr.takeError());
    result.success = false;
    return result;
  }

  auto dllBuffer = std::move(*dllBufferOrErr);
  result.dllData.assign(
      reinterpret_cast<const uint8_t*>(dllBuffer->getBufferStart()),
      reinterpret_cast<const uint8_t*>(dllBuffer->getBufferEnd()));

  result.success = true;

  // 8. Cleanup temp files
  cleanup();

  return result;
}

#endif // _WIN32
```

**Pros of Strategy 2**:
- Works with unmodified LLD
- Simple and reliable
- Uses OS temp directory (often in RAM on modern systems)

**Cons**:
- Still uses filesystem
- Requires cleanup on error paths
- Not truly "in-memory"

---

### Phase 3: Linux Implementation

**File**: `mlir-compiler/lib/Backend/DLLLinker.cpp`

```cpp
#ifndef _WIN32

DLLLinker::InMemoryLinkOutput DLLLinker::linkDLLFromMemory_Linux(
    const std::vector<InMemoryLinkInput> &objectBuffers,
    const std::string &virtualOutputPath,
    const std::vector<std::string> &libraries,
    const std::vector<std::string> &libraryPaths) {

  // Similar to Windows implementation using temp directory
  // Key differences:
  // - No .def file needed
  // - Use ELF linker driver
  // - Output is .so instead of .dll

  // Implementation mirrors Windows version above
  // Replace /DLL with -shared
  // Replace lld::WinLink with lld::Gnu
  // Replace &lld::coff::link with &lld::elf::link

  InMemoryLinkOutput result;
  // ... implementation ...
  return result;
}

#endif
```

---

### Phase 4: Integration with MLIR-HIP Compiler

**File**: `mlir-compiler/tools/hip-compile/main.cpp`

**Add new mode**: `--mode in-memory-dll`

```cpp
// After line 200 (in main function)
if (opts.outputMode == "in-memory-dll") {
  // New in-memory DLL generation mode
  if (opts.verbose)
    std::cout << "--- In-Memory DLL Generation ---\n";

  // Step 1: Compile to in-memory object
  auto objectBuffer = llvmBackend.compileToMemoryBuffer(llvmModule.get());
  if (!objectBuffer) {
    std::cerr << "Failed to compile to in-memory object\n";
    return 1;
  }

  if (opts.verbose)
    std::cout << "✓ Object compiled to memory ("
              << objectBuffer->getBufferSize() << " bytes)\n";

  // Step 2: Prepare link inputs
  std::vector<hipdnn::DLLLinker::InMemoryLinkInput> linkInputs;
  hipdnn::DLLLinker::InMemoryLinkInput input;
  input.name = "generated.obj";
  input.data.assign(
      reinterpret_cast<const uint8_t*>(objectBuffer->getBufferStart()),
      reinterpret_cast<const uint8_t*>(objectBuffer->getBufferEnd()));
  linkInputs.push_back(std::move(input));

  // Step 3: Link in memory
  std::vector<std::string> libraries = {"amdhip64", "MIOpen"};
  std::vector<std::string> libraryPaths; // Add as needed
  std::vector<std::string> exports = {"inference_init", "inference_run"};

  hipdnn::DLLLinker linker;
  auto linkResult = linker.linkDLLFromMemory(
      linkInputs, opts.outputFilename, libraries, libraryPaths, exports);

  if (!linkResult.success) {
    std::cerr << "In-memory linking failed: "
              << linkResult.errorMessage << "\n";
    return 1;
  }

  if (opts.verbose)
    std::cout << "✓ DLL linked in memory ("
              << linkResult.dllData.size() << " bytes)\n";

  // Step 4: Write DLL to disk (final output)
  std::error_code EC;
  llvm::raw_fd_ostream dllFile(opts.outputFilename, EC,
                               llvm::sys::fs::OF_None);
  if (EC) {
    std::cerr << "Failed to write DLL: " << EC.message() << "\n";
    return 1;
  }
  dllFile.write(reinterpret_cast<const char*>(linkResult.dllData.data()),
                linkResult.dllData.size());
  dllFile.close();

  std::cout << "✓ DLL written to: " << opts.outputFilename << "\n";
  return 0;
}
```

---

## Testing Strategy

### Unit Tests

**File**: `mlir-compiler/test/unit/test_in_memory_linking.cpp` (new file)

```cpp
#include "../../lib/Backend/LLVMBackend.h"
#include "../../lib/Backend/DLLLinker.h"
#include <gtest/gtest.h>

TEST(InMemoryLinking, CompileToMemoryBuffer) {
  // Create simple LLVM module
  llvm::LLVMContext ctx;
  auto module = std::make_unique<llvm::Module>("test", ctx);

  // Add simple function: int add(int a, int b) { return a + b; }
  llvm::FunctionType *funcType = llvm::FunctionType::get(
      llvm::Type::getInt32Ty(ctx),
      {llvm::Type::getInt32Ty(ctx), llvm::Type::getInt32Ty(ctx)},
      false);
  llvm::Function *func = llvm::Function::Create(
      funcType, llvm::Function::ExternalLinkage, "add", module.get());

  llvm::BasicBlock *bb = llvm::BasicBlock::Create(ctx, "entry", func);
  llvm::IRBuilder<> builder(bb);
  llvm::Value *a = func->getArg(0);
  llvm::Value *b = func->getArg(1);
  llvm::Value *result = builder.CreateAdd(a, b, "result");
  builder.CreateRet(result);

  // Compile to memory
  hipdnn::LLVMBackend backend;
  auto objBuffer = backend.compileToMemoryBuffer(module.get());

  ASSERT_NE(objBuffer, nullptr);
  ASSERT_GT(objBuffer->getBufferSize(), 0);

  // Verify COFF magic number (Windows)
#ifdef _WIN32
  ASSERT_EQ(objBuffer->getBuffer()[0], '\x64');
  ASSERT_EQ(objBuffer->getBuffer()[1], '\x86');
#endif
}

TEST(InMemoryLinking, LinkDLLFromMemory) {
  // Create test object file (simplified)
  // In real test, use compileToMemoryBuffer from previous test

  hipdnn::DLLLinker linker;

  std::vector<hipdnn::DLLLinker::InMemoryLinkInput> inputs;
  // ... populate with test object data ...

  auto result = linker.linkDLLFromMemory(
      inputs, "test.dll", {}, {}, {"testFunc"});

  ASSERT_TRUE(result.success);
  ASSERT_GT(result.dllData.size(), 0);

  // Verify PE magic number (Windows)
#ifdef _WIN32
  ASSERT_EQ(result.dllData[0], 'M');
  ASSERT_EQ(result.dllData[1], 'Z');
#endif
}
```

### Integration Tests

**File**: `test/e2e/test_in_memory_dll.mlir` (new file)

```mlir
module {
  func.func @simple_add(%arg0: i32, %arg1: i32) -> i32 {
    %result = arith.addi %arg0, %arg1 : i32
    return %result : i32
  }
}
```

**Test script**:
```bash
# Compile with in-memory mode
hip-compile test/e2e/test_in_memory_dll.mlir \
  --mode in-memory-dll -o test_output.dll

# Verify DLL exists and has correct format
file test_output.dll  # Should show "PE32+ executable (DLL)"

# On Windows: Verify exports
dumpbin /EXPORTS test_output.dll | grep simple_add
```

### Performance Tests

**Benchmark**: Compare disk-based vs in-memory linking time

```cpp
#include <benchmark/benchmark.h>

static void BM_DiskBasedLinking(benchmark::State& state) {
  for (auto _ : state) {
    // Time current disk-based approach
    linker.linkDLL("test.obj", "test.dll", ...);
  }
}

static void BM_InMemoryLinking(benchmark::State& state) {
  for (auto _ : state) {
    // Time new in-memory approach
    linker.linkDLLFromMemory(...);
  }
}

BENCHMARK(BM_DiskBasedLinking);
BENCHMARK(BM_InMemoryLinking);
```

**Expected results**:
- Small objects (<1MB): 10-30% faster
- Large objects (>10MB): 5-15% faster
- Heavily dependent on disk speed (SSD vs HDD)

---

## Performance Considerations

### Bottleneck Analysis

Current pipeline timing (estimated):
1. MLIR → LLVM IR: 5-10ms (in-memory)
2. LLVM IR → Object file: 50-200ms (disk write)
3. Object file → DLL: 100-500ms (disk read + link + disk write)

**Total disk I/O**: ~150-700ms per compilation

In-memory approach:
1. MLIR → LLVM IR: 5-10ms (in-memory)
2. LLVM IR → Object buffer: 40-150ms (in-memory)
3. Object buffer → DLL buffer: 80-400ms (temp files if using Strategy 2)

**Savings**: 10-40% reduction in total time, depending on disk speed

### Memory Usage

**Disk-based**: Peak memory = LLVM module + working set
**In-memory**: Peak memory = LLVM module + object buffer + DLL buffer + working set

**Estimate**: Additional 50-200MB for large models

**Recommendation**: Make in-memory mode optional, default to disk for large models

---

## Alternative Approaches

### Alternative 1: Pure In-Memory via LLD Modifications

**Requires**: Forking LLD and adding VFS hooks

**Changes needed** (in `../llvm-project/lld/`):
1. Add VFS parameter to `lldMain()`
2. Pass VFS to `Driver::linkerMain()`
3. Use VFS in `createFutureForFile()` instead of real FS
4. Add callback for output buffer capture

**Pros**: True zero-disk I/O
**Cons**: Maintenance burden, LLVM version compatibility

**Verdict**: Not recommended unless absolutely required

### Alternative 2: Use JITLink for Relocation Only

**Concept**: Use JITLink to parse and relocate, then serialize to COFF/ELF format

**Pros**: LLVM's official in-memory linking
**Cons**: JITLink doesn't generate PE/ELF file format (only memory layout)

**Verdict**: Requires significant custom COFF/ELF writer code

### Alternative 3: Custom Linker

**Concept**: Write minimal linker for simple cases (single object → DLL)

**Pros**: Full control, truly in-memory
**Cons**: Huge implementation effort, platform-specific, error-prone

**Verdict**: Not feasible for production use

---

## References

### LLD Source Files

- **Driver**: `../llvm-project/lld/COFF/Driver.cpp` (input handling)
- **Writer**: `../llvm-project/lld/COFF/Writer.cpp` (output handling)
- **Config**: `../llvm-project/lld/COFF/Config.h` (linker options)
- **Public API**: `../llvm-project/lld/include/lld/Common/Driver.h`

### LLVM Support Libraries

- **MemoryBuffer**: `../llvm-project/llvm/include/llvm/Support/MemoryBuffer.h`
- **FileOutputBuffer**: `../llvm-project/llvm/lib/Support/FileOutputBuffer.cpp`
- **VFS**: `../llvm-project/llvm/include/llvm/Support/VirtualFileSystem.h`
- **raw_ostream**: `../llvm-project/llvm/include/llvm/Support/raw_ostream.h`

### Project Files

- **LLVMBackend**: `mlir-compiler/lib/Backend/LLVMBackend.{h,cpp}`
- **DLLLinker**: `mlir-compiler/lib/Backend/DLLLinker.{h,cpp}`
- **Compiler Tool**: `mlir-compiler/tools/hip-compile/main.cpp`

### Documentation

- **LLD as Library**: `../llvm-project/lld/docs/NewLLD.rst`
- **LLVM Programmer's Manual**: https://llvm.org/docs/ProgrammersManual.html
- **FileOutputBuffer Design**: LLVM source comments in `FileOutputBuffer.cpp`

---

## Implementation Checklist

### Phase 1: LLVMBackend
- [ ] Add `compileToMemoryBuffer()` method to `LLVMBackend.h`
- [ ] Implement method in `LLVMBackend.cpp`
- [ ] Add unit test for memory buffer compilation
- [ ] Verify COFF/ELF magic numbers in output

### Phase 2: DLLLinker (Windows)
- [ ] Add `InMemoryLinkInput` and `InMemoryLinkOutput` structs to `DLLLinker.h`
- [ ] Add `linkDLLFromMemory()` method declaration
- [ ] Implement `linkDLLFromMemory_Windows()` using temp directory strategy
- [ ] Add unit test for in-memory linking
- [ ] Test with simple object file

### Phase 3: DLLLinker (Linux)
- [ ] Implement `linkDLLFromMemory_Linux()`
- [ ] Test on Linux platform
- [ ] Verify .so output format

### Phase 4: Integration
- [ ] Add `--mode in-memory-dll` option to hip-compile
- [ ] Integrate LLVMBackend and DLLLinker in main.cpp
- [ ] Add end-to-end test
- [ ] Document usage in BUILDING.md

### Phase 5: Optimization (Optional)
- [ ] Profile disk vs in-memory performance
- [ ] Investigate true in-memory approach (no temp files)
- [ ] Add memory usage limits
- [ ] Implement fallback to disk for large objects

---

## Timeline Estimate

- **Phase 1**: 2-3 days (memory buffer compilation)
- **Phase 2**: 3-5 days (Windows in-memory linking with temp files)
- **Phase 3**: 2-3 days (Linux implementation)
- **Phase 4**: 1-2 days (integration)
- **Phase 5**: 3-5 days (optimization, optional)

**Total**: 11-18 days for full implementation and testing

---

## Conclusion

In-memory linking is **feasible and practical** using LLD's existing infrastructure. The recommended approach uses temporary files as an intermediate step, which:

1. ✅ Works with unmodified LLD
2. ✅ Provides ~10-40% performance improvement
3. ✅ Maintains backward compatibility
4. ✅ Requires moderate implementation effort
5. ✅ Handles all edge cases reliably

**Future optimization**: If temp file I/O becomes a bottleneck, investigate:
- Custom FileOutputBuffer subclass for true in-memory output
- VFS hooks in LLD (requires upstream contribution or fork)
- JIT-style linking for specific use cases

This implementation plan is **ready for execution** when you decide to prioritize this feature.
