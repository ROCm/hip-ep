//===- DLLLinker.cpp - Native DLL linker for the HIP compiler - *- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.  All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
#include "hip/Target/LLVM/DLLLinker.h"

#include "hip/debug_log.h"

#include <llvm/Support/CommandLine.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/Path.h>
#include <llvm/Support/raw_ostream.h>

#include <fstream>
#include <sstream>

// LLD linker driver - use lldMain for crash recovery
#include "lld/Common/Driver.h"

// Still need to declare the link functions for driver registration
LLD_HAS_DRIVER(coff)
LLD_HAS_DRIVER(elf)

namespace hipdnn {

DLLLinker::DLLLinker() = default;
DLLLinker::~DLLLinker() = default;

bool DLLLinker::linkDLL(const std::string& objectFile,
                        const std::string& outputDLL,
                        const std::vector<std::string>& libraries,
                        const std::vector<std::string>& libraryPaths,
                        const std::vector<std::string>& exportSymbols) {
#ifdef _WIN32
  return linkDLL_Windows(objectFile, outputDLL, libraries, libraryPaths,
                         exportSymbols);
#else
  return linkDLL_Linux(objectFile, outputDLL, libraries, libraryPaths);
#endif
}

bool DLLLinker::linkDLLInMemory(const std::vector<uint8_t>& objectBytes,
                                std::vector<uint8_t>& outDLLBytes,
                                const std::vector<std::string>& libraries,
                                const std::vector<std::string>& libraryPaths,
                                const std::vector<std::string>& exportSymbols) {
  // LLD does not provide a pure in-memory linking API
  // Strategy: Use temporary files for linking, then read result into memory
  // This approach works for MVP; can be improved with MemoryModule later

  // Create unique temporary files atomically (avoids TOCTOU race of tmpnam)
  llvm::SmallString<128> objFile, dllFile;
  if (auto EC =
          llvm::sys::fs::createTemporaryFile("hip-link", "obj", objFile)) {
    llvm::errs() << "Failed to create temporary object file: " << EC.message()
                 << "\n";
    return false;
  }
  if (auto EC =
          llvm::sys::fs::createTemporaryFile("hip-link", "dll", dllFile)) {
    llvm::sys::fs::remove(objFile);
    llvm::errs() << "Failed to create temporary DLL file: " << EC.message()
                 << "\n";
    return false;
  }

  // Write object bytes to temporary file
  {
    std::error_code EC;
    llvm::raw_fd_ostream objOut(objFile, EC, llvm::sys::fs::OF_None);
    if (EC) {
      llvm::errs() << "Failed to open temporary object file: " << EC.message()
                   << "\n";
      llvm::sys::fs::remove(objFile);
      llvm::sys::fs::remove(dllFile);
      return false;
    }
    objOut.write(reinterpret_cast<const char*>(objectBytes.data()),
                 objectBytes.size());
  }

  // Link using existing file-based API
  std::string objStr(objFile), dllStr(dllFile);
  bool linkSuccess =
      linkDLL(objStr, dllStr, libraries, libraryPaths, exportSymbols);

  if (!linkSuccess) {
    llvm::sys::fs::remove(objFile);
    llvm::sys::fs::remove(dllFile);
    return false;
  }

  // Read DLL into memory
  {
    auto bufOrErr = llvm::MemoryBuffer::getFile(dllFile);
    if (!bufOrErr) {
      llvm::errs() << "Failed to read temporary DLL file: "
                   << bufOrErr.getError().message() << "\n";
      llvm::sys::fs::remove(objFile);
      llvm::sys::fs::remove(dllFile);
      return false;
    }
    auto& buf = *bufOrErr;
    const auto* data = reinterpret_cast<const uint8_t*>(buf->getBufferStart());
    outDLLBytes.assign(data, data + buf->getBufferSize());
  }

  // Cleanup temporary files
  llvm::sys::fs::remove(objFile);
  llvm::sys::fs::remove(dllFile);

  COMPILER_DEBUG_LOG("Linked DLL in memory: " << outDLLBytes.size()
                                              << " bytes\n");
  return true;
}

#ifdef _WIN32

bool DLLLinker::createModuleDefinitionFile(
    const std::string& defPath, const std::vector<std::string>& exportSymbols) {
  std::ofstream defFile(defPath);
  if (!defFile) {
    llvm::errs() << "Failed to create .def file: " << defPath << "\n";
    return false;
  }

  defFile << "EXPORTS\n";
  for (const auto& symbol : exportSymbols) {
    defFile << "    " << symbol << "\n";
  }

  defFile.close();
  return true;
}

bool DLLLinker::linkDLL_Windows(const std::string& objectFile,
                                const std::string& outputDLL,
                                const std::vector<std::string>& libraries,
                                const std::vector<std::string>& libraryPaths,
                                const std::vector<std::string>& exportSymbols) {
  // Create temporary .def file for exports
  std::string defFile = objectFile + ".def";
  if (!createModuleDefinitionFile(defFile, exportSymbols)) {
    return false;
  }

  // Build LLD-LINK command line arguments
  // Note: lldMain() requires argv[0] to be the program name
  std::vector<std::string> argStrings;
  argStrings.push_back("lld-link"); // argv[0] - program name
  argStrings.push_back("/DLL");     // Create DLL
  argStrings.push_back(
      "/ignore:4099"); // Suppress missing PDB warnings (LNK4099)
  argStrings.push_back("/OUT:" + outputDLL);
  argStrings.push_back("/DEF:" + defFile);
  argStrings.push_back(objectFile);

  // Add library paths
  for (const auto& libPath : libraryPaths) {
    argStrings.push_back("/LIBPATH:" + libPath);
  }

  // Add libraries
  for (const auto& lib : libraries) {
    if (lib.find('/') != std::string::npos ||
        lib.find('\\') != std::string::npos) {
      argStrings.push_back(lib); // full path — pass as-is
    } else if (lib.size() >= 4 && lib.substr(lib.size() - 4) == ".lib") {
      argStrings.push_back(lib);
    } else {
      argStrings.push_back(lib + ".lib");
    }
  }

  // Suppress all /defaultlib directives embedded in the .obj by the compiler
  // (e.g. libcmtd, vcruntimed from debug-mode LLVM codegen) and supply only
  // the release CRT import libraries explicitly.  This ensures the JIT-compiled
  // DLL loads on machines without VS installed (debug CRT DLLs such as
  // ucrtbased.dll and VCRUNTIME140D.dll are not redistributable).
  argStrings.push_back("/NODEFAULTLIB");
  for (const char* sysLib :
       {"msvcrt.lib", "ucrt.lib", "vcruntime.lib", "oldnames.lib",
        "libcpmt.lib", "libcmt.lib", "kernel32.lib", "user32.lib"})
    argStrings.push_back(sysLib);

  // Add default libraries and flags
  argStrings.push_back("/NOLOGO");
  argStrings.push_back("/MACHINE:X64");

  // Add debug flags to prevent optimization and get clear backtraces
  argStrings.push_back("/DEBUG");     // Generate debug info (.pdb)
  argStrings.push_back("/OPT:NOREF"); // Don't remove unreferenced code
  argStrings.push_back("/OPT:NOICF"); // Don't fold identical functions

  // Convert to C-style args for LLD
  std::vector<const char*> args;
  for (const auto& arg : argStrings) {
    args.push_back(arg.c_str());
  }

  COMPILER_DEBUG_LOG("LLD-LINK command (" << args.size() << " args):\n");
  for (size_t i = 0; i < args.size(); ++i) {
    COMPILER_DEBUG_LOG("  [" << i << "]='" << args[i] << "'\n");
  }

  // Call LLD linker library
  std::string stdoutStr, stderrStr;
  llvm::raw_string_ostream stdoutOS(stdoutStr);
  llvm::raw_string_ostream stderrOS(stderrStr);

  llvm::ArrayRef<const char*> argsRef(args);

  // Use lldMain for crash recovery instead of direct link() call
  // lldMain provides:
  // - CrashRecoveryContext for handling fatal() calls
  // - Proper cleanup via CommonLinkerContext::destroy()
  // - Safe for re-entry
  lld::Result result =
      lld::lldMain(argsRef, stdoutOS, stderrOS,
                   {{lld::WinLink, &lld::coff::link}} // Register COFF driver
      );

  if (!stdoutStr.empty()) {
    COMPILER_DEBUG_LOG(stdoutStr);
  }
  if (!stderrStr.empty()) {
    llvm::errs() << stderrStr;
  }

  if (result.retCode != 0) {
    llvm::errs() << "LLD-LINK failed with exit code: " << result.retCode
                 << "\n";
    if (!result.canRunAgain) {
      llvm::errs() << "  Warning: Linker crashed, cannot run again\n";
    }
    return false;
  }

  COMPILER_DEBUG_LOG("Successfully linked DLL: " << outputDLL << "\n");

  llvm::sys::fs::remove(defFile);

  return true;
}

#else // Linux

bool DLLLinker::linkDLL_Linux(const std::string& objectFile,
                              const std::string& outputDLL,
                              const std::vector<std::string>& libraries,
                              const std::vector<std::string>& libraryPaths) {
  // Build LLD-ELF command line arguments for shared library
  std::vector<std::string> argStrings;
  argStrings.push_back("ld.lld");  // Program name (required by LLD)
  argStrings.push_back("-shared"); // Create shared library
  argStrings.push_back("-o");
  argStrings.push_back(outputDLL);
  argStrings.push_back(objectFile);

  // Add library paths
  for (const auto& libPath : libraryPaths) {
    argStrings.push_back("-L" + libPath);
  }

  // Add libraries
  for (const auto& lib : libraries) {
    argStrings.push_back("-l" + lib);
  }

  // Add RPATH for runtime library search
  for (const auto& libPath : libraryPaths) {
    argStrings.push_back("-rpath");
    argStrings.push_back(libPath);
  }

  // Add default flags
  argStrings.push_back("--export-dynamic"); // Export all symbols by default
  argStrings.push_back("--no-undefined");   // Error on undefined symbols

  // Convert to C-style args for LLD
  std::vector<const char*> args;
  for (const auto& arg : argStrings) {
    args.push_back(arg.c_str());
  }

  // Call LLD linker library
  std::string stdoutStr, stderrStr;
  llvm::raw_string_ostream stdoutOS(stdoutStr);
  llvm::raw_string_ostream stderrOS(stderrStr);

  // Use lldMain for crash recovery instead of direct link() call
  lld::Result result =
      lld::lldMain(args, stdoutOS, stderrOS, {{lld::Gnu, &lld::elf::link}}
                   // Register ELF driver
      );

  if (!stdoutStr.empty()) {
    COMPILER_DEBUG_LOG(stdoutStr);
  }
  if (!stderrStr.empty()) {
    llvm::errs() << stderrStr;
  }

  if (result.retCode != 0) {
    llvm::errs() << "LLD-ELF failed with exit code: " << result.retCode << "\n";
    if (!result.canRunAgain) {
      llvm::errs() << "  Warning: Linker crashed, cannot run again\n";
    }
    return false;
  }

  COMPILER_DEBUG_LOG("Successfully linked shared library: " << outputDLL
                                                            << "\n");
  return true;
}

#endif

bool DLLLinker::verifyDLLExports(
    const std::string& dllPath,
    const std::vector<std::string>& requiredSymbols) {
  // This is a simplified implementation
  // A complete implementation would use LLVM's object file libraries
  // to parse the DLL/SO and verify exported symbols

  COMPILER_DEBUG_LOG("Verifying DLL exports for: " << dllPath << "\n");
  for (const auto& symbol : requiredSymbols) {
    COMPILER_DEBUG_LOG("  Required symbol: " << symbol << "\n");
  }

  if (!llvm::sys::fs::exists(dllPath)) {
    llvm::errs() << "DLL file does not exist: " << dllPath << "\n";
    return false;
  }

  return true;
}

} // namespace hipdnn
