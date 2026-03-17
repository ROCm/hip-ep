/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "hip/Target/LLVM/DLLLinker.h"

#include <llvm/Support/CommandLine.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/Path.h>
#include <llvm/Support/raw_ostream.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>

// LLD linker driver - use lldMain for crash recovery
#include "lld/Common/Driver.h"

// Still need to declare the link functions for driver registration
LLD_HAS_DRIVER(coff)
LLD_HAS_DRIVER(elf)

namespace hipdnn {

DLLLinker::DLLLinker() = default;
DLLLinker::~DLLLinker() = default;

bool DLLLinker::linkDLL(const std::string &objectFile,
                        const std::string &outputDLL,
                        const std::vector<std::string> &libraries,
                        const std::vector<std::string> &libraryPaths,
                        const std::vector<std::string> &exportSymbols) {
#ifdef _WIN32
  return linkDLL_Windows(objectFile, outputDLL, libraries, libraryPaths,
                         exportSymbols);
#else
  return linkDLL_Linux(objectFile, outputDLL, libraries, libraryPaths);
#endif
}

bool DLLLinker::linkDLLInMemory(const std::vector<uint8_t> &objectBytes,
                                std::vector<uint8_t> &outDLLBytes,
                                const std::vector<std::string> &libraries,
                                const std::vector<std::string> &libraryPaths,
                                const std::vector<std::string> &exportSymbols) {
  // LLD does not provide a pure in-memory linking API
  // Strategy: Use temporary files for linking, then read result into memory
  // This approach works for MVP; can be improved with MemoryModule later

  // Generate unique temp file names using std::tmpnam
  char tempObjPath[L_tmpnam];
  char tempDLLPath[L_tmpnam];

  if (!std::tmpnam(tempObjPath) || !std::tmpnam(tempDLLPath)) {
    std::cerr << "Failed to generate temporary file names\n";
    return false;
  }

  // Add appropriate extensions
  std::string objFile = std::string(tempObjPath) + ".obj";
  std::string dllFile = std::string(tempDLLPath) + ".dll";

  // Write object bytes to temporary file
  {
    std::ofstream objOut(objFile, std::ios::binary);
    if (!objOut) {
      std::cerr << "Failed to create temporary object file: " << objFile
                << "\n";
      return false;
    }
    objOut.write(reinterpret_cast<const char *>(objectBytes.data()),
                 objectBytes.size());
    objOut.close();
  }

  // Link using existing file-based API
  bool linkSuccess =
      linkDLL(objFile, dllFile, libraries, libraryPaths, exportSymbols);

  if (!linkSuccess) {
    // Cleanup temp object file
    llvm::sys::fs::remove(objFile);
    return false;
  }

  // Read DLL into memory
  {
    std::ifstream dllIn(dllFile, std::ios::binary | std::ios::ate);
    if (!dllIn) {
      std::cerr << "Failed to open temporary DLL file: " << dllFile << "\n";
      llvm::sys::fs::remove(objFile);
      llvm::sys::fs::remove(dllFile);
      return false;
    }

    std::streamsize dllSize = dllIn.tellg();
    dllIn.seekg(0, std::ios::beg);

    outDLLBytes.resize(static_cast<size_t>(dllSize));
    if (!dllIn.read(reinterpret_cast<char *>(outDLLBytes.data()), dllSize)) {
      std::cerr << "Failed to read temporary DLL file\n";
      llvm::sys::fs::remove(objFile);
      llvm::sys::fs::remove(dllFile);
      return false;
    }
  }

  // Cleanup temporary files
  llvm::sys::fs::remove(objFile);
  llvm::sys::fs::remove(dllFile);

  std::cout << "Linked DLL in memory: " << outDLLBytes.size() << " bytes\n";
  return true;
}

#ifdef _WIN32

bool DLLLinker::createModuleDefinitionFile(
    const std::string &defPath, const std::vector<std::string> &exportSymbols) {
  std::ofstream defFile(defPath);
  if (!defFile) {
    std::cerr << "Failed to create .def file: " << defPath << "\n";
    return false;
  }

  defFile << "EXPORTS\n";
  for (const auto &symbol : exportSymbols) {
    defFile << "    " << symbol << "\n";
  }

  defFile.close();
  return true;
}

bool DLLLinker::linkDLL_Windows(const std::string &objectFile,
                                const std::string &outputDLL,
                                const std::vector<std::string> &libraries,
                                const std::vector<std::string> &libraryPaths,
                                const std::vector<std::string> &exportSymbols) {
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
  argStrings.push_back("/OUT:" + outputDLL);
  argStrings.push_back("/DEF:" + defFile);
  argStrings.push_back(objectFile);

  // Add library paths
  for (const auto &libPath : libraryPaths) {
    argStrings.push_back("/LIBPATH:" + libPath);
  }

  // Add libraries
  for (const auto &lib : libraries) {
    if (lib.find('/') != std::string::npos ||
        lib.find('\\') != std::string::npos) {
      argStrings.push_back(lib); // full path — pass as-is
    } else if (lib.size() >= 4 && lib.substr(lib.size() - 4) == ".lib") {
      argStrings.push_back(lib);
    } else {
      argStrings.push_back(lib + ".lib");
    }
  }

  // Add Windows system libraries (C Runtime, entry point, etc.)
  // These provide malloc, free, printf, _DllMainCRTStartup, etc.
  // EXPERIMENTAL: Use DYNAMIC CRT (/MDd) to allow HIP runtime state sharing
  // between EXE and DLL. Static CRT (/MTd) creates isolated CRT instances.
  // See: THEROCK_DLL_BUG_ANALYSIS.md for details
  argStrings.push_back(
      "msvcrtd.lib"); // MSVC Runtime (Dynamic, Debug) - replaces libcmtd.lib
  argStrings.push_back(
      "ucrtd.lib"); // Universal CRT (Dynamic, Debug) - replaces libucrtd.lib
  argStrings.push_back("vcruntimed.lib"); // VC Runtime (Dynamic, Debug)
  argStrings.push_back("oldnames.lib");   // Compatibility names
  argStrings.push_back("kernel32.lib");   // Windows kernel
  argStrings.push_back("user32.lib");     // Windows user API

  // Add default libraries and flags
  argStrings.push_back("/NOLOGO");
  argStrings.push_back("/MACHINE:X64");

  // Add debug flags to prevent optimization and get clear backtraces
  argStrings.push_back("/DEBUG");     // Generate debug info (.pdb)
  argStrings.push_back("/OPT:NOREF"); // Don't remove unreferenced code
  argStrings.push_back("/OPT:NOICF"); // Don't fold identical functions

  // Convert to C-style args for LLD
  std::vector<const char *> args;
  for (const auto &arg : argStrings) {
    args.push_back(arg.c_str());
  }

  // TEMP: Save .obj file before LLD deletes it
  {
    std::string savedObj = objectFile + ".saved";
    llvm::sys::fs::copy_file(objectFile, savedObj);
    std::cout << "Saved copy of object file to: " << savedObj << "\n";
  }

  // Debug: Print LLD command line
  std::cout << "LLD-LINK command (" << args.size() << " args): ";
  for (size_t i = 0; i < args.size(); ++i) {
    std::cout << "[" << i << "]='" << args[i] << "' ";
  }
  std::cout << "\n";

  // Call LLD linker library
  std::string stdoutStr, stderrStr;
  llvm::raw_string_ostream stdoutOS(stdoutStr);
  llvm::raw_string_ostream stderrOS(stderrStr);

  // Create ArrayRef explicitly
  llvm::ArrayRef<const char *> argsRef(args);
  std::cout << "ArrayRef size: " << argsRef.size() << "\n";

  // Use lldMain for crash recovery instead of direct link() call
  // lldMain provides:
  // - CrashRecoveryContext for handling fatal() calls
  // - Proper cleanup via CommonLinkerContext::destroy()
  // - Safe for re-entry
  lld::Result result =
      lld::lldMain(argsRef, stdoutOS, stderrOS,
                   {{lld::WinLink, &lld::coff::link}} // Register COFF driver
      );

  // Print linker output
  if (!stdoutStr.empty()) {
    std::cout << stdoutStr;
  }
  if (!stderrStr.empty()) {
    std::cerr << stderrStr;
  }

  if (result.retCode != 0) {
    std::cerr << "LLD-LINK failed with exit code: " << result.retCode << "\n";
    if (!result.canRunAgain) {
      std::cerr << "  Warning: Linker crashed, cannot run again\n";
    }
    return false;
  }

  std::cout << "Successfully linked DLL: " << outputDLL << "\n";

  // Cleanup .def file
  // DISABLED: Keep intermediate files for debugging
  // llvm::sys::fs::remove(defFile);
  std::cout << "Kept intermediate files: " << objectFile << ", " << defFile
            << "\n";

  return true;
}

#else // Linux

bool DLLLinker::linkDLL_Linux(const std::string &objectFile,
                              const std::string &outputDLL,
                              const std::vector<std::string> &libraries,
                              const std::vector<std::string> &libraryPaths) {
  // Build LLD-ELF command line arguments for shared library
  std::vector<std::string> argStrings;
  argStrings.push_back("ld.lld");  // Program name (required by LLD)
  argStrings.push_back("-shared"); // Create shared library
  argStrings.push_back("-o");
  argStrings.push_back(outputDLL);
  argStrings.push_back(objectFile);

  // Add library paths
  for (const auto &libPath : libraryPaths) {
    argStrings.push_back("-L" + libPath);
  }

  // Add libraries
  for (const auto &lib : libraries) {
    argStrings.push_back("-l" + lib);
  }

  // Add RPATH for runtime library search
  for (const auto &libPath : libraryPaths) {
    argStrings.push_back("-rpath");
    argStrings.push_back(libPath);
  }

  // Add default flags
  argStrings.push_back("--export-dynamic"); // Export all symbols by default
  argStrings.push_back("--no-undefined");   // Error on undefined symbols

  // Convert to C-style args for LLD
  std::vector<const char *> args;
  for (const auto &arg : argStrings) {
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

  // Print linker output
  if (!stdoutStr.empty()) {
    std::cout << stdoutStr;
  }
  if (!stderrStr.empty()) {
    std::cerr << stderrStr;
  }

  if (result.retCode != 0) {
    std::cerr << "LLD-ELF failed with exit code: " << result.retCode << "\n";
    if (!result.canRunAgain) {
      std::cerr << "  Warning: Linker crashed, cannot run again\n";
    }
    return false;
  }

  std::cout << "Successfully linked shared library: " << outputDLL << "\n";
  return true;
}

#endif

bool DLLLinker::verifyDLLExports(
    const std::string &dllPath,
    const std::vector<std::string> &requiredSymbols) {
  // This is a simplified implementation
  // A complete implementation would use LLVM's object file libraries
  // to parse the DLL/SO and verify exported symbols

  std::cout << "Verifying DLL exports for: " << dllPath << "\n";
  for (const auto &symbol : requiredSymbols) {
    std::cout << "  Required symbol: " << symbol << "\n";
  }

  // TODO: Implement actual symbol verification using LLVM object file API
  // For now, just check if file exists
  if (!llvm::sys::fs::exists(dllPath)) {
    std::cerr << "DLL file does not exist: " << dllPath << "\n";
    return false;
  }

  return true;
}

} // namespace hipdnn
