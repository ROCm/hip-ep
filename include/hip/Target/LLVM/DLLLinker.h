/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef DLL_LINKER_H
#define DLL_LINKER_H

#include <cstddef>   // size_t
#include <cstdint>   // uint8_t
#include <string>
#include <vector>

namespace hipdnn {

// DLL Linker: Links object files with runtime library to produce native DLL
// Uses LLD (LLVM Linker) library APIs directly - NO external programs
// Supports Windows (PE/COFF) and Linux (ELF) platforms

class DLLLinker {
public:
  DLLLinker();
  ~DLLLinker();

  // Link object file with libraries to produce DLL/SO
  // Platform-specific:
  //   Windows: Produces .dll using LLD-LINK (PE/COFF)
  //   Linux: Produces .so using LLD-ELF (shared object)
  //
  // Parameters:
  //   objectFile: Input object file path (.obj on Windows, .o on Linux)
  //   outputDLL: Output DLL path (.dll on Windows, .so on Linux)
  //   libraries: Library names to link (e.g., "amdhip64", "MIOpen")
  //   libraryPaths: Directories to search for libraries
  //   exportSymbols: Symbol names to export from DLL (inference_init, etc.)
  //
  // Returns: true on success, false on failure
  bool linkDLL(const std::string &objectFile, const std::string &outputDLL,
               const std::vector<std::string> &libraries,
               const std::vector<std::string> &libraryPaths,
               const std::vector<std::string> &exportSymbols);

  // In-Memory Mode: Link object bytes to DLL in memory (for EPContext storage)
  // Parameters:
  //   objectBytes: Input object file data in memory
  //   outDLLBytes: Output DLL data in memory
  //   libraries: Library names to link (e.g., "amdhip64", "MIOpen")
  //   libraryPaths: Directories to search for libraries
  //   exportSymbols: Symbol names to export from DLL (inference_init, etc.)
  //
  // Returns: true on success, false on failure
  bool linkDLLInMemory(const std::vector<uint8_t> &objectBytes,
                       std::vector<uint8_t> &outDLLBytes,
                       const std::vector<std::string> &libraries,
                       const std::vector<std::string> &libraryPaths,
                       const std::vector<std::string> &exportSymbols);

  // Verify DLL has required exports (debugging utility)
  // Returns: true if all symbols are exported, false otherwise
  bool verifyDLLExports(const std::string &dllPath,
                        const std::vector<std::string> &requiredSymbols);

private:
  // Platform-specific linker implementations
#ifdef _WIN32
  bool linkDLL_Windows(const std::string &objectFile,
                       const std::string &outputDLL,
                       const std::vector<std::string> &libraries,
                       const std::vector<std::string> &libraryPaths,
                       const std::vector<std::string> &exportSymbols);

  // Create module definition file (.def) for Windows exports
  bool
  createModuleDefinitionFile(const std::string &defPath,
                             const std::vector<std::string> &exportSymbols);
#else
  bool linkDLL_Linux(const std::string &objectFile,
                     const std::string &outputDLL,
                     const std::vector<std::string> &libraries,
                     const std::vector<std::string> &libraryPaths);
#endif
};

} // namespace hipdnn

#endif // DLL_LINKER_H
