/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "hip/Target/LLVM/DLLLinker.h"

#include <llvm/Support/CommandLine.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/Path.h>
#include <llvm/Support/Program.h>
#include <llvm/Support/raw_ostream.h>

#include "hip/debug_log.h"

#include <fstream>
#include <sstream>

// In-process lld is only used by linkDLL_Windows (COFF). Linux uses a
// subprocess `clang++ -shared -fuse-ld=lld` instead (see linkDLL_Linux for
// the reason), so the ELF driver and lld::lldMain pull-in are scoped to
// _WIN32 to keep the Linux build free of liblldELF / liblldCommon link deps.
#ifdef _WIN32
#include "lld/Common/Driver.h"
LLD_HAS_DRIVER(coff)
#endif

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
    objOut.write(reinterpret_cast<const char *>(objectBytes.data()),
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
    auto &buf = *bufOrErr;
    const auto *data = reinterpret_cast<const uint8_t *>(buf->getBufferStart());
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
    const std::string &defPath, const std::vector<std::string> &exportSymbols) {
  std::ofstream defFile(defPath);
  if (!defFile) {
    llvm::errs() << "Failed to create .def file: " << defPath << "\n";
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
  argStrings.push_back(
      "/ignore:4099"); // Suppress missing PDB warnings (LNK4099)
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

  // Suppress all /defaultlib directives embedded in the .obj by the compiler
  // (e.g. libcmtd, vcruntimed from debug-mode LLVM codegen) and supply only
  // the release CRT import libraries explicitly.  This ensures the JIT-compiled
  // DLL loads on machines without VS installed (debug CRT DLLs such as
  // ucrtbased.dll and VCRUNTIME140D.dll are not redistributable).
  argStrings.push_back("/NODEFAULTLIB");
  for (const char *sysLib :
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
  std::vector<const char *> args;
  for (const auto &arg : argStrings) {
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

  llvm::ArrayRef<const char *> argsRef(args);

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

bool DLLLinker::linkDLL_Linux(const std::string &objectFile,
                              const std::string &outputDLL,
                              const std::vector<std::string> &libraries,
                              const std::vector<std::string> &libraryPaths) {
  // Delegate to `clang++ -shared` driver subprocess (vs in-process
  // lld::lldMain). Driver owns crt + sysroot + multiarch -L + libgcc + the
  // glibc 2.34 libpthread merge. Subprocess because lldMain SIGSEGVs on its
  // post-output cleanup when the shared library carrying this code is
  // dlopen'd into the EP host process — libhipgpu.so in an EP build,
  // libhip-compiler.so in a tools-only build (Windows linkDLL_Windows stays
  // in-process — lld-link doesn't have this cleanup bug). clang++ (not bare
  // clang) auto-links libstdc++ for the generated object's
  // __cxa_begin_catch / __cxa_rethrow.
  std::string clangPath;
#ifdef HIPDNN_CLANG_PATH
  clangPath = HIPDNN_CLANG_PATH;
#endif
  if (clangPath.empty() || !llvm::sys::fs::exists(clangPath)) {
    auto found = llvm::sys::findProgramByName("clang++");
    if (!found) {
      // HIPDNN_LLVM_MAJOR is injected by lib/Target/LLVM/CMakeLists.txt
      // from find_package(LLVM)'s LLVM_VERSION_MAJOR; stringify it so the
      // apt package hint tracks the LLVM version the project actually
      // configured against (instead of going stale on the next bump).
#define HIPDNN_STRINGIFY_(x) #x
#define HIPDNN_STRINGIFY(x) HIPDNN_STRINGIFY_(x)
      llvm::errs() << "clang++ not found on PATH and HIPDNN_CLANG_PATH is "
                      "unset or stale. Install clang (e.g. `apt install "
                      "clang-"
                   << HIPDNN_STRINGIFY(HIPDNN_LLVM_MAJOR)
                   << "`) or rebuild with HIPDNN_CLANG_PATH pointing to a "
                      "valid clang++ binary.\n";
#undef HIPDNN_STRINGIFY
#undef HIPDNN_STRINGIFY_
      return false;
    }
    clangPath = *found;
  }

  std::vector<std::string> args = {clangPath, "-shared", "-fuse-ld=lld",
                                   "-o",      outputDLL, objectFile};
  for (const auto &p : libraryPaths)
    args.push_back("-L" + p);
  // --start-group: user archives may mutually reference each other
  // (custom_kernels.a <-> libstdc++ via another user lib).
  args.push_back("-Wl,--start-group");
  for (const auto &lib : libraries) {
    // Mirror Windows: callers pass bare names or absolute paths.
    // `-l/abs/path` is malformed for ld; abs paths go through positional.
    if (lib.find('/') != std::string::npos)
      args.push_back(lib);
    else
      args.push_back("-l" + lib);
  }
  args.push_back("-Wl,--end-group");
  for (const auto &p : libraryPaths)
    args.push_back("-Wl,-rpath," + p);
  args.push_back("-Wl,--no-undefined");

  COMPILER_DEBUG_LOG("clang -shared command (" << args.size() << " args):\n");
  for (size_t i = 0; i < args.size(); ++i) {
    COMPILER_DEBUG_LOG("  [" << i << "]='" << args[i] << "'\n");
  }

  std::vector<llvm::StringRef> execArgs(args.begin(), args.end());
  std::string errMsg;
  bool execFailed = false;
  int rc = llvm::sys::ExecuteAndWait(clangPath, execArgs,
                                     /*Env=*/std::nullopt,
                                     /*Redirects=*/{},
                                     /*SecondsToWait=*/0,
                                     /*MemoryLimit=*/0, &errMsg, &execFailed);
  if (execFailed || rc != 0) {
    llvm::errs() << "clang -shared failed (rc=" << rc << "): " << errMsg
                 << "\n";
    return false;
  }

  COMPILER_DEBUG_LOG("Successfully linked shared library: " << outputDLL
                                                            << "\n");
  return true;
}

#endif

bool DLLLinker::verifyDLLExports(
    const std::string &dllPath,
    const std::vector<std::string> &requiredSymbols) {
  // This is a simplified implementation
  // A complete implementation would use LLVM's object file libraries
  // to parse the DLL/SO and verify exported symbols

  COMPILER_DEBUG_LOG("Verifying DLL exports for: " << dllPath << "\n");
  for (const auto &symbol : requiredSymbols) {
    COMPILER_DEBUG_LOG("  Required symbol: " << symbol << "\n");
  }

  if (!llvm::sys::fs::exists(dllPath)) {
    llvm::errs() << "DLL file does not exist: " << dllPath << "\n";
    return false;
  }

  return true;
}

} // namespace hipdnn
