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

// Resolve a gcc-managed file (crtbeginS.o, libgcc.a, ...) via
// `gcc -print-file-name=<name>`. Returns empty string if gcc isn't on PATH
// or returns the input unchanged (gcc's "not found" convention).
static std::string gccPrintFileName(const char *name) {
  std::string cmd =
      std::string("gcc -print-file-name=") + name + " 2>/dev/null";
  FILE *fp = popen(cmd.c_str(), "r");
  if (!fp)
    return {};
  char buf[1024];
  std::string result;
  if (fgets(buf, sizeof(buf), fp))
    result = buf;
  pclose(fp);
  while (!result.empty() && (result.back() == '\n' || result.back() == '\r'))
    result.pop_back();
  // gcc returns the input verbatim when it can't find the file; treat that
  // as "not found" so callers don't pass nonsense paths to the linker.
  if (result == name)
    return {};
  return result;
}

bool DLLLinker::linkDLL_Linux(const std::string &objectFile,
                              const std::string &outputDLL,
                              const std::vector<std::string> &libraries,
                              const std::vector<std::string> &libraryPaths) {
  // Resolve gcc CRT objects up-front so the link line below stays linear.
  // These bracket the user object files and provide the per-DSO startup +
  // teardown glue that `clang`/`gcc` driver normally injects but we have to
  // do by hand because we invoke ld.lld directly. Specifically:
  //
  //   crti.o       — opens _init / _fini function bodies (sentinel pair
  //                  with crtn.o). Strictly only needed when the DLL has
  //                  DT_INIT/DT_FINI, but harmless when paired with crtn.o.
  //   crtbeginS.o  — provides __dso_handle and __do_global_dtors_aux. The
  //                  init_array entry from this object registers
  //                  __cxa_atexit(__do_global_dtors_aux, ..., __dso_handle)
  //                  at load time so glibc has a hook to call when the DLL
  //                  is dlclose'd (the .fini_array reciprocal then walks
  //                  __cxa_finalize(&__dso_handle) and drains the
  //                  per-DSO destructor list).
  //   crtendS.o    — terminator counterpart for crtbeginS.o (closes the
  //                  .init_array / .fini_array sections).
  //   crtn.o       — closes _init / _fini bodies started by crti.o.
  //
  // Without crtbeginS.o + crtendS.o the model DLL has an empty (or missing)
  // .fini_array, so dlclose never invokes __cxa_finalize for the DLL. C++
  // global destructors stay registered in the global atexit list with
  // function pointers that point into the just-unmapped DLL. When libc's
  // exit() walks that list, it hits an unmapped PC and SIGSEGV's. See the
  // "S" variant of each object: those are the shared-library / PIC ones
  // (crtbegin.o vs crtbeginS.o etc.).
  const std::string crti = gccPrintFileName("crti.o");
  const std::string crtbeginS = gccPrintFileName("crtbeginS.o");
  const std::string crtendS = gccPrintFileName("crtendS.o");
  const std::string crtn = gccPrintFileName("crtn.o");

  // Build LLD-ELF command line arguments for shared library
  std::vector<std::string> argStrings;
  argStrings.push_back("ld.lld");  // Program name (required by LLD)
  argStrings.push_back("-shared"); // Create shared library
  argStrings.push_back("-o");
  argStrings.push_back(outputDLL);

  // CRT prologue: must precede the user objects so the .init_array entries
  // contributed by crtbeginS.o land before our _GLOBAL__sub_I_* ctors.
  if (!crti.empty())
    argStrings.push_back(crti);
  if (!crtbeginS.empty())
    argStrings.push_back(crtbeginS);

  // User object file (the per-model bitcode after LLVM codegen).
  argStrings.push_back(objectFile);

  // Add library paths
  for (const auto &libPath : libraryPaths) {
    argStrings.push_back("-L" + libPath);
  }

  // Add libraries. Mirror the Windows path's behaviour: callers can pass
  // either bare names ("MIOpen" -> "-lMIOpen") or absolute paths to a
  // specific archive/.so (e.g. install-prefix paths to
  // libhip_custom_kernels.a). Bare-`-l/abs/path` is a malformed lld arg that
  // fails with "unable to find library -l/abs/path" — pass full paths as
  // positional args.
  for (const auto &lib : libraries) {
    if (lib.find('/') != std::string::npos) {
      argStrings.push_back(lib);
    } else {
      argStrings.push_back("-l" + lib);
    }
  }

  // Add RPATH for runtime library search
  for (const auto &libPath : libraryPaths) {
    argStrings.push_back("-rpath");
    argStrings.push_back(libPath);
  }

  // Add system library search paths + standard C/C++ runtime libraries.
  //
  // We invoke ld.lld directly (via lld::lldMain) for crash-recovery, so unlike
  // a clang/gcc driver invocation NOTHING auto-pulls libc/libstdc++/libgcc.
  // The runtime bitcode (std::chrono, operator new, ...) and the custom-kernel
  // archive (fprintf/memcpy/__cxa_guard_acquire/...) reference dozens of libc
  // and libstdc++ symbols; without these the link fails with hundreds of
  // "undefined symbol" errors. The Windows path solves the same problem by
  // explicitly adding msvcrt.lib/ucrt.lib/vcruntime.lib above. This is the
  // ELF analogue.
  //
  // Multiarch path is hard-coded to x86_64-linux-gnu (the only Linux target
  // we currently support). The gcc internal libdir (libgcc.a, crtbeginS.o)
  // is discovered by parsing `gcc --print-libgcc-file-name` so we don't have
  // to hard-code the gcc version. On glibc 2.34+ libpthread/libdl/librt are
  // merged into libc, so they're omitted (would error "unable to find
  // -lpthread" because the .so files are gone).
  argStrings.push_back("-L/usr/lib/x86_64-linux-gnu");
  argStrings.push_back("-L/lib/x86_64-linux-gnu");
  if (FILE *fp = popen("gcc --print-libgcc-file-name 2>/dev/null", "r")) {
    char buf[1024];
    if (fgets(buf, sizeof(buf), fp)) {
      std::string libgccPath(buf);
      while (!libgccPath.empty() &&
             (libgccPath.back() == '\n' || libgccPath.back() == '\r'))
        libgccPath.pop_back();
      auto slash = libgccPath.find_last_of('/');
      if (slash != std::string::npos)
        argStrings.push_back("-L" + libgccPath.substr(0, slash));
    }
    pclose(fp);
  }
  for (const char *sysLib : {"stdc++", "m", "gcc_s", "gcc", "c"})
    argStrings.push_back(std::string("-l") + sysLib);

  // CRT epilogue: must come AFTER user libs (the symbols crtendS.o / crtn.o
  // provide bracket .init_array / .fini_array / _init / _fini).
  if (!crtendS.empty())
    argStrings.push_back(crtendS);
  if (!crtn.empty())
    argStrings.push_back(crtn);

  // Add default flags
  argStrings.push_back("--export-dynamic"); // Export all symbols by default
  argStrings.push_back("--no-undefined");   // Error on undefined symbols

  // Convert to C-style args for LLD
  std::vector<const char *> args;
  for (const auto &arg : argStrings) {
    args.push_back(arg.c_str());
  }

  // Invoke ld.lld as a SUBPROCESS rather than via lld::lldMain.
  //
  // Why: when libhip-compiler.so (which statically links lldELF) is loaded
  // into a long-lived host process via dlopen — exactly the path the
  // MorphiZen EP takes — lld::lldMain writes the .so output successfully
  // but then segfaults during its internal cleanup (likely a static-state
  // / atexit / llvm_shutdown interaction with the host process). The
  // crash happens AFTER the artifact is on disk, so the link "worked" but
  // the EP never sees a successful return. The same args invoked via the
  // ld.lld binary subprocess complete cleanly. Windows lldMain (lld-link
  // / COFF driver) does not exhibit this and stays on the in-process path.
  //
  // ld.lld discovery: HIPDNN_LD_LLD_PATH is baked at configure time
  // (lib/Target/LLVM/CMakeLists.txt). PATH lookup is the fallback so the
  // build still works in environments that don't have llvm-XX-dev's
  // ld.lld at the discovered location.
  std::string ldLldPath;
#ifdef HIPDNN_LD_LLD_PATH
  ldLldPath = HIPDNN_LD_LLD_PATH;
#endif
  if (ldLldPath.empty() || !llvm::sys::fs::exists(ldLldPath)) {
    auto found = llvm::sys::findProgramByName("ld.lld");
    if (!found) {
      llvm::errs() << "ld.lld not found on PATH and HIPDNN_LD_LLD_PATH "
                      "is unset. Install lld or rebuild with a working "
                      "find_program(ld.lld).\n";
      return false;
    }
    ldLldPath = *found;
  }

  // Skip argv[0] ("ld.lld" — the program name) when handing args to
  // ExecuteAndWait. The first element of the new argv is the path itself.
  std::vector<llvm::StringRef> execArgs;
  execArgs.reserve(argStrings.size());
  execArgs.emplace_back(ldLldPath);
  for (size_t i = 1; i < argStrings.size(); ++i)
    execArgs.emplace_back(argStrings[i]);

  std::string errMsg;
  bool execFailed = false;
  int rc = llvm::sys::ExecuteAndWait(ldLldPath, execArgs,
                                     /*Env=*/std::nullopt,
                                     /*Redirects=*/{},
                                     /*SecondsToWait=*/0,
                                     /*MemoryLimit=*/0, &errMsg, &execFailed);
  if (execFailed || rc != 0) {
    llvm::errs() << "ld.lld failed (rc=" << rc << "): " << errMsg << "\n";
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
