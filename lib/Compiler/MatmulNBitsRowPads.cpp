/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Compiler/MatmulNBitsRowPads.h"

#include "hip/debug_log.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#ifndef _WIN32
#include <dlfcn.h>
#endif

namespace hip::compiler {
namespace {

#ifdef _WIN32
// Declared rather than including <windows.h>, whose macros collide with the
// MLIR/LLVM headers; same pattern as RocMlirAutotune.cpp.
extern "C" __declspec(dllimport) int __stdcall GetModuleHandleExA(unsigned long,
                                                                  const char *,
                                                                  void **);
extern "C" __declspec(dllimport) unsigned long __stdcall GetModuleFileNameA(
    void *, char *, unsigned long);
extern "C" __declspec(dllimport) void *__stdcall LoadLibraryExA(const char *,
                                                                void *,
                                                                unsigned long);
extern "C" __declspec(dllimport) void *__stdcall GetProcAddress(void *,
                                                                const char *);
// From libloaderapi.h.
#define HIPDNN_MODULE_FROM_ADDRESS 0x00000004
#define HIPDNN_MODULE_UNCHANGED_REFCOUNT 0x00000002
// Resolve the library's own dependencies from its directory, since the path
// passed in is absolute.
#define HIPDNN_LOAD_WITH_ALTERED_SEARCH_PATH 0x00000008
#endif

// Directory of the module this code is linked into. The custom kernels
// library installs side by side with it.
std::string thisModuleDirectory() {
#ifdef _WIN32
  void *mod = nullptr;
  if (!GetModuleHandleExA(
          HIPDNN_MODULE_FROM_ADDRESS | HIPDNN_MODULE_UNCHANGED_REFCOUNT,
          reinterpret_cast<const char *>(&thisModuleDirectory), &mod))
    return {};
  char buf[1024] = {};
  const unsigned long n = GetModuleFileNameA(mod, buf, sizeof(buf));
  if (n == 0 || n >= sizeof(buf))
    return {};
  llvm::StringRef path(buf, n);
#else
  Dl_info info{};
  if (!dladdr(reinterpret_cast<void *>(&thisModuleDirectory), &info) ||
      !info.dli_fname)
    return {};
  llvm::StringRef path(info.dli_fname);
#endif
  const size_t slash = path.find_last_of("\\/");
  return slash == llvm::StringRef::npos ? std::string()
                                        : path.take_front(slash).str();
}

} // namespace

std::vector<std::string> queryMatmulNBitsRowPads() {
  std::string arch = hip_get_env("HIP_CUSTOM_KERNELS_ARCH");
#ifdef HIPDNN_CK_DEFAULT_ARCH
  if (arch.empty())
    arch = HIPDNN_CK_DEFAULT_ARCH;
#endif
  if (arch.empty())
    return {};
  std::string dir = hip_get_env("HIP_CUSTOM_KERNELS_DIR");
  if (dir.empty())
    dir = thisModuleDirectory();
  if (dir.empty())
    return {};

  using RowPadTableFn = const char *(*)();
  RowPadTableFn table = nullptr;
#ifdef _WIN32
  const std::string lib = dir + "\\custom_kernels_" + arch + ".dll";
  if (void *mod = LoadLibraryExA(lib.c_str(), nullptr,
                                 HIPDNN_LOAD_WITH_ALTERED_SEARCH_PATH))
    table = reinterpret_cast<RowPadTableFn>(
        GetProcAddress(mod, "hip_matmul_nbits_u4_row_pad_table"));
#else
  const std::string lib = dir + "/libcustom_kernels_" + arch + ".so";
  if (void *mod = dlopen(lib.c_str(), RTLD_LAZY))
    table = reinterpret_cast<RowPadTableFn>(
        dlsym(mod, "hip_matmul_nbits_u4_row_pad_table"));
#endif
  // The library stays loaded: it is the one the runtime loads for this model.
  if (!table) {
    COMPILER_DEBUG_LOG("[CompilerDriver] MatMulNBits row pads: no table in "
                       << lib << "; default rule\n");
    return {};
  }

  std::vector<std::string> entries;
  llvm::SmallVector<llvm::StringRef> parts;
  llvm::StringRef(table()).split(parts, ',', /*MaxSplit=*/-1,
                                 /*KeepEmpty=*/false);
  for (llvm::StringRef part : parts)
    entries.push_back(part.trim().str());
  COMPILER_DEBUG_LOG("[CompilerDriver] MatMulNBits row pads: "
                     << entries.size() << " entries from " << lib << "\n");
  return entries;
}

} // namespace hip::compiler
