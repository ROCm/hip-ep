/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Compiler/MatmulNBitsRowPads.h"

#include "hip/Support/GenericTarget.h"
#include "hip/debug_log.h"
#include "matmul_nbits_row_pad_abi.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <map>
#include <mutex>
#include <optional>
#include <utility>

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

std::string libraryPath(const std::string &dir, const std::string &arch) {
#ifdef _WIN32
  return dir + "\\custom_kernels_" + arch + ".dll";
#else
  return dir + "/libcustom_kernels_" + arch + ".so";
#endif
}

// The library's row-pad export, loading it on first use. The handle is kept
// for the process, one per path: unloading a HIP library unregisters its code
// objects, and the runtime loads the same library for the compiled model.
HipMatmulNBitsRowPadsFn rowPadsExport(const std::string &path) {
  static std::map<std::string, HipMatmulNBitsRowPadsFn> exports;
  auto it = exports.find(path);
  if (it != exports.end())
    return it->second;
  HipMatmulNBitsRowPadsFn fn = nullptr;
#ifdef _WIN32
  if (void *mod = LoadLibraryExA(path.c_str(), nullptr,
                                 HIPDNN_LOAD_WITH_ALTERED_SEARCH_PATH))
    fn = reinterpret_cast<HipMatmulNBitsRowPadsFn>(
        GetProcAddress(mod, "hip_matmul_nbits_u4_row_pads"));
#else
  if (void *mod = dlopen(path.c_str(), RTLD_LAZY | RTLD_LOCAL))
    fn = reinterpret_cast<HipMatmulNBitsRowPadsFn>(
        dlsym(mod, "hip_matmul_nbits_u4_row_pads"));
#endif
  exports.emplace(path, fn);
  return fn;
}

// `targetArch`'s row-pad policy from the library at `path` (empty arch: the
// running device). nullopt when it cannot be loaded, lacks the export, or has
// no table for the arch.
std::optional<MatmulNBitsRowPads> readLibrary(const std::string &path,
                                              const std::string &targetArch) {
  std::optional<MatmulNBitsRowPads> result;
  if (HipMatmulNBitsRowPadsFn query = rowPadsExport(path)) {
    const char *arch = targetArch.empty() ? nullptr : targetArch.c_str();
    int32_t defaultPad = 0;
    const int count = query(arch, &defaultPad, nullptr, 0);
    if (count >= 0) {
      std::vector<HipMatmulNBitsRowPad> rows(static_cast<size_t>(count));
      query(arch, &defaultPad, rows.data(), count);
      MatmulNBitsRowPads pads;
      pads.defaultPad = defaultPad;
      // The pass's `row-pads` option format.
      for (const HipMatmulNBitsRowPad &r : rows) {
        std::string entry;
        llvm::raw_string_ostream os(entry);
        os << r.n << ':' << r.k << ':' << r.block_size << ':' << r.zero_points
           << '=' << r.pad_bytes;
        pads.entries.push_back(std::move(entry));
      }
      result = std::move(pads);
    }
  }
  return result;
}

} // namespace

MatmulNBitsRowPads queryMatmulNBitsRowPads() {
  // The table is the compile target's: ROCK_ARCH when it names one (the same
  // override the artifact's rocMLIR target follows), else the running device
  // (device 0, as the runtime's own lookup).
  const std::string targetArch = hip_get_env("ROCK_ARCH");

  // Library: as the runtime loads it, custom_kernels_<arch> then its generic
  // ISA, with HIP_CUSTOM_KERNELS_ARCH first and the build's arch last; from
  // HIP_CUSTOM_KERNELS_DIR, else beside this module (installed), else the
  // build tree.
  std::vector<std::string> archs;
  auto addArch = [&](std::string arch) {
    if (!arch.empty() && llvm::find(archs, arch) == archs.end())
      archs.push_back(std::move(arch));
  };
  addArch(hip_get_env("HIP_CUSTOM_KERNELS_ARCH"));
  addArch(targetArch);
  addArch(genericTargetFor(targetArch));
#ifdef HIPDNN_CK_DEFAULT_ARCH
  addArch(HIPDNN_CK_DEFAULT_ARCH);
#endif
  std::vector<std::string> dirs;
  if (std::string dir = hip_get_env("HIP_CUSTOM_KERNELS_DIR"); !dir.empty()) {
    dirs.push_back(std::move(dir));
  } else {
    if (std::string dir = thisModuleDirectory(); !dir.empty())
      dirs.push_back(std::move(dir));
#ifdef HIPDNN_CK_LIB_DIR
    dirs.push_back(HIPDNN_CK_LIB_DIR);
#endif
  }

  // Compiling many models or subgraphs in one process asks the same library
  // the same question; answer it once per (library, target arch).
  static std::mutex mu;
  static std::map<std::pair<std::string, std::string>,
                  std::optional<MatmulNBitsRowPads>>
      cache;
  std::lock_guard<std::mutex> lock(mu);
  for (const std::string &dir : dirs) {
    for (const std::string &arch : archs) {
      const std::string path = libraryPath(dir, arch);
      if (!llvm::sys::fs::exists(path))
        continue;
      const auto key = std::make_pair(path, targetArch);
      auto it = cache.find(key);
      if (it == cache.end())
        it = cache.emplace(key, readLibrary(path, targetArch)).first;
      if (!it->second)
        continue;
      COMPILER_DEBUG_LOG("[CompilerDriver] MatMulNBits row pads: default "
                         << it->second->defaultPad << ", "
                         << it->second->entries.size() << " entries from "
                         << path << "\n");
      return *it->second;
    }
  }
  COMPILER_DEBUG_LOG("[CompilerDriver] MatMulNBits row pads: no table for "
                     << (targetArch.empty() ? "this device" : targetArch)
                     << " in any custom kernels library; weights stay dense\n");
  return {};
}

} // namespace hip::compiler
