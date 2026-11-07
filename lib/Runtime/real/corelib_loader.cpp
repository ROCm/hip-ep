/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "corelib_loader.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace hipdnn_ep::corelib {
namespace {

void *open_library(const std::string &path) {
#ifdef _WIN32
  // Search the DLL's own directory first so its dependencies beside it load.
  DWORD flags =
      path.find_first_of("/\\") != std::string::npos
          ? LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS
          : 0;
  return reinterpret_cast<void *>(LoadLibraryExA(path.c_str(), nullptr, flags));
#else
  return dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
#endif
}

void *find_symbol(void *lib, const char *name) {
#ifdef _WIN32
  return reinterpret_cast<void *>(
      GetProcAddress(reinterpret_cast<HMODULE>(lib), name));
#else
  return dlsym(lib, name);
#endif
}

std::mutex g_users_mu;
int g_users = 0;

} // namespace

std::string read_env(const char *name) {
#ifdef _WIN32
  char buf[1024];
  DWORD n = GetEnvironmentVariableA(name, buf, sizeof(buf));
  if (n == 0 || n >= sizeof(buf))
    return {};
  return std::string(buf, n);
#else
  const char *v = std::getenv(name);
  return v ? std::string(v) : std::string();
#endif
}

const CorelibApi *load_api() {
  static CorelibApi api;
  static const CorelibApi *loaded = []() -> const CorelibApi * {
#ifdef _WIN32
    const char *kSubdir = "/bin/";
    const char *kName = "ryzenai_corelib.dll";
#else
    const char *kSubdir = "/lib/";
    const char *kName = "libryzenai_corelib.so";
#endif
    std::vector<std::string> candidates;
    std::string root = read_env("RYZENAICORELIB_ROOT");
    if (!root.empty()) {
      candidates.push_back(root + kSubdir + kName);
      candidates.push_back(root + "/install" + kSubdir + kName);
    }
    candidates.push_back(kName);

    void *lib = nullptr;
    std::string loaded_path;
    for (const std::string &c : candidates)
      if ((lib = open_library(c))) {
        loaded_path = c;
        break;
      }
    if (!lib) {
      fprintf(stderr, "[corelib] %s not found (RYZENAICORELIB_ROOT='%s')\n",
              kName, root.c_str());
      return nullptr;
    }

#define HIPDNN_EP_CORELIB_RESOLVE(name)                                        \
  api.name = reinterpret_cast<decltype(api.name)>(find_symbol(lib, #name));    \
  if (!api.name) {                                                             \
    fprintf(stderr, "[corelib] %s has no symbol %s\n", loaded_path.c_str(),    \
            #name);                                                            \
    return nullptr;                                                            \
  }
    // The version comes first: another version's library may lack symbols,
    // and a mismatch names the real problem better than a missing symbol.
    // Below 1.0 any minor release may break the ABI.
    HIPDNN_EP_CORELIB_RESOLVE(ryzenai_corelib_get_version)
    uint32_t major = 0, minor = 0, patch = 0;
    api.ryzenai_corelib_get_version(&major, &minor, &patch);
    if (major != RYZENAI_CORELIB_VERSION_MAJOR ||
        (major == 0 && minor != RYZENAI_CORELIB_VERSION_MINOR)) {
      fprintf(stderr,
              "[corelib] %s is version %u.%u.%u, but hip-ep was built against "
              "%d.%d.%d\n",
              loaded_path.c_str(), major, minor, patch,
              RYZENAI_CORELIB_VERSION_MAJOR, RYZENAI_CORELIB_VERSION_MINOR,
              RYZENAI_CORELIB_VERSION_PATCH);
      return nullptr;
    }
    HIPDNN_EP_CORELIB_FUNCTIONS(HIPDNN_EP_CORELIB_RESOLVE)
#undef HIPDNN_EP_CORELIB_RESOLVE
    return &api;
  }();
  return loaded;
}

void add_user() {
  std::lock_guard<std::mutex> lock(g_users_mu);
  ++g_users;
}

void remove_user(const CorelibApi &api) {
  std::lock_guard<std::mutex> lock(g_users_mu);
  if (--g_users == 0)
    api.ryzenai_corelib_cleanup();
}

} // namespace hipdnn_ep::corelib
