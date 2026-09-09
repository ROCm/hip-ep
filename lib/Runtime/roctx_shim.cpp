/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
// Optional host-native ROCTx bridge for runtime.bc.
//
// The generated runtime is LLVM bitcode, so it must not import ROCTx directly:
// the JIT host may not have ROCTx installed, and native model DLLs must remain
// deployable without it.  This shim is linked natively (alongside tls_stream)
// and resolves the two range functions only when HIPDNN_EP_ROCTX is enabled.
// Missing libraries/symbols therefore degrade to a no-op instead of preventing
// the EP or a compiled model DLL from loading.

#include "hip/env.h"
#include "hipdnn_ep_runtime.h"

#include <cstdio>
#include <string>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace {

using RoctxRangePushFn = int (*)(const char *);
using RoctxRangePopFn = int (*)();

struct RoctxApi {
  RoctxRangePushFn push = nullptr;
  RoctxRangePopFn pop = nullptr;
};

bool roctxVerbose() {
  return hipdnn_ep::env_enabled("HIPDNN_EP_ROCTX_VERBOSE");
}

RoctxApi loadRoctxApi() {
  RoctxApi api;
  if (!hipdnn_ep::env_enabled("HIPDNN_EP_ROCTX"))
    return api;

  std::string configured = hipdnn_ep::env_string("HIPDNN_EP_ROCTX_LIB");

#ifdef _WIN32
  const char *library = configured.empty() ? "roctx64.dll" : configured.c_str();
  HMODULE module = ::LoadLibraryA(library);
  if (module) {
    api.push = reinterpret_cast<RoctxRangePushFn>(
        ::GetProcAddress(module, "roctxRangePushA"));
    api.pop = reinterpret_cast<RoctxRangePopFn>(
        ::GetProcAddress(module, "roctxRangePop"));
  }
#else
  const char *library =
      configured.empty() ? "libroctx64.so" : configured.c_str();
  void *module = ::dlopen(library, RTLD_NOW | RTLD_LOCAL);
  if (!module && configured.empty())
    module = ::dlopen("libroctx64.so.4", RTLD_NOW | RTLD_LOCAL);
  if (module) {
    api.push =
        reinterpret_cast<RoctxRangePushFn>(::dlsym(module, "roctxRangePushA"));
    api.pop =
        reinterpret_cast<RoctxRangePopFn>(::dlsym(module, "roctxRangePop"));
  }
#endif

  if (!api.push || !api.pop) {
    api = {};
    if (roctxVerbose())
      std::fprintf(
          stderr, "[hip-ep roctx] disabled: could not load range API from %s\n",
          library);
    return api;
  }

  if (roctxVerbose())
    std::fprintf(stderr, "[hip-ep roctx] enabled: %s\n", library);
  return api;
}

const RoctxApi &roctxApi() {
  // Process-lifetime cache.  Deliberately keep the library loaded because
  // runtime/JIT scopes may outlive individual model sessions.
  static const RoctxApi api = loadRoctxApi();
  return api;
}

} // namespace

extern "C" HIPDNN_EP_RT_EXPORT int
hipdnn_ep_roctx_range_push(const char *name) {
  const RoctxApi &api = roctxApi();
  if (!api.push || !name)
    return -1;
  return api.push(name);
}

extern "C" HIPDNN_EP_RT_EXPORT void hipdnn_ep_roctx_range_pop(void) {
  const RoctxApi &api = roctxApi();
  if (api.pop)
    (void)api.pop();
}
