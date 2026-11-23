/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once
// Optional ROCTx ranges for the EP bridge, one per compiled subgraph run.
//
// Same contract as hip-ep's lib/Runtime/roctx_shim.cpp: nothing is loaded
// unless HIPDNN_EP_ROCTX is enabled, HIPDNN_EP_ROCTX_LIB may name the library,
// and a missing library or symbol degrades to a no-op. ort-bridge resolves the
// library itself so it does not link against the hip-ep runtime.

#include <cstdlib>
#include <string>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace morphizen {
namespace roctx {

struct Api {
  int (*push)(const char *) = nullptr;
  int (*pop)() = nullptr;
};

inline std::string read_env(const char *name) {
#ifdef _WIN32
  char buf[1024];
  DWORD n = ::GetEnvironmentVariableA(name, buf, sizeof(buf));
  return (n > 0 && n < sizeof(buf)) ? std::string(buf, n) : std::string();
#else
  const char *v = std::getenv(name);
  return v ? std::string(v) : std::string();
#endif
}

inline Api load_api() {
  Api api;
  std::string enabled = read_env("HIPDNN_EP_ROCTX");
  if (enabled.empty() || enabled[0] < '1')
    return api;
  std::string configured = read_env("HIPDNN_EP_ROCTX_LIB");
#ifdef _WIN32
  HMODULE module =
      ::LoadLibraryA(configured.empty() ? "roctx64.dll" : configured.c_str());
  if (module) {
    api.push = reinterpret_cast<int (*)(const char *)>(
        ::GetProcAddress(module, "roctxRangePushA"));
    api.pop =
        reinterpret_cast<int (*)()>(::GetProcAddress(module, "roctxRangePop"));
  }
#else
  void *module = ::dlopen(
      configured.empty() ? "libroctx64.so" : configured.c_str(), RTLD_NOW);
  if (!module && configured.empty())
    module = ::dlopen("libroctx64.so.4", RTLD_NOW);
  if (module) {
    api.push = reinterpret_cast<int (*)(const char *)>(
        ::dlsym(module, "roctxRangePushA"));
    api.pop = reinterpret_cast<int (*)()>(::dlsym(module, "roctxRangePop"));
  }
#endif
  if (!api.push || !api.pop)
    api = {};
  return api;
}

// Process-lifetime cache; the library stays loaded.
inline const Api &api() {
  static const Api cached = load_api();
  return cached;
}

inline bool enabled() { return api().push != nullptr; }

// Balanced range for one scope. An empty label is a no-op, so callers build
// the label once (only when enabled()) and pay a single check per run.
class Range {
public:
  explicit Range(const std::string &label) {
    if (!label.empty() && api().push)
      active_ = api().push(label.c_str()) >= 0;
  }
  ~Range() {
    if (active_)
      api().pop();
  }
  Range(const Range &) = delete;
  Range &operator=(const Range &) = delete;

private:
  bool active_ = false;
};

} // namespace roctx
} // namespace morphizen
