/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once

#include "timing.h"

#include "llvm/Support/raw_ostream.h"
#include <cstdlib>
#include <string>

#ifdef _WIN32
// Static CRT (/MT) DLLs (the MorphiZen EP DLL is one) have their own CRT env
// table — std::getenv / _dupenv_s cannot see env vars set by the host process.
// Read the real process environment via the Win32 API instead. Mirrors the
// pattern in lib/Runtime/debug_log.h.
extern "C" __declspec(
    dllimport) unsigned long __stdcall GetEnvironmentVariableA(const char *,
                                                               char *,
                                                               unsigned long);
#endif

// Read an environment variable in a way that works from /MT DLLs on Windows.
// Returns empty string when unset. Use this instead of std::getenv anywhere
// the compiler library may end up linked into the EP DLL.
inline std::string hip_get_env(const char *name) {
#ifdef _WIN32
  char buf[1024];
  unsigned long n = GetEnvironmentVariableA(name, buf, sizeof(buf));
  if (n == 0)
    return {};
  if (n < sizeof(buf))
    return std::string(buf, n);
  // Value larger than stack buffer — re-query with sized heap buffer.
  std::string out(n, '\0');
  unsigned long n2 = GetEnvironmentVariableA(name, out.data(), n);
  if (n2 == 0)
    return {};
  out.resize(n2);
  return out;
#else
  const char *v = std::getenv(name);
  return v ? std::string(v) : std::string();
#endif
}

inline bool hipdnn_ep_debug_enabled() {
  static const bool enabled = [] {
    std::string v = hip_get_env("HIPDNN_EP_DEBUG");
    return !v.empty() && v[0] >= '1';
  }();
  return enabled;
}

#define COMPILER_DEBUG_LOG(expr)                                               \
  do {                                                                         \
    if (hipdnn_ep_debug_enabled())                                             \
      llvm::errs() << expr;                                                    \
  } while (0)
