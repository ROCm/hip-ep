/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#pragma once
// The single place the EP reads process environment variables.
//
// Why not std::getenv? The EP ships as a Windows static-CRT (/MT) DLL, whose
// CRT keeps its own environment copy snapshotted at init -- it does not see
// variables the host process (Python / onnxruntime / OGA) sets afterwards. And
// on the LLVM_IR JIT (bitcode) path the JIT cannot materialize the CRT env
// helpers at all. GetEnvironmentVariableA reads the real, kernel-maintained
// process environment, so it is correct in both the native and bitcode modes.
//
// Kept as plain C (char buffer, no <string>/<algorithm> search calls) so the
// runtime bitcode stays free of MSVC's vectorized __std_* helpers, which the
// JIT cannot resolve. This is the only Win32 call for env in the whole EP.
#include <cstdlib>
#include <string>

#ifdef _WIN32
extern "C" __declspec(
    dllimport) unsigned long __stdcall GetEnvironmentVariableA(const char *name,
                                                               char *buffer,
                                                               unsigned long
                                                                   size);
#endif

namespace hipdnn_ep {

// Copy env var `name` into `buf` (NUL-terminated); returns its length, 0 when
// unset. The one and only platform branch.
inline unsigned long read_env(const char *name, char *buf, unsigned long cap) {
#ifdef _WIN32
  return GetEnvironmentVariableA(name, buf, cap);
#else
  const char *v = std::getenv(name);
  if (v == nullptr)
    return 0;
  unsigned long n = 0;
  while (v[n] != '\0' && n + 1 < cap) {
    buf[n] = v[n];
    ++n;
  }
  buf[n] = '\0';
  return n;
#endif
}

// True when `name` is set to a value whose first character is >= '1'.
inline bool env_enabled(const char *name) {
  char buf[8];
  unsigned long n = read_env(name, buf, sizeof(buf));
  return n > 0 && buf[0] >= '1';
}

// Default-ON variant: enabled unless `name` is explicitly set to a value whose
// first character is '0'. For opt-in-turned-default paths that want the
// optimization active by default with an explicit "...=0" kill-switch.
inline bool env_enabled_default_on(const char *name) {
  char buf[8];
  unsigned long n = read_env(name, buf, sizeof(buf));
  return n == 0 || buf[0] != '0';
}

// Env var as a std::string; empty when unset.
inline std::string env_string(const char *name) {
  char buf[1024];
  unsigned long n = read_env(name, buf, sizeof(buf));
  return (n > 0 && n < sizeof(buf)) ? std::string(buf, n) : std::string();
}

} // namespace hipdnn_ep
