/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#pragma once
// Runtime debug logging gated on HIPDNN_EP_DEBUG env var (default: off)
// Set HIPDNN_EP_DEBUG=1 to enable all [Runtime DEBUG] output.
// Set HIPDNN_EP_PERF=1 to enable only [PERF] timing breakdown per inference.
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#ifndef _WIN32
#include <unistd.h>
#endif

#ifdef _WIN32
// Static CRT (/MT) DLLs have their own CRT env — _dupenv_s can't see env vars
// set by the host process. Use Win32 API to read the real process environment.
extern "C" __declspec(
    dllimport) unsigned long __stdcall GetEnvironmentVariableA(const char *,
                                                               char *,
                                                               unsigned long);

// Win32 stderr writer used by hipdnn_ep_log_emit (see the FILE*-free rationale
// there). Declared the same dllimport way as GetEnvironmentVariableA above,
// which the bitcode JIT already resolves successfully. Signatures match
// <windows.h> EXACTLY (HANDLE=void*, DWORD=unsigned long, BOOL=int,
// LPDWORD=unsigned long*, LPOVERLAPPED=struct _OVERLAPPED*) so that a TU which
// also pulls in <windows.h> (e.g. via another header) sees these as a
// compatible re-declaration rather than a conflicting one — we still avoid
// including <windows.h> ourselves (it drags in <intrin.h>, which conflicts with
// HIP's vector headers; see LlvmIrJit.cpp).
struct _OVERLAPPED;
extern "C" __declspec(dllimport) void *__stdcall GetStdHandle(unsigned long);
extern "C" __declspec(dllimport) int __stdcall WriteFile(void *, const void *,
                                                         unsigned long,
                                                         unsigned long *,
                                                         struct _OVERLAPPED *);

// Win32 module-resolution APIs used ONLY by the CRT-binding diagnostic
// (hipdnn_ep_diag_crt_binding, gated on HIPDNN_EP_DIAG_CRT=1). Same
// no-<windows.h> forward-decl approach as above; signatures match
// <libloaderapi.h> EXACTLY. NOTE: HMODULE is `struct HINSTANCE__*`, NOT
// `void*` (unlike HANDLE) — using void* here conflicts with <windows.h> if any
// other header in the TU pulls it in. Forward-declare the opaque struct so the
// pointer type matches exactly.
struct HINSTANCE__;
extern "C" __declspec(dllimport) int __stdcall GetModuleHandleExA(
    unsigned long, const char *, struct HINSTANCE__ **);
extern "C" __declspec(dllimport) unsigned long __stdcall GetModuleFileNameA(
    struct HINSTANCE__ *, char *, unsigned long);
// __acrt_iob_func(n) is what the UCRT's stdin/stdout/stderr macros expand to;
// stderr == __acrt_iob_func(2). Taking its address (and the FILE* it returns)
// reveals which loaded CRT instance the JIT bound this module's stdio to.
extern "C" FILE *__cdecl __acrt_iob_func(unsigned);

// Win32 STD_ERROR_HANDLE: the sentinel DWORD that GetStdHandle() maps to the
// process's stderr handle (winbase.h defines it as (DWORD)-12, i.e.
// 0xFFFFFFF4). Spelled out here so we don't pull in <windows.h> (it drags in
// <intrin.h>, which conflicts with HIP's vector headers — see LlvmIrJit.cpp).
constexpr unsigned long kWin32StdErrorHandle = static_cast<unsigned long>(-12);

namespace detail {
inline bool check_env(const char *name) {
  char buf[8];
  unsigned long n = GetEnvironmentVariableA(name, buf, sizeof(buf));
  return n > 0 && buf[0] >= '1';
}
} // namespace detail
#else
// POSIX-guaranteed file descriptor for stderr (also spelled STDERR_FILENO in
// <unistd.h>, included above). Named here to mirror the Win32 stderr-handle
// constant above.
constexpr int kPosixStderrFd = 2;
#endif

// Emit a formatted log line to stderr WITHOUT going through the C stdio FILE*
// table (no fprintf/fputs/stderr).
//
// Why: this header is compiled into the runtime bitcode, which the EP loads via
// LlvmIrJit (bitcode mode), NOT by linking a self-contained DLL. The JIT
// resolves external symbols by name against the whole process image
// (DynamicLibrarySearchGenerator::GetForCurrentProcess), so a `fprintf` call
// binds its `stderr` FILE* global and its stdio implementation from
// independently-resolved CRT state that was never wired together / initialized
// for this module — dereferencing that FILE* faults (0xC0000005) on the first
// HIPDNN_EP_DEBUG / HIPDNN_EP_PERF log. vsnprintf only formats into a caller
// buffer (no stdio stream state) and Win32 WriteFile / POSIX write target the
// OS stderr handle (fd 2) directly, so both resolve safely under the JIT. Same
// class of fix as using GetEnvironmentVariableA instead of getenv above.
inline void hipdnn_ep_log_emit(const char *fmt, ...) {
  char buf[2048];
  va_list ap;
  va_start(ap, fmt);
  int n = vsnprintf(buf, sizeof(buf), fmt, ap);
  va_end(ap);
  if (n <= 0)
    return;
  // vsnprintf returns the would-be length; clamp to the truncated buffer.
  size_t len = (static_cast<size_t>(n) < sizeof(buf)) ? static_cast<size_t>(n)
                                                      : sizeof(buf) - 1;
#ifdef _WIN32
  unsigned long written = 0;
  WriteFile(GetStdHandle(kWin32StdErrorHandle), buf,
            static_cast<unsigned long>(len), &written, nullptr);
#else
  ssize_t w = write(kPosixStderrFd, buf, len);
  (void)w;
#endif
}

// One-shot diagnostic: dump which loaded module backs each CRT symbol the JIT
// (or, when called from EP-native code, the loader) resolved for THIS module.
// Gated on HIPDNN_EP_DIAG_CRT=1. Purpose: pin down whether the JIT'd runtime's
// fprintf/stderr/getenv bind to the shared, bootstrapped UCRT (ucrtbase.dll) or
// to some other CRT instance whose per-context state was never set up for this
// module — the root-cause question behind the FILE*-free hipdnn_ep_log_emit.
// `tag` distinguishes call sites (e.g. "JIT-runtime" vs "EP-native") so the two
// bindings can be compared side by side. Output goes through the CRT-free
// hipdnn_ep_log_emit, and function-address lines are emitted BEFORE the
// (potentially faulting) __acrt_iob_func(2) call, so the key data is flushed
// even if resolving the stderr FILE* itself crashes.
#ifdef _WIN32
namespace detail {
inline void diag_emit_owner(const char *label, const void *addr) {
  // GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS (0x4) |
  // GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT (0x2) = 0x6: look up the
  // module owning `addr` without bumping its refcount.
  struct HINSTANCE__ *hmod = nullptr;
  char path[260];
  path[0] = '?';
  path[1] = '\0';
  if (GetModuleHandleExA(0x6u, static_cast<const char *>(addr), &hmod) && hmod)
    GetModuleFileNameA(hmod, path, sizeof(path));
  hipdnn_ep_log_emit("[DIAG CRT] %-24s @ %p  ->  %s\n", label, addr, path);
}
} // namespace detail

inline void hipdnn_ep_diag_crt_binding(const char *tag) {
  if (!detail::check_env("HIPDNN_EP_DIAG_CRT"))
    return;
  static int done = 0; // zero-init (.bss) => JIT-safe, no global constructor
  if (done)
    return;
  done = 1;
  hipdnn_ep_log_emit("[DIAG CRT] ==== CRT binding probe (%s) ====\n", tag);
  detail::diag_emit_owner("&fprintf", reinterpret_cast<const void *>(&fprintf));
  detail::diag_emit_owner("&vfprintf",
                          reinterpret_cast<const void *>(&vfprintf));
  detail::diag_emit_owner("&__acrt_iob_func",
                          reinterpret_cast<const void *>(&__acrt_iob_func));
  detail::diag_emit_owner("&getenv", reinterpret_cast<const void *>(&getenv));
  detail::diag_emit_owner("&malloc", reinterpret_cast<const void *>(&malloc));
  detail::diag_emit_owner(
      "&GetEnvironmentVariableA",
      reinterpret_cast<const void *>(&GetEnvironmentVariableA));
  // stdio writers: pin which CRT each resolves to. The HIPDNN_EP_DIAG_STDOUT
  // experiment showed printf SURVIVES but fputs/fwrite CRASH (delayed, 3/3)
  // when handed the UCRT `stdout` FILE*. Hypothesis: fputs/fwrite bind to a
  // different CRT (msvcrt.dll, where getenv/malloc resolved) than the FILE*
  // layout they receive (ucrtbase.dll). These lines confirm the cross-CRT
  // handoff.
  detail::diag_emit_owner("&printf", reinterpret_cast<const void *>(&printf));
  detail::diag_emit_owner("&fputs", reinterpret_cast<const void *>(&fputs));
  detail::diag_emit_owner("&fwrite", reinterpret_cast<const void *>(&fwrite));
  detail::diag_emit_owner("&putchar", reinterpret_cast<const void *>(&putchar));
  detail::diag_emit_owner("&fflush", reinterpret_cast<const void *>(&fflush));
  // Resolve the actual stderr FILE* (this CALLS the bound __acrt_iob_func; if
  // that faults, every line above is already on stderr via WriteFile).
  FILE *se = __acrt_iob_func(2);
  detail::diag_emit_owner("stderr FILE*", reinterpret_cast<const void *>(se));
  FILE *so = __acrt_iob_func(1);
  detail::diag_emit_owner("stdout FILE*", reinterpret_cast<const void *>(so));
  hipdnn_ep_log_emit("[DIAG CRT] ==== end probe (%s) ====\n", tag);
}
#else
inline void hipdnn_ep_diag_crt_binding(const char * /*tag*/) {}
#endif

inline bool hipdnn_ep_debug_enabled() {
  static const bool enabled = [] {
#ifdef _WIN32
    return detail::check_env("HIPDNN_EP_DEBUG");
#else
    const char *v = std::getenv("HIPDNN_EP_DEBUG");
    return v && v[0] >= '1';
#endif
  }();
  return enabled;
}

// Sync-isolated profiling mode (HIPDNN_EP_PERF_ISOLATE=1). Inserts a
// hipStreamSynchronize at every OP_PROFILE scope boundary so each op's
// reported GPU time is its true standalone runtime, with no carry-over from
// prior queued work. Implies HIPDNN_EP_PERF=1. Kills concurrency by design;
// only useful as a diagnostic to find ops whose real cost is being masked by
// stream-queue depth in normal profiling.
inline bool hipdnn_ep_perf_isolate_enabled() {
  static const bool enabled = [] {
#ifdef _WIN32
    return detail::check_env("HIPDNN_EP_PERF_ISOLATE");
#else
    const char *v = std::getenv("HIPDNN_EP_PERF_ISOLATE");
    return v && v[0] >= '1';
#endif
  }();
  return enabled;
}

inline bool hipdnn_ep_perf_enabled() {
  // PERF intentionally does NOT inherit from HIPDNN_EP_DEBUG: enabling PERF
  // forces a hipStreamSynchronize on every inference (so hipEventElapsedTime
  // can sample the H2D / Compute / D2H phases), which serializes the GPU
  // pipeline and skews measurements. Users who only want the per-call
  // [Runtime DEBUG] traces should not pay that cost.
  // ISOLATE implies PERF.
  static const bool enabled = [] {
#ifdef _WIN32
    if (detail::check_env("HIPDNN_EP_PERF"))
      return true;
    return detail::check_env("HIPDNN_EP_PERF_ISOLATE");
#else
    const char *v = std::getenv("HIPDNN_EP_PERF");
    if (v && v[0] >= '1')
      return true;
    const char *v2 = std::getenv("HIPDNN_EP_PERF_ISOLATE");
    return v2 && v2[0] >= '1';
#endif
  }();
  return enabled;
}

#define RUNTIME_DEBUG_LOG(fmt, ...)                                            \
  do {                                                                         \
    if (hipdnn_ep_debug_enabled())                                             \
      hipdnn_ep_log_emit(fmt, ##__VA_ARGS__);                                  \
  } while (0)

// Conditional fprintf to stderr, gated on HIPDNN_EP_PERF only (DEBUG does
// not enable PERF; see hipdnn_ep_perf_enabled() above for why). Used by the
// per-inference [PERF] timing breakdown emitted from
// hipdnn_ep_runtime_tensor.cpp. Arguments are only evaluated when PERF is
// enabled, so leaving HIPDNN_EP_PERF unset has zero overhead.
#define RUNTIME_PERF_LOG(fmt, ...)                                             \
  do {                                                                         \
    if (hipdnn_ep_perf_enabled())                                              \
      hipdnn_ep_log_emit(fmt, ##__VA_ARGS__);                                  \
  } while (0)
