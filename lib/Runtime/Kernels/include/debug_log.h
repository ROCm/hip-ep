// Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

#pragma once
// Custom-kernels debug logging gated on HIPDNN_EP_DEBUG env var (default: off)
// Set HIPDNN_EP_DEBUG=1 to enable all [custom_kernels] diagnostic output.
#include <cstdio>
#include <cstdlib>

#ifdef _WIN32
// Static CRT (/MT) DLLs have their own CRT env — _dupenv_s can't see env vars
// set by the host process. Use Win32 API to read the real process environment.
extern "C" __declspec(dllimport) unsigned long __stdcall
    GetEnvironmentVariableA(const char*, char*, unsigned long);
#endif

inline bool custom_kernels_debug_enabled() {
    static const bool enabled = [] {
#ifdef _WIN32
        char buf[8];
        unsigned long n = GetEnvironmentVariableA("HIPDNN_EP_DEBUG", buf, sizeof(buf));
        return n > 0 && buf[0] >= '1';
#else
        const char* v = getenv("HIPDNN_EP_DEBUG");
        return v && v[0] >= '1';
#endif
    }();
    return enabled;
}

#define CUSTOM_KERNELS_DEBUG_LOG(fmt, ...) \
    do { if (custom_kernels_debug_enabled()) fprintf(stderr, fmt, ##__VA_ARGS__); } while (0)

// MatMulNBits autotune logging, gated on its own HIPDNN_MATMUL_AUTOTUNE_LOG so
// the per-shape tuner candidate timings, the winning config, LUT hits and the
// cached selections can be captured on their own -- HIPDNN_MATMUL_LUT_LOG only
// covers the offline-table lookup (which AUTOTUNE_MODE=online bypasses), and
// HIPDNN_EP_DEBUG pulls in the full [custom_kernels] firehose. The full-debug
// switch still enables these lines for backward compatibility.
inline bool matmul_nbits_autotune_log_enabled() {
    static const bool enabled = [] {
#ifdef _WIN32
        char buf[8];
        unsigned long n =
            GetEnvironmentVariableA("HIPDNN_MATMUL_AUTOTUNE_LOG", buf, sizeof(buf));
        return n > 0 && buf[0] >= '1';
#else
        const char* v = getenv("HIPDNN_MATMUL_AUTOTUNE_LOG");
        return v && v[0] >= '1';
#endif
    }();
    return enabled;
}

#define MATMUL_NBITS_AUTOTUNE_LOG(fmt, ...)                                     \
    do {                                                                       \
        if (matmul_nbits_autotune_log_enabled() || custom_kernels_debug_enabled()) \
            fprintf(stderr, fmt, ##__VA_ARGS__);                              \
    } while (0)
