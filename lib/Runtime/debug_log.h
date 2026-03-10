// Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

#pragma once
// Runtime debug logging gated on HIPDNN_EP_DEBUG env var (default: off)
// Set HIPDNN_EP_DEBUG=1 to enable all [Runtime DEBUG] output.
#include <cstdio>
#include <cstdlib>

inline bool hipdnn_ep_debug_enabled() {
    static const bool enabled = [] {
        const char* v = getenv("HIPDNN_EP_DEBUG");
        return v && v[0] >= '1';
    }();
    return enabled;
}

#define RUNTIME_DEBUG_LOG(fmt, ...) \
    do { if (hipdnn_ep_debug_enabled()) fprintf(stderr, fmt, ##__VA_ARGS__); } while (0)
