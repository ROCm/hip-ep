/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once
// Profiling-only trace shim; see trace_shim.cpp. Declared only for builds with
// HIPDNN_EP_ENABLE_TRACE_SHIM=ON (HIPDNN_EP_TRACE_SHIM=1 on hipdnn-ep-tls).

#ifdef __cplusplus
extern "C" {
#endif

// Nonzero when the shim is collecting (HIPEP_TRACE attached to a collector
// ring) or a fence is configured (HIPEP_TRACE_FENCE).
int hipdnn_ep_trace_shim_active(void);

#ifdef __cplusplus
}
#endif
